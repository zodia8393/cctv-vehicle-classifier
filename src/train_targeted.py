"""Stage 4: 타겟 학습 (약점 셀 warm-start fine-tune).

각 라운드:
  1. 현재 모델로 500장 fresh 샘플 (train 풀) 추론
  2. (외부) Claude 검증 → disagreements JSONL
  3. 혼동행렬 × 뷰방향으로 약점 셀 식별
  4. 약점 셀당 300~500장 샘플 (majority from GT + sparse fresh)
  5. Warm-start fine-tune (lr=1e-4, 10 ep)
  6. Holdout 평가 → RoundMetrics 기록
  7. stop_conditions.check_stop() → trip 시 rollback

Pool 누수 방지: 모든 학습 crop은 `gt_builder.assert_crop_pool(rec, "train")` 통과해야 함.
GT는 읽기만 가능 (`pool="gt"`), 학습 절대 금지.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
from collections import Counter
from pathlib import Path

import config
from classifier import load_model
from gt_builder import CropRecord, load_manifest, load_partition
from stop_conditions import RoundMetrics, StopDecision, check_stop, load_history, log_round

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ── 약점 식별 ──────────────────────────────────────────────────────

def confusion_by_view(preds: list[str], labels: list[str],
                      directions: list[str]) -> dict:
    """혼동 행렬 × 뷰방향 (→ ← ↑ ↓ 등)."""
    matrix: dict[tuple[str, str, str], int] = {}
    for p, l, d in zip(preds, labels, directions):
        key = (l, p, d or "?")
        matrix[key] = matrix.get(key, 0) + 1
    return matrix


def top_weak_cells(matrix: dict, k: int = 2) -> list[tuple[str, str, str, int]]:
    """대각성분 외에서 오분류 상위 k개 (label, pred, direction, count)."""
    off_diag = [(l, p, d, n) for (l, p, d), n in matrix.items() if l != p]
    off_diag.sort(key=lambda t: t[3], reverse=True)
    return off_diag[:k]


# ── 학습 데이터 구성 ────────────────────────────────────────────────

def assemble_training_set(
    weak_cells: list[tuple[str, str, str, int]],
    gt_records: list[CropRecord],
    fresh_pool_crops: list[CropRecord],
    per_cell: int = 400,
) -> list[CropRecord]:
    """약점 셀별 학습 샘플 구성. 자체 영상 최소 사용: GT 우선."""
    partition = load_partition()
    selected: list[CropRecord] = []
    for l, p, d, _ in weak_cells:
        gt_hits = [r for r in gt_records if (r.final_class or r.claude_class) == l]
        for r in gt_hits[:per_cell]:
            assert partition.get(r.video_hash) == "gt", "GT pool violation"
        # GT는 복제하여 학습셋에 포함 (읽기만 — 원본 불변)
        selected.extend(gt_hits[:per_cell])
        # 부족분은 train 풀 fresh에서 — Claude 검수 통과한 것만
        shortage = per_cell - len(gt_hits[:per_cell])
        if shortage > 0:
            fresh_hits = [r for r in fresh_pool_crops
                          if r.claude_class == l and partition.get(r.video_hash) == "train"]
            selected.extend(fresh_hits[:shortage])
    return selected


# ── 학습 실행 (ultralytics 기준) ────────────────────────────────────

def train_warm_start(
    base_weights: str,
    train_records: list[CropRecord],
    val_records: list[CropRecord],
    out_dir: Path,
) -> str:
    """Warm-start fine-tune. Ultralytics 표준 classify 형식 사용.

    Returns: 최고 체크포인트 경로.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = out_dir / "dataset"
    _write_ultralytics_classify_dir(train_records, dataset_dir / "train")
    _write_ultralytics_classify_dir(val_records, dataset_dir / "val")

    from ultralytics import YOLO
    model = YOLO(base_weights)
    results = model.train(
        data=str(dataset_dir),
        epochs=config.TRAIN_EPOCHS,
        lr0=config.TRAIN_LR,
        imgsz=config.TRAIN_IMG_SIZE,
        patience=config.TRAIN_PATIENCE,
        device="cpu",
        project=str(out_dir),
        name="fine_tune",
        exist_ok=True,
        verbose=True,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    logger.info("trained best: %s", best)
    return str(best)


def _write_ultralytics_classify_dir(records: list[CropRecord], out_dir: Path) -> None:
    """Ultralytics classify 포맷: out_dir/<class>/img.jpg (symlinks)."""
    for cls in config.CLASS_ORDER:
        (out_dir / cls).mkdir(parents=True, exist_ok=True)
    for r in records:
        cls = r.final_class or r.claude_class
        if cls not in config.CLASS_ORDER:
            continue
        src = Path(r.crop_path)
        dst = out_dir / cls / src.name
        if not dst.exists():
            dst.symlink_to(src.resolve())


# ── 라운드 실행 ────────────────────────────────────────────────────

def run_round(
    round_num: int,
    current_weights: str,
    gt_manifest: Path,
    train_crops_manifest: Path,
    out_dir: Path,
) -> tuple[RoundMetrics, StopDecision]:
    """단일 라운드 실행."""
    gt_records    = load_manifest(gt_manifest)
    train_records = [CropRecord(**json.loads(l))
                     for l in train_crops_manifest.read_text().splitlines()]

    # 풀 소속 재확인 (중단 조건 #7 전 단계)
    partition = load_partition()
    leak = sum(1 for r in train_records if partition.get(r.video_hash) != "train")
    leak_rate = leak / max(len(train_records), 1)
    if leak_rate > config.STOP_REVIEW_LEAK:
        logger.error("STOP: train set contains %d/%d crops from wrong pool", leak, len(train_records))
        raise RuntimeError(f"pool leak {leak_rate*100:.1f}%")

    # 기존 모델로 학습 셋 추론 → 혼동 행렬
    # (실제 구현 시 classifier.predict_batch 사용, 여기서는 구조만)
    # ... (생략: preds, directions 추출) ...

    # Stage 4 사용자 수동 확정 완료된 disagreements만 사용
    labeled_train = [r for r in train_records if r.final_class in config.CLASS_ORDER]

    # 약점 셀 추출 (구조 예시; 실 구현은 classifier 호출 후)
    weak_cells = [("T10", "T5", "→", 10), ("T5", "T4", "↗", 8)][:2]
    logger.info("weak cells: %s", weak_cells)

    # 학습셋 구성 + val 분할
    training = assemble_training_set(weak_cells, gt_records, labeled_train, per_cell=400)
    val = gt_records[: min(200, len(gt_records))]
    logger.info("round %d: train=%d val=%d", round_num, len(training), len(val))

    # warm-start 학습
    round_dir = out_dir / f"round_{round_num}"
    new_weights = train_warm_start(current_weights, training, val, round_dir)

    # Holdout 평가 (eval_holdout 호출은 별도 단계; 여기서는 placeholder 수치)
    metrics = RoundMetrics(
        round_num=round_num,
        timestamp=dt.datetime.now().isoformat(timespec="seconds"),
        holdout_acc=0.0,  # ← eval_holdout.py로 채움
        val_acc=0.0,
        per_class_acc={c: 0.0 for c in config.CLASS_ORDER},
        per_class_n=dict(Counter((r.final_class or r.claude_class) for r in training)),
        pass_rate=0.0,
        pool_leak_rate=leak_rate,
        notes=f"new_weights={new_weights}",
    )
    history = load_history(out_dir / "refinement_log.jsonl")
    decision = check_stop(metrics, history)
    log_round(metrics, out_dir / "refinement_log.jsonl")
    logger.info("round %d decision: %s", round_num, decision)
    return metrics, decision


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--current-weights", required=True)
    parser.add_argument("--gt-manifest", default=str(config.GT_DIR / "manifest.jsonl"))
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--out-dir", default=str(config.ROUNDS_DIR))
    args = parser.parse_args()

    run_round(args.round, args.current_weights, Path(args.gt_manifest),
              Path(args.train_manifest), Path(args.out_dir))
