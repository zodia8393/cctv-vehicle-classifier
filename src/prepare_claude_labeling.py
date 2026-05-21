"""Claude 사전 라벨링 batch 분할.

Phase A 신규 흐름:
  pack → [prepare] → Claude 위임 → 머지 → GUI OX 검수

기능:
  1. crops.parquet에서 라벨 대상 N장 선정 (active 우선)
  2. 50장/batch 분할 → 각 batch별 jpg 디스크 export
  3. batch_index.json 생성 (subagent 위임용)

전략 (회의 결론, 2026-04-27):
  - active 우선: night + low_conf
  - 50장/batch (응답 토큰 8K 안정)
  - few-shot 앵커: T1/T3/T4/T5 각 3장 GT (헤더 고정)

출력:
  /pipeline/data/labeling_v1/claude_batches/
    batch_000/  *.jpg + meta.json
    batch_001/
    ...
    batch_index.json (전체 인덱스)

사용:
  python3 prepare_claude_labeling.py
  python3 prepare_claude_labeling.py --max 1000 --batch-size 50
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ── 경로 ──────────────────────────────────────────────────────────────
LABELING_DIR = Path("/workspace/prj_cctv/pipeline/data/labeling_v1")
DEFAULT_PARQUET = LABELING_DIR / "crops.parquet"
BATCH_DIR = LABELING_DIR / "claude_batches"
ANCHOR_SOURCE = Path("/workspace/prj_cctv/pipeline/data/final_dataset/train")  # GT 학습 데이터


# ── Few-shot 앵커 선정 (final_dataset에서 GT 가져오기) ──────────────
def find_anchor_crops(target_classes=("T1", "T2", "T3", "T4", "T5", "T10", "T13"),
                      per_class=3) -> dict[str, list[Path]]:
    """기존 GT 학습 데이터에서 클래스별 앵커 크롭 N장씩 선정."""
    anchors = {}
    for cls in target_classes:
        cls_dir = ANCHOR_SOURCE / cls
        if not cls_dir.exists():
            logger.warning("클래스 디렉토리 없음: %s", cls_dir)
            anchors[cls] = []
            continue
        jpgs = sorted(cls_dir.glob("*.jpg"))[:per_class]
        anchors[cls] = jpgs
        logger.info("앵커 %s: %d장 선정 (%s)", cls, len(jpgs), cls_dir.name)
    return anchors


# ── batch 분할 ───────────────────────────────────────────────────────
def select_active_samples(parquet_path: Path, max_n: int,
                          min_conf: float = 0.0,
                          mode: str = "active") -> list[dict]:
    """샘플링.

    mode='active': low_conf 우선 (능동학습)
    mode='high_conf': current_conf >= min_conf만, conf 내림차순
    """
    import pyarrow.parquet as pq
    table = pq.read_table(parquet_path, columns=[
        "crop_id", "ic", "video", "video_category",
        "tracker_id", "frame", "bbox",
        "current_class", "current_conf", "is_active_sample",
        "image_w", "image_h",
    ])
    rows = table.to_pylist()
    total = len(rows)

    # min_conf 필터
    if min_conf > 0:
        rows = [r for r in rows if r.get("current_conf", 0) >= min_conf]
        logger.info("conf >= %.2f 필터: %d / %d", min_conf, len(rows), total)

    if mode == "high_conf":
        # 카테고리 균형 유지하면서 conf 높은 순
        def priority(r: dict) -> tuple:
            cat_order = {"night": 0, "backlight": 1, "day": 2}.get(r.get("video_category"), 3)
            return (cat_order, -r.get("current_conf", 0))
    else:  # active
        def priority(r: dict) -> tuple:
            cat_order = {"night": 0, "backlight": 1, "day": 2}.get(r.get("video_category"), 3)
            is_active = 0 if r.get("is_active_sample") else 1
            return (cat_order, is_active, r.get("current_conf", 1.0))

    sorted_rows = sorted(rows, key=priority)
    selected = sorted_rows[:max_n]
    logger.info("선정: %d / 후보 %d", len(selected), len(rows))
    return selected


def export_batch(parquet_path: Path, batch_id: int, crop_ids: list[int],
                 batch_dir: Path) -> dict:
    """선정된 crop_ids의 image_bytes를 jpg로 디스크에 저장 + meta.json."""
    import pyarrow.parquet as pq
    import pyarrow.compute as pc

    batch_dir.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(parquet_path)
    # crop_id 필터
    mask = pc.is_in(table.column("crop_id"), value_set=pa_array(crop_ids))
    sub = table.filter(mask)

    metas = []
    for i in range(len(sub)):
        row = sub.slice(i, 1).to_pylist()[0]
        crop_id = row["crop_id"]
        img_bytes = row["image_bytes"]
        # jpg 저장
        jpg_path = batch_dir / f"crop_{crop_id:06d}.jpg"
        jpg_path.write_bytes(img_bytes)
        metas.append({
            "crop_id":         crop_id,
            "jpg_path":        str(jpg_path),
            "ic":              row["ic"],
            "video":           row["video"],
            "video_category":  row["video_category"],
            "current_class":   row["current_class"],
            "current_conf":    float(row["current_conf"]),
        })

    meta_path = batch_dir / "meta.json"
    meta_path.write_text(json.dumps({
        "batch_id": batch_id,
        "n_crops":  len(metas),
        "crops":    metas,
    }, ensure_ascii=False, indent=2))

    logger.info("[batch_%03d] %d 크롭 export → %s", batch_id, len(metas), batch_dir.name)
    return {"batch_id": batch_id, "n_crops": len(metas), "dir": str(batch_dir)}


def pa_array(values):
    """list → pyarrow array helper."""
    import pyarrow as pa
    return pa.array(values)


# ── 메인 ──────────────────────────────────────────────────────────────
def main(args):
    parquet_path = Path(args.parquet)
    if not parquet_path.exists():
        logger.error("Parquet 없음: %s", parquet_path)
        return

    BATCH_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 앵커 선정 (few-shot)
    logger.info("=== Few-shot 앵커 선정 ===")
    anchors = find_anchor_crops()
    anchor_dir = BATCH_DIR / "anchors"
    anchor_dir.mkdir(exist_ok=True)
    anchor_paths = {}
    for cls, paths in anchors.items():
        cls_dir = anchor_dir / cls
        cls_dir.mkdir(exist_ok=True)
        anchor_paths[cls] = []
        for i, p in enumerate(paths):
            dst = cls_dir / f"{cls}_{i}.jpg"
            if not dst.exists() and p.exists():
                dst.write_bytes(p.read_bytes())
            anchor_paths[cls].append(str(dst))

    # 2. 샘플 선정 (mode/min_conf 적용)
    logger.info("=== 샘플 선정 (mode=%s, min_conf=%.2f) ===", args.mode, args.min_conf)
    selected = select_active_samples(parquet_path, args.max,
                                     min_conf=args.min_conf, mode=args.mode)
    crop_ids = [r["crop_id"] for r in selected]

    # 3. batch 분할
    logger.info("=== Batch 분할 (%d장 / %d batch) ===",
                args.batch_size, (len(crop_ids) + args.batch_size - 1) // args.batch_size)
    batches = []
    for batch_id, start in enumerate(range(0, len(crop_ids), args.batch_size)):
        batch_crop_ids = crop_ids[start:start + args.batch_size]
        batch_dir = BATCH_DIR / f"batch_{batch_id:03d}"
        info = export_batch(parquet_path, batch_id, batch_crop_ids, batch_dir)
        batches.append(info)

    # 4. batch_index.json
    index = {
        "parquet_path":    str(parquet_path),
        "anchor_paths":    anchor_paths,
        "n_total":         len(crop_ids),
        "batch_size":      args.batch_size,
        "n_batches":       len(batches),
        "batches":         batches,
    }
    index_path = BATCH_DIR / "batch_index.json"
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2))
    logger.info("✓ batch_index.json: %s", index_path)

    # 5. 요약
    logger.info("=" * 60)
    logger.info("Batch 분할 완료:")
    logger.info("  앵커 총 %d장 (T1/T3/T4/T5 각 3장)",
                sum(len(v) for v in anchor_paths.values()))
    logger.info("  대상 크롭 %d장 → %d batch (50장씩)", len(crop_ids), len(batches))
    logger.info("  출력: %s", BATCH_DIR)
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", default=str(DEFAULT_PARQUET))
    parser.add_argument("--max", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--min-conf", type=float, default=0.0,
                        help="current_conf 최소 임계값 (예: 0.7)")
    parser.add_argument("--mode", choices=["active", "high_conf"], default="active",
                        help="active=low_conf 우선 / high_conf=conf 높은 순")
    args = parser.parse_args()
    main(args)
