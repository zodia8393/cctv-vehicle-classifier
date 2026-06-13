"""Claude subagent를 통한 사전 라벨링 실행기.

Phase A:
  prepare → [runner] → 머지 → GUI OX 검수

흐름:
  1. batch_index.json 로드 (20 batch)
  2. 메인 세션이 subagent에 batch 위임 (50장씩)
  3. subagent가 labels_b{N}.jsonl 직접 write
  4. 모든 batch 완료 후 → crops.parquet에 머지

이 스크립트는 batch 위임 정의서(prompt 템플릿)를 출력하여,
메인 Claude Code 세션에서 Agent 도구로 호출 시 사용.

사용:
  # 1. prompt 템플릿 출력 (메인이 Agent 호출에 사용)
  python3 claude_label_runner.py --print-prompt 0

  # 2. 결과 머지 (subagent들이 jsonl 작성한 후)
  python3 claude_label_runner.py --merge

  # 3. 진행 상황 확인
  python3 claude_label_runner.py --status
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ── 경로 ──────────────────────────────────────────────────────────────
LABELING_DIR = Path("/workspace/prj/cctv/pipeline/data/labeling_v1")
BATCH_DIR = LABELING_DIR / "claude_batches"
BATCH_INDEX = BATCH_DIR / "batch_index.json"
PARQUET_PATH = LABELING_DIR / "crops.parquet"


# ── Prompt 템플릿 ────────────────────────────────────────────────────
PROMPT_TEMPLATE = """\
당신은 한국 도로 CCTV 차종 분류 전문가입니다. 다음 batch의 차량 크롭 이미지를 분류해주세요.

## 분류 옵션 (정확히 하나 선택)

| 코드 | 설명 | 핵심 식별 |
|------|------|---------|
| T1 | 승용차 | 세단/SUV/미니밴, 4도어 ≤5인승, 적재함 없음 |
| T2 | 버스 | 길고 큰 객실, 다수 창문, 12m+ |
| T3 | 소형화물 | 1톤 트럭 (포터, 봉고), 좁은 적재함, 차폭 좁음 |
| T4 | 중형화물 | 2.5~5톤 트럭, 적재함 큼, 2축 |
| T5 | 대형화물 | 8톤+, 적재함 매우 큼, 3축 |
| T10 | 세미트레일러 | 트랙터+트레일러 분리형, 5축+ |
| T13 | 이륜차 | 오토바이, 스쿠터 |
| NOISE | 식별 불가 | bbox<32px, 50%+ 가림, 야간 흐림 |

## 결정 규칙 (헷갈릴 때)

- **T4 vs T5**: 축수로 구분 (T4=2축, T5=3축+)
- **T3 vs T1**: 차폭 좁음 + 후면 적재함 보이면 T3
- **야간 흐림 + conf<0.5**: NOISE 자동
- **헤드라이트만 보임 + 차체 흡수**: NOISE

## Few-shot 앵커 (참고)

각 클래스 GT 예시 이미지를 먼저 Read 도구로 확인하세요:
{anchor_paths}

## 작업

다음 50장 크롭을 분류하세요. 각 이미지를 Read 도구로 본 뒤,
**정확히 아래 JSON Lines 형식으로 한 줄씩** 출력하세요:

{{"crop_id": <int>, "claude_class": "T1|T2|T3|T4|T5|T10|T13|NOISE", "claude_conf": <0.0~1.0>, "claude_reason": "<한 줄 근거>"}}

## 분류 대상 이미지 (50장)

{crop_list}

## 출력 형식 (반드시 이것만)

위에서 본 50장에 대한 JSONL을 다음 파일에 저장:
**`{output_path}`**

(Bash로 `cat > {output_path} << 'EOF' ... EOF` 또는 Python으로 직접 write)

50장 모두 분류 후 끝내세요. 부가 설명 X.
"""


def build_prompt(batch_id: int) -> str:
    """batch_id에 대한 subagent prompt 생성."""
    if not BATCH_INDEX.exists():
        raise FileNotFoundError(f"batch_index.json 없음: {BATCH_INDEX}")
    index = json.loads(BATCH_INDEX.read_text())

    if batch_id >= len(index["batches"]):
        raise ValueError(f"batch_id {batch_id} 초과 (최대 {len(index['batches']) - 1})")

    batch_info = index["batches"][batch_id]
    batch_dir = Path(batch_info["dir"])
    meta_path = batch_dir / "meta.json"
    meta = json.loads(meta_path.read_text())

    # 앵커 경로 포맷
    anchor_lines = []
    for cls, paths in index["anchor_paths"].items():
        for p in paths:
            anchor_lines.append(f"  - [{cls}] {p}")
    anchor_paths_str = "\n".join(anchor_lines)

    # 크롭 리스트 포맷
    crop_lines = []
    for c in meta["crops"]:
        crop_lines.append(
            f"  - crop_id={c['crop_id']:>6}  "
            f"path={c['jpg_path']}  "
            f"current={c['current_class']} (conf={c['current_conf']:.2f})  "
            f"category={c['video_category']}"
        )
    crop_list_str = "\n".join(crop_lines)

    output_path = batch_dir / f"labels_b{batch_id:03d}.jsonl"

    return PROMPT_TEMPLATE.format(
        anchor_paths=anchor_paths_str,
        crop_list=crop_list_str,
        output_path=output_path,
    )


def status() -> None:
    """진행 상황 출력."""
    if not BATCH_INDEX.exists():
        logger.error("batch_index.json 없음 — prepare 먼저 실행")
        return
    index = json.loads(BATCH_INDEX.read_text())

    completed = []
    pending = []
    for b in index["batches"]:
        bdir = Path(b["dir"])
        label_file = bdir / f"labels_b{b['batch_id']:03d}.jsonl"
        if label_file.exists():
            n_lines = sum(1 for _ in label_file.open())
            completed.append((b["batch_id"], n_lines))
        else:
            pending.append(b["batch_id"])

    logger.info("=" * 60)
    logger.info("Claude 라벨링 진행 상황")
    logger.info("  완료: %d / %d batch", len(completed), len(index["batches"]))
    for bid, n in completed:
        logger.info("    [batch_%03d] %d 라벨", bid, n)
    if pending:
        logger.info("  대기: %d batch", len(pending))
        logger.info("    %s", pending[:10])
    logger.info("=" * 60)


def merge_to_parquet() -> int:
    """모든 batch의 labels_b*.jsonl → crops.parquet에 머지."""
    if not PARQUET_PATH.exists():
        logger.error("Parquet 없음: %s", PARQUET_PATH)
        return 0

    import pyarrow as pa
    import pyarrow.parquet as pq
    import pandas as pd

    if not BATCH_INDEX.exists():
        logger.error("batch_index.json 없음")
        return 0
    index = json.loads(BATCH_INDEX.read_text())

    # 모든 라벨 수집
    all_labels: dict[int, dict] = {}
    for b in index["batches"]:
        label_file = Path(b["dir"]) / f"labels_b{b['batch_id']:03d}.jsonl"
        if not label_file.exists():
            logger.warning("[batch_%03d] 라벨 파일 없음 (skip)", b["batch_id"])
            continue
        with label_file.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    all_labels[rec["crop_id"]] = rec
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning("[batch_%03d] 라인 무시: %s", b["batch_id"], e)

    logger.info("총 라벨 수집: %d개", len(all_labels))
    if not all_labels:
        return 0

    # Parquet 로드 + 머지
    table = pq.read_table(PARQUET_PATH)
    df = table.to_pandas()

    # claude_labeled_at 컬럼을 datetime64[ms]로 사전 캐스팅
    if "claude_labeled_at" in df.columns:
        df["claude_labeled_at"] = df["claude_labeled_at"].astype("datetime64[ms]")
    if "claude_class" in df.columns:
        df["claude_class"] = df["claude_class"].astype("object")
    if "claude_reason" in df.columns:
        df["claude_reason"] = df["claude_reason"].astype("object")

    n_updated = 0
    now = pd.Timestamp.now().floor("ms")
    for crop_id, label in all_labels.items():
        mask = df["crop_id"] == crop_id
        if not mask.any():
            continue
        df.loc[mask, "claude_class"] = label.get("claude_class")
        df.loc[mask, "claude_conf"] = float(label.get("claude_conf", 0.5))
        df.loc[mask, "claude_reason"] = label.get("claude_reason", "")
        df.loc[mask, "claude_labeled_at"] = now
        n_updated += 1

    # 다시 쓰기
    new_table = pa.Table.from_pandas(df, schema=table.schema, preserve_index=False)
    pq.write_table(new_table, PARQUET_PATH,
                   compression="zstd", compression_level=3)
    logger.info("✓ Parquet 머지: %d 라벨 업데이트", n_updated)
    return n_updated


# ── 메인 ──────────────────────────────────────────────────────────────
def main(args):
    if args.status:
        status()
    elif args.merge:
        merge_to_parquet()
    elif args.print_prompt is not None:
        prompt = build_prompt(args.print_prompt)
        print(prompt)
    else:
        # 기본: status 표시
        status()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-prompt", type=int, default=None,
                        help="batch_id의 subagent prompt 출력")
    parser.add_argument("--merge", action="store_true",
                        help="모든 라벨 jsonl → Parquet 머지")
    parser.add_argument("--status", action="store_true",
                        help="진행 상황 출력")
    args = parser.parse_args()
    main(args)
