"""크롭 jpg + 메타 → 단일 Parquet 통합.

신규 흐름의 Phase 0.5 (extract → pack → label):
  - extract_crops_for_labeling.py가 jpg 파일 + crops_meta.jsonl 생성
  - 이 스크립트가 둘을 묶어 단일 crops.parquet 생성
  - crop_review_gui.py가 crops.parquet 직접 로드

Parquet 스키마:
  crop_id, ic, video, video_category, tracker_id, frame, bbox,
  current_class, current_conf, is_active_sample,
  image_bytes (binary), image_w, image_h,
  manual_class, low_confidence, reviewer, reviewed_at  ← 라벨링 컬럼 (초기 NULL)

사용:
  python3 pack_crops_to_parquet.py
  python3 pack_crops_to_parquet.py --meta /path/to/crops_meta.jsonl --out crops.parquet
  python3 pack_crops_to_parquet.py --delete-jpg  # jpg 원본 삭제 (주의)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ── 경로 ──────────────────────────────────────────────────────────────
LABELING_DIR = Path("/workspace/prj/cctv/pipeline/data/labeling_v1")
DEFAULT_META = LABELING_DIR / "crops_meta.jsonl"
DEFAULT_OUTPUT = LABELING_DIR / "crops.parquet"


# ── 스키마 ────────────────────────────────────────────────────────────
SCHEMA = pa.schema([
    # 식별
    ("crop_id",          pa.int64()),
    ("crop_path_orig",   pa.string()),

    # 메타
    ("ic",               pa.string()),
    ("video",            pa.string()),
    ("video_path",       pa.string()),
    ("video_category",   pa.string()),
    ("tracker_id",       pa.int32()),
    ("frame",            pa.int32()),
    ("bbox",             pa.list_(pa.int32())),

    # Triple V2 예측
    ("current_class",    pa.string()),
    ("current_conf",     pa.float32()),
    ("is_active_sample", pa.bool_()),

    # 이미지 (바이너리)
    ("image_bytes",      pa.binary()),
    ("image_w",          pa.int16()),
    ("image_h",          pa.int16()),

    # Claude 사전 라벨링 (초기 NULL)
    ("claude_class",     pa.string()),
    ("claude_conf",      pa.float32()),
    ("claude_reason",    pa.string()),
    ("claude_labeled_at", pa.timestamp("ms")),

    # 사용자 라벨링 (OX 검수, 초기 NULL)
    ("manual_class",     pa.string()),
    ("low_confidence",   pa.bool_()),
    ("user_decision",    pa.string()),  # "O" (Claude 채택) / "X" (직접 수정)
    ("reviewer",         pa.string()),
    ("reviewed_at",      pa.timestamp("ms")),
])


def load_metas(meta_path: Path) -> list[dict]:
    """crops_meta.jsonl 로드."""
    metas = []
    with meta_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                metas.append(json.loads(line))
    return metas


def read_jpg_as_bytes(path: Path) -> tuple[bytes, int, int]:
    """JPG 파일 읽기 → (bytes, width, height)."""
    with open(path, "rb") as f:
        img_bytes = f.read()
    # 크기는 cv2로 디코딩하여 확인 (검증 + width/height 추출)
    img = cv2.imdecode(
        memoryview(img_bytes).tobytes() and __import__("numpy").frombuffer(img_bytes, dtype=__import__("numpy").uint8),
        cv2.IMREAD_COLOR,
    )
    if img is None:
        raise ValueError(f"디코딩 실패: {path}")
    h, w = img.shape[:2]
    return img_bytes, w, h


def build_parquet(meta_path: Path, output_path: Path, delete_jpg: bool = False) -> int:
    """crops_meta.jsonl + jpg 파일 → 단일 Parquet."""
    metas = load_metas(meta_path)
    logger.info("메타 로드: %d 크롭", len(metas))

    if not metas:
        logger.warning("메타 비어있음 — 스킵")
        return 0

    rows = []
    n_skipped = 0
    for i, m in enumerate(metas):
        if i % 200 == 0 and i > 0:
            logger.info("  처리 %d / %d", i, len(metas))

        crop_path = Path(m["crop_path"])
        if not crop_path.exists():
            logger.warning("파일 없음 (skip): %s", crop_path)
            n_skipped += 1
            continue

        try:
            img_bytes, w, h = read_jpg_as_bytes(crop_path)
        except Exception as e:
            logger.warning("읽기 실패 (skip): %s — %s", crop_path, e)
            n_skipped += 1
            continue

        rows.append({
            "crop_id":          i,
            "crop_path_orig":   str(crop_path),
            "ic":               m.get("ic", ""),
            "video":            m.get("video", ""),
            "video_path":       m.get("video_path", ""),
            "video_category":   m.get("video_category", ""),
            "tracker_id":       int(m.get("tracker_id", -1)),
            "frame":            int(m.get("frame", -1)),
            "bbox":             [int(x) for x in m.get("bbox", [0, 0, 0, 0])],
            "current_class":    m.get("current_class", ""),
            "current_conf":     float(m.get("current_conf", 0.0)),
            "is_active_sample": bool(m.get("is_active_sample", False)),
            "image_bytes":      img_bytes,
            "image_w":          w,
            "image_h":          h,
            "claude_class":     None,
            "claude_conf":      None,
            "claude_reason":    None,
            "claude_labeled_at": None,
            "manual_class":     None,
            "low_confidence":   None,
            "user_decision":    None,
            "reviewer":         None,
            "reviewed_at":      None,
        })

    logger.info("Parquet 작성 중... (%d 행, skip %d)", len(rows), n_skipped)
    table = pa.Table.from_pylist(rows, schema=SCHEMA)
    pq.write_table(table, output_path, compression="zstd", compression_level=3)

    file_size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info("✓ 저장: %s (%.1f MB)", output_path, file_size_mb)

    # 검증
    verify = pq.read_metadata(output_path)
    logger.info("검증 — Parquet 행 수: %d (기대 %d)", verify.num_rows, len(rows))

    # JPG 삭제 (옵션)
    if delete_jpg and len(rows) > 0:
        logger.info("jpg 원본 삭제 중...")
        n_deleted = 0
        for r in rows:
            p = Path(r["crop_path_orig"])
            if p.exists():
                p.unlink()
                n_deleted += 1
        logger.info("✓ jpg 삭제: %d개", n_deleted)

    return len(rows)


def show_summary(parquet_path: Path) -> None:
    """Parquet 통계 출력."""
    if not parquet_path.exists():
        logger.warning("Parquet 없음: %s", parquet_path)
        return

    table = pq.read_table(parquet_path, columns=[
        "ic", "video_category", "current_class", "current_conf", "is_active_sample"
    ])
    df = table.to_pandas()

    logger.info("=" * 60)
    logger.info("Parquet 요약: %s", parquet_path)
    logger.info("  총 크롭: %d", len(df))
    logger.info("  파일 크기: %.1f MB", parquet_path.stat().st_size / 1024 / 1024)
    logger.info("  IC 수: %d", df["ic"].nunique())
    logger.info("  카테고리:")
    for cat, n in df["video_category"].value_counts().items():
        logger.info("    %s: %d", cat, n)
    logger.info("  현재 예측 분포:")
    for cls, n in df["current_class"].value_counts().items():
        logger.info("    %s: %d", cls, n)
    logger.info("  Active sample (low conf): %d (%.1f%%)",
                df["is_active_sample"].sum(),
                df["is_active_sample"].mean() * 100)
    logger.info("  평균 conf: %.3f", df["current_conf"].mean())
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", default=str(DEFAULT_META))
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--delete-jpg", action="store_true",
                        help="Parquet 생성 후 원본 jpg 파일 삭제 (주의)")
    parser.add_argument("--summary-only", action="store_true",
                        help="기존 Parquet 통계만 출력")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.summary_only:
        show_summary(out_path)
    else:
        n = build_parquet(Path(args.meta), out_path, args.delete_jpg)
        logger.info("✓ 완료: %d 크롭 packed", n)
        show_summary(out_path)
