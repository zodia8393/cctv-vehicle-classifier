"""Holdout GT v2 구축 도구.

재설계 파이프라인의 핵심 — 학습 루프와 **완전 분리된** 홀드아웃 GT를
수동 검수 기반으로 구축한다. 영상을 GT 풀(600)과 학습 풀(1,981)로
해시 기반 결정론적 분할하여 누수를 사전 차단.

Subcommands:
    partition   — 전체 영상을 GT/학습 풀로 분할, SHA256 봉인
    sample      — GT 풀에서 crop 추출 (detector.py 사용)
    grid        — 4×4 그리드 이미지 조립 (Claude 검증용)
    review      — manifest에 사용자 수동 확정 기록
    seal        — 최종 manifest SHA256 봉인
    check       — crop 파일의 영상 풀 소속 검증 (학습측 훅)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ── 영상 해시 / 풀 분할 ─────────────────────────────────────────────

HASH_BYTES = 1 << 20  # 1 MiB — 영상 식별에 충분


def video_hash(path: Path) -> str:
    """영상 파일 식별용 SHA256 (앞 1MB).

    전체 해시는 GB 단위 파일에 과비용. 앞 1MB로도 영상간 충돌 없음.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        h.update(f.read(HASH_BYTES))
    return h.hexdigest()


def discover_videos(root: Path = config.VIDEO_SOURCE_DIR) -> list[Path]:
    """원본 디렉토리의 모든 AVI/MP4 수집 (정렬 결정적)."""
    paths = sorted([p for p in root.rglob("*") if p.suffix.lower() in {".avi", ".mp4"}])
    logger.info("discovered %d videos under %s", len(paths), root)
    return paths


def partition_videos(
    videos: list[Path],
    gt_pool_size: int = config.GT_POOL_SIZE,
    seed: int = config.PARTITION_SEED,
) -> tuple[dict[str, str], dict[str, str]]:
    """해시 기반 결정론적 분할.

    Returns
    -------
    (gt_pool, train_pool): 각각 {video_hash: video_path_str}
    """
    items = [(video_hash(p), str(p)) for p in videos]
    rng = random.Random(seed)
    rng.shuffle(items)
    gt   = dict(items[:gt_pool_size])
    trn  = dict(items[gt_pool_size:])
    logger.info("partition: GT=%d train=%d (seed=%d)", len(gt), len(trn), seed)
    return gt, trn


def write_partition(gt: dict[str, str], trn: dict[str, str], out: Path) -> str:
    """분할을 JSONL로 기록 + SHA256 봉인 반환."""
    out.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = out.with_suffix(".jsonl")
    with jsonl_path.open("w") as f:
        for h, p in gt.items():
            f.write(json.dumps({"hash": h, "path": p, "pool": "gt"}) + "\n")
        for h, p in trn.items():
            f.write(json.dumps({"hash": h, "path": p, "pool": "train"}) + "\n")
    seal = hashlib.sha256(jsonl_path.read_bytes()).hexdigest()
    out.write_text(f"{seal}  {jsonl_path.name}\n")
    logger.info("partition written: %s (%d total)", jsonl_path, len(gt) + len(trn))
    logger.info("partition sealed: %s = %s", out, seal)
    return seal


def load_partition(partition_file: Path = config.PARTITION_FILE) -> dict[str, str]:
    """Partition 로드 → {video_hash: pool_label}. SHA256 검증 필수."""
    jsonl_path = partition_file.with_suffix(".jsonl")
    expected = partition_file.read_text().split()[0]
    actual   = hashlib.sha256(jsonl_path.read_bytes()).hexdigest()
    if expected != actual:
        raise RuntimeError(f"partition SHA256 mismatch: expected {expected}, got {actual}")
    result: dict[str, str] = {}
    with jsonl_path.open() as f:
        for line in f:
            rec = json.loads(line)
            result[rec["hash"]] = rec["pool"]
    return result


# ── Crop 추출 ──────────────────────────────────────────────────────

@dataclass
class CropRecord:
    """단일 crop 메타데이터."""
    crop_path: str
    video_hash: str
    video_path: str
    pool: str               # "gt" | "train"
    frame_idx: int
    track_id: int | None
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    area: int
    coco_cls: int           # COCO detector class
    det_conf: float
    proposed_class: str | None = None   # 초안 (classifier 또는 COCO_TO_7CLASS)
    claude_class: str | None = None     # 세션 Claude 판정
    final_class: str | None = None      # 사용자 수동 확정
    reviewer: str = ""
    notes: str = ""


def iter_video_frames(
    video_path: Path, stride_seconds: float = 30.0, max_frames: int = 10
) -> Iterator[tuple[int, np.ndarray]]:
    """영상에서 stride 간격으로 프레임 읽기 (최대 max_frames)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("cannot open %s", video_path)
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride_frames = max(1, int(stride_seconds * fps))
    yielded = 0
    frame_idx = 0
    while yielded < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            break
        yield frame_idx, frame
        yielded += 1
        frame_idx += stride_frames
    cap.release()


MIN_CROP_SIDE  = 96   # 96×96 이하 crop은 분류 불가능 경험적 하한
MIN_CROP_ASPECT = 0.3  # 극단적 종횡비 (트레일러 측면 제외) 필터

def extract_crops_from_video(
    video_path: Path, video_h: str, pool: str,
    detector, out_dir: Path,
    stride_seconds: float = 30.0, max_frames: int = 10,
    max_crops_per_video: int = 1,
    stratify_by_coco: bool = True,
) -> list[CropRecord]:
    """한 영상에서 crop 추출.

    "자체 영상 최소 사용" 제약을 반영하여 영상당 기본 1 crop.
    검출 신뢰도 상위 + 면적 상위로 선정.

    stratify_by_coco: COCO 클래스(truck/bus/motorcycle 등)별로 우선 선정,
        T10/T13 등 희귀 클래스 비율 확보.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[tuple[CropRecord, np.ndarray]] = []
    for frame_idx, frame in iter_video_frames(video_path, stride_seconds, max_frames):
        detections = detector.detect_frame(frame)
        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det
            w, h = x2 - x1, y2 - y1
            if w < MIN_CROP_SIDE or h < MIN_CROP_SIDE:
                continue
            aspect = min(w, h) / max(w, h)
            if aspect < MIN_CROP_ASPECT and int(cls_id) != 7:  # truck(=세미트레일러 가능성) 제외
                continue
            area = int(w * h)
            rec = CropRecord(
                crop_path="", video_hash=video_h, video_path=str(video_path),
                pool=pool, frame_idx=frame_idx, track_id=None,
                bbox=(int(x1), int(y1), int(x2), int(y2)), area=area,
                coco_cls=int(cls_id), det_conf=float(conf),
                proposed_class=config.COCO_TO_7CLASS.get(int(cls_id)),
            )
            crop_img = frame[int(y1):int(y2), int(x1):int(x2)].copy()
            candidates.append((rec, crop_img))

    # 희귀 COCO 클래스 우선 (truck=7, bus=5, motorcycle=3) → car(=2)
    if stratify_by_coco:
        def score(t):
            rec, _ = t
            rarity = {3: 3.0, 5: 2.5, 7: 2.0, 2: 1.0}.get(rec.coco_cls, 1.0)
            return rarity * rec.det_conf * rec.area
        candidates.sort(key=score, reverse=True)
    else:
        candidates.sort(key=lambda t: t[0].det_conf * t[0].area, reverse=True)

    saved: list[CropRecord] = []
    for rec, img in candidates[:max_crops_per_video]:
        fname = f"{video_h[:12]}_f{rec.frame_idx:06d}_c{rec.coco_cls}_a{rec.area}.jpg"
        dest = out_dir / fname
        cv2.imwrite(str(dest), img)
        rec.crop_path = str(dest)
        saved.append(rec)
    return saved


# ── Grid 조립 (Claude 검증용) ──────────────────────────────────────

def build_grid(
    crops: list[CropRecord], grid_size: int = 4, cell_px: int = 224,
    out_path: Path | None = None,
) -> tuple[np.ndarray, dict]:
    """4×4 그리드 이미지 + 셀-crop 매핑 메타.

    각 셀에 crop_path 순서대로 배치. 부족한 셀은 회색으로 채움.
    메타에는 (row,col) → crop_path, bbox, video_hash 등.
    """
    assert len(crops) <= grid_size * grid_size, "crops too many for grid"
    canvas = np.full((grid_size * cell_px, grid_size * cell_px, 3), 128, dtype=np.uint8)
    cells = []
    for i, rec in enumerate(crops):
        r, c = divmod(i, grid_size)
        img = cv2.imread(rec.crop_path)
        if img is None:
            continue
        resized = cv2.resize(img, (cell_px, cell_px))
        # 라벨 오버레이 (번호)
        cv2.rectangle(resized, (0, 0), (30, 20), (0, 0, 0), -1)
        cv2.putText(resized, f"{i+1}", (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        canvas[r*cell_px:(r+1)*cell_px, c*cell_px:(c+1)*cell_px] = resized
        cells.append({"cell": i + 1, "row": r, "col": c,
                      "crop_path": rec.crop_path, "video_hash": rec.video_hash,
                      "bbox": list(rec.bbox), "proposed_class": rec.proposed_class})
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), canvas)
        meta_path = out_path.with_suffix(".json")
        meta_path.write_text(json.dumps({"grid_path": str(out_path), "cells": cells}, indent=2))
    return canvas, {"cells": cells}


# ── Manifest / Seal ────────────────────────────────────────────────

def write_manifest(records: list[CropRecord], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    logger.info("manifest: %d records → %s", len(records), out)


def seal_manifest(manifest: Path) -> str:
    """Manifest SHA256 봉인."""
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    seal_path = manifest.with_suffix(manifest.suffix + ".sha256")
    seal_path.write_text(f"{digest}  {manifest.name}\n")
    logger.info("manifest sealed: %s = %s", seal_path, digest)
    return digest


def verify_manifest(manifest: Path) -> bool:
    seal_path = manifest.with_suffix(manifest.suffix + ".sha256")
    if not seal_path.exists():
        return False
    expected = seal_path.read_text().split()[0]
    actual   = hashlib.sha256(manifest.read_bytes()).hexdigest()
    return expected == actual


def load_manifest(manifest: Path) -> list[CropRecord]:
    if not verify_manifest(manifest):
        raise RuntimeError(f"manifest SHA256 verify failed: {manifest}")
    records = []
    with manifest.open() as f:
        for line in f:
            rec = CropRecord(**json.loads(line))
            # tuple 복원
            rec.bbox = tuple(rec.bbox)
            records.append(rec)
    return records


# ── Pool 소속 검증 (학습측 훅) ──────────────────────────────────────

def assert_crop_pool(crop_record: CropRecord, expected_pool: str) -> None:
    """crop이 기대한 풀에 속하는지 강제 검증. 실패 시 RuntimeError."""
    partition = load_partition()
    actual = partition.get(crop_record.video_hash)
    if actual != expected_pool:
        raise RuntimeError(
            f"POOL VIOLATION: crop from video_hash={crop_record.video_hash[:12]} "
            f"is in pool='{actual}' but expected '{expected_pool}'. "
            f"This blocks Claude verification leak (stop condition #7)."
        )


# ── CLI ────────────────────────────────────────────────────────────

def cmd_partition(args):
    videos = discover_videos()
    if len(videos) < config.GT_POOL_SIZE + 100:
        logger.warning("only %d videos found; GT pool may exhaust source", len(videos))
    gt, trn = partition_videos(videos)
    write_partition(gt, trn, Path(args.out))


def cmd_sample(args):
    from detector import VehicleDetector
    partition = load_partition()
    pool_videos = [(h, Path(p)) for h, p in _read_partition_jsonl().items()
                   if partition[h] == args.pool]
    random.Random(args.seed).shuffle(pool_videos)
    pool_videos = pool_videos[:args.n_videos]
    det = VehicleDetector()
    all_recs: list[CropRecord] = []
    out_dir = Path(args.out_dir)
    for i, (h, p) in enumerate(pool_videos, 1):
        recs = extract_crops_from_video(
            p, h, args.pool, det, out_dir,
            stride_seconds=args.stride, max_frames=args.max_frames,
            max_crops_per_video=args.max_crops,
            stratify_by_coco=not args.no_stratify,
        )
        all_recs.extend(recs)
        if i % 10 == 0:
            logger.info("  progress %d/%d videos → %d crops", i, len(pool_videos), len(all_recs))
        if len(all_recs) >= args.target:
            break
    manifest = out_dir.parent / "crop_manifest.jsonl"
    write_manifest(all_recs, manifest)
    logger.info("sampled %d crops from %d videos", len(all_recs), len(pool_videos))


def _read_partition_jsonl() -> dict[str, str]:
    """partition.jsonl → {hash: path}."""
    jsonl = config.PARTITION_FILE.with_suffix(".jsonl")
    out = {}
    with jsonl.open() as f:
        for line in f:
            rec = json.loads(line)
            out[rec["hash"]] = rec["path"]
    return out


def cmd_grid(args):
    recs = load_manifest(Path(args.manifest)) if verify_manifest(Path(args.manifest)) \
        else [CropRecord(**json.loads(l)) for l in Path(args.manifest).read_text().splitlines()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = args.grid_size * args.grid_size
    for gi in range(0, len(recs), cells):
        batch = recs[gi:gi + cells]
        grid_path = out_dir / f"grid_{gi // cells + 1:03d}.png"
        build_grid(batch, grid_size=args.grid_size, cell_px=args.cell_px, out_path=grid_path)
    logger.info("grids written under %s", out_dir)


def cmd_seal(args):
    seal_manifest(Path(args.manifest))


def cmd_check(args):
    """단일 crop의 풀 소속 검증 (디버그용)."""
    rec = CropRecord(**json.loads(Path(args.crop_meta).read_text()))
    assert_crop_pool(rec, args.expected_pool)
    print(f"OK: crop in pool='{args.expected_pool}'")


def main():
    parser = argparse.ArgumentParser(description="Holdout GT v2 builder")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("partition", help="영상 풀 분할 + SHA256 봉인")
    p.add_argument("--out", default=str(config.PARTITION_FILE))
    p.set_defaults(func=cmd_partition)

    p = sub.add_parser("sample", help="풀에서 crop 추출")
    p.add_argument("--pool", choices=["gt", "train"], required=True)
    p.add_argument("--n-videos", type=int, default=400)
    p.add_argument("--stride", type=float, default=30.0)
    p.add_argument("--max-frames", type=int, default=10)
    p.add_argument("--max-crops", type=int, default=1, help="영상당 최대 crop 수")
    p.add_argument("--target", type=int, default=1200)
    p.add_argument("--seed", type=int, default=config.PARTITION_SEED)
    p.add_argument("--out-dir", default=str(config.GT_DIR / "raw_crops"))
    p.add_argument("--no-stratify", action="store_true",
                   help="COCO 희귀 클래스 우선 정렬 비활성")
    p.set_defaults(func=cmd_sample)

    p = sub.add_parser("grid", help="4×4 그리드 조립")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--grid-size", type=int, default=4)
    p.add_argument("--cell-px", type=int, default=224)
    p.set_defaults(func=cmd_grid)

    p = sub.add_parser("seal", help="manifest SHA256 봉인")
    p.add_argument("--manifest", required=True)
    p.set_defaults(func=cmd_seal)

    p = sub.add_parser("check", help="풀 소속 검증 (단일 crop)")
    p.add_argument("--crop-meta", required=True)
    p.add_argument("--expected-pool", required=True, choices=["gt", "train"])
    p.set_defaults(func=cmd_check)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
