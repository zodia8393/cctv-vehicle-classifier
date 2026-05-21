"""Sample NEW crops from GT pool videos that are NOT in existing manifest.

Blind test constraint: new crops must not overlap with training pool.
Strategy: re-sample same 600 GT videos at higher max_crops, exclude existing crop_paths.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import cv2
import numpy as np

import config
from gt_builder import (
    CropRecord, iter_video_frames, load_partition, _read_partition_jsonl,
    MIN_CROP_SIDE, MIN_CROP_ASPECT,
)
from detector import VehicleDetector

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exclude-manifest",
                   default=str(config.GT_DIR / "crop_manifest.jsonl"))
    p.add_argument("--n-videos", type=int, default=150)
    p.add_argument("--target", type=int, default=105)
    p.add_argument("--stride", type=float, default=20.0)  # finer stride
    p.add_argument("--max-frames", type=int, default=20)
    p.add_argument("--max-crops", type=int, default=5)  # more per video
    p.add_argument("--out-dir",
                   default="/workspace/prj_cctv/pipeline/data/holdout_gt_v3/new_crops")
    p.add_argument("--seed", type=int, default=98765)
    args = p.parse_args()

    # Existing crops to exclude
    existing_paths = set()
    for l in Path(args.exclude_manifest).read_text().splitlines():
        existing_paths.add(json.loads(l)['crop_path'])
    logger.info("excluding %d existing crop paths", len(existing_paths))

    partition = load_partition()
    pool = [(h, Path(p)) for h, p in _read_partition_jsonl().items()
            if partition[h] == 'gt']
    random.Random(args.seed).shuffle(pool)
    pool = pool[:args.n_videos]
    logger.info("scanning %d GT videos", len(pool))

    det = VehicleDetector()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    new_recs: list[CropRecord] = []
    for i, (vhash, vpath) in enumerate(pool, 1):
        candidates = []
        for fidx, frame in iter_video_frames(vpath, args.stride, args.max_frames):
            for det_tup in det.detect_frame(frame):
                x1, y1, x2, y2, conf, cls_id = det_tup
                w, h = x2 - x1, y2 - y1
                if w < MIN_CROP_SIDE or h < MIN_CROP_SIDE:
                    continue
                aspect = min(w, h) / max(w, h)
                if aspect < MIN_CROP_ASPECT and int(cls_id) != 7:
                    continue
                area = int(w * h)
                rec = CropRecord(
                    crop_path="", video_hash=vhash, video_path=str(vpath),
                    pool='gt', frame_idx=fidx, track_id=None,
                    bbox=(int(x1), int(y1), int(x2), int(y2)), area=area,
                    coco_cls=int(cls_id), det_conf=float(conf),
                    proposed_class=config.COCO_TO_7CLASS.get(int(cls_id)),
                )
                crop_img = frame[int(y1):int(y2), int(x1):int(x2)].copy()
                candidates.append((rec, crop_img))
        # Rare COCO first
        candidates.sort(
            key=lambda t: {3:3.0, 5:2.5, 7:2.0, 2:1.0}.get(t[0].coco_cls, 1.0)
                         * t[0].det_conf * t[0].area,
            reverse=True)

        added = 0
        for rec, img in candidates:
            if added >= args.max_crops:
                break
            fname = f"{vhash[:12]}_f{rec.frame_idx:06d}_c{rec.coco_cls}_a{rec.area}.jpg"
            dest = out_dir / fname
            # Skip if exists in excluded set by inspection of source path pattern
            # (existing crops live under raw_crops/, our new under new_crops/)
            if str(dest) in existing_paths:
                continue  # would clobber
            # Also skip if same video+frame was already sampled
            potential_existing = f"{vhash[:12]}_f{rec.frame_idx:06d}"
            if any(potential_existing in p for p in existing_paths):
                continue
            cv2.imwrite(str(dest), img)
            rec.crop_path = str(dest)
            new_recs.append(rec)
            added += 1

        if i % 20 == 0:
            logger.info("  %d/%d videos → %d new crops", i, len(pool), len(new_recs))
        if len(new_recs) >= args.target:
            break

    # Save manifest
    manifest = out_dir.parent / "new_crop_manifest.jsonl"
    from dataclasses import asdict
    manifest.write_text('\n'.join(
        json.dumps(asdict(r), ensure_ascii=False) for r in new_recs) + '\n')
    logger.info("collected %d new crops → %s", len(new_recs), manifest)


if __name__ == "__main__":
    main()
