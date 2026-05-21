"""AI Hub #165 시내도로 CCTV → 7-class crop 추출.

Label JSON 구조:
  images[].id ↔ annotations[].image_id
  annotations[].bbox = list of [x, y, w, h] (COCO format? 실제 검증 필요)
  annotations[].category_id = list (이미지에 여러 bbox)

카테고리 → T-code:
  1 승용차 → T1
  2 소형버스 → T2
  3 대형버스 → T2
  4 트럭 → area 기반 T3/T4/T5 (<30k=T3, <150k=T4, else=T5)
  5 대형 트레일러 → T10
  6 오토바이 → T13
  7 보행자 → skip
  8 분류없음 → skip
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

import config
from gt_builder import CropRecord

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

CAT_TO_T = {1: "T1", 2: "T2", 3: "T2", 4: "T_TRUCK_AREA", 5: "T10", 6: "T13"}
# T4 트럭은 area로 T3/T4/T5 분기

MIN_SIDE = 64  # 작은 객체 필터


def truck_by_area(area: int) -> str:
    if area < 3000:  return None   # too small
    if area < 15000: return "T3"   # small pickup
    if area < 80000: return "T4"   # medium cargo
    return "T5"                    # large cargo


def load_labels(label_path: Path) -> dict:
    """Load a single AI Hub label JSON."""
    return json.loads(label_path.read_text())


def iter_annotations(data: dict):
    """Yield (image_file, bbox, t_code) for each valid annotation."""
    images = {img["id"]: img for img in data["images"]}
    for ann in data["annotations"]:
        img = images.get(ann["image_id"])
        if not img:
            continue
        bboxes = ann.get("bbox") or []
        cids = ann.get("category_id", [])
        if isinstance(cids, int):
            cids = [cids]
        if isinstance(bboxes[0], (int, float)):
            bboxes = [bboxes]   # single bbox edge case
        for bbox, cid in zip(bboxes, cids):
            t = CAT_TO_T.get(cid)
            if t is None:
                continue
            # bbox format: AI Hub uses [x1, y1, x2, y2] per inspection
            if len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in bbox]
            if x2 <= x1 or y2 <= y1:
                continue
            yield img["file_name"], (x1, y1, x2, y2), t


def extract_crops_from_location(
    label_path: Path,
    images_root: Path,
    out_crops_dir: Path,
    max_per_class: dict = None,
) -> list[CropRecord]:
    """Extract crops for one location label file."""
    data = load_labels(label_path)
    cls_counts = {c: 0 for c in config.CLASS_ORDER}
    records: list[CropRecord] = []
    out_crops_dir.mkdir(parents=True, exist_ok=True)

    for fname, bbox, t in iter_annotations(data):
        # T_TRUCK_AREA needs area-based sub-classification
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        if w < MIN_SIDE or h < MIN_SIDE:
            continue
        area = int(w * h)
        if t == "T_TRUCK_AREA":
            t = truck_by_area(area)
            if t is None:
                continue
        if max_per_class and cls_counts.get(t, 0) >= max_per_class.get(t, 99999):
            continue

        img_path = images_root / fname
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]
        x1i, y1i = max(0, int(x1)), max(0, int(y1))
        x2i, y2i = min(W, int(x2)), min(H, int(y2))
        crop = img[y1i:y2i, x1i:x2i]
        if crop.size == 0:
            continue

        # unique synthetic "video_hash" — "aihub_{location}"
        video_hash = hashlib.sha256(str(label_path).encode()).hexdigest()
        crop_name = f"aihub_{t}_{img_path.stem}_{x1i}_{y1i}.jpg"
        dest = out_crops_dir / crop_name
        cv2.imwrite(str(dest), crop)

        rec = CropRecord(
            crop_path=str(dest),
            video_hash=video_hash,
            video_path=f"aihub_165:{label_path.parent.name}",
            pool="external",   # separate pool tag
            frame_idx=0, track_id=None,
            bbox=(x1i, y1i, x2i, y2i),
            area=area,
            coco_cls=-1,   # external
            det_conf=1.0,
            proposed_class=t,
            claude_class=t,
            final_class=t,
            reviewer="aihub_165_autolabel",
            notes=f"cat_id remapped; AI Hub Korean CCTV",
        )
        records.append(rec)
        cls_counts[t] = cls_counts.get(t, 0) + 1

    logger.info("extracted from %s: %s", label_path.name, cls_counts)
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True, help="AI Hub label JSON path")
    p.add_argument("--images", required=True, help="extracted source images root")
    p.add_argument("--out-dir", required=True, help="crop output directory")
    p.add_argument("--out-manifest", required=True, help="output JSONL manifest")
    p.add_argument("--max-t10", type=int, default=999999)
    p.add_argument("--max-t13", type=int, default=999999)
    p.add_argument("--max-t1", type=int, default=200)
    p.add_argument("--max-t2", type=int, default=500)
    p.add_argument("--max-t4", type=int, default=500)
    p.add_argument("--max-t5", type=int, default=500)
    args = p.parse_args()

    max_per_class = {"T1": args.max_t1, "T2": args.max_t2, "T4": args.max_t4,
                     "T5": args.max_t5, "T10": args.max_t10, "T13": args.max_t13,
                     "T3": 200}
    records = extract_crops_from_location(
        Path(args.label), Path(args.images), Path(args.out_dir),
        max_per_class=max_per_class,
    )
    out = Path(args.out_manifest)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(json.dumps(asdict(r), ensure_ascii=False) for r in records) + '\n')
    logger.info("manifest: %d records → %s", len(records), out)


if __name__ == "__main__":
    main()
