"""전체 2,581 영상 crop 수집 + 자동 재분류

run_batch_classify.py의 전체 영상 버전.
이미 staging에 있는 영상은 건너뛴다 (이어하기 가능).
"""

import csv
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from config import LABEL_DIR, VEHICLE_CLASSES, VIDEO_SOURCE_DIR
from classifier.detector import VehicleDetector
from classifier.video_reader import VideoReader
from classifier.tracker import VehicleTracker
from classifier.counter import _parse_location
from utils.ocr_timestamp import extract_timestamp_from_frame
from reclassify import apply_reclassification

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MIN_CROP_SIZE = 50


def auto_reclassify(staging: Path) -> dict[str, str]:
    meta_path = staging / "_meta.csv"
    if not meta_path.exists():
        return {}
    corrections = {}
    with open(meta_path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            fname = row["file"]
            yolo = row["yolo_cls"]
            area = int(row.get("bbox_area", 0) or 0)
            w = int(row.get("crop_w", 0) or 0)
            h = int(row.get("crop_h", 0) or 0)
            if yolo == "C4":
                if area < 12000:
                    corrections[fname] = "C3"
                elif area > 60000:
                    corrections[fname] = "C5"
            elif yolo == "C1" and h > 0 and w > 0:
                if w / h > 2.2 and area > 8000:
                    corrections[fname] = "C3"
    return corrections


def process_one(video_path: Path, detector: VehicleDetector):
    stem = video_path.stem
    staging = LABEL_DIR / "staging" / stem
    staging.mkdir(parents=True, exist_ok=True)
    for code, name in VEHICLE_CLASSES.items():
        (staging / f"{code}_{name}").mkdir(exist_ok=True)

    tracker = VehicleTracker()
    location, camera = _parse_location(video_path)
    ts_str = None
    start = time.time()

    with VideoReader(video_path) as reader:
        for i, (fidx, ts, frame) in enumerate(reader.frames()):
            if i == 0:
                ts_str = extract_timestamp_from_frame(frame)
            dets = detector.detect_frame(frame)
            tracker.update(dets, ts, frame=frame)

    elapsed = time.time() - start

    meta_rows = []
    saved = 0
    for tid, v in tracker.vehicles.items():
        if v.best_crop is None:
            continue
        h, w = v.best_crop.shape[:2]
        if w < MIN_CROP_SIZE or h < MIN_CROP_SIZE:
            continue
        fname = f"t{tid:04d}.jpg"
        folder = staging / f"{v.vehicle_cls}_{v.vehicle_name}"
        cv2.imwrite(str(folder / fname), v.best_crop)
        saved += 1
        meta_rows.append({
            "file": fname, "yolo_cls": v.vehicle_cls, "yolo_name": v.vehicle_name,
            "corrected_cls": "", "corrected_name": "",
            "direction": v.direction, "bbox_area": v.best_crop_area,
            "crop_w": w, "crop_h": h,
        })

    if meta_rows:
        with open(staging / "_meta.csv", "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=meta_rows[0].keys())
            writer.writeheader()
            writer.writerows(meta_rows)

    corrections = auto_reclassify(staging)
    apply_reclassification(stem, corrections)

    return stem, location, camera, ts_str, tracker.unique_count, saved, len(corrections), elapsed


def main():
    all_videos = sorted(VIDEO_SOURCE_DIR.rglob("*.avi"))
    total = len(all_videos)

    # 이미 처리된 영상 건너뛰기
    staging_dir = LABEL_DIR / "staging"
    existing = {d.name for d in staging_dir.iterdir()} if staging_dir.exists() else set()
    remaining = [v for v in all_videos if v.stem not in existing]

    logger.info("전체: %d | 완료: %d | 남은: %d", total, len(existing), len(remaining))

    detector = VehicleDetector()
    detector.load()

    start_all = time.time()
    for idx, video_path in enumerate(remaining, 1):
        try:
            stem, loc, cam, ts, unique, saved, changed, elapsed = process_one(video_path, detector)

            elapsed_all = time.time() - start_all
            avg = elapsed_all / idx
            eta_h = int(avg * (len(remaining) - idx)) // 3600
            eta_m = (int(avg * (len(remaining) - idx)) % 3600) // 60

            logger.info(
                "[%d/%d] %s | %s cam%s | %s | %d대 crop:%d 변경:%d | %.0f초 | ETA %dh%02dm",
                idx, len(remaining), stem, loc, cam, ts or "N/A",
                unique, saved, changed, elapsed, eta_h, eta_m,
            )
        except Exception as e:
            logger.error("[%d/%d] 실패: %s — %s", idx, len(remaining), video_path.name, e)

        # 500파일마다 현황 출력
        if idx % 500 == 0:
            _print_verified()

    total_elapsed = time.time() - start_all
    logger.info("=" * 60)
    logger.info("완료: %d파일, %.1f시간", len(remaining), total_elapsed / 3600)
    _print_verified()


def _print_verified():
    verified = LABEL_DIR / "verified"
    total = 0
    for code, name in VEHICLE_CLASSES.items():
        folder = verified / f"{code}_{name}"
        if folder.exists():
            cnt = len(list(folder.glob("*.jpg")))
            if cnt:
                logger.info("  %s_%s: %d장", code, name, cnt)
                total += cnt
    logger.info("  합계: %d장", total)


if __name__ == "__main__":
    main()
