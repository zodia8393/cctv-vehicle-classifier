"""filter_crops.py 경로 패치 버전 — verified_merged → verified_clean"""

import sys
from pathlib import Path

sys.path.insert(0, "/mnt/Expansion/CCTV차종분류/src")
import config

# 경로 패치
config.LABEL_DIR = Path("/workspace/CCTV차종분류_output/labels")

# filter_crops 모듈 import + 입력 디렉토리 변경
import filter_crops as fc

# main 패치: verified → verified_merged
import csv
import logging
import cv2
import numpy as np
from config import VEHICLE_CLASSES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

FRAME_AREA = 1920 * 1080
MIN_SIZE = 50
MIN_BRIGHTNESS = 40.0
MIN_SHARPNESS = 15.0


def main():
    verified = Path("/workspace/CCTV차종분류_output/labels/verified_merged")
    clean = Path("/workspace/CCTV차종분류_output/labels/verified_clean")

    clean.mkdir(parents=True, exist_ok=True)
    for code, name in VEHICLE_CLASSES.items():
        (clean / f"{code}_{name}").mkdir(exist_ok=True)

    stats = {"total": 0, "good": 0, "too_small": 0, "too_close": 0, "too_dark": 0, "too_blurry": 0}
    cls_stats = {}

    from tqdm import tqdm

    for cls_dir in sorted(verified.iterdir()):
        if not cls_dir.is_dir():
            continue
        cls_name = cls_dir.name
        cls_code = cls_name.split("_")[0]
        cls_stats[cls_name] = {"total": 0, "good": 0}

        max_ratio = 0.25 if cls_code == "C5" else 0.15
        files = list(cls_dir.glob("*.jpg"))

        for img_path in tqdm(files, desc=cls_name, unit="img"):
            stats["total"] += 1
            cls_stats[cls_name]["total"] += 1

            img = cv2.imread(str(img_path))
            if img is None:
                stats["too_small"] += 1
                continue

            h, w = img.shape[:2]
            area = w * h

            if w < MIN_SIZE or h < MIN_SIZE:
                stats["too_small"] += 1
                continue
            if area > FRAME_AREA * max_ratio:
                stats["too_close"] += 1
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if float(gray.mean()) < MIN_BRIGHTNESS:
                stats["too_dark"] += 1
                continue

            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if lap_var < MIN_SHARPNESS:
                stats["too_blurry"] += 1
                continue

            # OSD 마스킹
            osd_h = int(h * 0.1)
            if osd_h > 5:
                img[h - osd_h:, :] = 0

            cv2.imwrite(str(clean / cls_name / img_path.name), img)
            stats["good"] += 1
            cls_stats[cls_name]["good"] += 1

    logger.info("=" * 50)
    logger.info("필터링 완료")
    logger.info("  총: %d", stats["total"])
    logger.info("  양호: %d (%.1f%%)", stats["good"], stats["good"] / max(stats["total"], 1) * 100)
    for k in ("too_small", "too_close", "too_dark", "too_blurry"):
        if stats[k]:
            logger.info("  %s: %d (%.1f%%)", k, stats[k], stats[k] / max(stats["total"], 1) * 100)

    logger.info("클래스별:")
    for cls_name in sorted(cls_stats.keys()):
        s = cls_stats[cls_name]
        if s["total"]:
            logger.info("  %s: %d → %d (%.0f%%)", cls_name, s["total"], s["good"], s["good"] / s["total"] * 100)


if __name__ == "__main__":
    main()
