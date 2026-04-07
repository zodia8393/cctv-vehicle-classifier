"""배치 crop 수집 이어하기 — 경로 패치 버전

읽기: /mnt/Expansion (영상 원본 + 기존 모델)
쓰기: /workspace/CCTV차종분류_output (staging, verified, cache)
기존 staging 체크: /mnt/Expansion + /workspace 양쪽 모두 확인
"""

import sys
from pathlib import Path

sys.path.insert(0, "/mnt/Expansion/CCTV차종분류/src")
import config

# 읽기 경로 (영상, 모델)
config.VIDEO_SOURCE_DIR = Path("/mnt/Expansion/영상/220603_고속도로(경동)")
config.MODEL_DIR = Path("/mnt/Expansion/CCTV차종분류/data/models")
config.YOLO_MODEL = str(config.MODEL_DIR / "yolov8n.pt")

# 쓰기 경로 → /workspace
OUTPUT_BASE = Path("/workspace/CCTV차종분류_output")
config.LABEL_DIR = OUTPUT_BASE / "labels"
config.PROCESSED_DIR = OUTPUT_BASE / "processed"

# 기존 staging 목록 (양쪽 디렉토리 체크)
OLD_STAGING = Path("/mnt/Expansion/CCTV차종분류/data/labels/staging")

# run_batch_all_crops의 main을 패치해서 기존 staging도 체크
import run_batch_all_crops as batch_mod

_orig_main = batch_mod.main

def patched_main():
    # 기존 staging에서 처리된 영상 목록 추가 로드
    all_videos = sorted(config.VIDEO_SOURCE_DIR.rglob("*.avi"))

    staging_dir = config.LABEL_DIR / "staging"
    existing = set()
    # 새 경로
    if staging_dir.exists():
        existing.update(d.name for d in staging_dir.iterdir())
    # 기존 경로
    if OLD_STAGING.exists():
        existing.update(d.name for d in OLD_STAGING.iterdir())

    remaining = [v for v in all_videos if v.stem not in existing]

    import logging
    logger = logging.getLogger(__name__)
    logger.info("전체: %d | 기존완료: %d | 남은: %d", len(all_videos), len(existing), len(remaining))

    if not remaining:
        logger.info("모든 영상 처리 완료!")
        return

    from classifier.detector import VehicleDetector
    detector = VehicleDetector()
    detector.load()

    import time
    start_all = time.time()
    for idx, video_path in enumerate(remaining, 1):
        try:
            result = batch_mod.process_one(video_path, detector)
            elapsed_all = time.time() - start_all
            avg = elapsed_all / idx
            eta_h = int(avg * (len(remaining) - idx)) // 3600
            eta_m = (int(avg * (len(remaining) - idx)) % 3600) // 60
            stem, loc, cam, ts, unique, saved, changed, elapsed = result
            logger.info(
                "[%d/%d] %s | %s cam%s | %s | %d대 crop:%d 변경:%d | %.0f초 | ETA %dh%02dm",
                idx, len(remaining), stem, loc, cam, ts or "N/A",
                unique, saved, changed, elapsed, eta_h, eta_m,
            )
        except Exception as e:
            logger.error("[%d/%d] 실패: %s — %s", idx, len(remaining), video_path.name, e)

        if idx % 500 == 0:
            batch_mod._print_verified()

    total_elapsed = time.time() - start_all
    logger.info("완료: %d파일, %.1f시간", len(remaining), total_elapsed / 3600)
    batch_mod._print_verified()

patched_main()
