"""크롭 추출 — GUI 능동 라벨링용 데이터 준비.

신규 흐름의 Phase 0:
  크롭 추출 → GUI 라벨링 → 재학습 → CSV 확인

전략 (회의 결론, 2026-04-27):
  - 50영상 샘플링: 야간 30 + 역광 12 + 주간 8
  - 봉인셋(main_test_v3 + urban_blind) 영상 자동 익스클루드
  - Lazy v2 동일 검출+추적 → best-3 crop 디스크 저장
  - 분류는 안 함 (current_class/conf만 Triple V2로 기록)

출력:
  /workspace/prj/cctv/output/labeling/
    crops/<ic>/<video>_t<tid>_f<frame>.jpg
    crops_meta.jsonl              # 크롭별 메타정보
    extraction_summary.json       # 통계

사용:
  python3 extract_crops_for_labeling.py
  python3 extract_crops_for_labeling.py --n-night 30 --n-backlight 12 --n-day 8
  python3 extract_crops_for_labeling.py --workers 4
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import random
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

import config
from detector import VehicleDetector
from tracker import VehicleTracker
from ensemble_classifier import load_ensemble
from video_consistency import crop_quality_score

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ── 경로 ──────────────────────────────────────────────────────────────
VIDEO_ROOT = Path("/mnt/Expansion/영상/220603_고속도로(경동)")
OUTPUT_DIR = Path("/workspace/prj/cctv/output/labeling")
HOLDOUT_DIR = Path("/workspace/prj/cctv/pipeline/data/holdout_gt_v3")


# ── 봉인셋 익스클루드 ────────────────────────────────────────────────
def build_excluded_videos() -> set[str]:
    """봉인셋에 사용된 video_path를 추출해 익스클루드 리스트 빌드."""
    excluded: set[str] = set()
    for manifest_name in ["main_test_v3_manifest.jsonl", "urban_blind_test_manifest.jsonl"]:
        m = HOLDOUT_DIR / manifest_name
        if not m.exists():
            logger.warning("봉인셋 매니페스트 없음: %s", m)
            continue
        with m.open() as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    vp = rec.get("video_path", "")
                    if vp.startswith("/mnt/Expansion/"):
                        excluded.add(vp)
                except json.JSONDecodeError:
                    continue
    logger.info("봉인셋 익스클루드 영상: %d개", len(excluded))
    return excluded


# ── 영상 분류 (야간/역광/주간) ────────────────────────────────────────
NIGHT_KEYWORDS = ["hiv00401", "hiv00402", "hiv00403", "hiv00404", "hiv00405",
                  "hiv00406", "hiv00407", "hiv00408", "hiv00409", "hiv00410",
                  "hiv00411", "hiv00412", "hiv00413", "hiv00414", "hiv00415",
                  "hiv00416", "hiv00417", "hiv00418", "hiv00419"]
BACKLIGHT_IC_HINTS = ["11_양지IC사거리"]


def estimate_brightness(video_path: Path, n_samples: int = 5) -> float:
    """영상 평균 밝기 추정 (5 프레임 샘플링)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 128.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
    samples = []
    for i in range(n_samples):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * i / n_samples))
        ok, frame = cap.read()
        if ok:
            samples.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
    cap.release()
    return float(np.mean(samples)) if samples else 128.0


def classify_video(video_path: Path, brightness: float) -> str:
    """영상을 night / backlight / day 중 하나로 분류."""
    name = video_path.stem.lower()
    parent_ic = video_path.parent.parent.name if video_path.parent.name in ("1", "2") else video_path.parent.name

    # 야간: 파일명 또는 밝기 기준
    if any(k in name for k in NIGHT_KEYWORDS) or brightness < 90:
        return "night"
    # 역광: 11_양지IC 또는 비슷한 패턴 + 박명
    if parent_ic in BACKLIGHT_IC_HINTS or (90 <= brightness < 120):
        return "backlight"
    return "day"


# ── 영상 샘플링 ──────────────────────────────────────────────────────
def sample_videos(
    excluded: set[str],
    n_night: int, n_backlight: int, n_day: int,
    seed: int = 20260427,
) -> list[tuple[Path, str]]:
    """3 카테고리 균형 샘플링."""
    rng = random.Random(seed)

    all_videos: list[Path] = []
    for ic_dir in sorted(VIDEO_ROOT.iterdir()):
        if not ic_dir.is_dir():
            continue
        for sub in ic_dir.iterdir():
            if sub.is_dir():  # /1/ /2/ 서브디렉토리
                all_videos.extend(sub.glob("*.avi"))
            elif sub.suffix == ".avi":
                all_videos.append(sub)

    # 익스클루드 적용
    candidates = [v for v in all_videos if str(v) not in excluded]
    logger.info("후보 영상: %d개 (전체 %d, 익스클루드 %d)",
                len(candidates), len(all_videos), len(all_videos) - len(candidates))

    # 분류
    by_category: dict[str, list[Path]] = defaultdict(list)
    logger.info("영상 밝기 분류 중...")
    for i, vp in enumerate(candidates):
        if i % 100 == 0:
            logger.info("  %d / %d", i, len(candidates))
        b = estimate_brightness(vp, n_samples=3)
        cat = classify_video(vp, b)
        by_category[cat].append(vp)

    logger.info("분류 결과: night=%d backlight=%d day=%d",
                len(by_category["night"]), len(by_category["backlight"]), len(by_category["day"]))

    # 샘플링
    rng.shuffle(by_category["night"])
    rng.shuffle(by_category["backlight"])
    rng.shuffle(by_category["day"])

    selected: list[tuple[Path, str]] = []
    selected.extend([(v, "night") for v in by_category["night"][:n_night]])
    selected.extend([(v, "backlight") for v in by_category["backlight"][:n_backlight]])
    selected.extend([(v, "day") for v in by_category["day"][:n_day]])

    logger.info("최종 선정: %d영상 (요청 night=%d backlight=%d day=%d)",
                len(selected), n_night, n_backlight, n_day)
    return selected


# ── 영상 1개 처리 (worker) ───────────────────────────────────────────
def process_video_for_crops(args: tuple) -> dict:
    """영상에서 best-3 crop 추출 + 메타 기록."""
    video_path, category, output_root_str = args
    output_root = Path(output_root_str)
    ic_name = video_path.parent.parent.name if video_path.parent.name in ("1", "2") else video_path.parent.name
    video_name = video_path.stem

    crop_dir = output_root / "crops" / ic_name
    crop_dir.mkdir(parents=True, exist_ok=True)

    # 모델 로드 (worker 1회)
    detector = VehicleDetector()
    classifier = load_ensemble()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"video": video_name, "error": "cannot open", "n_crops": 0}
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
    tracker = VehicleTracker(fps=int(fps))

    track_buffer: dict[int, list[dict]] = defaultdict(list)

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        dets = detector.detect_frame(frame)
        det_dicts = [
            {"bbox": [d[0], d[1], d[2], d[3]], "conf": d[4], "coco_cls": d[5]}
            for d in dets
        ]
        tracked = tracker.update(det_dicts, timestamp=frame_idx / fps, frame=None)

        if tracked.tracker_id is not None and len(tracked.tracker_id) > 0:
            for xyxy, tid, coco_id, det_conf in zip(
                tracked.xyxy, tracked.tracker_id, tracked.class_id, tracked.confidence
            ):
                tid = int(tid)
                x1, y1, x2, y2 = map(int, xyxy)
                w, h = x2 - x1, y2 - y1
                if w < 48 or h < 48 or x2 <= x1 or y2 <= y1:
                    continue
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                quality = crop_quality_score(w, h, float(det_conf), frame_h, frame_w, cx, cy)
                track_buffer[tid].append({
                    "quality": quality, "frame": frame_idx,
                    "bbox": (x1, y1, x2, y2), "coco_cls": int(coco_id),
                    "det_conf": float(det_conf),
                })
        frame_idx += 1
    cap.release()

    # Top-N best crops per track + 분류 (Triple V2)
    crops_to_extract: list[tuple[int, tuple, int, dict]] = []  # (frame, bbox, tid, info)
    for tid, buf in track_buffer.items():
        if len(buf) < 6:  # min_track_frames
            continue
        sorted_buf = sorted(buf, key=lambda d: d["quality"], reverse=True)[:3]
        for item in sorted_buf:
            crops_to_extract.append((item["frame"], item["bbox"], tid, item))

    # 영상 재추출 + 분류
    metas = []
    if crops_to_extract:
        cap2 = cv2.VideoCapture(str(video_path))
        target_frames = sorted(set(f for f, _, _, _ in crops_to_extract))
        crops_by_frame = defaultdict(list)
        for f, bbox, tid, info in crops_to_extract:
            crops_by_frame[f].append((bbox, tid, info))

        cur = 0
        crop_imgs = {}  # (frame, tid) → image
        for target in target_frames:
            while cur <= target:
                ok, frame = cap2.read()
                if not ok:
                    break
                cur += 1
            if not ok:
                break
            for (x1, y1, x2, y2), tid, info in crops_by_frame[target]:
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    fname = f"{video_name}_t{tid:04d}_f{target:06d}.jpg"
                    fpath = crop_dir / fname
                    cv2.imwrite(str(fpath), crop)
                    crop_imgs[(target, tid)] = (str(fpath), info, (x1, y1, x2, y2))
        cap2.release()

        # 분류 (Triple V2)
        crop_paths = list(crop_imgs.keys())
        if crop_paths:
            imgs = [cv2.imread(crop_imgs[k][0]) for k in crop_paths]
            valid_idx = [i for i, im in enumerate(imgs) if im is not None]
            imgs = [imgs[i] for i in valid_idx]
            crop_paths = [crop_paths[i] for i in valid_idx]
            if imgs:
                preds, confs = classifier.predict_batch(imgs)
                for (frame, tid), pred, conf in zip(crop_paths, preds, confs):
                    fpath, info, bbox = crop_imgs[(frame, tid)]
                    metas.append({
                        "crop_path": fpath,
                        "ic": ic_name,
                        "video": video_name,
                        "video_path": str(video_path),
                        "video_category": category,
                        "tracker_id": tid,
                        "frame": frame,
                        "bbox": list(bbox),
                        "current_class": pred,
                        "current_conf": float(conf),
                        "is_active_sample": float(conf) < 0.6,
                    })

    return {
        "video": video_name,
        "ic": ic_name,
        "category": category,
        "n_total_tracks": len(track_buffer),
        "n_crops": len(metas),
        "metas": metas,
    }


# ── 메인 ──────────────────────────────────────────────────────────────
def main(args):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "crops").mkdir(exist_ok=True)

    # 1. 봉인셋 익스클루드
    excluded = build_excluded_videos()

    # 2. 50영상 샘플링
    selected = sample_videos(excluded, args.n_night, args.n_backlight, args.n_day, args.seed)

    # 샘플링 결과 저장
    sampling_log = OUTPUT_DIR / "sampling_log.json"
    sampling_log.write_text(json.dumps([
        {"video": str(v), "category": c} for v, c in selected
    ], ensure_ascii=False, indent=2))
    logger.info("샘플링 로그: %s", sampling_log)

    # 3. 병렬 추출
    logger.info("크롭 추출 시작 (workers=%d)...", args.workers)
    t0 = time.time()
    tasks = [(v, c, str(OUTPUT_DIR)) for v, c in selected]

    if args.workers <= 1:
        results = [process_video_for_crops(t) for t in tasks]
    else:
        with mp.Pool(args.workers) as pool:
            results = pool.map(process_video_for_crops, tasks)

    elapsed = time.time() - t0
    logger.info("크롭 추출 완료: %.1f분", elapsed / 60)

    # 4. 메타 통합
    all_metas = []
    summary = {
        "extraction_date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_videos": len(selected),
        "n_videos_by_category": {
            c: sum(1 for _, cat in selected if cat == c)
            for c in ["night", "backlight", "day"]
        },
        "n_crops_total": 0,
        "n_crops_by_category": defaultdict(int),
        "n_active_samples": 0,
        "elapsed_minutes": round(elapsed / 60, 1),
        "per_video": [],
    }

    for r in results:
        if "error" in r:
            logger.warning("[%s] error: %s", r["video"], r["error"])
            continue
        all_metas.extend(r["metas"])
        summary["per_video"].append({
            "video": r["video"], "ic": r["ic"], "category": r["category"],
            "n_total_tracks": r["n_total_tracks"], "n_crops": r["n_crops"],
        })
        summary["n_crops_total"] += r["n_crops"]
        summary["n_crops_by_category"][r["category"]] += r["n_crops"]
        summary["n_active_samples"] += sum(1 for m in r["metas"] if m["is_active_sample"])

    summary["n_crops_by_category"] = dict(summary["n_crops_by_category"])

    # 5. JSONL + summary 저장
    meta_path = OUTPUT_DIR / "crops_meta.jsonl"
    with meta_path.open("w") as f:
        for m in all_metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    logger.info("메타 저장: %s (%d 크롭)", meta_path, len(all_metas))

    summary_path = OUTPUT_DIR / "extraction_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    logger.info("요약 저장: %s", summary_path)

    # 6. 출력
    logger.info("=" * 60)
    logger.info("추출 완료 요약:")
    logger.info("  영상: %d개 (night=%d, backlight=%d, day=%d)",
                summary["n_videos"],
                summary["n_videos_by_category"]["night"],
                summary["n_videos_by_category"]["backlight"],
                summary["n_videos_by_category"]["day"])
    logger.info("  크롭: %d개 (active_samples=%d)",
                summary["n_crops_total"], summary["n_active_samples"])
    logger.info("  소요: %.1f분", elapsed / 60)
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-night", type=int, default=30)
    parser.add_argument("--n-backlight", type=int, default=12)
    parser.add_argument("--n-day", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260427)
    args = parser.parse_args()
    main(args)
