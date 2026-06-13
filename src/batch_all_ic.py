"""전체 2,581 AVI 배치 처리 — 23 IC, Triple Ensemble R15 + Tracker 다수결.

Config (CPU budget 관리):
- classify-every-n = 10 (기본 3 → sparser)
- max-frames = None (전체 영상 처리, 2025-04-22 수정)
- no TTA (속도 우선)
- min-lived = 5

출력:
- ic_summary.json: IC별 차종 분포 + 메타
- per_video CSV: 각 영상 track-level 결과
- weak_spots.md: IC × 차종 편차 리포트
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

# Multi-core CPU 활용 최대화 — R15 학습 때 2000%+ 수준
os.environ.setdefault("OMP_NUM_THREADS", "6")   # per-worker
os.environ.setdefault("MKL_NUM_THREADS", "6")

import torch
torch.set_num_threads(6)  # per worker: 6 threads × 3 workers = 18 cores

import config
from detector import VehicleDetector
from tracker import VehicleTracker
from ensemble_classifier import load_ensemble
from video_consistency import process_video

N_WORKERS = 3  # 3 병렬 workers × 6 threads = 18 cores (~1800% CPU)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

VIDEO_ROOT = Path("/mnt/Expansion/영상/220603_고속도로(경동)")
OUT_ROOT = Path("/workspace/prj/cctv/output/batch_all_ic")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# 경동 실무 기대 분포
EXPECTED = {"T1": 0.65, "T2": 0.075, "T3": 0.01, "T4": 0.20, "T5": 0.05, "T10": 0.02, "T13": 0.005}


# Worker-level globals: 앙상블을 worker 시작 시 1회만 로드
_WORKER_DET = None
_WORKER_CLF = None


def _worker_init():
    """ProcessPoolExecutor initializer — worker 프로세스 시작 시 1회 실행.
    여기서 detector + ensemble 로드 후 전역 변수로 재사용.
    """
    global _WORKER_DET, _WORKER_CLF
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "6")
    os.environ.setdefault("MKL_NUM_THREADS", "6")
    import torch
    torch.set_num_threads(6)

    from detector import VehicleDetector
    from ensemble_classifier import load_ensemble

    _WORKER_DET = VehicleDetector()
    _WORKER_CLF = load_ensemble()


def process_single_video(args):
    """Worker function — 재사용 detector/classifier 사용."""
    vpath_str, out_csv_str, ic_name = args
    from video_consistency import process_video
    try:
        result = process_video(
            Path(vpath_str), _WORKER_DET, _WORKER_CLF, Path(out_csv_str),
            classify_every_n=5, use_tta=False,
            min_lived_frames=5, max_frames=None,
        )
        return (ic_name, Path(vpath_str).stem, result, None)
    except Exception as e:
        return (ic_name, Path(vpath_str).stem, None, str(e))


def main():
    # 23 IC 디렉토리 스캔
    ic_dirs = sorted([d for d in VIDEO_ROOT.iterdir() if d.is_dir() and not d.name.startswith('.')])
    logger.info("Found %d IC dirs", len(ic_dirs))

    all_results = {}
    total_videos = 0
    processed = 0
    skipped = 0
    t0 = time.time()

    ic_videos = {}
    for ic_dir in ic_dirs:
        videos = list(ic_dir.rglob("*.avi")) + list(ic_dir.rglob("*.mp4"))
        ic_videos[ic_dir.name] = videos
        total_videos += len(videos)
    logger.info("Total videos to process: %d (N_WORKERS=%d)", total_videos, N_WORKERS)

    # Build work list: skip already-processed + collect pending
    work_items = []
    skip_counts = {}  # ic_name → Counter for skipped videos
    for ic_name, videos in ic_videos.items():
        ic_out = OUT_ROOT / ic_name
        ic_out.mkdir(parents=True, exist_ok=True)
        skip_counts[ic_name] = Counter()
        for vpath in videos:
            out_csv = ic_out / f"{vpath.stem}.csv"
            if out_csv.exists():
                import csv
                with out_csv.open() as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        skip_counts[ic_name][row['final_class']] += 1
                skipped += 1
                processed += 1
                continue
            work_items.append((str(vpath), str(out_csv), ic_name))

    logger.info("Work items: %d (skipped %d)", len(work_items), skipped)

    # IC별 누적
    ic_counts_all = {ic: Counter(skip_counts[ic]) for ic in ic_videos.keys()}

    # Parallel processing with initializer (ensemble loaded ONCE per worker)
    with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_worker_init) as ex:
        futures = [ex.submit(process_single_video, w) for w in work_items]
        for fut in as_completed(futures):
            ic_name, video_stem, result, err = fut.result()
            if err:
                logger.warning("failed %s: %s", video_stem, err)
                continue
            ic_counts_all[ic_name].update(result['vehicle_counts'])
            processed += 1
            if processed % 10 == 0:
                elapsed = time.time() - t0
                eta = elapsed / processed * (total_videos - processed) / 60
                logger.info("PROGRESS %d/%d (%.1f%%) | elapsed %.1fm | ETA %.1fm | %s",
                            processed, total_videos, processed/total_videos*100,
                            elapsed/60, eta, video_stem)

    # IC별 집계 마무리
    for ic_name, videos in ic_videos.items():
        ic_counts = ic_counts_all[ic_name]

        all_results[ic_name] = {
            "n_videos": len(videos),
            "vehicle_counts": dict(ic_counts),
            "total_vehicles": sum(ic_counts.values()),
        }

        # IC-level 리포트
        total = sum(ic_counts.values())
        ic_dist = {cls: ic_counts.get(cls, 0) / max(total, 1) for cls in config.CLASS_ORDER}
        dev = {cls: ic_dist[cls] - EXPECTED[cls] for cls in config.CLASS_ORDER}
        all_results[ic_name]["class_dist"] = ic_dist
        all_results[ic_name]["deviation"] = dev
        logger.info("[%s] %d vehicles, T1=%.0f%% T2=%.0f%% T4=%.0f%% T10=%.0f%%",
                    ic_name, total,
                    ic_dist.get("T1", 0)*100, ic_dist.get("T2", 0)*100,
                    ic_dist.get("T4", 0)*100, ic_dist.get("T10", 0)*100)


    # 종합 리포트
    summary_file = OUT_ROOT / "ic_summary.json"
    summary_file.write_text(json.dumps(all_results, ensure_ascii=False, indent=2))

    # Weak spots 분석
    weak_report = []
    weak_report.append("# 23 IC 배치 결과 — Weak Spots 분석\n")
    weak_report.append(f"총 영상: {total_videos}, 처리 {processed} (skip {skipped})\n")
    weak_report.append(f"총 소요: {(time.time()-t0)/3600:.1f}시간\n\n")

    # IC별 편차 테이블
    weak_report.append("## IC별 차종 분포 vs 기대\n")
    weak_report.append("| IC | 차량 | T1 | T2 | T3 | T4 | T5 | T10 | T13 |")
    weak_report.append("|----|---:|---:|---:|---:|---:|---:|---:|---:|")
    for ic_name, data in sorted(all_results.items()):
        total = data["total_vehicles"]
        d = data["class_dist"]
        weak_report.append(f"| {ic_name} | {total} | "
                           + " | ".join(f"{d.get(c,0)*100:.0f}%" for c in config.CLASS_ORDER) + " |")

    # Gate 평가 (Urban Engineer 기준 ±10% 허용)
    weak_report.append("\n## Weak Spots (차종별 편차 >10%p)\n")
    for ic_name, data in sorted(all_results.items()):
        for cls in config.CLASS_ORDER:
            dev_pct = data["deviation"][cls] * 100
            if abs(dev_pct) > 10 and data["total_vehicles"] > 20:
                sign = "+" if dev_pct > 0 else ""
                weak_report.append(f"- **{ic_name}** {cls}: {sign}{dev_pct:.1f}%p "
                                   f"(실측 {data['class_dist'][cls]*100:.1f}% / 기대 {EXPECTED[cls]*100:.1f}%)")

    weak_file = OUT_ROOT / "weak_spots.md"
    weak_file.write_text('\n'.join(weak_report))

    logger.info("=" * 60)
    logger.info("BATCH COMPLETE: %d videos in %.1f hours",
                processed, (time.time()-t0)/3600)
    logger.info("Summary: %s", summary_file)
    logger.info("Weak spots: %s", weak_file)


if __name__ == "__main__":
    main()
