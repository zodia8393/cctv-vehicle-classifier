"""단일 영상 데모: 트래킹 시각화 영상 + 차종분류 CSV 동시 출력."""

from __future__ import annotations

import csv
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from detector import VehicleDetector
from tracker import VehicleTracker
from ensemble_classifier import load_ensemble
from video_consistency import crop_quality_score, line_crossed

# BGR
CLASS_COLORS = {
    "T1": (0, 200, 0), "T2": (255, 165, 0), "T3": (0, 220, 220),
    "T4": (0, 140, 255), "T5": (0, 0, 255), "T10": (255, 0, 255),
    "T13": (200, 200, 0), "UNK": (128, 128, 128),
}
CLASS_NAMES = {
    "T1": "승용", "T2": "버스", "T3": "소형화물", "T4": "중형화물",
    "T5": "대형화물", "T10": "세미트레일러", "T13": "이륜",
}

FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"
_font_cache: dict[int, ImageFont.FreeTypeFont] = {}


def _get_font(size: int = 18) -> ImageFont.FreeTypeFont:
    if size not in _font_cache:
        _font_cache[size] = ImageFont.truetype(FONT_PATH, size)
    return _font_cache[size]


def _bgr_to_rgb(bgr: tuple[int, int, int]) -> tuple[int, int, int]:
    return (bgr[2], bgr[1], bgr[0])


def put_korean_text(img: np.ndarray, text: str, pos: tuple[int, int],
                    font_size: int = 18,
                    color_bgr: tuple[int, int, int] = (255, 255, 255),
                    bg_color_bgr: tuple[int, int, int] | None = None) -> np.ndarray:
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = _get_font(font_size)
    bbox = draw.textbbox(pos, text, font=font)
    if bg_color_bgr is not None:
        pad = 2
        draw.rectangle([bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad],
                        fill=_bgr_to_rgb(bg_color_bgr))
    draw.text(pos, text, font=font, fill=_bgr_to_rgb(color_bgr))
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def run(video_path: str, out_dir: str, max_frames: int | None = None):
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = video_path.stem
    out_video = out_dir / f"{stem}_tracking.mp4"
    out_csv = out_dir / f"{stem}_classification.csv"

    print(f"입력: {video_path.name}")
    print(f"출력: {out_video.name}, {out_csv.name}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"영상 열기 실패: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        total_frames = min(total_frames, max_frames)

    print(f"해상도: {frame_w}x{frame_h}, FPS: {fps:.1f}, 프레임: {total_frames}")
    print("모델 로딩 중...")

    det = VehicleDetector()
    clf = load_ensemble()
    tracker = VehicleTracker(fps=int(fps))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    virtual_line_y = frame_h / 2
    best_n = 3
    min_track_frames = 6
    min_trajectory_px = 80

    track_buffer: dict[int, list[dict]] = defaultdict(list)
    track_crossed: dict[int, bool] = defaultdict(bool)
    track_prev_pos: dict[int, tuple[float, float]] = {}
    track_meta: dict[int, dict] = defaultdict(dict)
    track_label: dict[int, str] = {}

    # ── Pass 1: 검출 + 추적 (렌더링 없이 메타만 수집) ──
    print("Pass 1: 검출 + 추적...")
    t0 = time.time()
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames and frame_idx >= max_frames:
            break

        dets = det.detect_frame(frame)
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
                if w < 48 or h < 48:
                    continue
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                if tid in track_prev_pos and not track_crossed[tid]:
                    if line_crossed(track_prev_pos[tid], (cx, cy), virtual_line_y):
                        track_crossed[tid] = True
                track_prev_pos[tid] = (cx, cy)

                quality = crop_quality_score(w, h, float(det_conf),
                                             frame_h, frame_w, cx, cy)
                track_buffer[tid].append({
                    "quality": quality, "frame": frame_idx,
                    "area": w * h, "bbox": (x1, y1, x2, y2),
                    "coco_cls": int(coco_id),
                    "aspect_ratio": max(w, h) / max(min(w, h), 1),
                    "cx": cx, "cy": cy,
                })
                track_meta[tid]["last"] = frame_idx
                track_meta[tid].setdefault("first", frame_idx)

        frame_idx += 1

        if frame_idx % 500 == 0:
            elapsed = time.time() - t0
            pct = frame_idx / total_frames * 100
            print(f"  {frame_idx}/{total_frames} ({pct:.0f}%) - {elapsed:.1f}s")

    cap.release()
    elapsed_pass1 = time.time() - t0

    # ── Pass 2: valid track 분류 ──
    print("Pass 2: 차종 분류...")
    t1 = time.time()

    valid_track_ids = []
    all_metas = []
    crop_to_track = []
    frames_to_extract: dict[int, list[tuple[int, tuple, int]]] = defaultdict(list)

    for tid, buffer in track_buffer.items():
        if len(buffer) < min_track_frames:
            continue
        first_cx, first_cy = buffer[0]["cx"], buffer[0]["cy"]
        last_cx, last_cy = buffer[-1]["cx"], buffer[-1]["cy"]
        traj = ((last_cx - first_cx) ** 2 + (last_cy - first_cy) ** 2) ** 0.5
        if traj < min_trajectory_px:
            continue

        sorted_buf = sorted(buffer, key=lambda d: d["quality"], reverse=True)
        best = sorted_buf[:best_n]
        valid_track_ids.append(tid)
        for item in best:
            meta_idx = len(all_metas)
            all_metas.append({
                "bbox_area": item["area"],
                "aspect_ratio": item["aspect_ratio"],
                "coco_cls": item["coco_cls"],
            })
            crop_to_track.append(tid)
            frames_to_extract[item["frame"]].append((meta_idx, item["bbox"], tid))

    all_crops: list = [None] * len(all_metas)
    if frames_to_extract:
        cap2 = cv2.VideoCapture(str(video_path))
        target_frames = sorted(frames_to_extract.keys())
        cur = 0
        for target in target_frames:
            while cur <= target:
                ok, frame = cap2.read()
                if not ok:
                    break
                cur += 1
            if not ok:
                break
            for meta_idx, (x1, y1, x2, y2), _ in frames_to_extract[target]:
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    all_crops[meta_idx] = crop
        cap2.release()

    valid_idx = [i for i, c in enumerate(all_crops) if c is not None]
    all_crops = [all_crops[i] for i in valid_idx]
    all_metas = [all_metas[i] for i in valid_idx]
    crop_to_track = [crop_to_track[i] for i in valid_idx]

    vehicle_counts = Counter()
    if all_crops:
        preds, confs = clf.predict_batch(all_crops)
        track_preds: dict[int, list[tuple[str, float]]] = defaultdict(list)
        for tid, p, c in zip(crop_to_track, preds, confs):
            track_preds[tid].append((p, c))

        with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["tracker_id", "차종코드", "차종명", "예측수",
                        "평균신뢰도", "관측프레임수", "시작프레임", "종료프레임",
                        "이동거리(px)"])
            for tid in valid_track_ids:
                preds_list = track_preds.get(tid, [])
                if not preds_list:
                    continue
                weighted = Counter()
                for p, c in preds_list:
                    weighted[p] += c
                final_cls = weighted.most_common(1)[0][0]
                mean_conf = sum(c for _, c in preds_list) / len(preds_list)
                buf = track_buffer[tid]
                traj = ((buf[-1]["cx"] - buf[0]["cx"]) ** 2
                        + (buf[-1]["cy"] - buf[0]["cy"]) ** 2) ** 0.5
                vehicle_counts[final_cls] += 1
                track_label[tid] = final_cls
                w.writerow([tid, final_cls, CLASS_NAMES.get(final_cls, final_cls),
                            len(preds_list), f"{mean_conf:.3f}",
                            len(buf),
                            track_meta[tid].get("first", -1),
                            track_meta[tid].get("last", -1),
                            f"{traj:.0f}"])

    elapsed_pass2 = time.time() - t1

    # ── Pass 3: 분류 결과 반영한 트래킹 영상 렌더링 (PIL 한글) ──
    print("Pass 3: 분류 결과 반영 렌더링...")
    t2 = time.time()

    cap3 = cv2.VideoCapture(str(video_path))
    writer2 = cv2.VideoWriter(str(out_video), fourcc, fps, (frame_w, frame_h))
    tracker2 = VehicleTracker(fps=int(fps))
    frame_idx = 0

    summary_text = "  ".join(f"{CLASS_NAMES.get(k,k)}: {v}대" for k, v in sorted(vehicle_counts.items()))
    summary_text += f"  합계: {sum(vehicle_counts.values())}대"

    while True:
        ok, frame = cap3.read()
        if not ok:
            break
        if max_frames and frame_idx >= max_frames:
            break

        dets = det.detect_frame(frame)
        det_dicts = [
            {"bbox": [d[0], d[1], d[2], d[3]], "conf": d[4], "coco_cls": d[5]}
            for d in dets
        ]
        tracked = tracker2.update(det_dicts, timestamp=frame_idx / fps, frame=None)

        vis = frame.copy()

        # bbox 먼저 그리기 (cv2로 빠르게)
        labels_to_draw: list[tuple[int, int, str, tuple]] = []
        if tracked.tracker_id is not None and len(tracked.tracker_id) > 0:
            for xyxy, tid in zip(tracked.xyxy, tracked.tracker_id):
                tid = int(tid)
                x1, y1, x2, y2 = map(int, xyxy)
                w, h = x2 - x1, y2 - y1
                if w < 48 or h < 48:
                    continue

                label = track_label.get(tid)
                if label:
                    color = CLASS_COLORS.get(label, (128, 128, 128))
                    name = CLASS_NAMES.get(label, label)
                    text = f"#{tid} {name}"
                else:
                    color = (200, 200, 200)
                    text = f"#{tid}"

                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                labels_to_draw.append((x1, y1, text, color))

        # PIL로 한글 텍스트 일괄 렌더링
        pil_img = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        font_label = _get_font(16)
        font_info = _get_font(20)

        for x1, y1, text, color_bgr in labels_to_draw:
            color_rgb = _bgr_to_rgb(color_bgr)
            bbox = draw.textbbox((x1, y1 - 20), text, font=font_label)
            draw.rectangle([bbox[0] - 1, bbox[1] - 1, bbox[2] + 1, bbox[3] + 1],
                            fill=color_rgb)
            draw.text((x1, y1 - 20), text, font=font_label, fill=(0, 0, 0))

        draw.text((10, 8), f"Frame {frame_idx}/{total_frames}", font=font_info, fill=(255, 255, 255))
        draw.text((10, frame_h - 32), summary_text, font=font_info, fill=(255, 255, 255))

        vis = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        writer2.write(vis)
        frame_idx += 1

        if frame_idx % 500 == 0:
            elapsed = time.time() - t2
            pct = frame_idx / total_frames * 100
            print(f"  {frame_idx}/{total_frames} ({pct:.0f}%) - {elapsed:.1f}s")

    cap3.release()
    writer2.release()
    elapsed_pass3 = time.time() - t2

    # ── 결과 요약 ──
    total_elapsed = elapsed_pass1 + elapsed_pass2 + elapsed_pass3
    print(f"\n{'='*50}")
    print(f"완료! 총 {total_elapsed:.1f}초")
    print(f"  Pass 1 (추적): {elapsed_pass1:.1f}s")
    print(f"  Pass 2 (분류): {elapsed_pass2:.1f}s")
    print(f"  Pass 3 (렌더): {elapsed_pass3:.1f}s")
    print(f"\n총 트랙: {len(track_buffer)}")
    print(f"유효 트랙: {len(valid_track_ids)}")
    print(f"\n차종 분포:")
    for cls, cnt in sorted(vehicle_counts.items()):
        print(f"  {cls} ({CLASS_NAMES.get(cls, cls)}): {cnt}대")
    print(f"  합계: {sum(vehicle_counts.values())}대")
    print(f"\n출력 파일:")
    print(f"  영상: {out_video}")
    print(f"  CSV:  {out_csv}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="단일 영상 데모")
    parser.add_argument("video", help="입력 영상 경로")
    parser.add_argument("--out-dir", default="/workspace/prj_cctv/output/demo")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()
    run(args.video, args.out_dir, args.max_frames)
