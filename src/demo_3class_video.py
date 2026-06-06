"""3종 차종분류 데모 영상 생성 (승용차/버스/트럭).

제안서 시각자료용. YOLO11n COCO 검출 + ByteTrack 추적으로
차량마다 안정적인 ID와 3종 라벨(승용차/버스/트럭)을 부여해
바운딩박스 + 한글 라벨 + 실시간 카운터가 표시되는 영상을 출력한다.

분류 방식:
  - COCO 차량 클래스를 3종으로 매핑 (car→승용차, bus→버스, truck→트럭).
  - 트랙 단위 다수결(majority vote)로 라벨 확정 → 프레임별 깜빡임 제거.

2-pass 구조:
  Pass 1: 검출 + 추적, 트랙별 COCO 클래스 투표 누적 → 트랙 라벨 확정.
  Pass 2: 동일 추적 재현 + 확정 라벨로 렌더링.
  (ByteTrack은 결정적 — 동일 입력에 동일 ID 부여하므로 2-pass 재현 일관)

사용:
  python3 demo_3class_video.py <video> [--out-dir DIR] [--max-frames N]
"""

from __future__ import annotations

import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))

from detector import VehicleDetector
from tracker import VehicleTracker

# ── 3종 분류 정의 ──────────────────────────────────────────────────
# COCO: car=2, bus=5, truck=7 (motorcycle=3은 3종에 미포함 → 제외)
COCO_TO_3CLASS = {2: "car", 5: "bus", 7: "truck"}
CLASS_KO = {"car": "승용차", "bus": "버스", "truck": "트럭"}
CLASS_ORDER = ["car", "bus", "truck"]
# BGR
CLASS_COLORS = {
    "car":   (0, 200, 0),     # 초록
    "bus":   (0, 165, 255),   # 주황
    "truck": (0, 0, 255),     # 빨강
}

FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"

# 노이즈 트랙 제거 기준
MIN_TRACK_FRAMES = 5      # 최소 관측 프레임
MIN_BBOX_SIDE = 24        # 너무 작은 검출 무시 (px)


def _get_font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_PATH, size)


def run(video_path: str, out_dir: str, max_frames: int | None = None):
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = video_path.stem
    tmp_video = out_dir / f"{stem}_3class_tmp.mp4"      # mp4v (중간)
    out_video = out_dir / f"{stem}_3class.mp4"          # H.264 (최종)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"영상 열기 실패: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        total_frames = min(total_frames, max_frames)
    cap.release()

    print(f"입력: {video_path.name}")
    print(f"해상도: {frame_w}x{frame_h}, FPS: {fps:.1f}, 프레임: {total_frames}")
    print("모델 로딩 중...")
    det = VehicleDetector()
    print(f"  검출 모델: {det.model_path}")

    # ── Pass 1: 검출 + 추적 + 트랙별 클래스 투표 ──────────────────
    print("Pass 1: 검출 + 추적...")
    t0 = time.time()
    tracker = VehicleTracker(fps=int(fps))
    # tid -> Counter({coco_cls: 표수})  (bbox 면적으로 가중 → 가까운 관측 우대)
    track_votes: dict[int, Counter] = defaultdict(Counter)
    track_frames: Counter = Counter()

    cap = cv2.VideoCapture(str(video_path))
    fidx = 0
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and fidx >= max_frames):
            break
        dets = det.detect_frame(frame)
        det_dicts = [
            {"bbox": [d[0], d[1], d[2], d[3]], "conf": d[4], "coco_cls": d[5]}
            for d in dets if d[5] in COCO_TO_3CLASS
        ]
        tracked = tracker.update(det_dicts, timestamp=fidx / fps, frame=None)
        if tracked.tracker_id is not None:
            for xyxy, tid, coco in zip(tracked.xyxy, tracked.tracker_id, tracked.class_id):
                tid = int(tid)
                x1, y1, x2, y2 = xyxy
                w, h = x2 - x1, y2 - y1
                if w < MIN_BBOX_SIDE or h < MIN_BBOX_SIDE:
                    continue
                cls3 = COCO_TO_3CLASS.get(int(coco))
                if cls3 is None:
                    continue
                # 면적 가중 투표: 차량이 클수록(가까울수록) 분류 신뢰 ↑
                track_votes[tid][cls3] += int(w * h)
                track_frames[tid] += 1
        fidx += 1
        if fidx % 500 == 0:
            print(f"  {fidx}/{total_frames} ({fidx/total_frames*100:.0f}%) - {time.time()-t0:.1f}s")
    cap.release()
    elapsed1 = time.time() - t0

    # 트랙 라벨 확정 (다수결) + 노이즈 트랙 필터
    track_label: dict[int, str] = {}
    class_counts: Counter = Counter()
    for tid, votes in track_votes.items():
        if track_frames[tid] < MIN_TRACK_FRAMES:
            continue
        label = votes.most_common(1)[0][0]
        track_label[tid] = label
        class_counts[label] += 1

    print(f"  유효 트랙: {len(track_label)}  "
          + "  ".join(f"{CLASS_KO[c]} {class_counts.get(c,0)}" for c in CLASS_ORDER))

    # ── Pass 2: 확정 라벨로 렌더링 ────────────────────────────────
    print("Pass 2: 렌더링...")
    t1 = time.time()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp_video), fourcc, fps, (frame_w, frame_h))
    tracker2 = VehicleTracker(fps=int(fps))

    summary = "  ".join(f"{CLASS_KO[c]}: {class_counts.get(c,0)}대" for c in CLASS_ORDER)
    summary += f"   합계: {sum(class_counts.values())}대"
    font_label = _get_font(15)
    font_info = _get_font(20)
    font_title = _get_font(22)

    cap = cv2.VideoCapture(str(video_path))
    fidx = 0
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and fidx >= max_frames):
            break
        dets = det.detect_frame(frame)
        det_dicts = [
            {"bbox": [d[0], d[1], d[2], d[3]], "conf": d[4], "coco_cls": d[5]}
            for d in dets if d[5] in COCO_TO_3CLASS
        ]
        tracked = tracker2.update(det_dicts, timestamp=fidx / fps, frame=None)

        vis = frame.copy()
        labels_to_draw = []
        if tracked.tracker_id is not None:
            for xyxy, tid in zip(tracked.xyxy, tracked.tracker_id):
                tid = int(tid)
                label = track_label.get(tid)
                if label is None:   # 노이즈 트랙은 미표시
                    continue
                x1, y1, x2, y2 = map(int, xyxy)
                if (x2 - x1) < MIN_BBOX_SIDE or (y2 - y1) < MIN_BBOX_SIDE:
                    continue
                color = CLASS_COLORS[label]
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                labels_to_draw.append((x1, y1, f"{CLASS_KO[label]}", color))

        # 한글 텍스트 일괄 렌더링 (PIL)
        pil = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        for x1, y1, text, color_bgr in labels_to_draw:
            color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
            ty = max(y1 - 19, 0)
            bb = draw.textbbox((x1, ty), text, font=font_label)
            draw.rectangle([bb[0] - 2, bb[1] - 1, bb[2] + 2, bb[3] + 1], fill=color_rgb)
            draw.text((x1, ty), text, font=font_label, fill=(255, 255, 255))

        # 상단 타이틀 + 하단 요약 (반투명 배경)
        draw.rectangle([0, 0, frame_w, 30], fill=(0, 0, 0))
        draw.text((8, 4), "차종 자동분류 (승용차 · 버스 · 트럭)", font=font_title, fill=(255, 255, 255))
        draw.rectangle([0, frame_h - 30, frame_w, frame_h], fill=(0, 0, 0))
        draw.text((8, frame_h - 27), summary, font=font_info, fill=(255, 255, 255))

        vis = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        writer.write(vis)
        fidx += 1
        if fidx % 500 == 0:
            print(f"  {fidx}/{total_frames} ({fidx/total_frames*100:.0f}%) - {time.time()-t1:.1f}s")
    cap.release()
    writer.release()
    elapsed2 = time.time() - t1

    # ── H.264 재인코딩 (PPT/범용 호환) ───────────────────────────
    print("H.264 재인코딩...")
    import subprocess
    ffmpeg = "/home/ybs/.local/bin/ffmpeg"
    r = subprocess.run(
        [ffmpeg, "-y", "-i", str(tmp_video), "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "medium", str(out_video)],
        capture_output=True, text=True)
    if r.returncode == 0:
        tmp_video.unlink(missing_ok=True)
    else:
        print("  ⚠️ H.264 재인코딩 실패 — mp4v 원본 유지:", r.stderr[-300:])
        out_video = tmp_video

    print(f"\n{'='*52}")
    print(f"완료! Pass1 {elapsed1:.1f}s + Pass2 {elapsed2:.1f}s")
    print(f"유효 트랙(고유 차량): {len(track_label)}대")
    for c in CLASS_ORDER:
        print(f"  {CLASS_KO[c]}: {class_counts.get(c, 0)}대")
    print(f"  합계: {sum(class_counts.values())}대")
    print(f"출력 영상: {out_video}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="3종 차종분류 데모 영상")
    p.add_argument("video")
    p.add_argument("--out-dir", default="/workspace/prj_cctv/output/demo_3class")
    p.add_argument("--max-frames", type=int, default=None)
    a = p.parse_args()
    run(a.video, a.out_dir, a.max_frames)
