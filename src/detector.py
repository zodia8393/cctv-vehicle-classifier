"""YOLO11n 기반 차량 검출기.

COCO 클래스 중 차량(car, motorcycle, bus, truck)만 필터.
분류는 별도 classifier.py가 담당 — detector는 bbox 생성까지만.

환경변수:
  USE_INT8_DETECTOR=1 — ONNX INT8 양자화 모델 사용 (~1.5× 가속)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)

ONNX_DIR = config.PRESERVED_DIR / "onnx"
DET_ONNX_FP32 = ONNX_DIR / "yolo11n.onnx"
DET_ONNX_INT8 = ONNX_DIR / "yolo11n_int8.onnx"


class VehicleDetector:
    """YOLO11n detector. 우선순위: INT8 ONNX > FP32 ONNX > Ultralytics YOLO."""

    def __init__(self, model_path: str | None = None, conf: float | None = None,
                 imgsz: int | None = None):
        self.conf       = conf   or config.DET_CONF
        # DET_IMGSZ 환경변수로 해상도 오버라이드 가능 (야간/저조도 영상에 고해상도 실험 용)
        env_imgsz       = os.environ.get("DET_IMGSZ")
        self.imgsz      = imgsz or (int(env_imgsz) if env_imgsz else config.DET_IMG)

        # INT8 dynamic quantization은 YOLO detection head에 부적합 (max conf -38%)
        # 필요 시 사용자가 USE_INT8_DETECTOR=1로 명시 활성화
        use_int8 = os.environ.get("USE_INT8_DETECTOR", "0") == "1"
        onnx_path = DET_ONNX_INT8 if (use_int8 and DET_ONNX_INT8.exists()) else DET_ONNX_FP32

        self._ort_session = None
        if model_path is None and onnx_path.exists():
            try:
                import onnxruntime as ort
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = int(os.environ.get("OMP_NUM_THREADS", "6"))
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self._ort_session = ort.InferenceSession(
                    str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
                self._input_name = self._ort_session.get_inputs()[0].name
                self.model_path = str(onnx_path)
                logger.info("Detector: ONNX %s (%s)",
                            onnx_path.name, "INT8" if use_int8 and "int8" in onnx_path.name else "FP32")
                return
            except Exception as e:
                logger.warning("ONNX detector load failed: %s — fallback", e)

        # Fallback: Ultralytics
        from ultralytics import YOLO
        self.model_path = model_path or config.YOLO_MODEL
        if not Path(self.model_path).exists():
            logger.warning("%s missing, falling back to %s", self.model_path, config.YOLO_FALLBACK)
            self.model_path = config.YOLO_FALLBACK
        self.model = YOLO(self.model_path)

    def _letterbox(self, img: np.ndarray, new_shape: int = 640):
        """YOLO 표준 letterbox — aspect ratio 보존 resize + pad."""
        h0, w0 = img.shape[:2]
        r = min(new_shape / h0, new_shape / w0)
        nh, nw = int(round(h0 * r)), int(round(w0 * r))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        top = (new_shape - nh) // 2
        left = (new_shape - nw) // 2
        out = np.full((new_shape, new_shape, 3), 114, dtype=np.uint8)
        out[top:top+nh, left:left+nw] = resized
        return out, r, (left, top)

    # ── 야간/저조도 전처리 ─────────────────────────────────────────────
    # 11_양지IC(hiv00401~417) 야간+헤드라이트+빛번짐 영상 대응
    # USE_NIGHT_PREPROCESS=0 환경변수로 비활성화 가능

    def _analyze_frame(self, frame: np.ndarray) -> tuple[float, float]:
        """프레임의 밝기 및 극단 밝음 비율 측정."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())
        # 극단 밝음 픽셀 비율 (헤드라이트 flare 지표)
        extreme_bright_ratio = float((gray > 240).sum()) / gray.size
        return brightness, extreme_bright_ratio

    def _mask_headlights(self, frame: np.ndarray) -> np.ndarray:
        """극단 밝음(헤드라이트 flare) 감쇠 — BGR 유지."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        # V > 240인 픽셀을 70%로 감쇠 (헤드라이트 blooming 제거)
        mask = v > 240
        if mask.any():
            hsv[:, :, 2] = np.where(mask, (v.astype(np.float32) * 0.7).clip(0, 255).astype(np.uint8), v)
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return frame

    def _exposure_correction(self, frame: np.ndarray) -> np.ndarray:
        """저조도 환경의 자동노출 보정: 극단 밝음 억제 + 극단 어두움 부스트."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l = lab[:, :, 0].astype(np.float32)
        # 극단 밝음 (>230): 기울기 0.5로 완만하게
        l = np.where(l > 230, 230 + (l - 230) * 0.5, l)
        # 극단 어두움 (<50): 살짝 부스트 (차체 가시성 ↑)
        l = np.where(l < 50, l * 1.25, l)
        # CLAHE로 지역 대비 개선
        l_uint8 = l.clip(0, 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_uint8)
        lab[:, :, 0] = l_enhanced
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _preprocess_if_night(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        """저조도/헤드라이트 환경에서만 전처리 + 적응형 conf 반환.

        반환: (전처리된 frame, 적용할 conf threshold)

        ⚠️ 실측 검증 결과 (hiv00406, 2026-04-23):
        - 전처리 ON (640px):   4 → 3 valid (-25%, 효과 없음)
        - 전처리 + imgsz=1280: 4 → 1 valid (5000프레임 비례 -67%, 악화)
        원인: 심야 영상은 (1) 실제 교통량 적음 (2) 빛 번짐으로 차체 흡수
              (3) YOLO 640 학습 → 해상도 변경 시 out-of-distribution
        결론: 기본 OFF. 필요 시 USE_NIGHT_PREPROCESS=1로 명시 활성화.
        """
        if os.environ.get("USE_NIGHT_PREPROCESS", "0") != "1":
            return frame, self.conf

        brightness, flare_ratio = self._analyze_frame(frame)

        # 주간(밝기≥100) + flare 없음: 전처리 스킵, 기존 conf 유지
        if brightness >= 100 and flare_ratio < 0.02:
            return frame, self.conf

        # 저조도/헤드라이트 환경 감지 → 전처리 적용
        processed = frame
        if flare_ratio > 0.01:
            processed = self._mask_headlights(processed)
        if brightness < 100:
            processed = self._exposure_correction(processed)

        # 적응형 conf: 극야(<50) + flare>5% → 0.12, 저조도 → 0.18, 경계 → 0.20
        if brightness < 50 and flare_ratio > 0.05:
            adaptive_conf = 0.12
        elif brightness < 50:
            adaptive_conf = 0.15
        elif brightness < 100:
            adaptive_conf = 0.18
        else:
            adaptive_conf = 0.22
        return processed, adaptive_conf

    def _detect_onnx(self, frame: np.ndarray, conf_override: float | None = None) -> list[tuple]:
        """직접 ONNX 추론 — INT8 호환. conf_override로 적응형 conf 적용 가능."""
        conf_th = conf_override if conf_override is not None else self.conf
        img, r, (padx, pady) = self._letterbox(frame, self.imgsz)
        x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = x.transpose(2, 0, 1)[None]   # BCHW
        raw = self._ort_session.run(None, {self._input_name: x})[0]
        # raw: (1, 84, 8400) → squeeze batch → (84, 8400) → transpose → (8400, 84)
        preds = raw[0].T   # (8400, 84)
        boxes = preds[:, :4]   # cx, cy, w, h
        class_scores = preds[:, 4:]   # 80 classes
        class_ids = class_scores.argmax(axis=1)
        confs = class_scores.max(axis=1)
        # 차량 클래스 필터 + conf 필터 (적응형)
        mask = confs >= conf_th
        for cid in range(80):
            if cid not in config.COCO_VEHICLE_IDS:
                mask &= (class_ids != cid)
        boxes = boxes[mask]
        confs = confs[mask]
        class_ids = class_ids[mask]
        if len(boxes) == 0:
            return []
        # cx,cy,w,h → x1,y1,x2,y2, padding 제거
        x1 = (boxes[:, 0] - boxes[:, 2] / 2 - padx) / r
        y1 = (boxes[:, 1] - boxes[:, 3] / 2 - pady) / r
        x2 = (boxes[:, 0] + boxes[:, 2] / 2 - padx) / r
        y2 = (boxes[:, 1] + boxes[:, 3] / 2 - pady) / r
        # NMS (simple — per-class IoU)
        from torchvision.ops import nms
        import torch
        boxes_xyxy = torch.tensor(np.stack([x1, y1, x2, y2], axis=1), dtype=torch.float32)
        scores = torch.tensor(confs, dtype=torch.float32)
        keep = nms(boxes_xyxy, scores, iou_threshold=0.45).numpy()
        detections = []
        for i in keep:
            detections.append((float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i]),
                               float(confs[i]), int(class_ids[i])))
        return detections

    def detect_frame(self, frame: np.ndarray) -> list[tuple[float, float, float, float, float, int]]:
        """Returns: list of (x1, y1, x2, y2, conf, coco_cls_id). 차량 클래스만.

        야간/저조도/헤드라이트 감지 시 자동으로 전처리 + 적응형 conf 적용.
        USE_NIGHT_PREPROCESS=0 환경변수로 비활성 가능.
        """
        # 야간/저조도 전처리 (주간엔 bypass)
        frame_proc, adaptive_conf = self._preprocess_if_night(frame)

        if self._ort_session is not None:
            return self._detect_onnx(frame_proc, conf_override=adaptive_conf)
        results = self.model.predict(frame_proc, conf=adaptive_conf, imgsz=self.imgsz,
                                     verbose=False, device="cpu")
        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls.item())
                if cls_id not in config.COCO_VEHICLE_IDS:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append((x1, y1, x2, y2, float(box.conf.item()), cls_id))
        return detections

    def batch_detect(self, frames: list[np.ndarray]) -> list[list[tuple]]:
        """배치 검출 (메모리 허용 범위)."""
        return [self.detect_frame(f) for f in frames]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    args = parser.parse_args()
    import cv2
    img = cv2.imread(args.image_path)
    det = VehicleDetector()
    dets = det.detect_frame(img)
    print(f"detected {len(dets)} vehicles")
    for d in dets:
        print(f"  bbox=({d[0]:.0f},{d[1]:.0f},{d[2]:.0f},{d[3]:.0f}) conf={d[4]:.2f} coco={d[5]}")
