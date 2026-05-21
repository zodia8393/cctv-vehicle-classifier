"""차종 분류기 통합 인터페이스.

3개 백엔드 지원 (Stage 1 후보):
- timm_mnv4          : MobileNetV4-Conv-S (timm, 2.5M, CPU 1.74ms)
- timm_efficientvit  : EfficientViT-B0 (timm, 2.14M, CPU 2.19ms)
- ultralytics_yolo11n: YOLO11n-cls (ultralytics, 1.54M, 0.4ms)

경동고속도로 7-class (T1/T2/T3/T4/T5/T10/T13) 분류. 모든 백엔드는 동일한
predict_batch(images) → (preds, confs) 인터페이스로 호출.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Protocol

import numpy as np

import config

logger = logging.getLogger(__name__)


class Classifier(Protocol):
    """백엔드 공통 인터페이스."""

    def predict_batch(self, images: list[np.ndarray]) -> tuple[list[str], list[float]]:
        ...

    def latency_ms(self, images: list[np.ndarray], n_iter: int = 10) -> float:
        ...


# ── timm 백엔드 (MobileNetV4, EfficientViT 공용) ────────────────────

class TimmClassifier:
    """timm 모델 래퍼. ImageNet pretrain → 7-class 헤드."""

    def __init__(self, model_id: str, weights_path: str | None = None,
                 imgsz: int = config.TRAIN_IMG_SIZE):
        import timm
        import torch
        self.torch = torch
        self.imgsz = imgsz
        self.model = timm.create_model(model_id, pretrained=(weights_path is None),
                                       num_classes=config.NUM_CLASSES)
        if weights_path and Path(weights_path).exists():
            state = torch.load(weights_path, map_location="cpu", weights_only=False)
            self.model.load_state_dict(state if isinstance(state, dict) else state.state_dict())
        self.model.eval()
        # ImageNet mean/std
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def _preprocess(self, images: list[np.ndarray]):
        import cv2
        batch = []
        for img in images:
            resized = cv2.resize(img, (self.imgsz, self.imgsz))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            batch.append(rgb.astype(np.float32) / 255.0)
        arr = np.stack(batch, axis=0).transpose(0, 3, 1, 2)  # NCHW
        tensor = self.torch.from_numpy(arr)
        return (tensor - self.mean) / self.std

    @property
    def _torch(self):
        return self.torch

    def predict_batch(self, images: list[np.ndarray]) -> tuple[list[str], list[float]]:
        with self.torch.no_grad():
            logits = self.model(self._preprocess(images))
            probs  = self.torch.softmax(logits, dim=1)
            confs, idx = probs.max(dim=1)
        preds = [config.IDX_TO_CLASS[int(i)] for i in idx.tolist()]
        return preds, [float(c) for c in confs.tolist()]

    def latency_ms(self, images: list[np.ndarray], n_iter: int = 10) -> float:
        """Batch=1 추론 지연 평균."""
        single = images[0:1]
        self.predict_batch(single)  # warmup
        t0 = time.perf_counter()
        for _ in range(n_iter):
            self.predict_batch(single)
        return (time.perf_counter() - t0) / n_iter * 1000.0


# ── Ultralytics 백엔드 (YOLO11n-cls) ────────────────────────────────

class UltralyticsClassifier:
    """Ultralytics YOLO11n-cls wrapper."""

    def __init__(self, weights_path: str, imgsz: int = config.TRAIN_IMG_SIZE):
        from ultralytics import YOLO
        if not Path(weights_path).exists():
            raise FileNotFoundError(weights_path)
        self.model = YOLO(weights_path)
        self.imgsz = imgsz
        # 헤드 이름이 T-code와 다를 수 있음 → 학습 후 names 속성 사용
        self.names = getattr(self.model, "names", None)

    def predict_batch(self, images: list[np.ndarray]) -> tuple[list[str], list[float]]:
        # Ultralytics는 list[ndarray] 직접 지원
        results = self.model.predict(images, imgsz=self.imgsz, verbose=False, device="cpu")
        preds, confs = [], []
        for r in results:
            p = r.probs
            top1 = int(p.top1)
            name = self.names.get(top1) if isinstance(self.names, dict) else str(top1)
            # 학습된 모델의 names가 T-code여야 함 (prepare_finetune 시 매핑)
            preds.append(name if name in config.CLASS_ORDER else f"idx{top1}")
            confs.append(float(p.top1conf.item()))
        return preds, confs

    def latency_ms(self, images: list[np.ndarray], n_iter: int = 10) -> float:
        single = images[0:1]
        self.predict_batch(single)
        t0 = time.perf_counter()
        for _ in range(n_iter):
            self.predict_batch(single)
        return (time.perf_counter() - t0) / n_iter * 1000.0


# ── Factory ─────────────────────────────────────────────────────────

def load_model(backend: str, model_id: str, weights_path: str | None = None,
               imgsz: int = config.TRAIN_IMG_SIZE) -> Classifier:
    """백엔드 문자열 → Classifier 인스턴스."""
    if backend == "timm_mnv4":
        return TimmClassifier(model_id, weights_path, imgsz)
    if backend == "timm_efficientvit":
        return TimmClassifier(model_id, weights_path, imgsz)
    if backend == "ultralytics_yolo11n":
        return UltralyticsClassifier(weights_path or model_id, imgsz)
    raise ValueError(f"unknown backend: {backend}")


def model_size_mb(weights_path: str) -> float:
    return Path(weights_path).stat().st_size / (1024 * 1024)
