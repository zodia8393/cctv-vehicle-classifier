"""Production Ensemble Classifier.

Round 8 검증 결과 v8lb_self + MobileNetV4 soft-voting 50/50이 단일 백본 대비
+8~10%p 개선을 보임 (50% → 60%). 이 모듈이 프로덕션 분류기 엔트리 포인트.

Design:
- 동일한 Classifier 인터페이스: `predict_batch(images) -> (preds, confs)`
- 두 백본 softmax 평균
- video_consistency.py 등에서 기존 `load_model()` 대신 `load_ensemble()` 호출
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import os
import cv2
import numpy as np

import config


def _preprocess_rgb_chw(images, imgsz, mean, std):
    """BGR → RGB → resize → CHW → normalize. 공통 전처리."""
    arr = np.stack([
        cv2.cvtColor(cv2.resize(img, (imgsz, imgsz)),
                     cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        for img in images
    ], axis=0).transpose(0, 3, 1, 2)
    return ((arr - mean) / std).astype(np.float32)


def _softmax(x: np.ndarray, axis=1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

logger = logging.getLogger(__name__)


class EnsembleClassifier:
    """Triple ensemble: v8lb (Ultralytics) + MNv4-R8 (128) + MNv4-R14 (192).

    R8 = 경동 자체 전용 학습 (128px)
    R14 = AI Hub 통합 학습 (192px, multi-domain)
    v8lb = R12 (경동 + AI Hub T10/T13)
    → Equal 1/3 weight triple이 HW 62.9, URB 75.3, gap 12.5 최적
    """

    def __init__(
        self,
        v8lb_weights: str = str(config.PRESERVED_DIR / "round_12_v8lb_best.pt"),
        mnv4_weights: str = str(config.PRESERVED_DIR / "round_15_mnv4_balanced.pt"),
        mnv4_r8_weights: str = str(config.PRESERVED_DIR / "round_8_mnv4_best.pt"),
        mnv4_model_id: str = "mobilenetv4_conv_small.e2400_r224_in1k",
        mnv4_imgsz: int = 192,
        mnv4_r8_imgsz: int = 128,
        imgsz: int = 128,
        weights: tuple[float, float, float] = (0.34, 0.33, 0.33),
        use_triple: bool = True,
        use_onnx: bool = True,  # .onnx 파일 있으면 ONNX Runtime 사용 (2-3x 가속)
    ):
        self.imgsz = imgsz
        self.mnv4_imgsz = mnv4_imgsz
        self.mnv4_r8_imgsz = mnv4_r8_imgsz
        self.weights = weights
        self.use_triple = use_triple
        self.use_onnx = use_onnx

        onnx_dir = config.PRESERVED_DIR / "onnx"
        v8lb_onnx = onnx_dir / "round_12_v8lb.onnx"
        mnv4_onnx = onnx_dir / "round_15_mnv4.onnx"
        r8_onnx = onnx_dir / "round_8_mnv4.onnx"

        # ONNX Runtime 가능한지 체크
        self._ort = None
        if use_onnx and v8lb_onnx.exists() and mnv4_onnx.exists():
            try:
                import onnxruntime as ort
                self._ort = ort
                sess_opts = ort.SessionOptions()
                sess_opts.intra_op_num_threads = int(os.environ.get("OMP_NUM_THREADS", "6"))
                sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self.v8lb_sess = ort.InferenceSession(
                    str(v8lb_onnx), sess_options=sess_opts, providers=["CPUExecutionProvider"])
                self.mnv4_sess = ort.InferenceSession(
                    str(mnv4_onnx), sess_options=sess_opts, providers=["CPUExecutionProvider"])
                if use_triple and r8_onnx.exists():
                    self.mnv4_r8_sess = ort.InferenceSession(
                        str(r8_onnx), sess_options=sess_opts, providers=["CPUExecutionProvider"])
                else:
                    self.mnv4_r8_sess = None
                # Sentinels for predict_batch checks (truthy when ONNX session exists)
                self.mnv4_r8 = self.mnv4_r8_sess
                self.mnv4 = self.mnv4_sess
                self.v8lb = self.v8lb_sess
                logger.info("Ensemble (ONNX) loaded: v8lb=%s mnv4=%s r8=%s weights=%s",
                            v8lb_onnx.name, mnv4_onnx.name,
                            r8_onnx.name if self.mnv4_r8_sess else "-", weights)
                # v8lb classes 매핑 (Ultralytics 모델 동일)
                from ultralytics import YOLO
                _yolo = YOLO(v8lb_weights)
                self._v8lb_idx_to_our = {
                    i: config.CLASS_TO_IDX[name]
                    for i, name in _yolo.names.items()
                    if name in config.CLASS_TO_IDX
                }
                del _yolo
                self._has_pt = False
                self.mean_np = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
                self.std_np  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
                return  # ONNX 로드 성공
            except Exception as e:
                logger.warning("ONNX load failed, fall back to PyTorch: %s", e)
                self._ort = None

        # Fallback: PyTorch/Ultralytics
        import torch, timm
        from ultralytics import YOLO
        self.torch = torch
        self._has_pt = True

        for p in (v8lb_weights, mnv4_weights):
            if not Path(p).exists():
                raise FileNotFoundError(f"missing: {p}")

        self.v8lb = YOLO(v8lb_weights)
        self._v8lb_idx_to_our = {
            i: config.CLASS_TO_IDX[name]
            for i, name in self.v8lb.names.items()
            if name in config.CLASS_TO_IDX
        }

        self.mnv4 = timm.create_model(mnv4_model_id, pretrained=False,
                                      num_classes=config.NUM_CLASSES)
        state = torch.load(mnv4_weights, map_location="cpu", weights_only=False)
        self.mnv4.load_state_dict(state)
        self.mnv4.eval()

        self.mnv4_r8 = None
        if use_triple and Path(mnv4_r8_weights).exists():
            self.mnv4_r8 = timm.create_model(mnv4_model_id, pretrained=False,
                                             num_classes=config.NUM_CLASSES)
            state_r8 = torch.load(mnv4_r8_weights, map_location="cpu", weights_only=False)
            self.mnv4_r8.load_state_dict(state_r8)
            self.mnv4_r8.eval()

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        logger.info("Ensemble (PyTorch) loaded: v8lb=%s, mnv4=%s, r8=%s (weights=%s)",
                    Path(v8lb_weights).name, Path(mnv4_weights).name,
                    Path(mnv4_r8_weights).name if self.mnv4_r8 else "-", weights)

    def _v8lb_probs(self, images: list[np.ndarray]) -> np.ndarray:
        """v8lb probs — ONNX or Ultralytics."""
        if self._ort is not None:
            # ONNX: normalize + forward
            arr = _preprocess_rgb_chw(images, self.imgsz, self.mean_np, self.std_np)
            logits = self.v8lb_sess.run(None, {self.v8lb_sess.get_inputs()[0].name: arr})[0]
            probs_raw = _softmax(logits, axis=1)
            # remap indices
            batch = np.zeros((len(images), config.NUM_CLASSES), dtype=np.float32)
            for i in range(probs_raw.shape[1]):
                if i in self._v8lb_idx_to_our:
                    batch[:, self._v8lb_idx_to_our[i]] = probs_raw[:, i]
            return batch
        # Fallback
        results = self.v8lb.predict(images, imgsz=self.imgsz, verbose=False, device="cpu")
        batch = np.zeros((len(images), config.NUM_CLASSES), dtype=np.float32)
        for n, r in enumerate(results):
            data = r.probs.data.cpu().numpy() if hasattr(r.probs.data, "cpu") \
                else r.probs.data.numpy()
            for i, p in enumerate(data):
                if i in self._v8lb_idx_to_our:
                    batch[n, self._v8lb_idx_to_our[i]] = p
        return batch

    def _mnv4_probs(self, images: list[np.ndarray]) -> np.ndarray:
        """MNv4 R15 probs — ONNX or timm."""
        if self._ort is not None:
            arr = _preprocess_rgb_chw(images, self.mnv4_imgsz, self.mean_np, self.std_np)
            logits = self.mnv4_sess.run(None, {self.mnv4_sess.get_inputs()[0].name: arr})[0]
            return _softmax(logits, axis=1)
        arr = np.stack([
            cv2.cvtColor(cv2.resize(img, (self.mnv4_imgsz, self.mnv4_imgsz)),
                         cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            for img in images
        ], axis=0).transpose(0, 3, 1, 2)
        tensor = self.torch.from_numpy(arr)
        tensor = (tensor - self.mean) / self.std
        with self.torch.no_grad():
            logits = self.mnv4(tensor)
            probs = self.torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def _mnv4_r8_probs(self, images: list[np.ndarray]) -> np.ndarray:
        """MNv4 R8 probs — ONNX or timm."""
        if self._ort is not None:
            arr = _preprocess_rgb_chw(images, self.mnv4_r8_imgsz, self.mean_np, self.std_np)
            logits = self.mnv4_r8_sess.run(None, {self.mnv4_r8_sess.get_inputs()[0].name: arr})[0]
            return _softmax(logits, axis=1)
        arr = np.stack([
            cv2.cvtColor(cv2.resize(img, (self.mnv4_r8_imgsz, self.mnv4_r8_imgsz)),
                         cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            for img in images
        ], axis=0).transpose(0, 3, 1, 2)
        tensor = self.torch.from_numpy(arr)
        tensor = (tensor - self.mean) / self.std
        with self.torch.no_grad():
            return self.torch.softmax(self.mnv4_r8(tensor), dim=1).cpu().numpy()

    def predict_batch(self, images: list[np.ndarray]) -> tuple[list[str], list[float]]:
        """Triple ensemble prediction via soft-voting."""
        if not images:
            return [], []
        v8 = self._v8lb_probs(images)
        mn = self._mnv4_probs(images)
        if self.use_triple and self.mnv4_r8 is not None:
            r8 = self._mnv4_r8_probs(images)
            w_v8, w_r8, w_r14 = self.weights if len(self.weights) == 3 else (0.34, 0.33, 0.33)
            ensemble_probs = w_v8 * v8 + w_r8 * r8 + w_r14 * mn
        else:
            w_v8 = self.weights[0]
            w_mn = self.weights[-1] if len(self.weights) == 2 else (1 - w_v8)
            ensemble_probs = w_v8 * v8 + w_mn * mn
        top = ensemble_probs.argmax(axis=1)
        preds = [config.IDX_TO_CLASS[int(i)] for i in top]
        confs = [float(ensemble_probs[n, i]) for n, i in enumerate(top)]
        return preds, confs

    # ── TTA ────────────────────────────────────────────────────────

    def _tta_views(self, img: np.ndarray) -> list[np.ndarray]:
        """hflip + center + TL crop + BR crop (4 views total)."""
        H, W = img.shape[:2]
        center = img
        flipped = img[:, ::-1, :].copy()
        # 80% crops
        ch, cw = int(H * 0.85), int(W * 0.85)
        tl = img[:ch, :cw]
        br = img[H - ch:, W - cw:]
        return [center, flipped, tl, br]

    def predict_batch_tta(self, images: list[np.ndarray]) -> tuple[list[str], list[float]]:
        """TTA + triple ensemble (4 views × 3 models)."""
        if not images:
            return [], []
        n = len(images)
        expanded = []
        for img in images:
            expanded.extend(self._tta_views(img))
        v8 = self._v8lb_probs(expanded).reshape(n, 4, config.NUM_CLASSES).mean(axis=1)
        mn = self._mnv4_probs(expanded).reshape(n, 4, config.NUM_CLASSES).mean(axis=1)
        if self.use_triple and self.mnv4_r8 is not None:
            r8 = self._mnv4_r8_probs(expanded).reshape(n, 4, config.NUM_CLASSES).mean(axis=1)
            w_v8, w_r8, w_r14 = self.weights if len(self.weights) == 3 else (0.34, 0.33, 0.33)
            ensemble_probs = w_v8 * v8 + w_r8 * r8 + w_r14 * mn
        else:
            w_v8 = self.weights[0]
            w_mn = self.weights[-1] if len(self.weights) == 2 else (1 - w_v8)
            ensemble_probs = w_v8 * v8 + w_mn * mn
        top = ensemble_probs.argmax(axis=1)
        preds = [config.IDX_TO_CLASS[int(i)] for i in top]
        confs = [float(ensemble_probs[n_, i]) for n_, i in enumerate(top)]
        return preds, confs

    # ── Rule-based Postprocess ─────────────────────────────────────

    @staticmethod
    def postprocess(
        pred: str,
        bbox_area: int | None = None,
        aspect_ratio: float | None = None,
        coco_cls: int | None = None,
        probs: np.ndarray | None = None,
    ) -> str:
        """Rule-based correction (applied after classifier).

        - T13 direct: coco_cls==3 AND area<15k → T13
        - T10 promotion: pred==T5 AND area>120k AND aspect>2.5 → T10
        - T2 suppression (Tier A.2): pred==T2 AND aspect<0.7 → 2nd-best class
          (버스는 세로로 길거나 정방에 가까움. aspect<0.7은 가로로 누운 형태 = 트럭/승용차 오분류)
        """
        if coco_cls == 3 and bbox_area is not None and bbox_area < 15000:
            return "T13"
        if pred == "T5" and bbox_area is not None and aspect_ratio is not None:
            if bbox_area > 120000 and aspect_ratio > 2.5:
                return "T10"
        # T2 AR rule은 테스트 결과 Urban 성능을 -14%p 하락시켜 폐기.
        # T2 편향은 class-level 가중 조정이나 학습 재수행이 더 안전.
        return pred

    def predict_batch_with_meta(
        self, images: list[np.ndarray],
        metas: list[dict] | None = None,
        use_tta: bool = False,
    ) -> tuple[list[str], list[float]]:
        """Full prediction with optional TTA + rule postprocess.

        metas[i] dict keys (optional): 'bbox_area', 'aspect_ratio', 'coco_cls'
        """
        # Tier A.2: T2 AR rule needs probs — save intermediate
        if use_tta:
            preds, confs, probs = self._predict_tta_with_probs(images)
        else:
            preds, confs, probs = self._predict_with_probs(images)
        if metas:
            preds = [
                self.postprocess(
                    p, bbox_area=m.get("bbox_area"),
                    aspect_ratio=m.get("aspect_ratio"),
                    coco_cls=m.get("coco_cls"),
                    probs=probs[i],
                )
                for i, (p, m) in enumerate(zip(preds, metas))
            ]
        return preds, confs

    def _predict_with_probs(self, images):
        """predict_batch + return probs for post-processing."""
        if not images:
            return [], [], np.zeros((0, config.NUM_CLASSES))
        v8 = self._v8lb_probs(images)
        mn = self._mnv4_probs(images)
        if self.use_triple and self.mnv4_r8 is not None:
            r8 = self._mnv4_r8_probs(images)
            w_v8, w_r8, w_r14 = self.weights if len(self.weights) == 3 else (0.34, 0.33, 0.33)
            ensemble_probs = w_v8 * v8 + w_r8 * r8 + w_r14 * mn
        else:
            w_v8 = self.weights[0]
            w_mn = self.weights[-1] if len(self.weights) == 2 else (1 - w_v8)
            ensemble_probs = w_v8 * v8 + w_mn * mn
        top = ensemble_probs.argmax(axis=1)
        preds = [config.IDX_TO_CLASS[int(i)] for i in top]
        confs = [float(ensemble_probs[n, i]) for n, i in enumerate(top)]
        return preds, confs, ensemble_probs

    def _predict_tta_with_probs(self, images):
        """predict_batch_tta + return probs."""
        if not images:
            return [], [], np.zeros((0, config.NUM_CLASSES))
        n = len(images)
        expanded = []
        for img in images:
            expanded.extend(self._tta_views(img))
        v8 = self._v8lb_probs(expanded).reshape(n, 4, config.NUM_CLASSES).mean(axis=1)
        mn = self._mnv4_probs(expanded).reshape(n, 4, config.NUM_CLASSES).mean(axis=1)
        if self.use_triple and self.mnv4_r8 is not None:
            r8 = self._mnv4_r8_probs(expanded).reshape(n, 4, config.NUM_CLASSES).mean(axis=1)
            w_v8, w_r8, w_r14 = self.weights if len(self.weights) == 3 else (0.34, 0.33, 0.33)
            ensemble_probs = w_v8 * v8 + w_r8 * r8 + w_r14 * mn
        else:
            w_v8 = self.weights[0]
            w_mn = self.weights[-1] if len(self.weights) == 2 else (1 - w_v8)
            ensemble_probs = w_v8 * v8 + w_mn * mn
        top = ensemble_probs.argmax(axis=1)
        preds = [config.IDX_TO_CLASS[int(i)] for i in top]
        confs = [float(ensemble_probs[n_, i]) for n_, i in enumerate(top)]
        return preds, confs, ensemble_probs

    def latency_ms(self, images: list[np.ndarray], n_iter: int = 10) -> float:
        single = images[0:1]
        self.predict_batch(single)  # warmup
        t0 = time.perf_counter()
        for _ in range(n_iter):
            self.predict_batch(single)
        return (time.perf_counter() - t0) / n_iter * 1000.0


def load_ensemble(
    v8lb_weights: str | None = None, mnv4_weights: str | None = None,
    mnv4_r8_weights: str | None = None,
    imgsz: int = 128,
    weights: tuple[float, float, float] = (0.34, 0.33, 0.33),
    use_triple: bool = True,
) -> EnsembleClassifier:
    """Triple ensemble 기본값: R12 v8lb + R8 MNv4 + R14 MNv4 + TTA."""
    v8lb = v8lb_weights or str(config.PRESERVED_DIR / "round_12_v8lb_best.pt")
    mnv4 = mnv4_weights or str(config.PRESERVED_DIR / "round_15_mnv4_balanced.pt")
    mnv4_r8 = mnv4_r8_weights or str(config.PRESERVED_DIR / "round_8_mnv4_best.pt")
    return EnsembleClassifier(
        v8lb, mnv4, mnv4_r8_weights=mnv4_r8,
        imgsz=imgsz, weights=weights, use_triple=use_triple,
    )


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("image_path")
    args = p.parse_args()
    img = cv2.imread(args.image_path)
    clf = load_ensemble()
    preds, confs = clf.predict_batch([img])
    print(f"{args.image_path}: {preds[0]} (conf={confs[0]:.3f})")
    lat = clf.latency_ms([img], n_iter=5)
    print(f"latency: {lat:.2f}ms")
