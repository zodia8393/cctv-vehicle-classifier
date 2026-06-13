"""ONNX export — 한 번만 실행, 이후 .onnx 파일 재사용.

생성 위치: /workspace/prj/cctv/output/_models_preserved/onnx/
- yolo11n.onnx         (detector, 640px)
- round_12_v8lb.onnx   (classifier v8lb, 128px)
- round_15_mnv4.onnx   (classifier MNv4 R15, 192px)
- round_8_mnv4.onnx    (classifier MNv4 R8, 128px)

ensemble_classifier.py는 .onnx 파일 존재하면 자동 사용 (ONNX Runtime).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ONNX_DIR = Path("/workspace/prj/cctv/output/_models_preserved/onnx")
ONNX_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = Path("/workspace/prj/cctv/pipeline/data/models")
PRESERVED = Path("/workspace/prj/cctv/output/_models_preserved")


def export_ultralytics(src_pt: Path, dst_onnx: Path, imgsz: int = 128):
    """Ultralytics YOLO/classify 모델 export."""
    if dst_onnx.exists():
        logger.info("skip (exists): %s", dst_onnx.name)
        return
    from ultralytics import YOLO
    logger.info("exporting %s → %s (imgsz=%d)", src_pt.name, dst_onnx.name, imgsz)
    model = YOLO(str(src_pt))
    # Ultralytics export returns path to .onnx
    out = model.export(format="onnx", imgsz=imgsz, simplify=True, dynamic=False, half=False, verbose=False)
    # 기본 위치에서 target으로 이동
    out = Path(out)
    if out.resolve() != dst_onnx.resolve():
        dst_onnx.parent.mkdir(parents=True, exist_ok=True)
        out.replace(dst_onnx)
    logger.info("✓ %s", dst_onnx)


def export_timm(weights: Path, dst_onnx: Path, imgsz: int = 192,
                model_id: str = "mobilenetv4_conv_small.e2400_r224_in1k"):
    """timm 모델 수동 export."""
    if dst_onnx.exists():
        logger.info("skip (exists): %s", dst_onnx.name)
        return
    import timm, torch
    logger.info("exporting %s → %s (imgsz=%d)", weights.name, dst_onnx.name, imgsz)
    model = timm.create_model(model_id, pretrained=False, num_classes=7)
    state = torch.load(str(weights), map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.eval()
    dummy = torch.randn(1, 3, imgsz, imgsz)
    torch.onnx.export(
        model, dummy, str(dst_onnx),
        opset_version=17,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    logger.info("✓ %s", dst_onnx)


def main():
    # 1. Detector YOLO11n (imgsz=640)
    export_ultralytics(
        MODEL_DIR / "yolo11n.pt",
        ONNX_DIR / "yolo11n.onnx",
        imgsz=640,
    )

    # 2. v8lb R12 classifier (imgsz=128)
    export_ultralytics(
        PRESERVED / "round_12_v8lb_best.pt",
        ONNX_DIR / "round_12_v8lb.onnx",
        imgsz=128,
    )

    # 3. MNv4 R15 (imgsz=192, main classifier)
    export_timm(
        PRESERVED / "round_15_mnv4_balanced.pt",
        ONNX_DIR / "round_15_mnv4.onnx",
        imgsz=192,
    )

    # 4. MNv4 R8 (imgsz=128)
    export_timm(
        PRESERVED / "round_8_mnv4_best.pt",
        ONNX_DIR / "round_8_mnv4.onnx",
        imgsz=128,
    )

    # 요약
    logger.info("=" * 40)
    for f in sorted(ONNX_DIR.glob("*.onnx")):
        size_mb = f.stat().st_size / (1024 * 1024)
        logger.info("  %s  %.1f MB", f.name, size_mb)
    logger.info("Done: ONNX_DIR=%s", ONNX_DIR)


if __name__ == "__main__":
    main()
