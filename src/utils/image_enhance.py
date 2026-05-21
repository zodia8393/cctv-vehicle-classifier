"""밝기 조건부 영상 전처리 (CLAHE + Gamma)

야간/저조도 영상에서 차량 검출 성능 향상.
실측: 야간(밝기<100)에서 +50~150% 검출률 향상.

주의: 밝은 주간(밝기>130)에서는 적용 금지 (오히려 -40%)
"""

import cv2
import numpy as np

# Gamma 0.5 LUT (한 번만 계산)
_GAMMA = 0.5
_GAMMA_TABLE = np.array(
    [((i / 255.0) ** (1.0 / _GAMMA)) * 255 for i in np.arange(256)]
).astype("uint8")

# CLAHE 객체 (재사용)
_CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


def enhance_frame(frame: np.ndarray) -> np.ndarray:
    """프레임 밝기에 따라 적응형 전처리.

    Args:
        frame: BGR 프레임

    Returns:
        개선된 BGR 프레임 (또는 원본)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())

    if brightness < 100:
        # 야간: CLAHE + Gamma 결합
        return _apply_clahe_gamma(frame)
    elif brightness < 130:
        # 새벽/흐림: CLAHE만
        return _apply_clahe(frame)
    else:
        # 주간: 원본
        return frame


def _apply_clahe(frame: np.ndarray) -> np.ndarray:
    """LAB 색공간에서 L 채널에 CLAHE 적용."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = _CLAHE.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _apply_clahe_gamma(frame: np.ndarray) -> np.ndarray:
    """CLAHE + Gamma 0.5 결합."""
    enhanced = _apply_clahe(frame)
    return cv2.LUT(enhanced, _GAMMA_TABLE)
