"""ByteTrack 기반 차량 추적

검출 결과에 차량 ID를 부여하여 동일 차량 중복 카운팅을 방지한다.
supervision 라이브러리의 ByteTrack 구현을 사용.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import supervision as sv

from config import TRACK_BUFFER, TRACK_THRESH, MATCH_THRESH

logger = logging.getLogger(__name__)


@dataclass
class TrackedVehicle:
    """추적 중인 차량 정보."""
    tracker_id: int
    vehicle_cls: str          # C1~C6
    vehicle_name: str
    first_seen: float         # timestamp (초)
    last_seen: float
    bbox: list[float] = field(default_factory=list)
    crossed_line: bool = False  # 카운팅 라인 통과 여부
    # 이동방향 추적
    first_center: tuple[float, float] | None = None   # 첫 출현 위치 (cx, cy)
    last_center: tuple[float, float] | None = None    # 마지막 위치
    direction: str = ""       # 이동방향: "→", "←", "↑", "↓", "↗", "↘", "↙", "↖"
    best_crop: np.ndarray | None = field(default=None, repr=False)  # 최선 crop (가장 큰 bbox)
    best_crop_area: int = 0   # best crop의 bbox 면적


class VehicleTracker:
    """ByteTrack 기반 차량 추적기."""

    def __init__(self, fps: int = 30):
        self.fps = int(fps)
        self._tracker = sv.ByteTrack(
            track_activation_threshold=TRACK_THRESH,
            lost_track_buffer=TRACK_BUFFER,
            minimum_matching_threshold=MATCH_THRESH,
            frame_rate=self.fps,
        )
        # tracker_id → TrackedVehicle
        self.vehicles: dict[int, TrackedVehicle] = {}

    def update(self, detections: list[dict], timestamp: float,
               frame: np.ndarray | None = None) -> sv.Detections:
        """검출 결과를 트래커에 전달하고 추적 ID를 부여한다.

        Args:
            detections: detector.detect_frame() 반환값
            timestamp: 현재 프레임 타임스탬프 (초)
            frame: 원본 프레임 (crop 저장용, None이면 crop 안 함)

        Returns:
            supervision.Detections (tracker_id 포함)
        """
        if not detections:
            # 빈 프레임도 트래커에 전달 (lost track 처리)
            empty = sv.Detections.empty()
            self._tracker.update_with_detections(empty)
            return empty

        # supervision Detections 구성
        xyxy = np.array([d["bbox"] for d in detections], dtype=np.float32)
        confidence = np.array([d["conf"] for d in detections], dtype=np.float32)
        class_id = np.array([d["coco_cls"] for d in detections], dtype=int)

        sv_dets = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
        )

        # ByteTrack 업데이트
        tracked = self._tracker.update_with_detections(sv_dets)

        # 차량 정보 갱신: tracked.class_id를 직접 사용 (ByteTrack이 보존)
        if tracked.tracker_id is not None:
            for i, tid in enumerate(tracked.tracker_id):
                tid = int(tid)
                coco_cls = int(tracked.class_id[i]) if tracked.class_id is not None else 2

                from config import COCO_TO_7CLASS, VEHICLE_CLASSES
                v_cls = COCO_TO_7CLASS.get(coco_cls, "T1")
                v_name = VEHICLE_CLASSES.get(v_cls, "미분류")

                x1, y1, x2, y2 = tracked.xyxy[i].tolist()
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                area = int((x2 - x1) * (y2 - y1))

                if tid in self.vehicles:
                    v = self.vehicles[tid]
                    v.last_seen = timestamp
                    v.bbox = [x1, y1, x2, y2]
                    v.last_center = (cx, cy)
                    # best crop 갱신 (가장 큰 bbox = 가장 가까운 순간)
                    if frame is not None and area > v.best_crop_area:
                        ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
                        crop = frame[iy1:iy2, ix1:ix2]
                        if crop.size > 0:
                            v.best_crop = crop.copy()
                            v.best_crop_area = area
                else:
                    v = TrackedVehicle(
                        tracker_id=tid,
                        vehicle_cls=v_cls,
                        vehicle_name=v_name,
                        first_seen=timestamp,
                        last_seen=timestamp,
                        bbox=[x1, y1, x2, y2],
                        first_center=(cx, cy),
                        last_center=(cx, cy),
                    )
                    # 첫 crop 저장
                    if frame is not None and area > 0:
                        ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
                        crop = frame[iy1:iy2, ix1:ix2]
                        if crop.size > 0:
                            v.best_crop = crop.copy()
                            v.best_crop_area = area
                    self.vehicles[tid] = v

        # 완료된 트랙에 이동방향 계산
        self._update_directions()

        return tracked

    def _update_directions(self):
        """각 차량의 이동방향을 first_center→last_center 벡터로 계산."""
        import math
        for v in self.vehicles.values():
            if v.direction or not v.first_center or not v.last_center:
                continue
            dx = v.last_center[0] - v.first_center[0]
            dy = v.last_center[1] - v.first_center[1]
            dist = math.hypot(dx, dy)
            if dist < 30:  # 이동 거리 너무 짧으면 판단 보류
                continue
            angle = math.degrees(math.atan2(-dy, dx))  # y축 반전 (이미지 좌표)
            if -22.5 <= angle < 22.5:
                v.direction = "→"
            elif 22.5 <= angle < 67.5:
                v.direction = "↗"
            elif 67.5 <= angle < 112.5:
                v.direction = "↑"
            elif 112.5 <= angle < 157.5:
                v.direction = "↖"
            elif angle >= 157.5 or angle < -157.5:
                v.direction = "←"
            elif -157.5 <= angle < -112.5:
                v.direction = "↙"
            elif -112.5 <= angle < -67.5:
                v.direction = "↓"
            else:
                v.direction = "↘"

    @property
    def unique_count(self) -> int:
        """지금까지 추적된 고유 차량 수."""
        return len(self.vehicles)

    def class_summary(self) -> dict[str, int]:
        """차종별 고유 차량 수."""
        counts: dict[str, int] = {}
        for v in self.vehicles.values():
            counts[v.vehicle_name] = counts.get(v.vehicle_name, 0) + 1
        return counts

    def reset(self):
        """트래커 초기화."""
        self._tracker.reset()
        self.vehicles.clear()


class LineCrossingCounter:
    """가상 라인 통과 카운터.

    화면을 가로지르는 가상 라인을 설정하고,
    차량이 라인을 통과할 때 카운팅한다.
    """

    def __init__(self, line_start: tuple[int, int], line_end: tuple[int, int]):
        self.line_start = line_start
        self.line_end = line_end
        self.crossed_ids: set[int] = set()
        self.counts: dict[str, int] = {}  # 차종별 카운트

    def update(self, tracker: VehicleTracker, tracked: sv.Detections):
        """추적 결과를 확인하여 라인 통과 차량을 카운팅."""
        if tracked.tracker_id is None:
            return

        for i, tid in enumerate(tracked.tracker_id):
            tid = int(tid)
            if tid in self.crossed_ids:
                continue

            # bbox 하단 중심점
            x1, y1, x2, y2 = tracked.xyxy[i]
            cx = (x1 + x2) / 2
            cy = y2  # 하단

            if self._point_crosses_line(cx, cy):
                self.crossed_ids.add(tid)
                vehicle = tracker.vehicles.get(tid)
                if vehicle:
                    vehicle.crossed_line = True
                    name = vehicle.vehicle_name
                    self.counts[name] = self.counts.get(name, 0) + 1

    def _point_crosses_line(self, px: float, py: float) -> bool:
        """점이 라인 근처(±허용오차)에 있는지 확인."""
        lx1, ly1 = self.line_start
        lx2, ly2 = self.line_end

        # 수평 라인인 경우 (y 기준)
        if abs(ly1 - ly2) < 10:
            target_y = (ly1 + ly2) / 2
            tolerance = 30  # 1fps이므로 넉넉히
            return (abs(py - target_y) < tolerance
                    and min(lx1, lx2) <= px <= max(lx1, lx2))

        # 수직 라인인 경우 (x 기준)
        if abs(lx1 - lx2) < 10:
            target_x = (lx1 + lx2) / 2
            tolerance = 30
            return (abs(px - target_x) < tolerance
                    and min(ly1, ly2) <= py <= max(ly1, ly2))

        # 일반 대각선: 점-직선 거리
        dist = abs((ly2 - ly1) * px - (lx2 - lx1) * py + lx2 * ly1 - ly2 * lx1)
        dist /= ((ly2 - ly1) ** 2 + (lx2 - lx1) ** 2) ** 0.5
        return dist < 30

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def _iou(box_a: list[float], box_b: list[float]) -> float:
    """두 bbox의 IoU 계산."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0
