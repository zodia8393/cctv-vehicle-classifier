"""재설계 파이프라인 상수 정의.

경동고속도로 23 IC CCTV 차종분류 시스템 (2026-04-21 재설계).
자동 라벨 오염으로 인한 성능 천장(GT 75.4%)을 해결하기 위해
수동 검수 기반 고순도 데이터로 처음부터 재구축하는 설계의 중심 설정.
"""

from pathlib import Path


# ── 경로 ────────────────────────────────────────────────────────────
ROOT_DIR          = Path("/workspace/prj_cctv")
PIPELINE_DIR      = ROOT_DIR / "pipeline"
DATA_DIR          = PIPELINE_DIR / "data"
OUTPUT_DIR        = ROOT_DIR / "output"

VIDEO_SOURCE_DIR  = Path("/mnt/Expansion/영상/220603_고속도로(경동)")
MODEL_DIR         = DATA_DIR / "models"
PRESERVED_DIR     = OUTPUT_DIR / "_models_preserved"

GT_DIR            = DATA_DIR / "holdout_gt_v2"
ROUNDS_DIR        = DATA_DIR / "rounds"
PARTITION_FILE    = DATA_DIR / "video_partition.sha256"


# ── 검출기 (YOLO11n) ────────────────────────────────────────────────
YOLO_MODEL        = str(MODEL_DIR / "yolo11n.pt")
YOLO_FALLBACK     = str(MODEL_DIR / "yolov8n.pt")
DET_CONF          = 0.25
DET_IMG           = 640
# COCO vehicle class indices (car=2, motorcycle=3, bus=5, truck=7)
COCO_VEHICLE_IDS  = {2, 3, 5, 7}


# ── 트래커 (ByteTrack via supervision) ─────────────────────────────
TRACK_THRESH      = 0.25
TRACK_BUFFER      = 30    # detect_interval 비활성 (기본값 복원)
MATCH_THRESH      = 0.8   # 0.85 시도는 association 실패 급증 (실측 -91%) → 원복
MIN_LIVED_FRAMES  = 5


# ── 분류 (7-class T-code) ──────────────────────────────────────────
VEHICLE_CLASSES = {
    "T1":  "승용차",
    "T2":  "버스",
    "T3":  "소형화물",
    "T4":  "중형화물",
    "T5":  "대형화물",
    "T10": "세미트레일러",
    "T13": "이륜차",
}
CLASS_ORDER       = ["T1", "T2", "T3", "T4", "T5", "T10", "T13"]
NUM_CLASSES       = len(CLASS_ORDER)
CLASS_TO_IDX      = {c: i for i, c in enumerate(CLASS_ORDER)}
IDX_TO_CLASS      = {i: c for c, i in CLASS_TO_IDX.items()}

# COCO fallback 매핑 (검출 전용, 최종 분류 아님)
COCO_TO_7CLASS    = {2: "T1", 3: "T13", 5: "T2", 7: "T4"}


# ── Holdout GT v2 ──────────────────────────────────────────────────
GT_PER_CLASS           = 150   # T10 제외 전 클래스
GT_T10                 = 300   # T10 특화 (24.8% → 60% 목표)
GT_AUTO_ACCEPT_RATE_MAX = 0.05  # Claude 자동 수용 비율 상한(나머지 수동 확정)


# ── Video partition ────────────────────────────────────────────────
GT_POOL_SIZE      = 600    # 2,581 중 GT 구축 전용
# 학습 풀은 나머지 (~1,981)
PARTITION_SEED    = 20260421


# ── 학습 ───────────────────────────────────────────────────────────
TRAIN_LR          = 1e-4
TRAIN_EPOCHS      = 10
TRAIN_IMG_SIZE    = 128     # classifier
TRAIN_PATIENCE    = 3
TRAIN_WEIGHTS_FROM = str(PRESERVED_DIR / "v8lb_self_best.pt")  # warm-start 기준선


# ── Stage 1 zero-shot 후보 ─────────────────────────────────────────
ZEROSHOT_CANDIDATES = [
    # 기준선: v8lb_self (이전 확정, 7-class 학습 완료, GT 75.4%)
    {"backend": "ultralytics_yolo11n", "model_id": str(PRESERVED_DIR / "v8lb_self_best.pt"),
     "label": "v8lb_self (baseline)"},
    # 후보 1: YOLO11n-cls ImageNet pretrained (head fine-tune 필요)
    {"backend": "ultralytics_yolo11n", "model_id": str(PRESERVED_DIR / "yolo11n-cls.pt"),
     "label": "yolo11n-cls (pretrained head)"},
    # 후보 2: MobileNetV4 (head random-init → 낮은 acc 예상, 지연만 비교)
    {"backend": "timm_mnv4", "model_id": "mobilenetv4_conv_small.e2400_r224_in1k",
     "label": "MobileNetV4-Conv-S"},
    # 후보 3: EfficientViT-B0
    {"backend": "timm_efficientvit", "model_id": "efficientvit_b0.r224_in1k",
     "label": "EfficientViT-B0"},
]
PILOT_SIZE         = 200
PILOT_T10_MIN      = 0.40     # T10 통과 조건
PILOT_LATENCY_MAX  = 3.0      # ms/img (CPU)


# ── Release gate ───────────────────────────────────────────────────
GATE_OVERALL      = 0.85
GATE_T10          = 0.60
GATE_VAL_GT_GAP   = 0.05
MAX_ROUNDS        = 5
ROUND_HARD_CAP    = 3    # 실제 운영 상한 (플랜상 3라운드)


# ── 중단 조건 (7개) ────────────────────────────────────────────────
STOP_DROP_HOLDOUT   = -0.02  # 홀드아웃 -2%p
STOP_CONF_INFLATE   = 0.05   # conf 인플레이션 +5%p
STOP_CLASS_LOSS_PCT = -0.30
STOP_CLASS_LOSS_ABS = 50
STOP_STAGNATION_EPS = 0.01
STOP_STAGNATION_N   = 2
STOP_GAP_REVERSAL   = 0.10   # val-GT 갭 재확대 (재발 감지)
STOP_REVIEW_LEAK    = 0.10   # Claude 검수 누수율 >10%


# ── 영상 일관성 (Stage 5) ──────────────────────────────────────────
VIDEO_VOTE_MIN_CONF = 0.15
VIDEO_VOTE_MIN_FRAMES = 5
VIDEO_VOTE_AREA_WEIGHT = True  # bbox 면적 가중
