# CCTV 영상 기반 차종 분류 시스템 — 프로젝트 계획서 (v2)

> 고속도로 IC 교차로 CCTV 영상에서 차량을 검출·추적·분류하여 차종별 교통량을 집계한다.

**최종 갱신:** 2026-04-07 | **상태:** Phase 4 진행 중 (전체 재처리 27%)

---

## 1. 프로젝트 개요

| 항목 | 내용 |
|------|------|
| **프로젝트명** | CCTV 영상 기반 차종 분류 시스템 |
| **목적** | CCTV 영상 → 차량 검출 → 차종 분류 → 교통량 집계 |
| **산출물** | 지점×시간대×차종×방향별 교통량 데이터 (CSV/Excel) + fine-tuned 분류 모델 |
| **소스 코드** | `/workspace/prj_cctv/pipeline/src/` (24개 스크립트) |
| **결과 데이터** | `/workspace/prj_cctv/output/` |
| **영상 데이터** | `/mnt/Expansion/영상/220603_고속도로(경동)/` |

---

## 2. 입력 데이터 현황

### 2.1 영상 데이터

| 항목 | 값 |
|------|------|
| 촬영일 | 2022.06.03 |
| 촬영처 | 경동고속도로 IC 교차로 |
| 지점 수 | **23개 IC** (34개 카메라) |
| 총 파일 수 | **2,581개 AVI** |
| 총 용량 | **409 GB** |
| 해상도 | 1920×1080 |
| 프레임 레이트 | 10 fps |
| 코덱 | MPEG-4 (FMP4) |
| 파일당 길이 | ~22분 (1,330초) |
| 총 영상 길이 | 864시간 |

### 2.2 촬영 지점 (23개, 카탈로그 완료)

부평IC교차로(3카메라), 부평IC사거리, 판교IC앞, 오산IC앞(2), 산업길사거리, 부천IC교차로(3), 한국화장품삼거리, 목감IC앞(2), 목감IC합류부, 양지IC사거리(2), 덕평IC앞, 서청주IC삼거리(2), 서청주교사거리, 안성IC삼거리, 신탄진IC앞, 판암IC앞(2), 유성IC삼거리(3), 북대전IC네거리, 중외공원입구삼거리, 문화사거리, 문흥지구입구교차로, 광산교차로, 나들목사거리

---

## 3. 실행 환경

| 항목 | 사양 |
|------|------|
| OS | Ubuntu 25.10 |
| CPU | Intel Core Ultra 9 285K (24코어) |
| RAM | 122 GB |
| GPU | **없음** (CPU only) |
| Python | 3.13.7 |
| PyTorch | 2.11.0 (CPU 전용) |

---

## 4. 차종 분류 체계 — 6종 (확정)

### 4.1 채택 분류

CCTV 외관 한계로 13종 → **6종 확정** (10종 트레일러는 일부 포함)

| 코드 | 분류 | 설명 |
|------|------|------|
| **T1** | 1종 승용차 | 세단, SUV, 경차 |
| **T2** | 2종 버스 | 시내/고속/관광버스 |
| **T4** | 4종 중형화물 | 2.5~8.5톤 탑차/카고 |
| **T5** | 5종 대형화물 | 8.5톤+, 덤프, 대형 단차 |
| **T10** | 10종 세미5축 | 컨테이너 트레일러 |
| **T13** | 13종 이륜차 | 오토바이, 스쿠터 |

### 4.2 제외/통합 사유

| 종 | 처리 | 사유 |
|----|------|------|
| 3종 소형화물 | T1로 흡수 | DINOv2 클러스터링 결과 90%가 작은 SUV/세단 |
| 6종 (4축 단차) | 미분류 | CCTV로 차축 판별 불가 |
| 7종 | 미분류 | 데이터 부족 |
| 8종 (4축 세미) | T4로 흡수 | 모델 분리 실패 (정확도 0%) |
| 9종 (풀트레일러) | 미분류 | 8종과 외관 동일 |
| 11/12종 | 미분류 | 데이터 부족 |

---

## 5. 시스템 아키텍처

### 5.1 처리 파이프라인 (최종)

```
AVI 영상
   ↓
[1] Frame Extraction
   - OpenCV VideoCapture
   - 1fps 샘플링 (grab+retrieve 최적화)
   - OSD 타임스탬프 OCR (easyocr)
   ↓
[2] Vehicle Detection
   - YOLOv8n COCO pretrained
   - car/bus/truck/motorcycle 4종
   - conf 0.35, iou 0.45
   ↓
[3] Tracking
   - ByteTrack (supervision)
   - 차량 ID + 이동방향(8방향) + best crop
   ↓
[4] Vehicle Classification (NEW)
   - YOLOv8n-cls fine-tuned (cls_v5_best.pt)
   - 6종 (T1, T2, T4, T5, T10, T13)
   - 트래킹된 차량의 best crop만 분류
   ↓
[5] Aggregation
   - 영상별 + 차량별 CSV
   - 시간/지점/차종/방향
   ↓
[6] Visualization
   - 지점별 교통량 차트
   - 차종 구성비 + 시간대별
```

### 5.2 디렉토리 구조 (현행)

```
/workspace/prj_cctv/pipeline/      # 코드 (쓰기 가능)
├── PROJECT_PLAN.md                    ← 본 문서
├── README.md
├── requirements.txt
├── docs/
├── src/                               # 24개 스크립트
│   ├── extract_embeddings.py          # DINOv2 임베딩
│   ├── recluster_k200.py              # 클러스터링
│   ├── confident_learning.py          # CL
│   ├── manual_correction.py           # 수동 보정
│   ├── build_dataset_v5.py            # 학습 데이터셋
│   ├── train_v5.py                    # fine-tune
│   ├── self_training.py               # Self-training
│   ├── run_final_inference.py         # 최종 추론
│   ├── render_tracking_video.py       # 트래킹 영상 렌더
│   └── ... (기타)
└── logs/                              # 10개 로그

/workspace/prj_cctv/output/        # 결과
├── embeddings.npz                     # DINOv2 임베딩 201MB
├── cluster_assignments_v2.npz         # K=200 클러스터
├── final_labels_v5_self.csv           # 최종 라벨
├── cls_v4_best.pt                     # v4 모델
├── cls_v5_best.pt                     # v5 모델 (최종)
├── tracking_demo.mp4                  # 트래킹 데모 81MB
├── final_v5/                          # 전체 추론 결과 (진행중)
│   ├── v5_files.csv
│   └── v5_vehicles.csv
├── labels/
│   ├── verified_clean/                # 147,521장 (필터링됨)
│   └── verified_13cls/                # 13종 분류 (심링크)
├── dataset_v4/, dataset_v5/           # 학습용 디렉토리
├── training_v4/, training_v5/         # 학습 결과
├── clusters_v2/                       # 200개 클러스터 그리드
└── gt_samples/, gt_v5/                # GT 검증 그리드

/mnt/Expansion/CCTV차종분류/           # 영상 + 모델 (읽기 전용)
├── data/
│   ├── models/yolov8n.pt              # 검출 모델
│   └── labels/verified/               # 162K crop (이전 단계)
└── ...
```

---

## 6. 단계별 실행 결과

### Phase 0: 검출 검증 ✅ (완료)

| 지표 | 결과 |
|------|------|
| 모델 | YOLOv8n COCO |
| 처리 속도 | 33-37 fps (CPU) |
| 검증 | 3개 지점 OK |

### Phase 1: 트래킹 + 카운팅 ✅ (완료)

| 지표 | 결과 |
|------|------|
| 처리 시간 | 15.9시간 (전체 2,581 영상) |
| 총 차량 | **496,502대 검출** |
| 차종 (4종) | 승용차 83.2%, 중형화물 13.3%, 버스 3.4%, 이륜차 0.2% |
| 산출물 | batch_all.csv, batch_pivot.csv, summary.xlsx |

### Phase 2: 6종 분류기 fine-tune ✅ (완료)

#### 2.1 Crop 수집

| 단계 | 결과 |
|------|------|
| 영상 처리 | 2,581 / 2,581 |
| 수집 crop | **162,545장** |
| 품질 필터 | 90.8% 통과 → **147,521장** (verified_clean) |

#### 2.2 라벨링 (DINOv2 + Clustering)

| 단계 | 결과 |
|------|------|
| 임베딩 (DINOv2-small) | 147K장 × 384dim, 27.6분 |
| UMAP 차원축소 | 384 → 32 |
| K-Means 클러스터링 | K=200 |
| bbox 필터 (< 8K) | 27.5% 노이즈 |
| Confident Learning | 1.6% 추가 정제 |
| 학습 가능 데이터 | **96,758장** |

#### 2.3 라벨 분포 (v5 최종)

| 라벨 | 수량 | 비율 |
|------|------|------|
| T1 1종 승용차 | 70,542 | 49.5% |
| T2 2종 버스 | 5,133 | 3.4% |
| T4 4종 중형화물 | 14,718 | 9.7% |
| T5 5종 대형화물 | 2,719 | 2.1% |
| T10 10종 세미5축 | 989 | 0.9% |
| T13 13종 이륜차 | 137 | 0.1% |
| NOISE (제외) | 50,523 | 34.3% |
| **학습 가능** | **94,238** | 65.7% |

#### 2.4 모델 학습 결과

| 모델 | Val Top-1 | GT 정확도 |
|------|-----------|----------|
| **cls_v4_best.pt** | 93.7% | 79% |
| **cls_v5_best.pt** | **99.3%** ⚡ | **93%** |

**v5 Confusion Matrix (정규화):**

| 클래스 | Recall |
|--------|--------|
| T1 승용차 | 100% |
| T2 버스 | 99% |
| T4 중형화물 | 100% |
| T5 대형화물 | 98% |
| T10 세미5축 | 99% |
| T13 이륜차 | 100% |

### Phase 3: 검증 ✅ (완료)

| 검증 방식 | 결과 |
|----------|------|
| Val 분할 (학습 시) | 99.3% |
| GT 시각 검증 (Claude, 120장) | **93%** |
| Self-training v4↔v5 일치율 | **98.2%** |

### Phase 4: 전체 재처리 🔄 (진행 중)

**현재 상태 (2026-04-07 16:17 기준):**

| 항목 | 값 |
|------|------|
| 진행 | **693 / 2,581** (26.8%) |
| 경과 | 4시간 15분 |
| 영상당 | ~22초 |
| ETA | ~11시간 34분 (내일 새벽 4시경) |
| PID | 538159 |
| CPU | 778% (정상) |

**산출물 (예정):**
- `final_v5/v5_files.csv` — 영상별 + 차종별
- `final_v5/v5_vehicles.csv` — 개별 차량 (시간/방향/차종/confidence)

### Phase 5: 최종 보고서 ⏳ (대기)

- 시간대별 교통량 차트
- 지점×차종 피벗
- 방향별 회전 교통량
- 정확도 검증 리포트
- 한글 폰트 적용된 시각화

---

## 7. 핵심 기술 결정 (ADR — 최종)

### ADR-001: 차종 분류 — 6종

- **결정:** 13종 → 6종 (T1, T2, T4, T5, T10, T13)
- **근거:** CCTV 외관으로는 차축 판별 불가, GT 검증으로 5/6/7/8/9/11/12종 분리 실패

### ADR-002: 검출 모델 — YOLOv8n

- **결정:** YOLOv8n (nano)
- **근거:** CPU 환경에서 실시간 처리 가능, 정확도 충분

### ADR-003: 트래커 — ByteTrack

- **결정:** supervision의 ByteTrack
- **근거:** 1fps 샘플링에서도 안정적

### ADR-004: 처리 방식 — 오프라인 배치

- **결정:** 단일 프로세스 순차 (멀티프로세싱 경합 발견)
- **근거:** 8워커 시 CPU 경합으로 처리 시간 100배 증가

### ADR-005: 프레임 샘플링 — 1fps + grab() 최적화

- **결정:** 10fps → 1fps + skip 프레임에 grab() 사용
- **근거:** 디코딩 비용 절감

### ADR-006: 라벨링 — DINOv2 + KMeans + Confident Learning ⭐ NEW

- **결정:** Self-supervised 임베딩 + 클러스터링 + CL
- **근거:** 자동 분류기보다 높은 품질, 사람 노력 1/100

### ADR-007: 분류 모델 — YOLOv8n-cls fine-tuned ⭐ NEW

- **결정:** YOLOv8n-cls (warm start v4 → v5)
- **근거:** 2.9MB 경량, val 99.3% / GT 93% 달성

### ADR-008: Self-training 적용 ⭐ NEW

- **결정:** v4 모델로 재라벨링 → v5 학습
- **근거:** 일치율 98.2%로 안정 수렴

---

## 8. 성과 지표

### 8.1 검출 성능

| 메트릭 | 목표 | 실측 |
|--------|------|------|
| Detection mAP@50 | 0.75+ | YOLOv8n 기본 ~0.75 |
| 처리 속도 | 1fps 실시간 | 6 fps (CPU) ✅ |

### 8.2 분류 성능

| 메트릭 | 목표 | 실측 |
|--------|------|------|
| 6종 top-1 accuracy | 0.85+ | **0.993** ⚡ |
| GT 검증 평균 | 0.85+ | **0.93** ⚡ |
| 클래스별 최저 (Recall) | 0.75+ | 0.98 (T5) |

### 8.3 모델 사양

| 항목 | 값 |
|------|------|
| Parameters | ~1.5M |
| Model size | 2.9 MB |
| Input | 128×128 RGB |
| Inference (CPU) | ~0.3 ms/image |
| Inference (RTX 5090 추정) | ~0.04 ms/image |

---

## 9. 발견된 문제 + 대응 이력

| # | 문제 | 대응 |
|---|------|------|
| 1 | 8워커 멀티프로세싱 시 CPU 경합 | 단일 프로세스 순차로 전환 |
| 2 | 디스크 마운트 변경 (`/media/ybs` → `/mnt/Expansion`) | resume_batch.py로 monkeypatch |
| 3 | exfat ybs1 권한 문제 | 출력 경로 `/workspace`로 우회 |
| 4 | C3 소형화물 90% 오분류 (자동재분류 룰 한계) | 수동 보정으로 T1 흡수 |
| 5 | T8 세미트레일러 0% 정확도 | T4로 흡수 |
| 6 | 한글 폰트 깨짐 | NotoSansCJK TTC 직접 로딩 |
| 7 | OCR 실패율 | easyocr 4배 확대 + 정규식 보정 |
| 8 | tesseract 미설치 (sudo 없음) | easyocr 우선 사용 |

---

## 10. GPU 마이그레이션 대비 (RTX 5090 기준)

| 항목 | CPU (현재) | RTX 5090 |
|------|----------|---------|
| YOLO 검출 | 30 fps | 3,000 fps (FP32) |
| 분류기 | 0.3ms | 0.04ms |
| 우리 파이프라인 | 6 fps | **600 fps** |
| 23채널 동시 처리 | 1fps만 | 10fps 원본 (사용률 5%) |
| 전체 배치 (15시간) | - | **~15분** 추정 |

**마이그레이션 작업:** `device="cuda"` 1줄, 선택적 TensorRT 변환

---

## 11. 산출물 목록 (현재까지)

### 데이터
- `embeddings.npz` (201MB) — DINOv2 임베딩 147K
- `cluster_assignments_v2.npz` — K=200 클러스터
- `final_labels_v5_self.csv` — 최종 라벨

### 모델
- `cls_v4_best.pt` (2.9MB) — fine-tuned v4
- **`cls_v5_best.pt` (2.9MB)** — fine-tuned v5 (운영용)

### 결과
- `tracking_demo.mp4` (81MB) — 트래킹 시각화
- `final_v5/v5_files.csv` — 영상별 (진행 중)
- `final_v5/v5_vehicles.csv` — 차량별 (진행 중)

### 시각화
- `gt_v5/` — 클래스별 GT 검증 그리드
- `clusters_v2/grids/` — 200개 클러스터 그리드
- `training_v5/run1/confusion_matrix_normalized.png`

---

## 12. 의존성

```
# 핵심 ML
ultralytics>=8.3        # YOLO 검출 + 분류
torch>=2.11             # PyTorch
transformers>=5.0       # DINOv2
timm>=1.0               # 비전 모델

# 라벨링/클러스터링
umap-learn>=0.5
scikit-learn>=1.7
hdbscan>=0.8
cleanlab>=2.9           # Confident Learning

# 영상 처리
opencv-python>=4.13
supervision>=0.27       # ByteTrack

# OCR
easyocr>=1.7            # OSD 타임스탬프

# 데이터/시각화
pandas, openpyxl, matplotlib, tqdm
```

---

## 13. 다음 단계

### 즉시 (Phase 4 완료 후)
1. 시간대별 교통량 분석
2. 지점×차종 피벗 + 시각화
3. 방향별 회전 교통량 산출
4. 최종 보고서 (PDF)

### 중기 (실무 배포)
1. GPU 환경 마이그레이션 (`cuda` 1줄)
2. RTSP 실시간 스트림 입력
3. DB 적재 (PostgreSQL / TimescaleDB)
4. 운영 대시보드 (Grafana)

### 장기 (정확도 개선)
1. 수동 GT 1,000장 라벨링 (실측 정확도)
2. T8/T9/T10 세분류 위해 측면 카메라 데이터 확보
3. 야간 영상 전용 모델 (저조도 fine-tune)
4. Hard negative mining (오분류 케이스 재학습)
