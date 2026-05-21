# CCTV 차종분류 프로젝트 — 종합 핸드오프

> **작성일**: 2026-04-09 09:49 KST  
> **작성자**: Antigravity AI (이전 세션에서 전체 작업 수행)  
> **대상 독자**: Claude Code (후속 작업용)

---

## 0. 워크스페이스 전체 구조

```
/workspace/
├── AI교통량조사/              ← 상위 프로젝트 (사고보고서 파싱/챗봇/HWP생성)
│   ├── docs/차종분류_참고자료/ ← ★ 12종 분류 기준 원본 이미지 및 지침
│   └── src/                   ← 사고보고서 관련 코드 (본 작업과 무관)
│
├── CCTV차종분류/pipeline/     ← ★ 메인 작업 디렉토리 (차종분류 파이프라인)
│   ├── src/                   ← 파이프라인 소스 코드 전체
│   ├── data/                  ← 학습 데이터셋, 모델 가중치
│   └── PROJECT_PLAN.md        ← 초기 프로젝트 계획서
│
├── CCTV차종분류/output/       ← ★ 모든 산출물 (영상, 로그, 모델, 리포트)
│   ├── tracking_hiv00134.mp4  ← 메인 트래킹 영상
│   ├── batch_videos/          ← 배치 처리된 추가 영상들
│   ├── finetune/              ← 학습된 모델 가중치들
│   └── reports/               ← 분석 리포트
│
└── test.py                    ← 사용자의 별도 스크립트 (본 작업과 무관)
```

---

## 1. 프로젝트 목표

고속도로 CCTV 영상(AVI, 10fps, 1920×1080)에서 차량을 자동으로:
1. **검출** (YOLO)
2. **추적** (ByteTrack)
3. **차종 분류** (YOLOv8-cls)

하여 도로교통량 조사지침의 **12종 분류 체계**에 따른 교통량 데이터를 산출하는 것.

---

## 2. 영상 데이터 위치

```
/mnt/Expansion/영상/220603_고속도로(경동)/12_덕평IC앞교차로/
├── hiv00084.avi ~ hiv00143.avi  (총 60개 영상, 각 약 200MB)
```

- 10fps, 1920×1080, 영상당 약 15,000~22,000프레임
- 고속도로 덕평IC 앞 교차로 지점을 촬영한 CCTV 영상

---

## 3. 파이프라인 아키텍처

```
AVI 영상
 ↓
[1단계] YOLO 검출 (yolov8n.pt + 고해상도 패치)
 │  ★ 몽키패치: imgsz=1280, conf=0.15 (원거리 소형 객체 탐지 극대화)
 │  ★ 나이트비전: 평균 조도 분석 후 저조도시 CLAHE 화질 보정 적용
 │  car(2), motorcycle(3), bus(5), truck(7) 검출
 ↓
[2단계] ByteTrack 추적 (lost_track_buffer=60, match_thresh=0.7)
 │  각 차량에 고유 tracker_id 부여, 궤적 추적
 ↓
[3단계] 하이브리드 분류 v3.2 (5중 교차 검증)
 │  차량 crop → T1~T13 분류 (cls_v5_best.pt)
 │  1. 정면 화물차 방어: 종횡비 작아도 면적 크면 T4/T5 강제 보정
 │  2. 오토바이 방어: YOLO=MOTO면 무조건 T13
 │  3. 버스 오판 방어: YOLO=BUS면 승용 오판 무시하고 T2 강제 환원
 │  4. 노이즈 방어: 30px 미만 초소형 Bbox는 종횡비 기록에서 제외
 │  5. UNK 폴백: 3초 대기 후 타임아웃 시 크기별 COCO 폴백
 ↓
[출력] MP4 시각화 영상 + CSV 로그
```

### 현재 지원하는 차종 클래스

| 코드 | 명칭 | 12종 대응 | 설명 |
|---|---|---|---|
| T1 | 1종 승용차 | 1종 | 승용차, 16인승 이하 승합, 2.5톤 미만 화물 |
| T2 | 2종 버스 | 2종 | 17인~23인 승합, 2.5~5.5톤 화물 |
| T3 | 3종 소형트럭 | 3종 | 포터/봉고급, 1~2.5톤 미만 화물 |
| T4 | 4종 중형화물 | 4종 | 2.5~8.5톤 미만 화물 |
| T5 | 5종 대형화물 | 5종 | 8.5톤 이상 |
| T10 | 10종 세미트레일러 | 10종 | 5축 세미 트레일러 |
| T13 | 13종 이륜차 | — | 오토바이 |

---

## 4. 핵심 소스 파일 (중요도순)

### 4.1 메인 파이프라인
| 파일 | 경로 | 역할 |
|---|---|---|
| **render_tracking_video.py** | `CCTV차종분류/pipeline/src/` | ★ 핵심. 나이트비전, 고해상도 패치, 하이브리드 보정 로직 통합본 |
| **batch_render_videos.py** | 같은 디렉토리 | 다중 영상 순차 처리 래퍼 |

### 4.2 데이터 정제 및 진단
| 파일 | 역할 |
|---|---|
| **refine_tracking.py** | 로그 사후정제 (노이즈 트랙 제거, 단절 보간) |
| **diagnose_tracking.py** | 품질 진단 (Gap, ID Switching, 노이즈 비율) |

### 4.3 모델 학습 관련 (이력 참고용)
| 파일 | 역할 |
|---|---|
| massive_extraction.py | 10개 영상에서 포터 이미지 자동 채집 (Pseudo-labeling) |
| balance_dataset.py | 클래스별 오버샘플링 |
| finetune_model.py | 분류 모델 추가학습 |
| extract_t3_samples.py | T3 샘플 이미지 추출 |

### 4.4 원본 프로젝트 모듈 (읽기 전용)
| 경로 | 역할 |
|---|---|
| `/mnt/Expansion/CCTV차종분류/src/config.py` | 전역 설정 (YOLO 파라미터 등) |
| `/mnt/Expansion/CCTV차종분류/src/classifier/detector.py` | YOLO 검출기 래퍼 |
| `/mnt/Expansion/CCTV차종분류/src/classifier/tracker.py` | ByteTrack 추적기 래퍼 |

---

## 5. 모델 가중치

| 모델 | 경로 | 상태 | 비고 |
|---|---|---|---|
| YOLO 검출기 | `CCTV차종분류/pipeline/data/models/yolov8n.pt` | ✅ 사용중 | COCO pretrained |
| **cls_v5 (원본)** | **`CCTV차종분류/output/cls_v5_best.pt`** | **✅ 현재 사용중** | 6클래스(T1,T2,T4,T5,T10,T13). 안정적 |
| cls_v4 | `CCTV차종분류/output/cls_v4_best.pt` | ❌ 구버전 | v5로 대체됨 |
| porter_v1 | `finetune/porter_v1/weights/best.pt` | ❌ 폐기 | T3 추가했으나 불균형 데이터 |
| porter_final_v2 | `finetune/porter_final_v2/weights/best.pt` | ❌ 폐기 | 불균형 해소 시도 |
| **porter_final_v22** | **`finetune/porter_final_v22/weights/best.pt`** | **❌ 폐기** | **오버샘플링 부작용으로 T3 과잉분류 발생** |

> **중요**: `porter_final_v22` 모델은 사용하지 마세요. T3를 과도하게 분류하는 치명적 결함이 있습니다. 현재 시스템은 `cls_v5_best.pt` + COCO 폴백 하이브리드 방식을 사용합니다.

---

## 6. 작업 이력 (시간순)

### Phase 1-6: 트래킹 파이프라인 구축 및 모델 재학습 시도 (실패)
- ByteTrack 도입 및 정지차량(STABLE) 로직 구현
- T3(포터) 재학습 시도했으나 데이터 불균형으로 과적합 발생하여 폐기

### Phase 7: 하이브리드 v3 및 5중 방어 로직 (현재 - 2026.04.09)
- **버스-화물차 오분류 해결**: YOLO=BUS이나 덩치 크면 T2 강제 환원 로직 추가
- **동적 재분류**: 역대 최대 종횡비(`max_aspect`) 기억 및 매 프레임 재평가 도입
- **나이트 비전**: 저조도/심야 영상 자동 인식 및 CLAHE 화질 보정 엔진 탑재
- **고해상도 패치**: 원거리 소형 객체 탐지를 위해 `imgsz=1280`, `conf=0.15` 고감도 분석 모드 적용 (탐지율 2.7배 향상 확인)
- **전지역 테스트**: `hiv00329` 등 신규 지역 영상 교차 검증 완료 (UNK 0.8% 미만)

---

## 7. 현재 알려진 문제점 및 개선 과제

### 🔴 P0 (Critical)
1. **T3 과잉분류 검증 필요**: 하이브리드 v3로 전환했으나, 최종 결과 아직 미확인 (배치 처리 진행 중)
2. **UNK 폴백 정확도**: COCO 폴백이 실제로 정확한지 Ground Truth 대비 검증 필요

### 🟠 P1 (High)
3. **12종 중 6~9종, 11~12종 미지원**: 대형 특장차, 세미/풀 트레일러 분류 불가. 학습 데이터 자체가 없음
4. **TCS 2종 정합성**: 중형 승합(스타렉스, 카운티)이 T1로 오분류될 가능성

### 🟡 P2 (Medium)
5. **3종/4종 경계 모호**: CCTV로는 톤수 판별 불가, 외형 기반 근사 분류만 가능
6. **Ground Truth 부재**: 정량적 Precision/Recall 측정 불가
7. **ID 스위칭 시 라벨 불일치**: ByteTrack이 추적을 놓쳤다가 새 ID로 재분류할 때 같은 차량인데 다른 라벨이 붙는 현상 발생. 해결 방안: 사라진 위치 근처에서 새로 나타난 트랙에 이전 분류 결과를 상속시키는 로직 필요 (spatial re-identification)

---

## 8. 12종 분류 참고 자료 위치

```
/workspace/AI교통량조사/docs/차종분류_참고자료/
├── TCS 차종 구분.png                ← TCS 7종 분류 기준표
├── [별표 2] 12종 차종분류 체계.hwp  ← 도로교통량 조사지침 원문
└── 한국도로공사_교통량조사 차종별 대표 이미지_20250601/
    ├── 1종 대표적 차체.png ~ 12종 대표적 차체.png
    ├── 1종 차축배열.png ~ 12종 차축배열.png
    └── 1종(승용차, 미니트럭)_1단위2축_샘플.png ~ 12종 샘플.png
```

**TCS 분류 기준 요약**:
| TCS 종 | 정의 |
|---|---|
| 1종 | 승용차, 16인승 이하 승합, 2.5톤 미만 화물 |
| 2종 | 17~23인 승합, 2.5~5.5톤 화물 |
| 3종 | 33인 이상 승합, 5.5~10톤 미만 화물 |
| 4종 | 10~20톤 미만 화물 |
| 5종 | 20톤 이상 화물 |
| 6종 | 경차 |
| 기타 | 면제차량 |

> ⚠️ **주의**: TCS 분류(7종)와 도로교통량 조사지침 분류(12종)는 기준이 다릅니다. 현재 시스템은 조사지침 12종 체계를 따르되, 일부만 지원합니다.

---

## 9. 실행 방법

```bash
# 단일 영상 트래킹 (기본: hiv00134)
cd /workspace/prj_cctv/pipeline/src
python3 render_tracking_video.py

# 배치 처리 (TARGET_VIDEOS 목록 수정 후)
python3 batch_render_videos.py

# 함수 직접 호출
from render_tracking_video import run_tracking_render
run_tracking_render(
    video_path="/mnt/Expansion/영상/.../hiv00134.avi",
    output_path="/workspace/prj_cctv/output/tracking_hiv00134.mp4",
    log_path="/workspace/prj_cctv/output/tracking_hiv00134_log.csv"
)

# 로그 정제 (노이즈 제거)
python3 refine_tracking.py

# 품질 진단
python3 diagnose_tracking.py
```

---

## 10. 환경 정보

- Python 3.13.7
- PyTorch 2.11.0+cu130
- Ultralytics 8.4.33
- OpenCV (cv2)
- CPU 연산 (GPU 미사용, Intel Core Ultra 9 285K)
- 영상 1개(22,000프레임) 처리 시간: 약 9~10분

---

## 11. 현재 진행 중인 작업

**하이브리드 v3 배치 처리 진행 중** (2026-04-09 09:42 시작)
- 대상: hiv00134.avi, hiv00140.avi, hiv00135.avi
- 출력: `/workspace/prj_cctv/output/batch_videos/`
- 예상 완료: 약 09:55~10:00 경
- 완료 후 해야 할 일: 영상별 클래스 분포 확인하여 T3 과잉분류가 해소되었는지 검증
