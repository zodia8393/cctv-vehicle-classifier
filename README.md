# CCTV 차량 분류 pipeline

경동고속도로 CCTV 영상에서 차량을 검출·추적하고 7개 T-code(`T1`, `T2`, `T3`, `T4`, `T5`, `T10`, `T13`)로 분류하기 위한 라벨링, 학습, holdout 평가 파이프라인이다.

이 디렉터리는 상위 작업공간과 분리된 독립 Git 저장소이며 remote는 `cctv-vehicle-classifier`다. 공통 평가 산출물 기본 위치 `../output/`은 이 저장소 밖이다.

## 구조

| 경로 | 역할 |
|---|---|
| `src/config.py` | canonical project/data/output 경로, class, threshold, release gate |
| `src/detector.py`, `tracker.py`, `classifier.py` | 검출·추적·분류 기본 구성 |
| `src/gt_builder.py` | video partition과 holdout GT 구축 |
| `src/extract_crops_for_labeling.py` | 검수용 crop 추출 |
| `src/prepare_*_review.py`, `crop_review_*.py` | 검수 queue와 GUI/Web UI |
| `src/train_*.py` | zero-shot, MobileNetV4, targeted refit 학습 |
| `src/eval_holdout*.py`, `stop_conditions.py` | sealed holdout 평가와 중단 조건 |
| `src/video_consistency.py`, `batch_all_ic.py` | 영상 단위 일관성 평가와 전체 IC batch |
| `data/` | 모델, manifest, crop, review queue, round 데이터의 symlink 작업면 |
| `docs/progress/` | 저장소 주간 진행 기록 |

## 환경

프로젝트 루트에서 실행한다.

~~~bash
cd "/workspace/prj/work/AI기반 교통상황 대응 기술 개발 연구/pipeline"
python3 -m pip install -r requirements.txt
~~~

`requirements.txt`는 공통 의존성이다. 선택한 script에 따라 `pyarrow`, `supervision`, `streamlit`, `ultralytics` 또는 `timm`이 추가로 필요할 수 있으므로 실제 import error와 해당 script의 `--help`를 기준으로 설치한다.

기본 project root는 `/workspace/prj/work/AI기반 교통상황 대응 기술 개발 연구`다. 다른 checkout을 쓸 때만 환경변수 `CCTV_PRJ_ROOT`를 지정한다.

## 작업 흐름

1. `gt_builder.py`로 GT/train video partition과 sealed holdout을 준비한다.
2. `extract_crops_for_labeling.py`, `prepare_stratified_review.py`, `crop_review_*.py`로 crop을 추출·검수한다.
3. `train_stage1_zeroshot.py`, `train_mnv4_full.py`, `train_mnv4_refit.py`, `train_targeted.py`로 후보 모델을 학습한다.
4. `eval_holdout_v2.py`와 `stop_conditions.py`로 class별 성능, validation-gap, 중단 조건을 확인한다.
5. `video_consistency.py`와 `batch_all_ic.py`로 영상·IC 단위 동작을 확인한다.

각 script의 필수 인자는 실행 전에 확인한다.

~~~bash
python3 src/gt_builder.py --help
python3 src/train_mnv4_full.py --help
python3 src/eval_holdout_v2.py --help
python3 src/video_consistency.py --help
~~~

## 데이터 경계

- `data/` 아래 대용량 파일은 대부분 `/DATA/HJ/prj/AI기반 교통상황 대응 기술 개발 연구/pipeline/data/`를 가리키는 symlink다.
- 일부 round dataset 링크는 같은 canonical working tree의 crop 링크를 한 번 더 참조한다. 2026-07-10 점검 시 전체 링크가 최종 대상까지 정상 해석됐다.
- 원천 영상 기본 경로는 `/mnt/Expansion/영상/220603_고속도로(경동)`이다.
- 공통 모델 보존과 평가 출력은 상위 `output/`에 기록된다. 이 경로는 현재 Git 저장소 밖이다.
- 외부 mount나 `/DATA/HJ`가 준비되지 않은 환경에서는 학습·평가를 시작하지 않는다.

## 품질 제약

- CPU 추론을 배포 기준으로 삼는다.
- train/validation/holdout split과 manifest hash를 유지하고, sealed holdout을 학습이나 label review에 사용하지 않는다.
- YOLO detection head 동적 INT8, ByteTrack `detect_interval > 1`, `conf >= 0.8` 자동 재라벨링은 금지한다.
- 모델 한계 결론은 학습곡선과 class별 holdout 결과를 확인한 뒤 내린다.
