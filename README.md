# CCTV 차종분류 파이프라인

CCTV 영상에서 추출한 차종 crop 이미지를 13종 차종 분류로 자동 라벨링하는 파이프라인.

## 디렉터리 구조

```
/workspace/
├── CCTV차종분류 → /mnt/Expansion/CCTV차종분류   # 입력 (외장, 심볼릭)
├── CCTV차종분류_pipeline/                       # 코드 (이 디렉터리)
│   ├── src/         # 18개 .py 스크립트
│   ├── logs/        # 실행 로그 6종
│   ├── docs/        # 분석 보고서, 방법론
│   └── README.md
└── CCTV차종분류_output/                          # 산출물
    ├── embeddings.npz                            # 임베딩 벡터
    ├── clusters/, clusters_v2/                   # 클러스터링 결과
    ├── labels/, gt_samples/                      # 라벨 작업 디렉터리
    ├── final_labels_v4.csv      ← **정본 (147,521행)**
    ├── cluster_v2_labels.csv    # 클러스터 메타 (200개)
    ├── suspicious_cases.csv     # KNN 검증 의심 케이스 (46,260행)
    ├── dataset_v4/, training_v4/                 # 학습 데이터
    └── _archive_v1_v3/                           # 폐기 라벨 (v1, v2, v3, clustered)
```

## 데이터 흐름 (DAG)

```
[CCTV 원본 영상]
       │
       ▼  run_batch_all_crops.py
[crop 이미지 (외장 디스크)]
       │
       ▼  extract_embeddings.py
[embeddings.npz]
       │
       ▼  cluster_embeddings.py
[clusters/]
       │
       ▼  apply_cluster_labels.py
[clustered_labels.csv]                  → _archive_v1_v3/
       │
       ▼  finalize_labels.py
[final_labels.csv (v1, 5컬럼)]          → _archive_v1_v3/
       │
       ▼  recluster_k200.py + auto_label_v2.py
[clusters_v2/, cluster_v2_labels.csv]
       │
       ▼  finalize_v2.py
[final_labels_v2.csv (5컬럼)]           → _archive_v1_v3/
       │
       ▼  verify_labels.py + confident_learning.py
[suspicious_cases.csv]
       │
       ▼  manual_correction.py
[final_labels_v3_clean.csv (7컬럼)]     → _archive_v1_v3/
       │
       ▼  subcluster_c33.py + verify_v2.py
[final_labels_v4.csv (9컬럼)]           ← **현재 정본**
       │
       ▼  build_dataset_v4.py
[dataset_v4/]
       │
       ▼  train_v4.py
[training_v4/]
```

## 정본 라벨

- **`final_labels_v4.csv`** (2026-04-07 11:12 기준)
  - 행 수: 147,521
  - 컬럼: `path, v2_label, knn_label, confidence, is_issue, action, final_label, v4_label, v4_name`
  - v4_label = 최종 13종 분류 (T1~T13)
  - 학습 시 `v4_label` 컬럼 사용

## 폐기 라벨 (참조용)

| 파일 | 단계 | 컬럼 수 | 비고 |
|---|---|---|---|
| `clustered_labels.csv` | 1차 클러스터링 결과 | 6 | 폐기 |
| `final_labels.csv` | v1 최종(클러스터→T코드 매핑) | 5 | 폐기 |
| `final_labels_v2.csv` | v2 (재클러스터링 후) | 5 | 폐기 |
| `final_labels_v3_clean.csv` | v3 (KNN 검증 + manual correction) | 7 | 폐기 |

## 알려진 이슈 / TODO

- [ ] 테스트 코드 0건 — 최소 데이터 무결성 어설션 추가 필요
- [ ] requirements.txt 부재 — 의존성 명세 (cv2, numpy, tqdm 등) 필요
- [ ] 매직넘버가 파일명에 박힘 (`recluster_k200`, `subcluster_c33`) — CLI 인자화 검토
- [ ] v3 vs v4 차이 정량 비교 보고서 미작성

## 변경 이력

- 2026-04-07: /workspace 루트에서 이 디렉터리로 코드·로그 이관 (회의 [meeting_20260407_workspace_진단.md](../.claude/context/meetings/meeting_20260407_workspace_진단.md))
