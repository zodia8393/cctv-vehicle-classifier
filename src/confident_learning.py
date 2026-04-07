"""Confident Learning으로 라벨 정제

방법:
  1. v2 라벨 + DINOv2 임베딩 로드
  2. KNN 분류기 학습 (자기 자신 라벨로)
  3. KNN 예측 vs 현재 라벨 비교
  4. 일치 + 고확신: KEEP
  5. 불일치 + 저확신: NOISE
  6. 클러스터 응집도 + KNN confidence 결합
"""

import csv
import logging
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def main():
    # 임베딩 로드
    emb_path = Path("/workspace/CCTV차종분류_output/embeddings.npz")
    emb_data = np.load(emb_path, allow_pickle=True)
    paths = emb_data["paths"]
    embeddings = emb_data["embeddings"]
    n = len(paths)
    logger.info("임베딩 로드: %d장", n)

    # v2 라벨 로드
    v2_labels = {}
    with open("/workspace/CCTV차종분류_output/final_labels_v2.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            v2_labels[r["path"]] = r["label_13cls"]

    # NOISE 제외하고 학습 가능 데이터만
    valid_idx = []
    valid_labels = []
    for i, p in enumerate(paths):
        lbl = v2_labels.get(p, "NOISE_unknown")
        if not lbl.startswith("NOISE"):
            valid_idx.append(i)
            valid_labels.append(lbl)

    valid_idx = np.array(valid_idx)
    valid_emb = embeddings[valid_idx]
    valid_labels = np.array(valid_labels)
    logger.info("학습 가능: %d장", len(valid_idx))

    # L2 정규화
    valid_emb = valid_emb / np.maximum(np.linalg.norm(valid_emb, axis=1, keepdims=True), 1e-8)

    # 라벨 인코딩
    unique_labels = sorted(set(valid_labels))
    label_to_id = {l: i for i, l in enumerate(unique_labels)}
    id_to_label = {i: l for l, i in label_to_id.items()}
    y = np.array([label_to_id[l] for l in valid_labels])
    logger.info("라벨: %s", unique_labels)

    # KNN 예측 (cross-validated)
    logger.info("KNN cross-val 예측 (k=20)...")
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_predict

    knn = KNeighborsClassifier(n_neighbors=20, metric="cosine", n_jobs=-1)
    pred_proba = cross_val_predict(knn, valid_emb, y, cv=3, method="predict_proba", n_jobs=-1)
    logger.info("KNN 예측 완료: shape=%s", pred_proba.shape)

    # Confident Learning
    logger.info("Confident Learning 적용...")
    from cleanlab.filter import find_label_issues
    issues = find_label_issues(
        labels=y,
        pred_probs=pred_proba,
        return_indices_ranked_by="self_confidence",
        filter_by="prune_by_noise_rate",
    )
    logger.info("노이즈 라벨 후보: %d장 (%.1f%%)", len(issues), len(issues) / len(y) * 100)

    # KNN이 자체 예측한 라벨 (self_label)
    self_pred = pred_proba.argmax(axis=1)
    self_conf = pred_proba.max(axis=1)

    # 정제 결과 만들기
    issues_set = set(issues.tolist())

    rows = []
    cleaned_counts = Counter()
    for i, idx in enumerate(valid_idx):
        path = paths[idx]
        v2_label = valid_labels[i]
        knn_label = id_to_label[self_pred[i]]
        confidence = float(self_conf[i])
        is_issue = i in issues_set

        # 정제 규칙:
        # 1. issue 없음 + KNN과 일치 → KEEP
        # 2. issue 있음 + 신뢰도 낮음 → NOISE_cleanlab
        # 3. issue 있음 + KNN 신뢰도 높음(>0.7) → KNN 라벨로 변경
        # 4. issue 없음 + KNN과 불일치 → KEEP (애매)
        if not is_issue:
            final = v2_label
            action = "KEEP"
        elif confidence > 0.7 and knn_label != v2_label:
            final = knn_label
            action = "RELABEL"
        else:
            final = "NOISE_cleanlab"
            action = "NOISE"

        rows.append({
            "path": path,
            "v2_label": v2_label,
            "knn_label": knn_label,
            "confidence": f"{confidence:.3f}",
            "is_issue": "Y" if is_issue else "N",
            "action": action,
            "final_label": final,
        })
        cleaned_counts[final] += 1

    # 추가: NOISE_size, NOISE_mixed 보존
    noise_count = 0
    for i, p in enumerate(paths):
        if i in valid_idx:
            continue
        lbl = v2_labels.get(p, "NOISE_unknown")
        rows.append({
            "path": p,
            "v2_label": lbl,
            "knn_label": "",
            "confidence": "",
            "is_issue": "",
            "action": "NOISE_PRESERVE",
            "final_label": lbl,
        })
        cleaned_counts[lbl] += 1
        noise_count += 1

    # CSV 저장
    csv_path = Path("/workspace/CCTV차종분류_output/final_labels_v3_clean.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    logger.info("저장: %s", csv_path)

    # 통계
    logger.info("=" * 60)
    logger.info("Confident Learning 결과:")

    actions = Counter(r["action"] for r in rows)
    for k, v in actions.most_common():
        logger.info("  %s: %d", k, v)

    logger.info("=" * 60)
    logger.info("최종 라벨 분포:")
    train_total = 0
    for k in sorted(cleaned_counts.keys()):
        v = cleaned_counts[k]
        logger.info("  %-20s: %d (%.1f%%)", k, v, v / n * 100)
        if not k.startswith("NOISE"):
            train_total += v
    logger.info("  학습 가능: %d장", train_total)


if __name__ == "__main__":
    main()
