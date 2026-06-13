"""Crop 검수 — Streamlit 웹 GUI (Claude 사전 라벨링 + OX 검수).

PyQt5 대안. 노트북 브라우저에서 접속 가능.

실행 (호스트 PC):
  streamlit run crop_review_streamlit.py --server.port 8501 --server.address 0.0.0.0

접속 (노트북 브라우저):
  http://<호스트IP>:8501

기능:
  - Claude 라벨 표시 (큰 글씨 + 근거)
  - O / X 버튼 (또는 키보드 단축키)
  - X 시 직접 차종 선택
  - 진행 추적 + 분포 표시
  - Parquet in-place 업데이트
"""

from __future__ import annotations

import io
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import streamlit as st


# ── 경로 ──────────────────────────────────────────────────────────────
LABELING_DIR = Path("/workspace/prj/cctv/pipeline/data/labeling_v1")
PARQUET_PATH = LABELING_DIR / "crops.parquet"

CLASSES = [
    ("T1",    "승용차"),
    ("T2",    "버스"),
    ("T3",    "소형화물"),
    ("T4",    "중형화물"),
    ("T5",    "대형화물"),
    ("T10",   "세미트레일러"),
    ("T13",   "이륜차"),
    ("NOISE", "식별불가"),
]


# ── 데이터 로드 (캐시) ────────────────────────────────────────────────
@st.cache_data
def load_parquet() -> dict:
    """Parquet 전체 로드 → 검수 대상만 반환 (이미지 제외 메타만 정렬용)."""
    import pyarrow.parquet as pq
    table = pq.read_table(PARQUET_PATH)
    rows = table.to_pylist()

    # 검수 대상: Claude 라벨 있고 user_decision 없는 것
    pending_idx = [
        i for i, r in enumerate(rows)
        if r.get("claude_class") and not r.get("user_decision")
    ]

    # 정렬: 야간 + low_conf 우선
    def priority(i):
        r = rows[i]
        is_night = 0 if r.get("video_category") == "night" else 1
        conf = r.get("claude_conf") or 1.0
        return (is_night, conf)

    pending_idx.sort(key=priority)

    # 분포
    cls_counter = Counter(
        r["manual_class"] for r in rows if r.get("manual_class")
    )
    o_count = sum(1 for r in rows if r.get("user_decision") == "O")
    x_count = sum(1 for r in rows if r.get("user_decision") == "X")

    return {
        "rows":         rows,
        "pending_idx":  pending_idx,
        "cls_counter":  dict(cls_counter),
        "o_count":      o_count,
        "x_count":      x_count,
        "total":        len(rows),
        "n_claude":     sum(1 for r in rows if r.get("claude_class")),
    }


def append_temp_label(record: dict) -> None:
    """JSONL append."""
    today = datetime.now().strftime("%Y%m%d")
    temp = LABELING_DIR / f"temp_labels_{today}.jsonl"
    with temp.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        f.flush()


def _save(row, manual_class, decision):
    """라벨 저장 + 다음으로 이동 (Streamlit 콜백)."""
    import streamlit as st
    record = {
        "crop_id":         row["crop_id"],
        "manual_class":    manual_class,
        "low_confidence":  False,
        "user_decision":   decision,
        "reviewer":        "ybs",
        "reviewed_at":     datetime.now().isoformat(timespec="milliseconds"),
    }
    append_temp_label(record)

    st.session_state.session_count += 1
    if decision == "O":
        st.session_state.data["o_count"] += 1
    else:
        st.session_state.data["x_count"] += 1
    st.session_state.data["cls_counter"][manual_class] = (
        st.session_state.data["cls_counter"].get(manual_class, 0) + 1
    )

    st.session_state.idx += 1
    st.session_state.x_mode = False

    # 10개마다 자동 저장
    if st.session_state.session_count % 10 == 0:
        merge_to_parquet()


def merge_to_parquet() -> int:
    """temp JSONL → Parquet in-place 머지."""
    today = datetime.now().strftime("%Y%m%d")
    temp = LABELING_DIR / f"temp_labels_{today}.jsonl"
    if not temp.exists():
        return 0

    import pyarrow as pa
    import pyarrow.parquet as pq
    import pandas as pd

    latest = {}
    with temp.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                latest[rec["crop_id"]] = rec

    if not latest:
        return 0

    table = pq.read_table(PARQUET_PATH)
    df = table.to_pandas()
    if "reviewed_at" in df.columns:
        df["reviewed_at"] = df["reviewed_at"].astype("datetime64[ms]")

    n = 0
    for crop_id, label in latest.items():
        mask = df["crop_id"] == crop_id
        if not mask.any():
            continue
        df.loc[mask, "manual_class"] = label.get("manual_class")
        df.loc[mask, "low_confidence"] = label.get("low_confidence", False)
        df.loc[mask, "user_decision"] = label.get("user_decision")
        df.loc[mask, "reviewer"] = label.get("reviewer", "ybs")
        if label.get("reviewed_at"):
            df.loc[mask, "reviewed_at"] = pd.Timestamp(label["reviewed_at"]).floor("ms")
        n += 1

    new_table = pa.Table.from_pandas(df, schema=table.schema, preserve_index=False)
    pq.write_table(new_table, PARQUET_PATH, compression="zstd", compression_level=3)
    return n


# ── Streamlit 앱 ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="CCTV 차종 검수",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 세션 state
if "idx" not in st.session_state:
    st.session_state.idx = 0
if "x_mode" not in st.session_state:
    st.session_state.x_mode = False
if "session_count" not in st.session_state:
    st.session_state.session_count = 0
if "data" not in st.session_state:
    st.session_state.data = None

# 데이터 로드
if st.session_state.data is None:
    with st.spinner("Parquet 로드 중..."):
        st.session_state.data = load_parquet()

data = st.session_state.data
rows = data["rows"]
pending_idx = data["pending_idx"]


# ── 사이드바 ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📊 진행")

    total_pending = len(pending_idx)
    progress = st.session_state.idx / max(total_pending, 1)
    st.progress(progress, text=f"{st.session_state.idx} / {total_pending}")

    o = data["o_count"]
    x = data["x_count"]
    total_dec = o + x
    x_ratio = x / max(total_dec, 1) * 100

    col1, col2 = st.columns(2)
    col1.metric("O 채택", o, f"{100-x_ratio:.0f}%" if total_dec else "—")
    col2.metric("X 수정", x, f"{x_ratio:.0f}%" if total_dec else "—")

    st.metric("이번 세션", st.session_state.session_count)

    st.divider()
    st.subheader("🚗 누적 차종 분포")
    cls_counter = data["cls_counter"]
    if cls_counter:
        total_cls = sum(cls_counter.values())
        for cls, name in CLASSES:
            n = cls_counter.get(cls, 0)
            pct = n / max(total_cls, 1) * 100
            st.text(f"{cls:<5} {n:>4}  ({pct:>4.1f}%)")

    st.divider()
    if st.button("💾 저장 & 새로고침"):
        n = merge_to_parquet()
        st.success(f"✓ {n} 라벨 머지")
        load_parquet.clear()
        st.session_state.data = load_parquet()
        st.rerun()

    st.divider()
    st.caption(f"Parquet: {PARQUET_PATH.name}")
    st.caption(f"Claude 라벨: {data['n_claude']:,} / {data['total']:,}")


# ── 메인 ──────────────────────────────────────────────────────────────
if not pending_idx:
    st.success("✅ 모든 검수 완료!")
    st.stop()

if st.session_state.idx >= len(pending_idx):
    st.balloons()
    st.success("🎉 이번 세션 완료!")
    if st.button("저장하고 종료"):
        n = merge_to_parquet()
        st.info(f"✓ {n} 라벨 저장됨. 브라우저 닫아도 됨.")
    st.stop()

cur_row_idx = pending_idx[st.session_state.idx]
r = rows[cur_row_idx]


# 레이아웃
col_img, col_meta = st.columns([2, 1])

# ── 좌측: 이미지 ──────────────────────────────────────────────────────
with col_img:
    st.subheader(f"crop_id {r['crop_id']} — {r.get('ic', '?')} / {r.get('video', '?')}")
    img_bytes = r.get("image_bytes", b"")
    if img_bytes:
        st.image(img_bytes, use_container_width=True)
    else:
        st.error("이미지 없음")

    bbox = r.get("bbox", [0, 0, 0, 0])
    bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
    cat = r.get("video_category", "?")
    cat_emoji = {"night": "🌙", "backlight": "🌗", "day": "☀️"}.get(cat, "?")
    st.caption(
        f"{cat_emoji} {cat}  |  bbox {bbox} ({bw}×{bh}px)  |  "
        f"frame {r.get('frame', '?')}, track {r.get('tracker_id', '?')}"
    )


# ── 우측: 라벨 ────────────────────────────────────────────────────────
with col_meta:
    claude_cls = r.get("claude_class")
    claude_conf = r.get("claude_conf")
    claude_reason = r.get("claude_reason", "")

    # Claude 추천
    if claude_cls:
        cls_name = next((n for c, n in CLASSES if c == claude_cls), "?")
        conf_text = f" ({claude_conf:.2f})" if claude_conf else ""
        low_warn = " ⚠️" if claude_conf and claude_conf < 0.7 else ""
        st.markdown(f"### 🤖 Claude 추천: **{claude_cls}** — {cls_name}{conf_text}{low_warn}")
        st.caption(f"근거: _{claude_reason}_")
    else:
        st.markdown("### (Claude 라벨 없음 — 직접 선택)")

    st.divider()

    # OX 버튼 (Claude 라벨 있을 때)
    if claude_cls and not st.session_state.x_mode:
        col_o, col_x = st.columns(2)
        if col_o.button("✅ O (채택)", key="btn_o", use_container_width=True, type="primary"):
            _save(r, claude_cls, "O")
            st.rerun()
        if col_x.button("❌ X (수정)", key="btn_x", use_container_width=True):
            st.session_state.x_mode = True
            st.rerun()

    # 직접 모드 (X 누른 후 또는 Claude 없을 때)
    if st.session_state.x_mode or not claude_cls:
        st.markdown("**직접 차종 선택**")
        cols = st.columns(2)
        for i, (cls, name) in enumerate(CLASSES):
            col = cols[i % 2]
            if col.button(f"{cls}  {name}", key=f"btn_{cls}", use_container_width=True):
                decision = "X" if claude_cls else "O"
                _save(r, cls, decision)
                st.rerun()

    st.divider()

    # Triple V2 참고
    st.caption(f"Triple V2 예측: **{r.get('current_class')}** (conf {r.get('current_conf', 0):.2f})")

    st.divider()

    # 네비게이션
    nav1, nav2, nav3 = st.columns(3)
    if nav1.button("⬅ 이전", disabled=(st.session_state.idx == 0)):
        st.session_state.idx -= 1
        st.session_state.x_mode = False
        st.rerun()
    if nav2.button("➡ 다음 (스킵)"):
        st.session_state.idx += 1
        st.session_state.x_mode = False
        st.rerun()
    if nav3.button("💾 저장 & 종료", type="secondary"):
        n = merge_to_parquet()
        st.success(f"✓ {n} 라벨 저장됨")


# (_save 함수는 파일 상단으로 이동됨)
