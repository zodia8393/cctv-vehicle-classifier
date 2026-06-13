"""웹 기반 크롭 검수 GUI — 도로공사 1종~12종 전체 참고이미지 기반 직접 분류."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from collections import Counter

import streamlit as st
import pyarrow.parquet as pq
import pyarrow as pa

LABELING_DIR = Path("/workspace/prj/cctv/pipeline/data/labeling_v1")
PARQUET_PATH = LABELING_DIR / "crops.parquet"
REF_DIR = Path(
    "/workspace/AI교통량조사/docs/차종분류_참고자료"
    "/한국도로공사_교통량조사 차종별 대표 이미지_20250601"
)

# 도로공사 12종 + 13종(이륜) + NOISE
CLASSES = [
    ("1",  "1종 승용차/미니트럭",      "1종 대표적 차체.png"),
    ("2",  "2종 버스",                 "2종 대표적 차체.png"),
    ("3",  "3종 소형화물(~2.5t)",      "3종 대표적 차체.png"),
    ("4",  "4종 중형화물(2.5~8.5t)",   "4종 대표적 차체.png"),
    ("5",  "5종 대형 3축",             "5종 대표적 차체.png"),
    ("6",  "6종 대형 4축",             "6종 대표적 차체.png"),
    ("7",  "7종 대형 5축",             "7종 대표적 차체.png"),
    ("8",  "8종 세미트레일러 4축",      "8종 대표적 차체.png"),
    ("9",  "9종 풀트레일러 4축",        "9종 대표적 차체.png"),
    ("10", "10종 세미트레일러 5축",     "10종 대표적 차체.png"),
    ("11", "11종 풀트레일러 5축",       "11종 대표적 차체.png"),
    ("12", "12종 세미트레일러 6축",     "12종 대표적 차체.png"),
    ("13", "13종 이륜차",              None),
    ("NOISE", "식별불가",              None),
]


@st.cache_data(show_spinner="Parquet 로딩 중...")
def load_data():
    table = pq.read_table(PARQUET_PATH)
    df = table.to_pandas()
    active = df[df["is_active_sample"] == True].copy()
    active = active.sort_values("current_conf", ascending=False)
    return active.reset_index(drop=True)


@st.cache_data
def load_ref_image(filename: str) -> bytes | None:
    path = REF_DIR / filename
    if path.exists():
        return path.read_bytes()
    return None


def save_label(crop_id: int, manual_class: str):
    today = datetime.now().strftime("%Y%m%d")
    temp_path = LABELING_DIR / f"temp_labels_{today}.jsonl"
    record = {
        "crop_id": crop_id,
        "manual_class": manual_class,
        "low_confidence": False,
        "user_decision": "O",
        "reviewer": "ybs",
        "reviewed_at": datetime.now().isoformat(timespec="milliseconds"),
    }
    with temp_path.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def merge_to_parquet():
    today = datetime.now().strftime("%Y%m%d")
    temp_path = LABELING_DIR / f"temp_labels_{today}.jsonl"
    if not temp_path.exists():
        return 0

    latest: dict[int, dict] = {}
    with temp_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                latest[rec["crop_id"]] = rec

    if not latest:
        return 0

    table = pq.read_table(PARQUET_PATH)
    df = table.to_pandas()

    n = 0
    for crop_id, label in latest.items():
        mask = df["crop_id"] == crop_id
        if not mask.any():
            continue
        df.loc[mask, "manual_class"] = label["manual_class"]
        df.loc[mask, "user_decision"] = label["user_decision"]
        df.loc[mask, "reviewer"] = label["reviewer"]
        df.loc[mask, "reviewed_at"] = label.get("reviewed_at")
        n += 1

    new_table = pa.Table.from_pandas(df, schema=table.schema, preserve_index=False)
    pq.write_table(new_table, PARQUET_PATH, compression="zstd", compression_level=3)
    return n


def select_class(cls: str):
    row = st.session_state._current_row
    crop_id = int(row["crop_id"])
    save_label(crop_id, cls)
    st.session_state.labels[crop_id] = cls
    st.session_state.history.append(st.session_state.idx)
    st.session_state.idx += 1


def main():
    st.set_page_config(page_title="차종 라벨링", layout="wide")

    st.markdown("""
    <style>
    div[data-testid="stHorizontalBlock"] button {
        min-height: 2.8rem; font-size: 0.85rem; font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

    df = load_data()

    if "idx" not in st.session_state:
        st.session_state.idx = 0
    if "labels" not in st.session_state:
        st.session_state.labels = {}
    if "history" not in st.session_state:
        st.session_state.history = []

    idx = st.session_state.idx
    total = len(df)
    labeled_count = len(st.session_state.labels)

    if idx >= total:
        st.success(f"전체 {total}장 검수 완료! ({labeled_count}장 라벨링)")
        if st.button("Parquet에 저장", type="primary"):
            n = merge_to_parquet()
            st.success(f"{n}건 Parquet 반영 완료")
        return

    row = df.iloc[idx]
    st.session_state._current_row = row

    # ── 상단: 진행 + 네비게이션 ──
    c1, c2, c3, c4, c5 = st.columns([4, 1, 1, 1, 1])
    with c1:
        st.progress(idx / total, text=f"**{idx + 1} / {total}**  (완료: {labeled_count})")
    with c2:
        if st.button("⬅ 이전", use_container_width=True) and idx > 0:
            st.session_state.idx -= 1
            st.rerun()
    with c3:
        if st.button("↩ 되돌리기", use_container_width=True) and st.session_state.history:
            st.session_state.idx = st.session_state.history.pop()
            st.rerun()
    with c4:
        if st.button("➡ 건너뛰기", use_container_width=True):
            st.session_state.idx += 1
            st.rerun()
    with c5:
        if st.button("💾 저장", type="primary", use_container_width=True):
            n = merge_to_parquet()
            st.toast(f"{n}건 저장")

    st.divider()

    # ── 메인: 크롭 이미지(좌) + 분류 버튼(우) ──
    col_img, col_btns = st.columns([1, 2])

    with col_img:
        img_bytes = row.get("image_bytes", b"")
        if img_bytes and len(img_bytes) > 0:
            st.image(img_bytes, use_container_width=True)
        else:
            st.error("이미지 없음")

    with col_btns:
        # 1~4종: 1행
        cols = st.columns(4)
        for col, (cls, name, ref_file) in zip(cols, CLASSES[0:4]):
            with col:
                if ref_file:
                    ref = load_ref_image(ref_file)
                    if ref:
                        st.image(ref, use_container_width=True)
                st.button(name, key=f"b{cls}", use_container_width=True,
                          on_click=select_class, args=(cls,))

        # 5~7종: 2행
        cols = st.columns(4)
        for i, (cls, name, ref_file) in enumerate(CLASSES[4:7]):
            with cols[i]:
                if ref_file:
                    ref = load_ref_image(ref_file)
                    if ref:
                        st.image(ref, use_container_width=True)
                st.button(name, key=f"b{cls}", use_container_width=True,
                          on_click=select_class, args=(cls,))

        # 8~11종: 3행
        cols = st.columns(4)
        for col, (cls, name, ref_file) in zip(cols, CLASSES[7:11]):
            with col:
                if ref_file:
                    ref = load_ref_image(ref_file)
                    if ref:
                        st.image(ref, use_container_width=True)
                st.button(name, key=f"b{cls}", use_container_width=True,
                          on_click=select_class, args=(cls,))

        # 12종 + 13종 + NOISE: 4행
        cols = st.columns(4)
        for i, (cls, name, ref_file) in enumerate(CLASSES[11:14]):
            with cols[i]:
                if ref_file:
                    ref = load_ref_image(ref_file)
                    if ref:
                        st.image(ref, use_container_width=True)
                st.button(name, key=f"b{cls}", use_container_width=True,
                          on_click=select_class, args=(cls,))

    # ── 사이드바: 현황 ──
    with st.sidebar:
        st.header("라벨링 현황")
        cls_counts = Counter(st.session_state.labels.values())
        if cls_counts:
            import pandas as pd
            chart_data = pd.DataFrame([
                {"종": f"{c}종" if c.isdigit() else c, "건수": n}
                for c, n in sorted(cls_counts.items(), key=lambda x: (not x[0].isdigit(), x[0].zfill(3)))
            ])
            st.bar_chart(chart_data, x="종", y="건수")
            st.divider()
            for c, n in sorted(cls_counts.items(), key=lambda x: (not x[0].isdigit(), x[0].zfill(3))):
                label = f"{c}종" if c.isdigit() else c
                st.text(f"{label}: {n}장")
        st.divider()
        st.metric("총 라벨링", labeled_count)
        st.metric("남은 수", total - idx)


if __name__ == "__main__":
    main()
