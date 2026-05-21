"""Crop 검수 GUI — Claude 사전 라벨링 + OX 검수 모드.

신규 흐름 (2026-04-27 회의 합의):
  pack → Claude 사전 라벨링 → [GUI OX 검수] → 재학습

UI 모드 (자동 분기):
  - Claude 라벨 있는 행: OX 모드
    [Y/Enter] = O (Claude 라벨 채택)
    [N/Space] = X (직접 수정 모드 진입)
  - Claude 라벨 없는 행: 직접 모드 (1~7/0)

X 누르면 직접 모드 활성:
  [1] T1  [2] T2  [3] T3  [4] T4  [5] T5  [6] T10  [7] T13  [0] NOISE

저장 (중도 종료 안전):
  - 매 라벨 → temp_labels_<날짜>.jsonl 한 줄 append
  - [Q] 종료 시 → Parquet 머지

사용:
  python3 crop_review_gui.py
  python3 crop_review_gui.py --max 1000
  python3 crop_review_gui.py --strategy active
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ── 경로 ──────────────────────────────────────────────────────────────
LABELING_DIR = Path("/workspace/prj_cctv/pipeline/data/labeling_v1")
DEFAULT_PARQUET = LABELING_DIR / "crops.parquet"


# ── 차종 ──────────────────────────────────────────────────────────────
CLASSES = [
    ("T1",    "승용차",       "1"),
    ("T2",    "버스",         "2"),
    ("T3",    "소형화물",     "3"),
    ("T4",    "중형화물",     "4"),
    ("T5",    "대형화물",     "5"),
    ("T10",   "세미트레일러", "6"),
    ("T13",   "이륜차",       "7"),
    ("NOISE", "식별불가",     "0"),
]


# ── Parquet 로드 + 샘플링 ─────────────────────────────────────────────
def load_parquet_rows(parquet_path: Path) -> list[dict]:
    import pyarrow.parquet as pq
    table = pq.read_table(parquet_path)
    rows = table.to_pylist()
    logger.info("Parquet 로드: %d 행", len(rows))
    return rows


def sort_rows(rows: list[dict], strategy: str = "active") -> list[dict]:
    """Claude 라벨 우선 → 미라벨링만 → 능동학습 정렬."""
    pending = [r for r in rows if r.get("user_decision") is None]
    logger.info("미검수: %d / 전체 %d", len(pending), len(rows))

    if strategy == "random":
        import random
        rng = random.Random(20260428)
        rng.shuffle(pending)
        return pending

    # active sample 최우선 → conf 높은 순
    def priority(r: dict) -> tuple:
        is_active = 0 if r.get("is_active_sample") else 1
        return (is_active, -r.get("current_conf", 0.0))

    return sorted(pending, key=priority)


def append_temp_label(temp_jsonl: Path, record: dict) -> None:
    with temp_jsonl.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        f.flush()


def merge_labels_into_parquet(parquet_path: Path, temp_jsonl: Path) -> int:
    if not temp_jsonl.exists():
        return 0

    import pyarrow as pa
    import pyarrow.parquet as pq

    latest: dict[int, dict] = {}
    with temp_jsonl.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                latest[rec["crop_id"]] = rec

    if not latest:
        return 0

    table = pq.read_table(parquet_path)
    df = table.to_pandas()

    n_updated = 0
    for crop_id, label in latest.items():
        mask = df["crop_id"] == crop_id
        if not mask.any():
            continue
        df.loc[mask, "manual_class"] = label.get("manual_class")
        df.loc[mask, "low_confidence"] = label.get("low_confidence", False)
        df.loc[mask, "user_decision"] = label.get("user_decision")  # "O" or "X"
        df.loc[mask, "reviewer"] = label.get("reviewer", "ybs")
        if label.get("reviewed_at"):
            df.loc[mask, "reviewed_at"] = pa.scalar(
                label["reviewed_at"], pa.timestamp("ms")
            ).as_py()
        n_updated += 1

    new_table = pa.Table.from_pandas(df, schema=table.schema, preserve_index=False)
    pq.write_table(new_table, parquet_path,
                   compression="zstd", compression_level=3)
    logger.info("Parquet 업데이트: %d 라벨", n_updated)
    return n_updated


# ── PyQt5 GUI ─────────────────────────────────────────────────────────
def run_gui(parquet_path: Path, strategy: str, max_n: int) -> None:
    try:
        from PyQt5.QtCore import Qt, QSize
        from PyQt5.QtGui import QPixmap, QKeySequence, QFont
        from PyQt5.QtWidgets import (
            QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
            QLabel, QPushButton, QShortcut, QProgressBar, QGroupBox, QStatusBar,
        )
    except ImportError:
        logger.error("PyQt5 미설치. sudo apt install python3-pyqt5")
        sys.exit(1)

    rows = load_parquet_rows(parquet_path)
    pending = sort_rows(rows, strategy)[:max_n]
    if not pending:
        logger.warning("검수할 크롭 없음 (모두 검수 완료)")
        return

    from collections import Counter
    cls_counter = Counter(r["manual_class"] for r in rows if r.get("manual_class"))
    o_count = sum(1 for r in rows if r.get("user_decision") == "O")
    x_count = sum(1 for r in rows if r.get("user_decision") == "X")

    today = datetime.now().strftime("%Y%m%d")
    temp_jsonl = LABELING_DIR / f"temp_labels_{today}.jsonl"

    class LabelerWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle(f"Crop Review — {parquet_path.name}")
            self.resize(1200, 800)
            self.rows = pending
            self.idx = 0
            self.session_count = 0
            self.session_start = datetime.now()
            self.cls_counter = cls_counter
            self.o_count = o_count
            self.x_count = x_count
            self.x_mode = False  # X 누르면 True (직접 수정 모드)
            self.last_idx: int | None = None

            self._build_ui()
            self._connect_shortcuts()
            self._show_current()

        def _build_ui(self):
            central = QWidget()
            self.setCentralWidget(central)
            main_layout = QHBoxLayout(central)

            # ── 좌측: 이미지 ────────────────────────────────
            left = QVBoxLayout()
            self.image_label = QLabel("(loading)")
            self.image_label.setAlignment(Qt.AlignCenter)
            self.image_label.setMinimumSize(QSize(580, 580))
            self.image_label.setStyleSheet(
                "border: 2px solid #444; background: #1a1a1a;"
            )
            left.addWidget(self.image_label)
            self.bbox_info = QLabel("")
            self.bbox_info.setStyleSheet("color: #888; font-size: 11px;")
            left.addWidget(self.bbox_info)
            main_layout.addLayout(left, stretch=2)

            # ── 우측 ────────────────────────────────────────
            right = QVBoxLayout()

            # Claude 추천 (큰 글씨)
            self.claude_box = QGroupBox("🤖 Claude 추천")
            cb_layout = QVBoxLayout()
            self.claude_label = QLabel("")
            self.claude_label.setFont(QFont("monospace", 14, QFont.Bold))
            self.claude_label.setWordWrap(True)
            cb_layout.addWidget(self.claude_label)
            self.claude_box.setLayout(cb_layout)
            right.addWidget(self.claude_box)

            # OX 버튼
            self.ox_box = QGroupBox("OX 검수 (Claude 라벨)")
            ox_layout = QHBoxLayout()
            self.btn_O = QPushButton("[Y/Enter] O\n채택")
            self.btn_O.setStyleSheet(
                "padding: 16px; background: #ddffdd; font-size: 14px; font-weight: bold;"
            )
            self.btn_O.clicked.connect(self._accept_claude)
            self.btn_X = QPushButton("[N/Space] X\n수정")
            self.btn_X.setStyleSheet(
                "padding: 16px; background: #ffdddd; font-size: 14px; font-weight: bold;"
            )
            self.btn_X.clicked.connect(self._enter_x_mode)
            ox_layout.addWidget(self.btn_O)
            ox_layout.addWidget(self.btn_X)
            self.ox_box.setLayout(ox_layout)
            right.addWidget(self.ox_box)

            # 직접 수정 옵션 (X 모드 시 활성)
            self.direct_box = QGroupBox("직접 수정 (X 모드)")
            direct_layout = QVBoxLayout()
            self.direct_buttons = {}
            for cls, name, key in CLASSES:
                btn = QPushButton(f"[{key}]  {cls:<5}  {name}")
                btn.setStyleSheet("text-align: left; padding: 6px;")
                btn.clicked.connect(lambda _, c=cls: self._save_direct(c))
                btn.setEnabled(False)
                self.direct_buttons[cls] = btn
                direct_layout.addWidget(btn)
            self.direct_box.setLayout(direct_layout)
            right.addWidget(self.direct_box)

            # 메타 (간단히)
            meta_box = QGroupBox("크롭 정보")
            ml = QVBoxLayout()
            self.meta_label = QLabel("")
            self.meta_label.setFont(QFont("monospace", 9))
            ml.addWidget(self.meta_label)
            meta_box.setLayout(ml)
            right.addWidget(meta_box)

            # 진행 / 분포
            prog_box = QGroupBox("진행 / 누적")
            pl = QVBoxLayout()
            self.progress_bar = QProgressBar()
            self.progress_bar.setMaximum(len(self.rows))
            pl.addWidget(self.progress_bar)
            self.dist_label = QLabel("")
            self.dist_label.setFont(QFont("monospace", 9))
            pl.addWidget(self.dist_label)
            prog_box.setLayout(pl)
            right.addWidget(prog_box)

            right.addStretch()
            main_layout.addLayout(right, stretch=1)

            # 상태바
            self.status = QStatusBar()
            self.setStatusBar(self.status)

        def _connect_shortcuts(self):
            # OX (Claude 라벨 있을 때)
            QShortcut(QKeySequence("Y"), self, activated=self._accept_claude)
            QShortcut(QKeySequence("Return"), self, activated=self._accept_claude)
            QShortcut(QKeySequence("Enter"), self, activated=self._accept_claude)
            QShortcut(QKeySequence("N"), self, activated=self._enter_x_mode)
            QShortcut(QKeySequence("Space"), self, activated=self._enter_x_mode)

            # 직접 라벨 (X 모드 또는 Claude 라벨 없을 때)
            for cls, _, key in CLASSES:
                QShortcut(QKeySequence(key), self,
                          activated=lambda c=cls: self._save_direct(c))

            # 네비게이션
            QShortcut(QKeySequence("Right"), self, activated=self._next)
            QShortcut(QKeySequence("Left"), self, activated=self._prev)
            QShortcut(QKeySequence("U"), self, activated=self._undo)
            QShortcut(QKeySequence("Q"), self, activated=self._save_and_quit)

        def _show_current(self):
            if self.idx >= len(self.rows):
                self._save_and_quit()
                return

            r = self.rows[self.idx]
            from PyQt5.QtGui import QPixmap
            from PyQt5.QtCore import Qt

            # 이미지
            pix = QPixmap()
            img_bytes = r.get("image_bytes", b"")
            if img_bytes:
                pix.loadFromData(img_bytes)
            if pix.isNull():
                self.image_label.setText("(이미지 로드 실패)")
            else:
                self.image_label.setPixmap(pix.scaled(
                    580, 580, Qt.KeepAspectRatio, Qt.SmoothTransformation
                ))

            # Claude 라벨 표시
            claude_cls = r.get("claude_class")
            claude_conf = r.get("claude_conf")
            claude_reason = r.get("claude_reason", "")

            if claude_cls:
                cls_name = next((n for c, n, _ in CLASSES if c == claude_cls), "?")
                conf_str = f" (conf {claude_conf:.2f})" if claude_conf else ""
                low = " ⚠️ low" if claude_conf and claude_conf < 0.7 else ""
                self.claude_label.setText(
                    f"<b>{claude_cls}</b> — {cls_name}{conf_str}{low}\n"
                    f"<i>{claude_reason}</i>"
                )
                self.claude_box.setTitle("Claude 추천")
                self.ox_box.setVisible(True)
                self.btn_O.setEnabled(True)
                self.btn_X.setEnabled(True)
            else:
                cur = r.get("current_class", "?")
                cur_conf = r.get("current_conf", 0)
                cls_name = next((n for c, n, _ in CLASSES if c == cur), "?")
                self.claude_label.setText(
                    f"<b>{cur}</b> — {cls_name} (conf {cur_conf:.3f})\n"
                    f"<i>[Enter] 수용  /  [1~7,0] 직접 선택</i>"
                )
                self.claude_box.setTitle("모델 예측 (Triple V2)")
                self.ox_box.setVisible(False)
                self._activate_direct_buttons(True)

            # X 모드 → 직접 버튼 활성
            self._activate_direct_buttons(self.x_mode or not claude_cls)

            # 메타
            cur = r.get("current_class", "?")
            cur_conf = r.get("current_conf", 0)
            cat = r.get("video_category", "?")
            cat_emoji = {"night": "🌙", "backlight": "🌗", "day": "☀️"}.get(cat, "?")
            self.meta_label.setText(
                f"crop_id: {r.get('crop_id')}\n"
                f"IC: {r.get('ic', '?')}\n"
                f"video: {r.get('video', '?')} f={r.get('frame', '?')} tid={r.get('tracker_id', '?')}\n"
                f"category: {cat_emoji} {cat}\n"
                f"────────\n"
                f"Triple V2: {cur} (conf {cur_conf:.3f})"
            )

            bbox = r.get("bbox", [0, 0, 0, 0])
            bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
            self.bbox_info.setText(
                f"bbox: {bbox} ({bw}×{bh}px)  image: {r.get('image_w', 0)}×{r.get('image_h', 0)}"
            )

            self.progress_bar.setValue(self.idx)
            self._update_dist()

            mode = "직접" if (self.x_mode or not claude_cls) else "OX"
            self.status.showMessage(
                f"[{self.idx + 1}/{len(self.rows)}]  crop_id={r.get('crop_id')}  "
                f"모드: {mode}  세션: {self.session_count}장  "
                f"({(datetime.now() - self.session_start).seconds // 60}분)"
            )

        def _activate_direct_buttons(self, active: bool):
            for btn in self.direct_buttons.values():
                btn.setEnabled(active)
            self.direct_box.setEnabled(active)

        def _update_dist(self):
            total_decisions = self.o_count + self.x_count
            x_ratio = self.x_count / max(total_decisions, 1) * 100
            lines = [
                f"검수 {total_decisions}장",
                f"  O (채택): {self.o_count} ({100-x_ratio:.1f}%)",
                f"  X (수정): {self.x_count} ({x_ratio:.1f}%)",
                f"───────",
                "차종 누적:",
            ]
            total_cls = sum(self.cls_counter.values())
            for cls, _, _ in CLASSES:
                n = self.cls_counter.get(cls, 0)
                pct = n / max(total_cls, 1) * 100
                lines.append(f"  {cls:<5} {n:>4}  ({pct:>4.1f}%)")
            self.dist_label.setText("\n".join(lines))

        def _accept_claude(self):
            """O — Claude 라벨 채택 또는 모델 예측 수용."""
            if self.idx >= len(self.rows):
                return
            r = self.rows[self.idx]
            claude_cls = r.get("claude_class")
            if claude_cls:
                self._save(claude_cls, decision="O")
            else:
                cur = r.get("current_class")
                if cur:
                    self._save(cur, decision="O")

        def _enter_x_mode(self):
            """X — 직접 수정 모드 진입."""
            self.x_mode = True
            self._activate_direct_buttons(True)
            self.status.showMessage("X 모드 — 직접 차종 선택 (1~7/0)")

        def _save_direct(self, manual_class: str):
            """직접 라벨 저장 (X 모드 또는 Claude 없음)."""
            if self.idx >= len(self.rows):
                return
            r = self.rows[self.idx]
            decision = "X" if r.get("claude_class") else "O"  # Claude 없으면 그냥 직접 = O
            self._save(manual_class, decision=decision)

        def _save(self, manual_class: str, decision: str):
            r = self.rows[self.idx]
            record = {
                "crop_id":         r["crop_id"],
                "manual_class":    manual_class,
                "low_confidence":  False,
                "user_decision":   decision,
                "reviewer":        "ybs",
                "reviewed_at":     datetime.now().isoformat(timespec="milliseconds"),
            }
            append_temp_label(temp_jsonl, record)
            self.cls_counter[manual_class] += 1
            if decision == "O":
                self.o_count += 1
            else:
                self.x_count += 1
            self.session_count += 1
            self.last_idx = self.idx
            self.x_mode = False  # 다음 크롭은 OX 모드로 리셋
            self.idx += 1
            self._show_current()

        def _next(self):
            if self.idx < len(self.rows):
                self.idx += 1
                self.x_mode = False
                self._show_current()

        def _prev(self):
            if self.idx > 0:
                self.idx -= 1
                self.x_mode = False
                self._show_current()

        def _undo(self):
            if self.last_idx is not None and self.idx > 0:
                self.idx = self.last_idx
                self.x_mode = False
                self._show_current()

        def _save_and_quit(self):
            n = merge_labels_into_parquet(parquet_path, temp_jsonl)
            self.status.showMessage(f"저장 완료: +{n} 라벨")
            logger.info("종료 — Parquet 머지: +%d 라벨", n)
            QApplication.quit()

        def closeEvent(self, event):
            self._save_and_quit()
            event.accept()

    app = QApplication(sys.argv)
    win = LabelerWindow()
    win.show()
    app.exec_()


def main(args):
    parquet_path = Path(args.parquet)
    if not parquet_path.exists():
        logger.error("Parquet 없음: %s", parquet_path)
        sys.exit(1)
    run_gui(parquet_path, args.strategy, args.max)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", default=str(DEFAULT_PARQUET))
    parser.add_argument("--strategy", choices=["active", "random"], default="active")
    parser.add_argument("--max", type=int, default=1000)
    args = parser.parse_args()
    main(args)
