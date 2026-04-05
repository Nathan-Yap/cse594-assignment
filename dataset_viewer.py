"""
dataset_viewer.py  —  Interactive vehicle image viewer with label editing & JSON export.

Usage:
    python dataset_viewer.py
    python dataset_viewer.py --csv data/labels.csv --img_dir ./data
    python dataset_viewer.py --csv data/labels.csv --img_dir ./data --out export.json

CSV expected columns (case-insensitive):
    imageName  — filename of the image
    description — the label / description text

    Optional extra columns that map to the export format:
    is_correct  — 1/True/yes  →  marks this as the ground-truth description
    extra_descriptions — semicolon-separated list of distractor descriptions

Export JSON format (ready for the labelling service /api/admin/upload_dataset):
    [
      {
        "image_name": "car_001.jpg",
        "descriptions": [
          {"text": "A red sedan with chrome trim", "is_correct": true},
          {"text": "A blue SUV with roof rack",    "is_correct": false}
        ]
      },
      ...
    ]

Keyboard shortcuts:
    ←  /  →   Previous / Next image
    Ctrl+S    Save (export JSON)
    Ctrl+D    Add a new blank description row
    Delete    Remove the focused description (when ≥ 2 exist)
"""

import argparse
import json
import os
import sys

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.widgets import Button, TextBox

# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Vehicle image viewer with label editing")
    parser.add_argument("--csv",     default="data/labels.csv", help="Path to CSV file")
    parser.add_argument("--img_dir", default="./data",          help="Image directory")
    parser.add_argument("--out",     default="export.json",     help="Output JSON path")
    return parser.parse_args()

# ── Colour palette ─────────────────────────────────────────────────────────────

COLORS = {
    "bg":           "#1a1d24",
    "surface":      "#22262f",
    "accent":       "#e8f04a",
    "accent2":      "#4af0c8",
    "text":         "#e8eaf0",
    "muted":        "#6b7280",
    "border":       "#2e3340",
    "correct_bg":   "#0d2e22",
    "correct_edge": "#4af0c8",
    "btn_bg":       "#2a2f3d",
    "btn_hover":    "#3a4050",
    "export_bg":    "#2a3a18",
    "export_edge":  "#e8f04a",
}

MAX_DESC_ROWS = 6   # Maximum editable description slots shown at once

# ── Data helpers ───────────────────────────────────────────────────────────────

def load_csv(csv_path: str) -> list[dict]:
    """
    Load the CSV and normalise into a list of records:
      {
        "image_name": str,
        "descriptions": [{"text": str, "is_correct": bool}, ...]
      }
    """
    df = pd.read_csv(csv_path)
    # Normalise column names: strip whitespace, lower-case lookup
    df.columns = [c.strip() for c in df.columns]
    col_map = {c.lower(): c for c in df.columns}

    def col(name):
        return col_map.get(name.lower())

    img_col  = col("imagename") or col("image_name") or col("filename")
    desc_col = col("description") or col("desc") or col("label")
    corr_col = col("is_correct") or col("correct")
    extra_col = col("extra_descriptions") or col("distractors")

    if img_col is None or desc_col is None:
        print("ERROR: CSV must have 'imageName' and 'description' columns.")
        print(f"  Found columns: {list(df.columns)}")
        sys.exit(1)

    records = []
    # Group by image name so multiple rows per image merge into one record
    for img_name, group in df.groupby(img_col, sort=False):
        descs = []
        for _, row in group.iterrows():
            text = str(row[desc_col]).strip() if pd.notna(row[desc_col]) else ""
            is_correct = False
            if corr_col:
                val = str(row[corr_col]).strip().lower()
                is_correct = val in ("1", "true", "yes", "y")
            if text:
                descs.append({"text": text, "is_correct": is_correct})

            # Extra semicolon-separated distractors column
            if extra_col and pd.notna(row.get(extra_col, None)):
                for extra in str(row[extra_col]).split(";"):
                    extra = extra.strip()
                    if extra:
                        descs.append({"text": extra, "is_correct": False})

        # If nothing marked correct, mark the first one by default
        if descs and not any(d["is_correct"] for d in descs):
            descs[0]["is_correct"] = True

        records.append({"image_name": str(img_name).strip(), "descriptions": descs})

    return records


def export_json(records: list[dict], out_path: str):
    """Write records to the labelling-service JSON format."""
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"Exported {len(records)} records → {out_path}")


# ── Viewer ─────────────────────────────────────────────────────────────────────

class ImageViewer:
    def __init__(self, records: list[dict], img_dir: str, out_path: str):
        self.records  = records
        self.img_dir  = img_dir
        self.out_path = out_path
        self.idx      = 0
        self.total    = len(records)
        self._unsaved = False

        # ── Figure setup ───────────────────────────────────────────────────────
        plt.rcParams.update({
            "figure.facecolor":  COLORS["bg"],
            "axes.facecolor":    COLORS["bg"],
            "text.color":        COLORS["text"],
            "axes.labelcolor":   COLORS["text"],
            "xtick.color":       COLORS["text"],
            "ytick.color":       COLORS["text"],
        })

        self.fig = plt.figure(figsize=(13, 8), facecolor=COLORS["bg"])
        self.fig.canvas.manager.set_window_title("Vehicle Dataset Viewer")

        # Left column: image (60% width)
        self.ax_img = self.fig.add_axes([0.02, 0.15, 0.54, 0.80])
        self.ax_img.set_facecolor(COLORS["bg"])
        self.ax_img.axis("off")

        # Right column: description editor (starts at x=0.58, leaves margin)
        # Populated dynamically in _rebuild_desc_widgets()
        self._desc_axes   = []   # list of axes for each description row
        self._desc_boxes  = []   # TextBox widgets
        self._corr_axes   = []   # correct-toggle button axes
        self._corr_btns   = []   # Button widgets for correct toggle
        self._del_axes    = []
        self._del_btns    = []

        # ── Navigation buttons ─────────────────────────────────────────────────
        ax_prev   = self.fig.add_axes([0.02, 0.04, 0.10, 0.06])
        ax_next   = self.fig.add_axes([0.13, 0.04, 0.10, 0.06])
        ax_add    = self.fig.add_axes([0.30, 0.04, 0.14, 0.06])
        ax_export = self.fig.add_axes([0.46, 0.04, 0.14, 0.06])
        ax_status = self.fig.add_axes([0.62, 0.04, 0.36, 0.06])

        self.btn_prev   = Button(ax_prev,   "◀  Prev",   color=COLORS["btn_bg"], hovercolor=COLORS["btn_hover"])
        self.btn_next   = Button(ax_next,   "Next  ▶",   color=COLORS["btn_bg"], hovercolor=COLORS["btn_hover"])
        self.btn_add    = Button(ax_add,    "+ Add Desc", color=COLORS["btn_bg"], hovercolor=COLORS["btn_hover"])
        self.btn_export = Button(ax_export, "⬇  Export JSON", color=COLORS["export_bg"], hovercolor="#3a5020")

        for btn in (self.btn_prev, self.btn_next, self.btn_add, self.btn_export):
            btn.label.set_color(COLORS["text"])
            btn.label.set_fontsize(9)
        self.btn_export.label.set_color(COLORS["accent"])

        ax_status.axis("off")
        ax_status.set_facecolor(COLORS["bg"])
        self._status_text = ax_status.text(
            0.0, 0.5, "", transform=ax_status.transAxes,
            fontsize=9, color=COLORS["muted"], va="center"
        )

        self.btn_prev.on_clicked(self._prev)
        self.btn_next.on_clicked(self._next)
        self.btn_add.on_clicked(self._add_description)
        self.btn_export.on_clicked(self._export)

        # ── Keyboard shortcuts ─────────────────────────────────────────────────
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        # ── Initial render ─────────────────────────────────────────────────────
        self._rebuild_desc_widgets()
        self._show_image()

    # ── Navigation ────────────────────────────────────────────────────────────

    def _prev(self, _=None):
        self._flush_textboxes()
        self._clear_desc_widgets()
        self.idx = (self.idx - 1) % self.total
        self._rebuild_desc_widgets()
        self._show_image()

    def _next(self, _=None):
        self._flush_textboxes()
        self._clear_desc_widgets()
        self.idx = (self.idx + 1) % self.total
        self._rebuild_desc_widgets()
        self._show_image()

    def _on_key(self, event):
        if event.key == "right":
            self._next()
        elif event.key == "left":
            self._prev()
        elif event.key in ("ctrl+s", "cmd+s"):
            self._export()

    # ── Image display ─────────────────────────────────────────────────────────

    def _show_image(self):
        self.ax_img.clear()
        self.ax_img.axis("off")

        rec      = self.records[self.idx]
        img_name = rec["image_name"]
        img_path = os.path.join(self.img_dir, img_name)

        try:
            img = mpimg.imread(img_path)
            self.ax_img.imshow(img)
            title = f"[{self.idx + 1} / {self.total}]  {img_name}"
        except (FileNotFoundError, OSError):
            self.ax_img.text(
                0.5, 0.5, f"Image not found:\n{img_name}",
                ha="center", va="center", fontsize=13,
                color="#f87171", transform=self.ax_img.transAxes,
            )
            title = f"[{self.idx + 1} / {self.total}]  ⚠ Missing: {img_name}"

        self.ax_img.set_title(
            title, fontsize=11, pad=8,
            color=COLORS["text"], fontweight="bold",
        )
        saved_label = "  •  unsaved changes" if self._unsaved else ""
        self._status_text.set_text(
            f"{self.total} images total  |  ←/→ to navigate  |  Ctrl+S to export{saved_label}"
        )
        self.fig.canvas.draw_idle()

    # ── Description widget management ─────────────────────────────────────────

    def _clear_desc_widgets(self):
        """Remove all existing description row axes from the figure."""
        for axes_list in (self._desc_axes, self._corr_axes, self._del_axes):
            for ax in axes_list:
                ax.remove()
        self._desc_axes  = []
        self._desc_boxes = []
        self._corr_axes  = []
        self._corr_btns  = []
        self._del_axes   = []
        self._del_btns   = []

    def _rebuild_desc_widgets(self):
        """Recreate the description TextBox rows for the current record."""
        rec   = self.records[self.idx]
        descs = rec["descriptions"]
        self._flush_textboxes()
        self._clear_desc_widgets()

        n     = min(len(descs), MAX_DESC_ROWS)

        # Layout constants
        RIGHT_START = 0.58   # x start of right panel
        ROW_H       = 0.088  # height per row
        ROW_GAP     = 0.012
        TOP         = 0.93   # y of first row top edge
        TB_W        = 0.28   # TextBox width
        CORR_W      = 0.055
        DEL_W       = 0.035
        X_CORR      = RIGHT_START + TB_W + 0.008
        X_DEL       = X_CORR + CORR_W + 0.005

        # Header label
        header_ax = self.fig.add_axes([RIGHT_START, TOP, TB_W + CORR_W + DEL_W + 0.02, 0.03])
        header_ax.axis("off")
        header_ax.set_facecolor(COLORS["bg"])
        header_ax.text(
            0.0, 0.5,
            "DESCRIPTIONS   (click ✓ to toggle correct  •  ✕ to delete)",
            fontsize=8, color=COLORS["muted"], va="center",
            transform=header_ax.transAxes,
        )
        self._desc_axes.append(header_ax)   # reuse list for cleanup

        for i in range(n):
            d      = descs[i]
            y_top  = TOP - 0.03 - i * (ROW_H + ROW_GAP)
            y_bot  = y_top - ROW_H

            is_correct = d.get("is_correct", False)

            # ── TextBox ──────────────────────────────────────────────────────
            ax_tb = self.fig.add_axes([RIGHT_START, y_bot, TB_W, ROW_H])
            ax_tb.set_facecolor(COLORS["correct_bg"] if is_correct else COLORS["surface"])
            for spine in ax_tb.spines.values():
                spine.set_edgecolor(COLORS["correct_edge"] if is_correct else COLORS["border"])
                spine.set_linewidth(1.5 if is_correct else 0.8)

            tb = TextBox(
                ax_tb, label="", initial=d["text"],
                color=COLORS["correct_bg"] if is_correct else COLORS["surface"],
                hovercolor=COLORS["correct_bg"] if is_correct else "#2a2f3d",
                label_pad=0.02,
            )
            tb.text_disp.set_color(COLORS["text"])
            tb.text_disp.set_fontsize(9)

            # Capture index in closure
            def _make_submit(idx_record, idx_desc):
                def _on_submit(text):
                    self.records[idx_record]["descriptions"][idx_desc]["text"] = text
                    self._unsaved = True
                    self._show_image()
                return _on_submit

            tb.on_submit(_make_submit(self.idx, i))
            self._desc_axes.append(ax_tb)
            self._desc_boxes.append(tb)

            # ── Correct toggle ────────────────────────────────────────────────
            ax_corr = self.fig.add_axes([X_CORR, y_bot, CORR_W, ROW_H])
            corr_color = COLORS["correct_bg"] if is_correct else COLORS["btn_bg"]
            btn_corr = Button(ax_corr, "✓", color=corr_color, hovercolor="#1a4030")
            btn_corr.label.set_color(COLORS["accent2"] if is_correct else COLORS["muted"])
            btn_corr.label.set_fontsize(11)

            def _make_toggle(idx_record, idx_desc):
                def _toggle(_):
                    descs_local = self.records[idx_record]["descriptions"]
                    currently   = descs_local[idx_desc]["is_correct"]
                    # If turning on: turn off all others
                    if not currently:
                        for d2 in descs_local:
                            d2["is_correct"] = False
                    descs_local[idx_desc]["is_correct"] = not currently
                    self._unsaved = True
                    self._rebuild_desc_widgets()
                    self._show_image()
                return _toggle

            btn_corr.on_clicked(_make_toggle(self.idx, i))
            self._corr_axes.append(ax_corr)
            self._corr_btns.append(btn_corr)

            # ── Delete button ─────────────────────────────────────────────────
            ax_del = self.fig.add_axes([X_DEL, y_bot, DEL_W, ROW_H])
            btn_del = Button(ax_del, "✕", color=COLORS["btn_bg"], hovercolor="#3a1a1a")
            btn_del.label.set_color("#f87171")
            btn_del.label.set_fontsize(10)

            def _make_delete(idx_record, idx_desc):
                def _delete(_):
                    d_list = self.records[idx_record]["descriptions"]
                    if len(d_list) <= 1:
                        self._set_status("⚠  Cannot delete: at least one description required.")
                        return
                    d_list.pop(idx_desc)
                    # Ensure at least one is_correct
                    if not any(d2["is_correct"] for d2 in d_list):
                        d_list[0]["is_correct"] = True
                    self._unsaved = True
                    self._rebuild_desc_widgets()
                    self._show_image()
                return _delete

            btn_del.on_clicked(_make_delete(self.idx, i))
            self._del_axes.append(ax_del)
            self._del_btns.append(btn_del)

        # Truncation notice
        if len(descs) > MAX_DESC_ROWS:
            y_notice = TOP - 0.03 - n * (ROW_H + ROW_GAP) - 0.01
            ax_note = self.fig.add_axes([RIGHT_START, y_notice, TB_W, 0.04])
            ax_note.axis("off")
            ax_note.set_facecolor(COLORS["bg"])
            ax_note.text(
                0.0, 0.5,
                f"+ {len(descs) - MAX_DESC_ROWS} more descriptions (not shown — edit JSON directly)",
                fontsize=8, color=COLORS["muted"], va="center",
                transform=ax_note.transAxes,
            )
            self._desc_axes.append(ax_note)

        self.fig.canvas.draw_idle()

    # ── Description add ────────────────────────────────────────────────────────

    def _add_description(self, _=None):
        rec = self.records[self.idx]
        rec["descriptions"].append({"text": "New description", "is_correct": False})
        self._unsaved = True
        self._rebuild_desc_widgets()
        self._show_image()

    # ── Flush textbox values to records ───────────────────────────────────────

    def _flush_textboxes(self):
        """Read current TextBox text into records before navigating away.

        tb.text returns the *initial* value set at construction and only
        updates after the user presses Enter.  The live edited string is
        always available via the underlying Text artist tb.text_disp.
        """
        rec   = self.records[self.idx]
        descs = rec["descriptions"]
        for i, tb in enumerate(self._desc_boxes):
            if i < len(descs):
                try:
                    live_text = tb.text_disp.get_text()
                    descs[i]["text"] = live_text
                except Exception:
                    pass

    # ── Export ────────────────────────────────────────────────────────────────

    def _export(self, _=None):
        self._flush_textboxes()
        export_json(self.records, self.out_path)
        self._unsaved = False
        self._set_status(f"✓  Exported {self.total} records to {self.out_path}")

    def _set_status(self, msg: str):
        self._status_text.set_text(msg)
        self.fig.canvas.draw_idle()


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args   = parse_args()
    records = load_csv(args.csv)
    print(f"Loaded {len(records)} records from {args.csv}")

    viewer = ImageViewer(records, args.img_dir, args.out)
    plt.show()
