"""Interview deck: ``artifacts/credit_default_summary.pptx`` (cover + 4 content slides).

    uv run python -m src.reporting.ppt          # also runs as the pipeline's `report` stage

Every number is read from the JSON/CSV files the pipeline writes to ``artifacts/``; nothing
is hard-coded. The layout is a fixed grid (no overlapping boxes by construction), and every
text box / table cell is checked against a conservative text-fit estimate while building,
so an overflowing deck fails loudly instead of rendering badly.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

from src.common.config import Settings, get_settings
from src.common.logging import get_logger, log_path_io, stage

log = get_logger(__name__)

DECK_NAME = "credit_default_summary.pptx"
INPUTS = {
    "eda": "eda_summary.json",
    "iv": "iv_table.csv",
    "cv": "cv_comparison.csv",
    "grids": "model_grids.json",
    "oof": "train_oof_metrics.json",
    "test": "test_metrics.json",
    "deciles": "train_decile_analysis.csv",
    "test_deciles": "test_decile_analysis.csv",
}

TITLES = [
    "Credit Card Default Prediction — End-to-End ML & MLOps Pipeline",
    "Important EDA Observations",
    "Candidate Models & Cross-Validation Hyperparameters",
    "Top 3 Models & Winning Hyperparameters",
    "Training (OOF) vs. Test Results & Decile-Wise Performance",
]

# ----------------------------------------------------------------------------- theme
NAVY = RGBColor(0x1F, 0x2A, 0x44)
SLATE = RGBColor(0x4A, 0x55, 0x68)
TEAL = RGBColor(0x0F, 0x9D, 0x9A)
ROW_ALT = RGBColor(0xEE, 0xF2, 0xF6)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
COVER_SUB = RGBColor(0xC9, 0xD6, 0xE3)
FONT = "Calibri"

TITLE_PT, COVER_TITLE_PT, BODY_PT, NOTE_PT = 28, 40, 15, 14
HEADER_PT, CELL_PT, DECILE_PT = 12, 11, 10.5

SLIDE_W, SLIDE_H = 13.333, 7.5
MARGIN = 0.5
CONTENT_W = SLIDE_W - 2 * MARGIN  # 12.333"
TITLE_BOX = (MARGIN, 0.35, CONTENT_W, 0.85)
ACCENT_BAR = (MARGIN, 1.24, 1.3, 0.06)
TOP = 1.5  # content starts below the accent bar
BOTTOM = 7.1


class LayoutError(ValueError):
    """Raised when content would not fit its box (fail the build, don't overflow)."""


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    w: float
    h: float

    @property
    def bottom(self) -> float:
        return self.y + self.h


# ------------------------------------------------------------------- text-fit estimate
def _lines_needed(text: str, width_in: float, pt: float) -> int:
    # Average Calibri glyph ≈ 0.47em; 0.52em keeps the estimate conservative.
    chars_per_line = max(1, int(width_in / (0.52 * pt / 72)))
    return sum(max(1, math.ceil(len(part) / chars_per_line)) for part in text.split("\n"))


def _text_height(paragraphs: list[str], width_in: float, pt: float, spacing: float) -> float:
    lines = sum(_lines_needed(p, width_in, pt) for p in paragraphs)
    gaps = (len(paragraphs) - 1) * pt * 0.45 / 72  # space-after between paragraphs
    return lines * pt * spacing / 72 + gaps


def _check_fit(paragraphs: list[str], box: Box, pt: float, what: str, spacing=1.2) -> None:
    inner_w = box.w - 0.2  # default left+right insets 0.1"
    need = _text_height(paragraphs, inner_w, pt, spacing) + 0.1
    if need > box.h:
        raise LayoutError(f"{what}: needs {need:.2f}in of height, box has {box.h:.2f}in")


# ------------------------------------------------------------------------ primitives
def _font(run, pt: float, color: RGBColor, bold: bool = False) -> None:
    run.font.name = FONT
    run.font.size = Pt(pt)
    run.font.bold = bold
    run.font.color.rgb = color


def _textbox(
    slide,
    box: Box,
    paragraphs: list[str],
    pt: float,
    color: RGBColor,
    name: str,
    bold: bool = False,
    align=PP_ALIGN.LEFT,
    bullets: bool = False,
) -> None:
    _check_fit(paragraphs, box, pt, name)
    shape = slide.shapes.add_textbox(Inches(box.x), Inches(box.y), Inches(box.w), Inches(box.h))
    shape.name = name
    tf = shape.text_frame
    tf.word_wrap = True
    for i, text in enumerate(paragraphs):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align
        para.space_after = Pt(pt * 0.45)
        run = para.add_run()
        run.text = f"• {text}" if bullets else text
        _font(run, pt, color, bold)


def _title(slide, text: str) -> None:
    _textbox(slide, Box(*TITLE_BOX), [text], TITLE_PT, NAVY, "title", bold=True)
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, *(Inches(v) for v in ACCENT_BAR))
    bar.name = "accent"
    bar.fill.solid()
    bar.fill.fore_color.rgb = TEAL
    bar.line.fill.background()


def _table(
    slide,
    box: Box,
    header: list[str],
    rows: list[list[str]],
    col_w: list[float],
    name: str,
    cell_pt: float = CELL_PT,
    header_h: float = 0.45,
    highlight: int | None = None,
    align_right_from: int | None = None,
    row_min: float = 0.36,
) -> float:
    """Add a styled table; returns its height. ``highlight`` = data-row index to accent."""
    if not rows:
        raise LayoutError(f"{name}: no rows")
    if abs(sum(col_w) - box.w) > 0.01:
        raise LayoutError(f"{name}: column widths sum to {sum(col_w):.2f}, box is {box.w:.2f}")
    row_h = []
    for row in rows:  # each row as tall as its tallest cell needs
        need = max(
            _text_height([c], w - 0.12, cell_pt, 1.15) for c, w in zip(row, col_w, strict=True)
        )
        row_h.append(max(row_min, need + 0.1))
    height = header_h + sum(row_h)
    if height > box.h + 1e-6:
        raise LayoutError(f"{name}: table needs {height:.2f}in, box has {box.h:.2f}in")

    gf = slide.shapes.add_table(
        len(rows) + 1, len(header), Inches(box.x), Inches(box.y), Inches(box.w), Inches(height)
    )
    gf.name = name
    table = gf.table
    table.first_row = True
    table.horz_banding = False
    for j, w in enumerate(col_w):
        table.columns[j].width = Inches(w)
    table.rows[0].height = Inches(header_h)
    for i, h in enumerate(row_h, start=1):
        table.rows[i].height = Inches(h)

    def style(cell, text: str, fill: RGBColor, color: RGBColor, pt: float, bold: bool, right: bool):
        cell.fill.solid()
        cell.fill.fore_color.rgb = fill
        cell.margin_left = cell.margin_right = Inches(0.06)
        cell.margin_top = cell.margin_bottom = Inches(0.03)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf = cell.text_frame
        tf.word_wrap = True
        for k, line in enumerate(text.split("\n")):
            para = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
            para.alignment = PP_ALIGN.RIGHT if right else PP_ALIGN.LEFT
            run = para.add_run()
            run.text = line
            _font(run, pt, color, bold)

    for j, text in enumerate(header):
        right = align_right_from is not None and j >= align_right_from
        style(table.cell(0, j), text, NAVY, WHITE, HEADER_PT, True, right)
    for i, row in enumerate(rows):
        hl = highlight is not None and i == highlight
        fill = TEAL if hl else (WHITE if i % 2 == 0 else ROW_ALT)
        for j, text in enumerate(row):
            right = align_right_from is not None and j >= align_right_from
            style(table.cell(i + 1, j), text, fill, WHITE if hl else SLATE, cell_pt, hl, right)
    return height


# --------------------------------------------------------------------------- inputs
def load_inputs(artifacts_dir: Path) -> dict:
    missing = [f for f in INPUTS.values() if not (artifacts_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"missing report inputs in {artifacts_dir}: {missing}")
    out = {}
    for key, fname in INPUTS.items():
        path = artifacts_dir / fname
        log_path_io(log, "read", path)
        out[key] = (
            json.loads(path.read_text(encoding="utf-8"))
            if path.suffix == ".json"
            else pd.read_csv(path)
        )
    return out


def _display_names(grids: dict) -> dict[str, str]:
    return {m["name"]: m["display"] for m in grids["models"]}


def _fmt_params(params: dict) -> str:
    def v(x):
        return "None" if x is None else (f"{x:g}" if isinstance(x, float) else str(x))

    return ", ".join(f"{k}={v(val)}" for k, val in params.items())


def _champion(inp: dict) -> dict:
    cv = inp["cv"].sort_values("cv_f1", ascending=False).iloc[0]
    names = _display_names(inp["grids"])
    return {"name": cv["model"], "display": names.get(cv["model"], cv["model"]), "cv": cv}


# --------------------------------------------------------------------------- slides
def _cover(prs, inp: dict) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = NAVY
    eda, test, oof = inp["eda"], inp["test"]["at_registered_threshold"], inp["oof"]
    sh, tb = eda["shapes"], eda["target_balance"]
    champ = _champion(inp)
    _textbox(
        slide, Box(0.9, 1.35, 11.5, 1.6), [TITLES[0]], COVER_TITLE_PT, WHITE, "title", bold=True
    )
    bar = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0.9), Inches(3.05), Inches(1.6), Inches(0.07)
    )
    bar.name = "accent"
    bar.fill.solid()
    bar.fill.fore_color.rgb = TEAL
    bar.line.fill.background()
    _textbox(
        slide,
        Box(0.9, 3.3, 11.5, 0.55),
        ["UCI Default of Credit Card Clients (dataset id 350) · Taiwan, Apr–Sep 2005"],
        20,
        COVER_SUB,
        "subtitle",
    )
    facts = [
        f"{sh['full'][0]:,} accounts · {sh['full'][1] - 1} canonical features · "
        f"{tb['default_rate_full']:.1%} default rate",
        f"80/20 stratified split: {sh['train'][0]:,} train / {sh['test'][0]:,} holdout "
        "(random_state = 42)",
        f"Champion: {champ['display']} — CV F1 {champ['cv']['cv_f1']:.3f} ± "
        f"{champ['cv']['cv_f1_std']:.3f} (3-fold stratified CV)",
        f"Holdout F1 {test['f1']:.3f} · ROC-AUC {test['roc_auc']:.3f} at threshold "
        f"{test['threshold']:.3f} (flag riskiest {oof['top_percent_flagged']}%)",
        "Leak-free sklearn pipelines · MLflow Registry @champion · FastAPI · Evidently",
    ]
    _textbox(slide, Box(0.9, 4.1, 11.5, 2.9), facts, 17, WHITE, "facts", bullets=True)


def _eda_slide(prs, inp: dict) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _title(slide, TITLES[1])
    eda, iv = inp["eda"], inp["iv"]
    sh, tb = eda["shapes"], eda["target_balance"]
    edu = pd.DataFrame(eda["code_tables"]["EDUCATION"])
    mar = pd.DataFrame(eda["code_tables"]["MARRIAGE"])
    pay = pd.DataFrame(eda["code_tables"]["PAY_1"]).set_index("PAY_1")["default_rate"]
    unk_edu = edu[edu["EDUCATION"].isin([0, 5, 6])]
    named_edu = edu[edu["EDUCATION"].isin([1, 2, 3])]["default_rate"]
    code4 = float(edu.loc[edu["EDUCATION"] == 4, "default_rate"].iloc[0])
    mar0 = mar[mar["MARRIAGE"] == 0].iloc[0]
    top = iv.head(5)
    util = iv[iv["feature"].str.startswith("UTIL")].iloc[0]
    raw_bill_iv = iv[iv["feature"].str.startswith("BILL_AMT")]["iv"].max()
    bullets = [
        f"{sh['full'][0]:,} rows × {sh['full'][1] - 1} canonical features · "
        f"{eda['total_missing']} missing values · {eda['duplicate_rows']} duplicate rows kept",
        f"Imbalanced target: {tb['default_rate_full']:.1%} default "
        f"({tb['imbalance_ratio_neg_to_pos']:.1f}:1) → F1 / PR-AUC + explicit cutoff",
        f"EDUCATION {{0,5,6}}: {int(unk_edu['count'].sum())} rows "
        f"({unk_edu['count'].sum() / sh['full'][0]:.2%}), default {unk_edu['default_rate'].min():.0%}–"
        f"{unk_edu['default_rate'].max():.0%} ≈ code 4 ({code4:.1%}) vs codes 1–3 "
        f"{named_edu.min():.0%}–{named_edu.max():.0%} → remapped to 4",
        f"MARRIAGE {{0}}: {int(mar0['count'])} rows, default {mar0['default_rate']:.1%} → "
        "merged into 3 'others' (too rare for its own level)",
        f"PAY_1: −2 inactive {pay[-2]:.1%} and 0 revolving {pay[0]:.1%} default less than "
        f"−1 paid in full {pay[-1]:.1%} → codes kept distinct",
        "Top IV drivers: " + ", ".join(f"{r.feature} {r.iv:.2f}" for r in top.itertuples()),
        f"Utilisation beats raw bills: {util['feature']} IV {util['iv']:.2f} vs raw "
        f"BILL_AMT ≤ {raw_bill_iv:.3f}",
    ]
    _textbox(
        slide, Box(MARGIN, TOP, 7.0, BOTTOM - TOP), bullets, BODY_PT, SLATE, "body", bullets=True
    )

    seg = eda["domain_rule_checks"]["segments"]
    labels = {
        "base": "All accounts",
        "never_late": "Never late",
        "cured_now_current": "Cured (2+ → current)",
        "roll_forward": "Roll-forward (worsening)",
        "stagnant": "Stagnant at 2+ months",
        "over_limit": "Over limit (bill > limit)",
        "credit_balance": "Credit balance (bill < 0)",
        "paydown": "Paydown (paid > bill)",
    }
    rows = [
        [lab, f"{seg[k]['n']:,}", f"{seg[k]['default_rate']:.1%}"]
        for k, lab in labels.items()
        if k in seg
    ]
    x = MARGIN + 7.0 + 0.3
    _table(
        slide,
        Box(x, TOP, SLIDE_W - MARGIN - x, BOTTOM - TOP),
        ["Behavioural segment", "Accounts", "Default rate"],
        rows,
        [2.73, 1.1, 1.2],
        "table:segments",
        align_right_from=1,
    )


def _candidates_slide(prs, inp: dict) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _title(slide, TITLES[2])
    grids = inp["grids"]
    cv = grids["cv"]
    total = sum(m["n_candidates"] for m in grids["models"])
    setup = (
        f"{cv['splitter']}(n_splits={cv['n_splits']}, shuffle=True, "
        f'random_state={cv["random_state"]}) · {cv["search"]}(scoring="{cv["scoring"]}", '
        f"n_jobs={cv['n_jobs']}) · {total} candidates · cutoff tuned on OOF deciles"
    )
    _textbox(slide, Box(MARGIN, TOP, CONTENT_W, 0.8), [setup], BODY_PT, SLATE, "cv_setup")
    imbalance_keys = ("class_weight", "is_unbalance", "scale_pos_weight")
    rows = []
    for m in grids["models"]:
        fixed = dict(m["fixed_params"])
        imb = m["imbalance"]
        if isinstance(fixed.get("scale_pos_weight"), int | float):
            imb += f" ({fixed['scale_pos_weight']:.2f})"
        for k in imbalance_keys:
            fixed.pop(k, None)
        grid = "\n".join(
            f"{k}: {', '.join('None' if v is None else f'{v:g}' if isinstance(v, float) else str(v) for v in vals)}"
            for k, vals in m["grid"].items()
        )
        rows.append(
            [
                f"{m['display']}\n({m['n_candidates']} candidates)",
                m["preprocessing"],
                imb,
                grid,
                _fmt_params(fixed),
            ]
        )
    y = TOP + 0.95
    _table(
        slide,
        Box(MARGIN, y, CONTENT_W, BOTTOM - y),
        ["Model", "Pipeline", "Imbalance handling", "Grid (searched)", "Fixed params"],
        rows,
        [1.9, 3.0, 2.2, 2.6, 2.633],
        "table:candidates",
    )


def _top3_slide(prs, inp: dict) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _title(slide, TITLES[3])
    names = _display_names(inp["grids"])
    cv = inp["cv"].sort_values("cv_f1", ascending=False).reset_index(drop=True)
    top3 = cv.head(3)
    rows = []
    for i, r in top3.iterrows():
        rows.append(
            [
                str(i + 1),
                names.get(r["model"], r["model"]),
                _fmt_params(json.loads(r["best_params"])),
                f"{r['cv_f1']:.4f} ± {r['cv_f1_std']:.4f}",
                f"{r['cv_roc_auc']:.4f}",
                f"{r['cv_pr_auc']:.4f}",
                f"{r['cv_precision']:.4f}",
                f"{r['cv_recall']:.4f}",
            ]
        )
    h = _table(
        slide,
        Box(MARGIN, TOP, CONTENT_W, 3.6),
        [
            "Rank",
            "Model",
            "Selected hyperparameters",
            "CV F1 (mean ± std)",
            "ROC-AUC",
            "PR-AUC",
            "Precision",
            "Recall",
        ],
        rows,
        [0.7, 1.8, 3.2, 1.9, 1.15, 1.15, 1.2, 1.233],
        "table:top3",
        highlight=0,
        align_right_from=3,
    )
    spread = top3["cv_f1"].iloc[0] - top3["cv_f1"].iloc[-1]
    rest = cv.iloc[3:]
    notes = [
        f"Champion: {names.get(top3['model'].iloc[0])} — highest mean CV F1; the top 3 lie "
        f"within {spread:.4f} F1 (≈ {spread / top3['cv_f1_std'].iloc[0]:.1f}× the champion's "
        "fold std), so they are close to tied",
        "Selection metric = CV F1 at the default 0.5 cutoff on class-weighted models; the "
        "operating threshold is then tuned separately on out-of-fold predictions",
    ]
    if len(rest):
        notes.append(
            f"Not in top 3: {', '.join(names.get(m, m) for m in rest['model'])} "
            f"(CV F1 {rest['cv_f1'].max():.4f})"
        )
    y = TOP + h + 0.35
    _textbox(
        slide, Box(MARGIN, y, CONTENT_W, BOTTOM - y), notes, BODY_PT, SLATE, "notes", bullets=True
    )


def _decile_rows(d: pd.DataFrame) -> list[list[str]]:
    return [
        [
            str(int(r.decile)),
            f"{int(r.n):,}",
            f"{int(r.n_default):,}",
            f"{100 * r.bad_rate:.1f}%",
            f"{100 * r.cum_capture:.1f}%",
            f"{r.lift:.2f}",
            f"{r.f1:.3f}",
        ]
        for r in d.itertuples()
    ]


def _results_slide(prs, inp: dict) -> None:
    """Top: Train (OOF) vs Holdout metrics strip. Bottom: train and holdout decile tables
    side by side; the holdout uses the *training* score bands, so rows line up."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _title(slide, TITLES[4])
    oof, test = inp["oof"], inp["test"]["at_registered_threshold"]
    metrics = [
        ("Threshold", "threshold"),
        ("F1", "f1"),
        ("ROC-AUC", "roc_auc"),
        ("PR-AUC", "pr_auc"),
        ("Precision", "precision"),
        ("Recall", "recall"),
    ]
    strip = [
        ["Train (3-fold OOF)", *(f"{oof[k]:.4f}" for _, k in metrics)],
        ["Holdout (20%)", *(f"{test[k]:.4f}" for _, k in metrics)],
    ]
    h = _table(
        slide,
        Box(MARGIN, TOP, CONTENT_W, 1.2),
        ["Dataset", *(m for m, _ in metrics)],
        strip,
        [2.333] + [1.6667] * 6,
        "table:metrics",
        header_h=0.42,
        align_right_from=1,
    )

    k = math.ceil(oof["top_percent_flagged"] / 10)  # operating decile
    psi = inp["test"].get("score_psi_vs_train_bands")
    gap = 0.4
    half = (CONTENT_W - gap) / 2
    label_y = TOP + h + 0.15
    labels = [
        f"Train (3-fold OOF) · decile {k} = riskiest {oof['top_percent_flagged']}%",
        "Holdout (train score bands)"
        + (f" · PSI {psi:.3f}" if psi is not None else "")
        + f" · ΔF1 {test['f1'] - oof['f1']:+.3f}",
    ]
    table_y = label_y + 0.42
    header = ["Decile", "Accounts", "Defaults", "Bad rate", "Cum. capture", "Lift", "Cutoff F1"]
    widths = [0.62, 0.9, 0.85, 0.85, 1.05, 0.66, 1.0365]
    for i, (frame, label, name) in enumerate(
        [
            (inp["deciles"], labels[0], "deciles_train"),
            (inp["test_deciles"], labels[1], "deciles_test"),
        ]
    ):
        x = MARGIN + i * (half + gap)
        _textbox(
            slide, Box(x, label_y, half, 0.4), [label], NOTE_PT, NAVY, f"label:{name}", bold=True
        )
        _table(
            slide,
            Box(x, table_y, half, BOTTOM - table_y),
            header,
            _decile_rows(frame),
            widths,
            f"table:{name}",
            cell_pt=DECILE_PT,
            header_h=0.5,
            highlight=k - 1,
            align_right_from=1,
            row_min=0.33,
        )


# ---------------------------------------------------------------------------- build
def build_deck(inp: dict, out_path: Path) -> Path:
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(SLIDE_W), Inches(SLIDE_H)
    for builder in (_cover, _eda_slide, _candidates_slide, _top3_slide, _results_slide):
        builder(prs, inp)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out_path))
    log_path_io(log, "write", out_path)
    return out_path


def build_from_artifacts(settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    inp = load_inputs(settings.artifacts_dir)
    return build_deck(inp, settings.artifacts_dir / DECK_NAME)


def main() -> None:
    with stage("report"):
        build_from_artifacts()


if __name__ == "__main__":
    main()
