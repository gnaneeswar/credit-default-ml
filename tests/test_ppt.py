"""Interview deck: structure, geometry, styling, and data-driven content (offline)."""

import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pptx import Presentation
from pptx.util import Emu

from src.common.config import Settings
from src.reporting import ppt
from src.training.models import MODEL_SPECS

TOP_PERCENT = 26  # -> operating decile 3


def write_synthetic_artifacts(d: Path, holdout_f1: float = 0.6123) -> None:
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    def codes(col, spec):
        return [
            {col: c, "count": n, "share": n / 30000, "default_rate": r, "documented": True}
            for c, n, r in spec
        ]

    eda = {
        "shapes": {"full": [30000, 24], "train": [24000, 24], "test": [6000, 24]},
        "total_missing": 0,
        "duplicate_rows": 35,
        "target_balance": {"default_rate_full": 0.2212, "imbalance_ratio_neg_to_pos": 3.52},
        "code_tables": {
            "EDUCATION": codes(
                "EDUCATION",
                [
                    (0, 14, 0.0),
                    (1, 10585, 0.19),
                    (2, 14030, 0.24),
                    (3, 4917, 0.25),
                    (4, 123, 0.057),
                    (5, 280, 0.064),
                    (6, 51, 0.157),
                ],
            ),
            "MARRIAGE": codes("MARRIAGE", [(0, 54, 0.093), (1, 13659, 0.23), (3, 323, 0.26)]),
            "PAY_1": codes("PAY_1", [(-2, 2759, 0.132), (-1, 5686, 0.168), (0, 14737, 0.128)]),
        },
        "domain_rule_checks": {
            "segments": {
                k: {"description": k, "n": 1000 + i, "default_rate": 0.1 + 0.05 * i}
                for i, k in enumerate(
                    [
                        "base",
                        "never_late",
                        "cured_now_current",
                        "roll_forward",
                        "stagnant",
                        "over_limit",
                        "credit_balance",
                        "paydown",
                    ]
                )
            }
        },
    }
    (d / "eda_summary.json").write_text(json.dumps(eda))
    pd.DataFrame(
        {
            "feature": ["PAY_1", "N_DELINQ", "MAX_DELINQ_STREAK", "UTIL_MEAN", "BILL_AMT1"],
            "iv": [0.88, 0.87, 0.85, 0.13, 0.015],
            "n_bins": [6, 6, 6, 10, 10],
            "strength": ["very strong (verify)"] * 3 + ["medium", "useless"],
            "selected": [True, True, True, True, False],
        }
    ).to_csv(d / "iv_table.csv", index=False)
    pd.DataFrame(
        [
            {
                "model": m,
                "family": "f",
                "cv_f1": f1,
                "cv_roc_auc": 0.78,
                "cv_pr_auc": 0.55,
                "cv_precision": 0.5,
                "cv_recall": 0.6,
                "cv_f1_std": 0.005,
                "n_candidates": 4,
                "seconds": 1.0,
                "best_params": json.dumps(bp),
            }
            for m, f1, bp in [
                ("random_forest", 0.549, {"max_depth": 12, "min_samples_leaf": 10}),
                ("xgboost", 0.545, {"learning_rate": 0.05, "max_depth": 3}),
                ("lightgbm", 0.543, {"learning_rate": 0.05, "num_leaves": 31}),
                ("logistic_regression", 0.535, {"C": 10.0, "class_weight": "balanced"}),
            ]
        ]
    ).to_csv(d / "cv_comparison.csv", index=False)
    grids = {
        "cv": {
            "splitter": "StratifiedKFold",
            "n_splits": 3,
            "shuffle": True,
            "random_state": 42,
            "search": "GridSearchCV",
            "scoring": "f1",
            "n_jobs": -1,
        },
        "models": [s.describe(Settings(), 3.52) for s in MODEL_SPECS.values()],
    }
    (d / "model_grids.json").write_text(json.dumps(grids))
    metrics = {
        "threshold": 0.478,
        "f1": 0.5488,
        "precision": 0.508,
        "recall": 0.597,
        "roc_auc": 0.787,
        "pr_auc": 0.557,
    }
    (d / "train_oof_metrics.json").write_text(
        json.dumps({**metrics, "top_percent_flagged": TOP_PERCENT, "deciles_flagged": 3})
    )
    (d / "test_metrics.json").write_text(
        json.dumps(
            {
                "at_registered_threshold": {**metrics, "f1": holdout_f1, "roc_auc": 0.78},
                "score_psi_vs_train_bands": 0.0123,
            }
        )
    )
    bad = np.sort(rng.random(10))[::-1] * 0.7
    pd.DataFrame(
        {
            "decile": range(1, 11),
            "n": 2400,
            "n_default": (bad * 2400).astype(int),
            "bad_rate": bad,
            "lift": bad / 0.22,
            "cum_capture": np.linspace(0.3, 1.0, 10),
            "f1": rng.random(10) * 0.5,
        }
    ).to_csv(d / "train_decile_analysis.csv", index=False)
    pd.DataFrame(  # holdout on the train bands: uneven counts, one deliberately tiny band
        {
            "decile": range(1, 11),
            "n": [612, 590, 601, 587, 610, 598, 605, 594, 600, 3],
            "n_default": (bad * 600).astype(int),
            "bad_rate": bad,
            "lift": bad / 0.22,
            "cum_capture": np.linspace(0.29, 1.0, 10),
            "f1": rng.random(10) * 0.5,
        }
    ).to_csv(d / "test_decile_analysis.csv", index=False)


@pytest.fixture(scope="module")
def deck(tmp_path_factory):
    d = tmp_path_factory.mktemp("artifacts")
    write_synthetic_artifacts(d)
    path = ppt.build_deck(ppt.load_inputs(d), d / ppt.DECK_NAME)
    return Presentation(str(path))


def _tables(slide):
    return [s for s in slide.shapes if s.has_table]


def test_slide_count_titles_and_widescreen(deck) -> None:
    assert len(deck.slides) == 5
    assert (Emu(deck.slide_width).inches, Emu(deck.slide_height).inches) == pytest.approx(
        (13.333, 7.5), abs=1e-3
    )
    titles = [next(s for s in sl.shapes if s.name == "title").text_frame.text for sl in deck.slides]
    assert titles == ppt.TITLES


def test_every_content_slide_has_a_populated_table(deck) -> None:
    for slide in list(deck.slides)[1:]:
        tables = _tables(slide)
        assert tables, "content slide without a table"
        for shape in tables:
            t = shape.table
            assert len(t.rows) > 1
            assert all(
                t.cell(r, c).text.strip() for r in range(len(t.rows)) for c in range(len(t.columns))
            ), f"empty cell in {shape.name}"


def test_shapes_inside_slide_and_never_overlap(deck) -> None:
    W, H = deck.slide_width, deck.slide_height
    for i, slide in enumerate(deck.slides):
        boxes = []
        for s in slide.shapes:
            assert s.left >= 0 and s.top >= 0, (i, s.name)
            assert s.left + s.width <= W and s.top + s.height <= H, (i, s.name)
            boxes.append((s.name, s.left, s.top, s.left + s.width, s.top + s.height))
        for a, b in combinations(boxes, 2):
            overlap = a[1] < b[3] and b[1] < a[3] and a[2] < b[4] and b[2] < a[4]
            assert not overlap, f"slide {i}: {a[0]} overlaps {b[0]}"


def test_minimum_font_sizes(deck) -> None:
    for slide in deck.slides:
        for shape in slide.shapes:
            if shape.has_table:
                for r in range(len(shape.table.rows)):
                    for c in range(len(shape.table.columns)):
                        for p in shape.table.cell(r, c).text_frame.paragraphs:
                            assert all(run.font.size.pt >= 10.5 for run in p.runs)
            elif shape.has_text_frame and shape.name != "accent":
                floor = 22 if shape.name == "title" else 14
                for p in shape.text_frame.paragraphs:
                    assert all(run.font.size.pt >= floor for run in p.runs), shape.name


@pytest.mark.parametrize("name", ["table:deciles_train", "table:deciles_test"])
def test_alternating_fills_and_operating_decile_highlight(deck, name: str) -> None:
    deciles = next(s for s in deck.slides[4].shapes if s.name == name).table
    assert len(deciles.rows) == 11 and deciles.cell(0, 0).fill.fore_color.rgb == ppt.NAVY
    k = math.ceil(TOP_PERCENT / 10)  # operating decile, highlighted in both tables
    for i in range(10):
        expected = ppt.TEAL if i == k - 1 else (ppt.WHITE if i % 2 == 0 else ppt.ROW_ALT)
        assert deciles.cell(i + 1, 0).fill.fore_color.rgb == expected, f"decile row {i + 1}"


def test_results_slide_shows_train_and_holdout_deciles(deck) -> None:
    shapes = {s.name: s for s in deck.slides[4].shapes}
    assert {"table:metrics", "table:deciles_train", "table:deciles_test"} <= set(shapes)
    train, test = shapes["table:deciles_train"], shapes["table:deciles_test"]
    assert train.top == test.top and train.left < test.left  # side by side
    assert test.table.cell(1, 1).text == "612" and test.table.cell(10, 1).text == "3"
    assert "PSI 0.012" in shapes["label:deciles_test"].text_frame.text
    assert "decile 3 = riskiest 26%" in shapes["label:deciles_train"].text_frame.text


def test_numbers_come_from_artifacts(tmp_path) -> None:
    write_synthetic_artifacts(tmp_path, holdout_f1=0.6123)
    prs = Presentation(str(ppt.build_deck(ppt.load_inputs(tmp_path), tmp_path / "d.pptx")))
    table = next(s for s in prs.slides[4].shapes if s.name == "table:metrics").table
    f1_col = next(c for c in range(len(table.columns)) if table.cell(0, c).text == "F1")
    holdout = next(r for r in range(len(table.rows)) if table.cell(r, 0).text.startswith("Hold"))
    assert table.cell(holdout, f1_col).text == "0.6123"
    cover = " ".join(s.text_frame.text for s in prs.slides[0].shapes if s.has_text_frame)
    assert "0.612" in cover and "Random Forest" in cover


def test_missing_inputs_and_overflow_fail_loudly(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="missing report inputs"):
        ppt.load_inputs(tmp_path)
    write_synthetic_artifacts(tmp_path)
    inp = ppt.load_inputs(tmp_path)
    inp["grids"]["models"][0]["preprocessing"] = "very long pipeline description " * 40
    with pytest.raises(ppt.LayoutError):
        ppt.build_deck(inp, tmp_path / "overflow.pptx")
