"""Phase 2: WoE/IV correctness and leak-free behaviour inside cross-validation."""

import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.common.config import Settings
from src.data.ingest import load_dataset
from src.data.split import split_xy, train_test
from src.features.domain import DomainCleaner, DomainFeatureEngineer
from src.features.woe import WoEIVSelector, iv_strength, woe_iv


@pytest.fixture(scope="module")
def xy() -> tuple[pd.DataFrame, pd.Series]:
    train, _ = train_test(load_dataset(Settings()))
    X, y = split_xy(train.sample(6000, random_state=0))
    pre = Pipeline([("clean", DomainCleaner()), ("fe", DomainFeatureEngineer())])
    return pre.fit_transform(X).reset_index(drop=True), y.reset_index(drop=True)


def lr_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("clean", DomainCleaner()),
            ("fe", DomainFeatureEngineer()),
            ("woe", WoEIVSelector()),
            ("lr", LogisticRegression(max_iter=1000)),
        ]
    )


# ------------------------------------------------------------------ correctness
def test_woe_iv_matches_hand_computation() -> None:
    bins = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    y = np.array([1, 0, 0, 1, 1, 1, 0, 1], dtype=float)
    woe, iv = woe_iv(bins, y, n_bins=2, alpha=0.0)
    # bin0: 1 bad / 2 good; bin1: 4 bad / 1 good. Totals: 5 bad, 3 good.
    dg, db = np.array([2 / 3, 1 / 3]), np.array([1 / 5, 4 / 5])
    assert woe == pytest.approx(np.log(dg / db))
    assert iv == pytest.approx(((dg - db) * np.log(dg / db)).sum())


def test_iv_table_ranks_known_signal_and_drops_weak(xy) -> None:
    X, y = xy
    sel = WoEIVSelector().fit(X, y)
    t = sel.iv_table_
    assert t.loc[0, "feature"] in {"PAY_0", "N_DELINQ", "MAX_DELINQ"}
    assert t["iv"].is_monotonic_decreasing
    assert set(sel.selected_features_) == set(t.loc[t["iv"] >= 0.02, "feature"])
    assert "SEX" not in sel.selected_features_
    assert list(sel.transform(X).columns) == [f"WOE_{c}" for c in sel.selected_features_]


def test_rare_bins_are_merged(xy) -> None:
    X, y = xy
    sel = WoEIVSelector(min_bin_frac=0.01).fit(X, y)
    counts = np.bincount(np.searchsorted(sel.cuts_["PAY_0"], X["PAY_0"], side="left"))
    assert counts.min() >= 0.01 * len(X)
    assert sel.bin_table("PAY_0")["high_incl"].iloc[-1] == np.inf


def test_unseen_and_missing_values_are_handled(xy) -> None:
    X, y = xy
    sel = WoEIVSelector().fit(X, y)
    probe = X.head(2).copy()
    probe.loc[0, "PAY_0"] = 99  # beyond training range -> top edge bin
    probe.loc[1, "PAY_0"] = np.nan  # missing -> WoE 0
    out = sel.transform(probe)
    assert out.loc[0, "WOE_PAY_0"] == pytest.approx(sel.woe_["PAY_0"][-1])
    assert out.loc[1, "WOE_PAY_0"] == 0.0


def test_iv_strength_labels() -> None:
    assert iv_strength(0.01) == "useless" and iv_strength(0.05) == "weak"
    assert iv_strength(0.2) == "medium" and iv_strength(0.6).startswith("very strong")


def test_rejects_bad_targets(xy) -> None:
    X, _ = xy
    with pytest.raises(ValueError, match="binary"):
        WoEIVSelector().fit(X.head(3), [0, 1, 2])
    with pytest.raises(ValueError, match="both classes"):
        WoEIVSelector().fit(X.head(3), [0, 0, 0])


# --------------------------------------------------------------------- leakage
def test_transform_ignores_held_out_labels(xy) -> None:
    """Encoding of fold B depends only on fold A: B's labels cannot change it."""
    X, y = xy
    a, b = X.index[:4000], X.index[4000:]
    sel = WoEIVSelector().fit(X.loc[a], y.loc[a])
    before = sel.transform(X.loc[b])
    y_tampered = y.copy()
    y_tampered.loc[b] = 1 - y_tampered.loc[b]
    refit = WoEIVSelector().fit(X.loc[a], y_tampered.loc[a])
    pd.testing.assert_frame_equal(before, refit.transform(X.loc[b]))
    # ...whereas fitting on A∪B (the leaky way) does change the encoding.
    leaky = WoEIVSelector().fit(X, y_tampered)
    assert not np.allclose(
        leaky.iv_table_.set_index("feature")["iv"], sel.iv_table_.set_index("feature")["iv"]
    )


def test_cv_refits_woe_on_each_training_fold_only() -> None:
    """In cross-validation each fold's fitted WoE equals a fit on that fold's train rows."""
    train, _ = train_test(load_dataset(Settings()))
    X, y = split_xy(train.sample(6000, random_state=1).reset_index(drop=True))
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    res = cross_validate(lr_pipeline(), X, y, cv=cv, return_estimator=True, scoring="roc_auc")
    for est, (tr_idx, _) in zip(res["estimator"], cv.split(X, y), strict=True):
        fold_woe = est.named_steps["woe"]
        Xtr = Pipeline([("c", DomainCleaner()), ("f", DomainFeatureEngineer())]).fit_transform(
            X.iloc[tr_idx]
        )
        manual = WoEIVSelector().fit(Xtr, y.iloc[tr_idx])
        pd.testing.assert_frame_equal(fold_woe.iv_table_, manual.iv_table_)
        for col in manual.selected_features_:
            np.testing.assert_array_equal(fold_woe.cuts_[col], manual.cuts_[col])
    assert np.mean(res["test_score"]) > 0.70  # sanity: the pipeline actually learns


def test_shuffled_target_has_no_information(xy) -> None:
    X, y = xy
    rng = np.random.default_rng(0)
    sel = WoEIVSelector().fit(X, rng.permutation(y.to_numpy()))
    assert sel.iv_table_["iv"].max() < 0.05
    assert len(sel.selected_features_) < 0.25 * X.shape[1]


# ------------------------------------------------------------------ sklearn API
def test_clone_get_params_and_pickle(xy) -> None:
    X, y = xy
    est = WoEIVSelector(iv_min=0.05, max_bins=8)
    assert clone(est).get_params() == est.get_params()
    fitted = est.fit(X, y)
    restored = pickle.loads(pickle.dumps(fitted))
    pd.testing.assert_frame_equal(restored.transform(X), fitted.transform(X))
    assert list(restored.get_feature_names_out()) == list(fitted.transform(X).columns)
