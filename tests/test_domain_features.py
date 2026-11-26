"""Phase 2: stateless domain cleaning and feature engineering."""

import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from src.data.schema import FEATURES
from src.features.domain import ENGINEERED, DomainCleaner, DomainFeatureEngineer


def make_row(**overrides) -> dict:
    row = {c: 0 for c in FEATURES}
    row.update(LIMIT_BAL=100_000, SEX=1, EDUCATION=2, MARRIAGE=1, AGE=35)
    row.update({f"BILL_AMT{i}": 10_000 for i in range(1, 7)})
    row.update({f"PAY_AMT{i}": 1_000 for i in range(1, 7)})
    row.update(overrides)
    return row


def frame(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=FEATURES)


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    return DomainFeatureEngineer().fit(df).transform(df)


# ---------------------------------------------------------------- DomainCleaner
@pytest.mark.parametrize(("raw", "expected"), [(0, 4), (5, 4), (6, 4), (1, 1), (4, 4)])
def test_education_remap(raw: int, expected: int) -> None:
    out = DomainCleaner().fit_transform(frame(make_row(EDUCATION=raw)))
    assert out.loc[0, "EDUCATION"] == expected


@pytest.mark.parametrize(("raw", "expected"), [(0, 3), (1, 1), (2, 2), (3, 3)])
def test_marriage_remap(raw: int, expected: int) -> None:
    out = DomainCleaner().fit_transform(frame(make_row(MARRIAGE=raw)))
    assert out.loc[0, "MARRIAGE"] == expected


def test_pay_codes_preserved_not_collapsed() -> None:
    codes = [-2, -1, 0, 1, 2, 8]
    df = frame(*[make_row(PAY_1=c, PAY_2=c) for c in codes])
    out = DomainCleaner().fit_transform(df)
    assert out["PAY_1"].tolist() == codes and out["PAY_2"].tolist() == codes


def test_cleaner_does_not_mutate_input() -> None:
    df = frame(make_row(EDUCATION=0, MARRIAGE=0))
    DomainCleaner().fit_transform(df)
    assert df.loc[0, "EDUCATION"] == 0 and df.loc[0, "MARRIAGE"] == 0


def test_requires_dataframe_with_columns() -> None:
    with pytest.raises(TypeError):
        DomainCleaner().fit(np.zeros((2, 23)))
    with pytest.raises(ValueError, match="missing required columns"):
        DomainFeatureEngineer().fit(frame(make_row()).drop(columns=["BILL_AMT3"]))


# ---------------------------------------------------------- DomainFeatureEngineer
def test_engineered_columns_and_names_out() -> None:
    df = frame(make_row())
    fe = DomainFeatureEngineer().fit(df)
    out = fe.transform(df)
    assert list(out.columns) == [*FEATURES, *ENGINEERED]
    assert list(fe.get_feature_names_out()) == list(out.columns)


def test_utilisation_and_coverage_values() -> None:
    out = engineer(frame(make_row(BILL_AMT1=50_000, BILL_AMT2=20_000, PAY_AMT1=5_000)))
    assert out.loc[0, "UTIL_1"] == pytest.approx(0.5)
    assert out.loc[0, "UTIL_MAX"] == pytest.approx(0.5)
    # PAY_AMT1 (paid in Sep) settles the Aug statement BILL_AMT2.
    assert out.loc[0, "PAY_COVER_1"] == pytest.approx(5_000 / 20_000)


def test_ratio_safety_with_zero_and_negative_bills() -> None:
    row = make_row(**{f"BILL_AMT{i}": v for i, v in enumerate([0, -500, 0, 0, -1, 0], 1)})
    row.update(PAY_AMT1=10**7)  # huge prepayment against a credit balance
    extreme = make_row(LIMIT_BAL=0, BILL_AMT1=-(10**6), PAY_AMT1=10**8, PAY_1=8, PAY_2=-2)
    out = engineer(frame(row, make_row(LIMIT_BAL=0), extreme))
    values = out[ENGINEERED].to_numpy()
    assert np.isfinite(values).all(), out[ENGINEERED].columns[~np.isfinite(values).all(0)]
    assert out.loc[0, "PAY_COVER_1"] == 1.0  # nothing owed -> fully covered
    assert out[[f"PAY_COVER_{i}" for i in range(1, 6)]].to_numpy().max() <= 5.0
    assert out.loc[0, "N_BILL_NONPOS"] == 6


def test_trends_sign() -> None:
    # BILL_AMT1 is the most recent month: increasing balances towards Sep -> positive slope.
    rising = make_row(**{f"BILL_AMT{i}": 60_000 - 10_000 * i for i in range(1, 7)})
    falling = make_row(**{f"BILL_AMT{i}": 10_000 * i for i in range(1, 7)})
    out = engineer(frame(rising, falling))
    assert out.loc[0, "BILL_TREND"] > 0 > out.loc[1, "BILL_TREND"]
    # slope = 10k per month / 100k limit
    assert out.loc[0, "BILL_TREND"] == pytest.approx(0.1)


def test_delinquency_counts() -> None:
    row = make_row(PAY_1=3, PAY_2=1, PAY_3=0, PAY_4=-1, PAY_5=-2, PAY_6=-2)
    out = engineer(frame(row, make_row(**dict.fromkeys(["PAY_1", "PAY_2", "PAY_3"], -1))))
    r = out.loc[0]
    assert (r["N_DELINQ"], r["MAX_DELINQ"], r["N_REVOLVING"]) == (2, 3, 1)
    assert (r["N_PAID_FULL"], r["N_INACTIVE"]) == (1, 2)
    assert out.loc[1, "MAX_DELINQ"] == 0  # never negative


def test_inactive_status_and_nonpositive_bill_are_independent_signals() -> None:
    # PAY_n == -2 with a non-zero bill (common in the data) must not count as a zero bill.
    out = engineer(frame(make_row(**dict.fromkeys(["PAY_1", "PAY_2"], -2))))
    assert out.loc[0, "N_INACTIVE"] == 2 and out.loc[0, "N_BILL_NONPOS"] == 0


def test_stateless_fit_data_does_not_matter() -> None:
    a = frame(make_row(BILL_AMT1=1), make_row(BILL_AMT1=99_999))
    b = frame(make_row(BILL_AMT1=12_345))
    fitted_on_a = DomainFeatureEngineer().fit(a).transform(b)
    fitted_on_b = DomainFeatureEngineer().fit(b).transform(b)
    pd.testing.assert_frame_equal(fitted_on_a, fitted_on_b)


def test_clone_and_pickle_roundtrip() -> None:
    df = frame(make_row(EDUCATION=5), make_row(PAY_1=2))
    for est in (DomainCleaner(), DomainFeatureEngineer()):
        fitted = est.fit(df)
        clone(est)  # sklearn API compliance
        restored = pickle.loads(pickle.dumps(fitted))
        pd.testing.assert_frame_equal(restored.transform(df), fitted.transform(df))


# ------------------------------------------------- billing-cycle accounting features
def test_feature_count_is_disciplined() -> None:
    assert len(ENGINEERED) == 45 and len(set(ENGINEERED)) == 45  # 23 raw + 45 = 68 columns


def test_brief_paydown_example() -> None:
    # Brief §5.A.3, re-indexed to the latest month: start with BILL_AMT2 = 1,000, spend 4,000,
    # pay 5,000 before the statement closes -> BILL_AMT1 = 0 and PAY_AMT1 (5,000) > BILL_AMT1.
    out = engineer(frame(make_row(BILL_AMT2=1_000, PAY_AMT1=5_000, BILL_AMT1=0)))
    r = out.loc[0]
    assert r["SPEND_1"] == pytest.approx(4_000 / 100_000)  # implied spend = 0 - (1000 - 5000)
    assert r["PAYDOWN_1"] == 1 and r["N_PAYDOWN"] >= 1
    assert r["MAX_PAYDOWN_EXCESS"] == pytest.approx(5_000 / 100_000)


def test_implied_spend_identity_and_rolling_means() -> None:
    rng = np.random.default_rng(0)
    rows = [
        make_row(
            **{f"BILL_AMT{i}": int(v) for i, v in enumerate(rng.integers(-5_000, 90_000, 6), 1)},
            **{f"PAY_AMT{i}": int(v) for i, v in enumerate(rng.integers(0, 20_000, 6), 1)},
        )
        for _ in range(20)
    ]
    df = frame(*rows)
    out = engineer(df)
    bills = df[[f"BILL_AMT{i}" for i in range(1, 7)]].to_numpy(float)
    pays = df[[f"PAY_AMT{i}" for i in range(1, 7)]].to_numpy(float)
    spend = bills[:, :5] - (bills[:, 1:] - pays[:, :5])
    np.testing.assert_allclose(out["SPEND_1"], spend[:, 0] / 100_000)
    np.testing.assert_allclose(out["SPEND_MEAN_3"], spend[:, :3].mean(1) / 100_000)
    np.testing.assert_allclose(out["SPEND_MEAN_5"], spend.mean(1) / 100_000)


def test_credit_balance_is_preserved_not_clipped() -> None:
    out = engineer(frame(make_row(BILL_AMT1=-500, BILL_AMT3=-2_000)))
    r = out.loc[0]
    assert r["UTIL_1"] == pytest.approx(-0.005)  # signed utilisation kept negative
    assert (r["HAS_CREDIT_BAL_1"], r["N_CREDIT_BAL"]) == (1, 2)
    assert r["MIN_CREDIT_BAL_RATIO"] == pytest.approx(-0.02)


def test_over_limit_flags_and_excess() -> None:
    out = engineer(frame(make_row(BILL_AMT1=120_000, BILL_AMT4=105_000)))
    r = out.loc[0]
    assert (r["OVER_LIMIT_1"], r["N_OVER_LIMIT"]) == (1, 2)
    assert r["MAX_OVER_LIMIT_EXCESS"] == pytest.approx(0.2)


# ------------------------------------------------------------ roll-rate features
def status_row(*chronological: int) -> dict:
    """Statuses given oldest -> newest (Apr ... Sep) = PAY_6 ... PAY_1."""
    assert len(chronological) == 6
    return make_row(**{f"PAY_{6 - i}": v for i, v in enumerate(chronological)})


ROLL = [
    "CURED_RECENT",
    "N_CURES",
    "ROLL_FWD_RECENT",
    "N_ROLL_FWD",
    "MAX_ROLL_FWD_STEP",
    "N_DORMANT_TO_REVOLVER",
    "ROLL_BACK_RECENT",
    "N_ROLL_BACK",
    "STAGNANT_RECENT",
    "N_STAGNANT",
    "MAX_DELINQ_STREAK",
]


@pytest.mark.parametrize(
    ("sequence", "expected"),
    [
        # never late: every trajectory signal off
        ((0, 0, 0, 0, 0, 0), {}),
        # recently cured: 2 -> 2 -> 0; the cure is the LATEST transition -> recent flag only
        (
            (0, 0, 0, 2, 2, 0),
            {
                "CURED_RECENT": 1,
                "N_ROLL_FWD": 1,
                "MAX_ROLL_FWD_STEP": 2,
                "N_STAGNANT": 1,
                "MAX_DELINQ_STREAK": 2,
            },
        ),
        # cured earlier in the history window -> counted in N_CURES, no recent flag
        (
            (0, 2, 2, 0, 0, 0),
            {
                "N_CURES": 1,
                "N_ROLL_FWD": 1,
                "MAX_ROLL_FWD_STEP": 2,
                "N_STAGNANT": 1,
                "MAX_DELINQ_STREAK": 2,
            },
        ),
        # roll forward 0 -> 2 in the latest month: recent flag only, history counts stay 0
        ((0, 0, 0, 0, 0, 2), {"ROLL_FWD_RECENT": 1, "MAX_DELINQ_STREAK": 1}),
        # dormant straight to late in history: -2 -> 2 is a step of 2, not 4
        (
            (-2, -2, -2, -2, 2, 2),
            {"N_ROLL_FWD": 1, "MAX_ROLL_FWD_STEP": 2, "STAGNANT_RECENT": 1, "MAX_DELINQ_STREAK": 2},
        ),
        # roll backward 3 -> 2 in the latest month (still past due)
        (
            (0, 0, 0, 0, 3, 2),
            {
                "N_ROLL_FWD": 1,
                "MAX_ROLL_FWD_STEP": 3,
                "ROLL_BACK_RECENT": 1,
                "MAX_DELINQ_STREAK": 2,
            },
        ),
        # stagnation 2 -> 2 -> 2: one stagnant step in history + the recent flag
        (
            (0, 0, 0, 2, 2, 2),
            {
                "N_ROLL_FWD": 1,
                "MAX_ROLL_FWD_STEP": 2,
                "STAGNANT_RECENT": 1,
                "N_STAGNANT": 1,
                "MAX_DELINQ_STREAK": 3,
            },
        ),
        # 1 -> 0 is deliberately neither a cure (needs prev >= 2) nor a roll-back (needs cur > 0)
        ((0, 0, 0, 0, 1, 0), {"N_ROLL_FWD": 1, "MAX_ROLL_FWD_STEP": 1}),
        # dormant -> revolver activation is NOT a roll forward
        ((-2, 0, 0, 0, 0, 0), {"N_DORMANT_TO_REVOLVER": 1}),
        # -2 -> -1 (dormant -> paid in full) is neither
        ((-2, -1, -1, -1, -1, -1), {}),
    ],
    ids=[
        "never-late",
        "cured-recent",
        "cured-history",
        "roll-fwd-recent",
        "dormant-to-late",
        "roll-back",
        "stagnant",
        "one-to-zero",
        "dormant-to-revolver",
        "dormant-to-paid",
    ],
)
def test_roll_rate_trajectories(sequence: tuple, expected: dict) -> None:
    r = engineer(frame(status_row(*sequence))).loc[0]
    got = {k: r[k] for k in ROLL}
    assert got == {k: expected.get(k, 0) for k in ROLL}


def test_never_late_and_cured_differ() -> None:
    """The brief's motivating case: single-month snapshots can't tell these apart."""
    out = engineer(frame(status_row(0, 0, 0, 0, 0, 0), status_row(0, 0, 0, 2, 2, 0)))
    assert out.loc[0, "PAY_1"] == out.loc[1, "PAY_1"] == 0
    assert out.loc[0, "CURED_RECENT"] == 0 and out.loc[1, "CURED_RECENT"] == 1


def test_feature_catalogues_cover_every_feature_exactly_once() -> None:
    from src.data.schema import RAW_FEATURE_GROUPS
    from src.features.domain import FEATURE_GROUPS

    raw = [f for _, feats, _ in RAW_FEATURE_GROUPS for f in feats]
    eng = [f for _, feats, _ in FEATURE_GROUPS for f in feats]
    assert sorted(raw) == sorted(FEATURES) and len(raw) == len(set(raw))
    assert sorted(eng) == sorted(ENGINEERED) and len(eng) == len(set(eng))


def test_latest_transition_is_never_counted_in_history() -> None:
    # Every event happens only in the latest transition (PAY_2 -> PAY_1).
    out = engineer(frame(status_row(0, 0, 0, 0, 3, 2), status_row(0, 0, 0, 0, 2, 0)))
    for col in ("N_CURES", "N_ROLL_BACK", "N_STAGNANT", "N_DORMANT_TO_REVOLVER"):
        assert (out[col] == 0).all(), col
    assert out.loc[0, "ROLL_BACK_RECENT"] == 1 and out.loc[1, "CURED_RECENT"] == 1


@pytest.mark.parametrize(
    "overrides",
    [{"PAY_1": np.nan}, {"BILL_AMT2": np.inf}, {"LIMIT_BAL": 1e-300}, {"BILL_AMT1": 1e39}],
    ids=["nan-input", "inf-input", "tiny-limit", "beyond-float32"],
)
def test_non_finite_inputs_or_outputs_raise(overrides: dict) -> None:
    """No silent imputation: callers bypassing the API get an explicit error (C2/M6)."""
    with pytest.raises(ValueError, match="non-finite|float32"):
        engineer(frame(make_row(**overrides)))
