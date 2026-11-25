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
    df = frame(*[make_row(PAY_0=c, PAY_2=c) for c in codes])
    out = DomainCleaner().fit_transform(df)
    assert out["PAY_0"].tolist() == codes and out["PAY_2"].tolist() == codes


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
    out = engineer(frame(row, make_row(LIMIT_BAL=0)))
    values = out[ENGINEERED].to_numpy()
    assert np.isfinite(values).all()
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
    row = make_row(PAY_0=3, PAY_2=1, PAY_3=0, PAY_4=-1, PAY_5=-2, PAY_6=-2)
    out = engineer(frame(row, make_row(**dict.fromkeys(["PAY_0", "PAY_2", "PAY_3"], -1))))
    r = out.loc[0]
    assert (r["N_DELINQ"], r["MAX_DELINQ"], r["N_REVOLVING"]) == (2, 3, 1)
    assert (r["N_PAID_FULL"], r["N_INACTIVE"]) == (1, 2)
    assert out.loc[1, "MAX_DELINQ"] == 0  # never negative


def test_inactive_status_and_nonpositive_bill_are_independent_signals() -> None:
    # PAY_n == -2 with a non-zero bill (common in the data) must not count as a zero bill.
    out = engineer(frame(make_row(**dict.fromkeys(["PAY_0", "PAY_2"], -2))))
    assert out.loc[0, "N_INACTIVE"] == 2 and out.loc[0, "N_BILL_NONPOS"] == 0


def test_stateless_fit_data_does_not_matter() -> None:
    a = frame(make_row(BILL_AMT1=1), make_row(BILL_AMT1=99_999))
    b = frame(make_row(BILL_AMT1=12_345))
    fitted_on_a = DomainFeatureEngineer().fit(a).transform(b)
    fitted_on_b = DomainFeatureEngineer().fit(b).transform(b)
    pd.testing.assert_frame_equal(fitted_on_a, fitted_on_b)


def test_clone_and_pickle_roundtrip() -> None:
    df = frame(make_row(EDUCATION=5), make_row(PAY_0=2))
    for est in (DomainCleaner(), DomainFeatureEngineer()):
        fitted = est.fit(df)
        clone(est)  # sklearn API compliance
        restored = pickle.loads(pickle.dumps(fitted))
        pd.testing.assert_frame_equal(restored.transform(df), fitted.transform(df))
