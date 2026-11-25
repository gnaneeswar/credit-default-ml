"""Stratified holdout split and the single CV splitter shared by every model."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.common.config import Settings, get_settings
from src.common.logging import get_logger
from src.data import schema

log = get_logger(__name__)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[schema.FEATURES], df[schema.TARGET]


def train_test(
    df: pd.DataFrame, settings: Settings | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified 80/20 split (random_state=42). The 20% test set is touched only once,
    by the registered @champion, after all tuning and threshold selection is done."""
    settings = settings or get_settings()
    train, test = train_test_split(
        df,
        test_size=settings.test_size,
        stratify=df[schema.TARGET],
        random_state=settings.seed,
    )
    for name, part in (("train", train), ("test", test)):
        log.info(
            "split[%s]: %s rows, default rate %.4f",
            name,
            len(part),
            part[schema.TARGET].mean(),
        )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def make_cv(settings: Settings | None = None) -> StratifiedKFold:
    """3-fold stratified CV: every training row gets exactly one out-of-fold prediction."""
    settings = settings or get_settings()
    return StratifiedKFold(n_splits=settings.n_splits, shuffle=True, random_state=settings.seed)
