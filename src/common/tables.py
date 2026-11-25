"""Render compact summary tables for terminal logs."""

from __future__ import annotations

import logging

import pandas as pd
from tabulate import tabulate


def render_table(df: pd.DataFrame, floatfmt: str = ".4f", index: bool = False) -> str:
    return tabulate(df, headers="keys", tablefmt="github", floatfmt=floatfmt, showindex=index)


def log_table(
    logger: logging.Logger, title: str, df: pd.DataFrame, floatfmt: str = ".4f", index: bool = False
) -> None:
    logger.info("%s\n%s", title, render_table(df, floatfmt=floatfmt, index=index))


def log_shape(logger: logging.Logger, label: str, df: pd.DataFrame) -> None:
    logger.info("shape[%s] = %s rows x %s cols", label, df.shape[0], df.shape[1])
