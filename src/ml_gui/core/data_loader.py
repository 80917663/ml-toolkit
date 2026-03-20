from __future__ import annotations

import os
from typing import Optional, Tuple

import pandas as pd


def load_dataset(
    path: str,
    has_header: bool,
    csv_delimiter: str = ",",
    encoding: Optional[str] = None,
) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    header = 0 if has_header else None

    if ext == ".csv":
        return pd.read_csv(path, sep=csv_delimiter, header=header, encoding=encoding)
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path, header=header)

    raise ValueError(f"不支持的文件类型：{ext}")


def normalize_columns_if_no_header(df: pd.DataFrame) -> pd.DataFrame:
    """
    When users load files without header, pandas will create default integer columns.
    We normalize to string columns for consistent UI display.
    """
    if all(isinstance(c, int) for c in df.columns):
        df = df.copy()
        df.columns = [f"col_{i}" for i in range(len(df.columns))]
    else:
        # Still stringify to avoid mixed-type issues in combobox
        df = df.copy()
        df.columns = [str(c) for c in df.columns]
    return df


def split_features_target(
    df: pd.DataFrame,
    target_col: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise KeyError(f"目标列不存在：{target_col}")
    y = df[target_col]
    mask = y.notna()
    X = df.loc[mask].drop(columns=[target_col])
    y = y.loc[mask]
    return X, y

