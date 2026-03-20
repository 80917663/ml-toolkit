from __future__ import annotations

from typing import Dict, Any, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def split_numeric_categorical(X: pd.DataFrame) -> Tuple[list[str], list[str]]:
    numeric_cols = []
    categorical_cols = []
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            numeric_cols.append(c)
        else:
            categorical_cols.append(c)
    return numeric_cols, categorical_cols


def build_preprocessor(
    X: pd.DataFrame,
    *,
    missing_num: str = "median",
    missing_cat: str = "most_frequent",
    scale_numeric: bool = True,
) -> ColumnTransformer:
    numeric_cols, categorical_cols = split_numeric_categorical(X)

    numeric_steps = [("imputer", SimpleImputer(strategy=missing_num))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_transformer = numeric_steps[0][1] if len(numeric_steps) == 1 else _make_pipeline(numeric_steps)

    categorical_transformer = _make_pipeline(
        [
            ("imputer", SimpleImputer(strategy=missing_cat)),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_transformer, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_transformer, categorical_cols))

    if not transformers:
        # Fallback: treat all as numeric
        transformers.append(("num", SimpleImputer(strategy=missing_num), list(X.columns)))

    return ColumnTransformer(transformers=transformers)


def _make_pipeline(steps: list[tuple[str, Any]]):
    # Local import to keep import graph small
    from sklearn.pipeline import Pipeline

    return Pipeline(steps=steps)

