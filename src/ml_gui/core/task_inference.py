from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def _is_integer_like(values: np.ndarray) -> bool:
    values = values[~np.isnan(values)]
    if values.size == 0:
        return False
    return np.all(np.isclose(values, np.round(values)))


def infer_task_type(y: pd.Series, *, max_class_count: int = 20) -> Dict[str, Any]:
    """
    Heuristic inference for v1:
    - If y is non-numeric -> classification
    - If y is numeric:
      - If it looks like integer labels with limited unique values -> classification
      - Else -> regression
    """
    y_no_na = y.dropna()
    if y_no_na.empty:
        return {"task_type": None, "n_classes": None, "classes": None}

    if pd.api.types.is_numeric_dtype(y_no_na):
        uniq = np.unique(y_no_na.to_numpy())
        # If integer-like with small unique values => classification
        if uniq.size <= max_class_count and _is_integer_like(uniq):
            classes = sorted(list(uniq.tolist()))
            return {
                "task_type": "classification",
                "n_classes": int(len(classes)),
                "classes": classes,
            }
        return {"task_type": "regression", "n_classes": None, "classes": None}

    # Non-numeric => classification
    classes = sorted(list(y_no_na.unique().tolist()), key=lambda v: str(v))
    return {
        "task_type": "classification",
        "n_classes": int(len(classes)),
        "classes": classes,
    }

