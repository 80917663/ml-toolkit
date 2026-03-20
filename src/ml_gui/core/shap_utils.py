from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np


def pick_class_shap_matrix(
    shap_values: Any, class_index: Optional[int] = None
) -> Tuple[np.ndarray, Optional[int]]:
    """
    Return 2D SHAP matrix (n_samples, n_features) for the chosen output class / head.
    For multiclass tree output (list of arrays), class_index selects the class.
    """
    if isinstance(shap_values, list):
        n = len(shap_values)
        if n == 0:
            return np.array([]).reshape(0, 0), None
        idx = 0 if class_index is None else int(class_index)
        idx = max(0, min(idx, n - 1))
        return np.asarray(shap_values[idx], dtype=float), idx

    arr = np.asarray(shap_values, dtype=float)
    if arr.ndim == 2:
        return arr, None
    if arr.ndim == 3:
        # (n_samples, n_features, n_outputs)
        c = 0 if class_index is None else int(class_index)
        c = max(0, min(c, arr.shape[2] - 1))
        return arr[:, :, c], c
    if arr.ndim == 1:
        return arr.reshape(1, -1), None
    raise ValueError(f"Unexpected shap_values shape: {arr.shape}")


def pick_base_values(expected_value: Any, class_idx: Optional[int], n_samples: int) -> np.ndarray:
    """Per-sample base value vector for Explanation.base_values."""
    ev = expected_value
    if isinstance(ev, (list, np.ndarray)):
        eva = np.asarray(ev, dtype=float).ravel()
        if eva.size == 0:
            b = 0.0
        elif class_idx is not None and eva.size > 1:
            ci = max(0, min(int(class_idx), eva.size - 1))
            b = float(eva[ci])
        else:
            b = float(eva[0])
    else:
        try:
            b = float(ev)
        except (TypeError, ValueError):
            b = 0.0
    return np.full(n_samples, b, dtype=float)


def build_shap_explanation(
    shap_values: Any,
    X_arr: np.ndarray,
    feature_names: list[str],
    expected_value: Any,
    class_index: Optional[int] = None,
) -> Tuple[Any, Optional[int]]:
    """Build shap.Explanation for one model output (class)."""
    import shap

    sv, cidx = pick_class_shap_matrix(shap_values, class_index)
    if sv.size == 0:
        raise ValueError("Empty SHAP matrix")
    n = sv.shape[0]
    # Explainer API may already provide one base value per sample
    ev = expected_value
    if isinstance(ev, np.ndarray) and ev.size == n and ev.ndim == 1:
        base = np.asarray(ev, dtype=float).reshape(-1)
    elif isinstance(ev, np.ndarray) and ev.shape == (n, 1):
        base = np.asarray(ev, dtype=float).reshape(-1)
    else:
        base = pick_base_values(ev, cidx, n)
    expl = shap.Explanation(
        values=sv,
        base_values=base,
        data=X_arr,
        feature_names=feature_names,
    )
    return expl, cidx


def mean_abs_shap_per_feature(shap_values: Any) -> np.ndarray:
    """
    Reduce SHAP values to one importance score per feature (for bar plot).
    Handles:
    - (n_samples, n_features)
    - list of (n_samples, n_features) for multiclass tree models
    - (n_samples, n_features, n_outputs) for some explainers
    """
    if isinstance(shap_values, list):
        # Multiclass tree: average mean(|SHAP|) across classes
        stacks = [np.abs(np.asarray(s, dtype=float)) for s in shap_values]
        if not stacks:
            return np.array([])
        per_class_mean = [s.mean(axis=0) for s in stacks]
        return np.mean(np.stack(per_class_mean, axis=0), axis=0)

    arr = np.asarray(shap_values, dtype=float)
    if arr.ndim == 1:
        return np.abs(arr)
    if arr.ndim == 2:
        return np.abs(arr).mean(axis=0)
    if arr.ndim == 3:
        # Heuristic: (n_samples, n_features, n_outputs) or (n_samples, n_outputs, n_features)
        s0, s1, s2 = arr.shape
        if s2 <= s1:
            return np.abs(arr).mean(axis=(0, 2))
        return np.abs(arr).mean(axis=(0, 1))
    return np.abs(arr).reshape(arr.shape[0], -1).mean(axis=0)
