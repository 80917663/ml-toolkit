from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.metrics import roc_auc_score


def classification_metrics(
    y_true,
    y_pred,
    *,
    y_proba: Optional[np.ndarray] = None,
    average_for_f1: str = "weighted",
    labels: Optional[list[Any]] = None,
    roc_auc_labels: Optional[list[Any]] = None,
) -> Dict[str, Any]:
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=average_for_f1, zero_division=0, labels=labels)
    recall = recall_score(y_true, y_pred, average=average_for_f1, zero_division=0, labels=labels)
    f1 = f1_score(y_true, y_pred, average=average_for_f1, zero_division=0, labels=labels)

    metrics: Dict[str, Any] = {
        "accuracy": float(acc),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
    }

    # Optional ROC-AUC
    if y_proba is not None:
        try:
            n_classes = y_proba.shape[1] if y_proba.ndim == 2 else 2
            if n_classes == 2:
                # binary: proba[:, 1] corresponds to the "second" class in sklearn's internal ordering.
                # We optionally pass roc_auc_labels to ensure mapping stability.
                pos_label = None
                if roc_auc_labels is not None and len(roc_auc_labels) == 2:
                    pos_label = roc_auc_labels[1]
                if pos_label is not None:
                    metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1], pos_label=pos_label))
                else:
                    metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
            else:
                metrics["roc_auc_ovr_weighted"] = float(
                    roc_auc_score(
                        y_true,
                        y_proba,
                        multi_class="ovr",
                        average="weighted",
                        labels=roc_auc_labels,
                    )
                )
        except Exception:
            # Keep UI robust: ROC-AUC is "best effort"
            pass

    return metrics


def regression_metrics(y_true, y_pred) -> Dict[str, Any]:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2),
    }


def compute_confusion(y_true, y_pred, labels: Optional[list[Any]] = None) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=labels)

