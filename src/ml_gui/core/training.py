from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)

from ml_gui.core.metrics import classification_metrics, regression_metrics, compute_confusion
from ml_gui.core.preprocessing import build_preprocessor


MODEL_MAP: Dict[str, Dict[str, Any]] = {
    "classification": {
        "Logistic Regression": LogisticRegression,
        "SVC (probability)": SVC,
        "KNN": KNeighborsClassifier,
        "Random Forest": RandomForestClassifier,
        "Gradient Boosting": GradientBoostingClassifier,
    },
    "regression": {
        "Linear Regression": LinearRegression,
        "Ridge": Ridge,
        "Lasso": Lasso,
        "ElasticNet": ElasticNet,
        "SVR": SVR,
        "Random Forest Regressor": RandomForestRegressor,
        "Gradient Boosting Regressor": GradientBoostingRegressor,
    },
}


def _make_model(task_type: str, model_name: str, *, random_state: int) -> Any:
    model_cls = MODEL_MAP[task_type].get(model_name)
    if model_cls is None:
        raise ValueError(f"未知模型：{model_name}")

    if task_type == "classification" and model_name == "Logistic Regression":
        return model_cls(max_iter=2000, n_jobs=None)
    if task_type == "classification" and model_name.startswith("SVC"):
        return model_cls(probability=True)
    if task_type == "classification" and model_name == "KNN":
        return model_cls(n_neighbors=5)
    if task_type == "classification" and model_name == "Random Forest":
        return model_cls(n_estimators=300, random_state=random_state)
    if task_type == "classification" and model_name == "Gradient Boosting":
        return model_cls(random_state=random_state)

    if task_type == "regression" and model_name == "Ridge":
        return model_cls(alpha=1.0)
    if task_type == "regression" and model_name == "Lasso":
        return model_cls(alpha=0.01, max_iter=5000)
    if task_type == "regression" and model_name == "ElasticNet":
        return model_cls(alpha=0.01, l1_ratio=0.5, max_iter=5000)
    if task_type == "regression" and model_name == "SVR":
        return model_cls()
    if task_type == "regression" and model_name == "Random Forest Regressor":
        return model_cls(n_estimators=300, random_state=random_state)
    if task_type == "regression" and model_name == "Gradient Boosting Regressor":
        return model_cls(random_state=random_state)

    return model_cls()


@dataclass
class TrainResult:
    metrics: Dict[str, Any]
    confusion: Optional[np.ndarray]
    y_test: Any
    X_test: Any
    y_pred: Any
    y_proba: Optional[np.ndarray]
    model_name: str
    task_type: str
    pipeline: Any


def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    task_type: str,
    model_name: str,
    test_size: float,
    random_state: int,
) -> TrainResult:
    if task_type not in ("classification", "regression"):
        raise ValueError(f"task_type 无效：{task_type}")

    # Ensure numeric target for regression
    y_use = y
    if task_type == "regression":
        y_use = pd.to_numeric(y_use, errors="coerce")
        mask = y_use.notna()
        X = X.loc[mask]
        y_use = y_use.loc[mask]

    stratify = y_use if task_type == "classification" else None
    # For classification, stratify supports multiclass too
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_use,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    preprocessor = build_preprocessor(X_train)
    model = _make_model(task_type, model_name, random_state=random_state)
    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    y_proba = None
    try:
        y_proba = pipeline.predict_proba(X_test)  # type: ignore[attr-defined]
    except Exception:
        y_proba = None

    # Best-effort roc labels mapping stability
    roc_labels = None
    if task_type == "classification" and y_proba is not None:
        try:
            roc_labels = list(getattr(pipeline.named_steps["model"], "classes_"))
        except Exception:
            roc_labels = None

    if task_type == "classification":
        # labels param is only used for f1/precision/recall when non-binary
        labels = None
        try:
            labels = list(roc_labels) if roc_labels is not None else None
        except Exception:
            labels = None
        confusion = compute_confusion(y_test, y_pred, labels=labels)  # type: ignore[arg-type]
        metrics = classification_metrics(
            y_test,
            y_pred,
            y_proba=y_proba,
            average_for_f1="weighted",
            labels=labels,
            roc_auc_labels=roc_labels,
        )
    else:
        confusion = None
        metrics = regression_metrics(y_test, y_pred)

    return TrainResult(
        metrics=metrics,
        confusion=confusion,
        y_test=y_test,
        X_test=X_test,
        y_pred=y_pred,
        y_proba=y_proba,
        model_name=model_name,
        task_type=task_type,
        pipeline=pipeline,
    )

