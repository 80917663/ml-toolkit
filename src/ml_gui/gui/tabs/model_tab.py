from __future__ import annotations

from typing import Any, Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from sklearn.metrics import ConfusionMatrixDisplay

from ml_gui.core.training import train_and_evaluate


class ModelTab(QWidget):
    """
    Train / evaluate.
    Qt labels may be English for consistency with exported figures (SCI-style).
    All Matplotlib axis/title text is English only.
    """

    def __init__(self, state: Any):
        super().__init__()
        self.state = state
        self.on_trained = None

        self.banner = QLabel()
        self.banner.setWordWrap(True)
        self.banner.setObjectName("taskBanner")
        self.banner.setStyleSheet(
            "#taskBanner { padding: 10px; border-radius: 6px; background: #e8f4fc; "
            "font-weight: 600; font-size: 13px; }"
        )

        self.grp = QGroupBox()
        self.cmb_model = QComboBox()
        self.cmb_model.setMinimumWidth(280)

        self.sp_test_size = QDoubleSpinBox()
        self.sp_test_size.setRange(0.05, 0.5)
        self.sp_test_size.setSingleStep(0.05)
        self.sp_test_size.setValue(0.2)

        self.sp_random_state = QComboBox()
        self.sp_random_state.addItems(["42", "0", "1", "7", "123"])
        self.sp_random_state.setCurrentText("42")

        self.btn_train = QPushButton("Train & evaluate")
        self.btn_train.clicked.connect(self._on_train_clicked)

        self.txt_metrics = QTextEdit()
        self.txt_metrics.setReadOnly(True)
        self.txt_metrics.setMinimumHeight(140)

        self.lbl_figure = QLabel(
            "Figure (export): classification → confusion matrix; regression → predicted vs. true"
        )
        self.lbl_figure.setWordWrap(True)

        self.canvas = FigureCanvas(Figure(figsize=(7, 5.5)))

        form = QFormLayout()
        form.addRow("Model", self.cmb_model)
        form.addRow("Test set fraction", self.sp_test_size)
        form.addRow("Random state", self.sp_random_state)

        grp_layout = QVBoxLayout(self.grp)
        grp_layout.addLayout(form)
        grp_layout.addWidget(self.btn_train)
        grp_layout.addWidget(QLabel("Metrics (test set)"))
        grp_layout.addWidget(self.txt_metrics)
        grp_layout.addWidget(self.lbl_figure)
        grp_layout.addWidget(self.canvas, 1)

        root = QVBoxLayout(self)
        root.addWidget(self.banner)
        root.addWidget(self.grp, 1)

        self.refresh_from_state()

    def refresh_from_state(self) -> None:
        task_type = getattr(self.state, "task_type", None)
        self.cmb_model.clear()

        if task_type == "classification":
            nc = getattr(self.state, "n_classes", None)
            extra = f"{nc} classes" if nc is not None else "multiclass / binary"
            self.banner.setText(
                f"<b>Task: Classification</b> — {extra}. "
                f"Choose a classifier below. Figure export uses English labels."
            )
            self.grp.setTitle("Classification — training & evaluation")
            self.cmb_model.addItems(
                [
                    "Logistic Regression",
                    "SVC (probability)",
                    "KNN",
                    "Random Forest",
                    "Gradient Boosting",
                ]
            )
            self.cmb_model.setCurrentIndex(0)
            self.btn_train.setEnabled(True)
        elif task_type == "regression":
            self.banner.setText(
                "<b>Task: Regression</b> — continuous target. "
                "Choose a regressor below. Figure export uses English labels."
            )
            self.grp.setTitle("Regression — training & evaluation")
            self.cmb_model.addItems(
                [
                    "Linear Regression",
                    "Ridge",
                    "Lasso",
                    "ElasticNet",
                    "SVR",
                    "Random Forest Regressor",
                    "Gradient Boosting Regressor",
                ]
            )
            self.cmb_model.setCurrentIndex(0)
            self.btn_train.setEnabled(True)
        else:
            self.banner.setText(
                "<b>Task: not set</b> — complete the <b>Data</b> tab (target column and task type)."
            )
            self.grp.setTitle("Model — training & evaluation")
            self.btn_train.setEnabled(False)

        self._clear_canvas()
        self.txt_metrics.setPlainText("Load data and set target on the Data tab first.")

    def _clear_canvas(self) -> None:
        self.canvas.figure.clear()
        self.canvas.draw()

    def _plot_confusion_matrix(self, confusion: np.ndarray, labels: Optional[list[Any]] = None) -> None:
        fig = self.canvas.figure
        fig.clear()

        n = int(confusion.shape[0])
        # Wider/taller figure for many classes; reserve margins for tick labels
        w = max(6.5, min(14.0, 4.0 + 0.55 * n))
        h = max(5.5, min(12.0, 4.0 + 0.55 * n))
        fig.set_size_inches(w, h)

        ax = fig.add_subplot(111)
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=labels)
        disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=True)
        ax.set_title("Confusion matrix")
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")

        tick_fs = max(6, min(10, int(14 - 0.35 * n)))
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")
            label.set_fontsize(tick_fs)
        for label in ax.get_yticklabels():
            label.set_fontsize(tick_fs)

        fig.subplots_adjust(left=0.18, right=0.88, bottom=0.22, top=0.92)
        self.canvas.draw()

    def _plot_regression_scatter(self, y_true: Any, y_pred: Any) -> None:
        fig = self.canvas.figure
        fig.clear()
        fig.set_size_inches(6.5, 5.5)
        ax = fig.add_subplot(111)
        y_true_arr = np.asarray(y_true, dtype=float)
        y_pred_arr = np.asarray(y_pred, dtype=float)
        ax.scatter(y_true_arr, y_pred_arr, s=22, alpha=0.7, edgecolors="none")
        mn = float(np.nanmin([y_true_arr.min(), y_pred_arr.min()]))
        mx = float(np.nanmax([y_true_arr.max(), y_pred_arr.max()]))
        ax.plot([mn, mx], [mn, mx], "r--", linewidth=1, label="Ideal (y = x)")
        ax.set_xlabel("True value")
        ax.set_ylabel("Predicted value")
        ax.set_title("Predicted vs. true (test set)")
        ax.legend(loc="upper left", fontsize=9)
        fig.subplots_adjust(left=0.12, right=0.97, bottom=0.12, top=0.90)
        self.canvas.draw()

    def _current_model_name(self) -> str:
        return self.cmb_model.currentText()

    def _on_train_clicked(self) -> None:
        if self.state.X is None or self.state.y is None or self.state.task_type is None:
            QMessageBox.warning(self, "Notice", "Please finish the Data tab first.")
            return

        task_type = self.state.task_type
        model_name = self._current_model_name()
        test_size = float(self.sp_test_size.value())
        random_state = int(self.sp_random_state.currentText())

        try:
            res = train_and_evaluate(
                self.state.X,
                self.state.y,
                task_type=task_type,
                model_name=model_name,
                test_size=test_size,
                random_state=random_state,
            )

            self.state.pipeline = res.pipeline
            previous = self.state.last_results or {}
            self.state.last_results = {
                **previous,
                "metrics": res.metrics,
                "model_name": res.model_name,
                "task_type": res.task_type,
                "confusion": res.confusion,
                "y_test": res.y_test,
                "y_pred": res.y_pred,
                "y_proba": res.y_proba,
                "X_test": res.X_test,
                "test_size": test_size,
                "random_state": random_state,
            }

            lines = [f"Model: {res.model_name}", f"Task: {res.task_type}", ""]
            for k, v in res.metrics.items():
                lines.append(f"{k} = {v:.6g}" if isinstance(v, (int, float)) else f"{k} = {v}")
            self.txt_metrics.setPlainText("\n".join(lines))

            if task_type == "classification" and res.confusion is not None:
                labels = None
                try:
                    labels = list(getattr(res.pipeline.named_steps["model"], "classes_"))
                except Exception:
                    labels = None
                self._plot_confusion_matrix(res.confusion, labels=labels)
            else:
                self._plot_regression_scatter(res.y_test, res.y_pred)

            self.state.last_results["figure_model"] = self.canvas.figure

            if self.on_trained is not None:
                self.on_trained()

        except Exception as e:
            QMessageBox.critical(self, "Training failed", str(e))
