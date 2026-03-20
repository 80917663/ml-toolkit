from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from scipy import sparse as sp_sparse

from sklearn.inspection import permutation_importance

from ml_gui.core.shap_utils import (
    build_shap_explanation,
    mean_abs_shap_per_feature,
    pick_class_shap_matrix,
)


def _to_dense_array(X: Any) -> np.ndarray:
    if sp_sparse.issparse(X):
        return X.toarray()
    return np.asarray(X)


@dataclass
class _ShapBundle:
    """Cached SHAP computation for switching plot types without full recompute."""

    shap_values: Any
    X_arr: np.ndarray
    feat_names: list[str]
    expected_value: Any
    explainer: Any
    model: Any
    interaction_values: Any | None = None


class ExplainTab(QWidget):
    """
    Permutation importance + multiple SHAP visualizations (publication-style English labels).
    """

    SHAP_VIZ_ITEMS = [
        "Bar (mean |SHAP|)",
        "Summary beeswarm",
        "Summary bar (native)",
        "Heatmap (samples × features)",
        "Dependence scatter",
        "Dependence + interaction color",
        "Dependence (legacy, auto interaction)",
        "Interaction matrix (tree, subset)",
        "Waterfall (one instance)",
        "Force plot (matplotlib)",
    ]

    def __init__(self, state: Any):
        super().__init__()
        self.state = state
        self._bundle: _ShapBundle | None = None
        self._bundle_pipeline_id: int | None = None
        self._bundle_k_eff: int | None = None

        self.canvas = FigureCanvas(Figure(figsize=(8, 6)))

        self.txt_info = QTextEdit()
        self.txt_info.setReadOnly(True)

        self.cmb_method = QComboBox()
        self.cmb_method.addItems(["Permutation importance", "SHAP"])
        self.cmb_method.currentIndexChanged.connect(self._update_method_ui)

        self.cmb_shap_viz = QComboBox()
        for t in self.SHAP_VIZ_ITEMS:
            self.cmb_shap_viz.addItem(t)

        self.sp_top_k = QSpinBox()
        self.sp_top_k.setRange(5, 100)
        self.sp_top_k.setValue(20)

        self.sp_shap_samples = QSpinBox()
        self.sp_shap_samples.setRange(20, 2000)
        self.sp_shap_samples.setValue(200)

        self.sp_max_display = QSpinBox()
        self.sp_max_display.setRange(5, 40)
        self.sp_max_display.setValue(15)

        self.sp_class_index = QSpinBox()
        self.sp_class_index.setRange(0, 50)
        self.sp_class_index.setValue(0)
        self.sp_class_index.setToolTip("Output class index for multiclass SHAP (tree/list outputs).")

        self.cmb_dep_feature = QComboBox()
        self.cmb_color_feature = QComboBox()

        self.sp_instance_index = QSpinBox()
        self.sp_instance_index.setRange(0, 9999)
        self.sp_instance_index.setValue(0)
        self.sp_instance_index.setToolTip("Row index within the SHAP sample (0 = first).")

        self.btn_run = QPushButton("Run explainability")
        self.btn_run.clicked.connect(self._on_run_clicked)

        self.grp_shap = QGroupBox("SHAP options")
        form_shap = QFormLayout()
        form_shap.addRow("Plot type", self.cmb_shap_viz)
        form_shap.addRow("Max samples", self.sp_shap_samples)
        form_shap.addRow("Max features displayed", self.sp_max_display)
        form_shap.addRow("Class index (multiclass)", self.sp_class_index)
        form_shap.addRow("Dependence: feature", self.cmb_dep_feature)
        form_shap.addRow("Color by feature (interaction)", self.cmb_color_feature)
        form_shap.addRow("Instance index (waterfall / force)", self.sp_instance_index)
        self.grp_shap.setLayout(form_shap)

        form_top = QFormLayout()
        form_top.addRow("Method", self.cmb_method)
        form_top.addRow("Top-K (permutation only)", self.sp_top_k)

        layout = QVBoxLayout(self)
        layout.addLayout(form_top)
        layout.addWidget(self.grp_shap)
        layout.addWidget(self.btn_run)
        layout.addWidget(QLabel("Log"))
        layout.addWidget(self.txt_info)
        layout.addWidget(QLabel("Figure (English labels)"))
        layout.addWidget(self.canvas, 1)

        self._update_method_ui()
        self.refresh_from_state()

    def _update_method_ui(self) -> None:
        is_shap = self.cmb_method.currentText() == "SHAP"
        self.grp_shap.setEnabled(is_shap)
        self.sp_top_k.setEnabled(not is_shap)

    def refresh_from_state(self) -> None:
        pipeline = getattr(self.state, "pipeline", None)
        if pipeline is None:
            self.btn_run.setEnabled(False)
            self.sp_class_index.setEnabled(False)
            self.txt_info.setPlainText("Train a model on the Model tab first.")
            self.canvas.figure.clear()
            self.canvas.draw()
            return

        self.btn_run.setEnabled(True)
        self.txt_info.setPlainText("Ready. Choose method and run.")
        self.canvas.figure.clear()
        self.canvas.draw()

        self.sp_class_index.setEnabled(getattr(self.state, "task_type", None) == "classification")

        pid = id(pipeline)
        if self._bundle_pipeline_id != pid:
            self._bundle = None
            self._bundle_k_eff = None
        self._bundle_pipeline_id = pid

    def _fill_feature_combos(self, names: list[str]) -> None:
        self.cmb_dep_feature.blockSignals(True)
        self.cmb_color_feature.blockSignals(True)
        self.cmb_dep_feature.clear()
        self.cmb_color_feature.clear()
        for n in names:
            self.cmb_dep_feature.addItem(str(n))
            self.cmb_color_feature.addItem(str(n))
        if self.cmb_dep_feature.count() > 1:
            self.cmb_color_feature.setCurrentIndex(1)
        self.cmb_dep_feature.blockSignals(False)
        self.cmb_color_feature.blockSignals(False)

    def _on_run_clicked(self) -> None:
        pipeline = self.state.pipeline
        X_test = None
        y_test = None
        if self.state.last_results:
            X_test = self.state.last_results.get("X_test")
            y_test = self.state.last_results.get("y_test")

        if pipeline is None or X_test is None or y_test is None:
            QMessageBox.warning(self, "Notice", "Please run training on the Model tab first.")
            return

        if self.cmb_method.currentText().startswith("Permutation"):
            self._compute_permutation_importance(
                pipeline, X_test, y_test, top_k=int(self.sp_top_k.value())
            )
            return

        self._run_shap_flow(pipeline, X_test, y_test)

    def _compute_permutation_importance(
        self, pipeline: Any, X_test: pd.DataFrame, y_test: pd.Series, top_k: int
    ) -> None:
        self.canvas.figure.clear()
        fig = self.canvas.figure
        ax = fig.add_subplot(111)

        task_type = getattr(self.state, "task_type", None)
        scoring = "accuracy" if task_type == "classification" else "r2"

        try:
            res = permutation_importance(
                pipeline,
                X_test,
                y_test,
                scoring=scoring,
                n_repeats=10,
                random_state=42,
            )
            importances = res.importances_mean
            feature_names = list(X_test.columns)
        except Exception as e:
            QMessageBox.critical(self, "Failed", str(e))
            return

        order = np.argsort(importances)[::-1]
        top_idx = order[: min(top_k, len(feature_names))]
        top_features = [feature_names[i] for i in top_idx]
        top_values = importances[top_idx]

        ax.barh(range(len(top_features)), top_values[::-1], color="#F58518")
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features[::-1], fontsize=9)
        ax.set_xlabel("Mean decrease in score")
        ax.set_title("Permutation feature importance (test set)")

        lines = ["Top features (permutation, test set):", ""]
        for f, v in zip(top_features[:10], top_values[:10]):
            lines.append(f"- {f}: {float(v):.6g}")
        self.txt_info.setPlainText("\n".join(lines))

        fig.tight_layout()
        self.canvas.draw()

        self.state.last_results = self.state.last_results or {}
        self.state.last_results["figure_explain"] = self.canvas.figure

    def _ensure_shap_bundle(self, pipeline: Any, X_test: pd.DataFrame) -> _ShapBundle | None:
        import shap

        pid = id(pipeline)
        k = int(self.sp_shap_samples.value())
        k_eff = min(k, len(X_test))
        if (
            self._bundle is not None
            and self._bundle_pipeline_id == pid
            and self._bundle_k_eff == k_eff
        ):
            return self._bundle

        self.txt_info.setPlainText("Computing SHAP values (may take a while)...")
        self.canvas.draw()

        try:
            preprocessor = pipeline.named_steps["preprocess"]
            model = pipeline.named_steps["model"]
        except Exception as e:
            self.txt_info.setPlainText(f"Unexpected pipeline: {e}")
            return None

        n = len(X_test)
        if n > k_eff:
            idx = np.random.RandomState(42).choice(n, size=k_eff, replace=False)
            X_sample = X_test.iloc[idx]
        else:
            X_sample = X_test

        X_trans = preprocessor.transform(X_sample)
        X_arr = _to_dense_array(X_trans)

        try:
            feat_names = list(preprocessor.get_feature_names_out())
        except Exception:
            feat_names = [f"f{i}" for i in range(X_arr.shape[1])]

        n_bg = min(100, X_arr.shape[0])
        X_bg = X_arr[:n_bg]

        shap_values = None
        expected_value = None
        explainer = None
        last_err: str | None = None

        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_arr)
            expected_value = explainer.expected_value
        except Exception as e:
            last_err = str(e)

        if shap_values is None:
            try:
                explainer = shap.LinearExplainer(model, X_bg)
                shap_values = explainer.shap_values(X_arr)
                expected_value = getattr(explainer, "expected_value", None)
            except Exception as e:
                last_err = str(e)

        if shap_values is None:
            try:
                masker = shap.maskers.Independent(X_bg)
                explainer = shap.Explainer(model, masker)
                out = explainer(X_arr)
                shap_values = out.values
                expected_value = out.base_values
                if isinstance(expected_value, np.ndarray) and expected_value.ndim == 2:
                    expected_value = expected_value[:, 0]
            except Exception as e:
                last_err = str(e)

        if shap_values is None or explainer is None:
            self.txt_info.setPlainText(
                "SHAP computation failed.\n"
                f"Last error: {last_err}\n"
                "Try Random Forest / Gradient Boosting or use Permutation importance."
            )
            return None

        self._bundle = _ShapBundle(
            shap_values=shap_values,
            X_arr=X_arr,
            feat_names=feat_names,
            expected_value=expected_value,
            explainer=explainer,
            model=model,
            interaction_values=None,
        )
        self._bundle_pipeline_id = pid
        self._bundle_k_eff = k_eff
        self._fill_feature_combos(feat_names)
        return self._bundle

    def _class_idx_for_plots(self, bundle: _ShapBundle) -> int | None:
        if getattr(self.state, "task_type", None) != "classification":
            return None
        if isinstance(bundle.shap_values, list):
            n = len(bundle.shap_values)
            if n <= 1:
                return 0
            return int(self.sp_class_index.value()) % n
        arr = np.asarray(bundle.shap_values)
        if arr.ndim == 3:
            return int(self.sp_class_index.value()) % arr.shape[2]
        return None

    def _run_shap_flow(self, pipeline: Any, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        import shap

        bundle = self._ensure_shap_bundle(pipeline, X_test)
        if bundle is None:
            self.canvas.draw()
            return

        viz = self.cmb_shap_viz.currentText()
        max_disp = int(self.sp_max_display.value())
        class_idx = self._class_idx_for_plots(bundle)
        ci_for_expl = class_idx  # None (regression / single-output) or class index

        try:
            if viz == "Bar (mean |SHAP|)":
                self._plot_shap_bar_custom(bundle, class_idx, max_disp)
            elif viz == "Summary beeswarm":
                expl, _ = build_shap_explanation(
                    bundle.shap_values,
                    bundle.X_arr,
                    bundle.feat_names,
                    bundle.expected_value,
                    ci_for_expl,
                )
                self._plot_on_canvas(lambda ax: shap.plots.beeswarm(
                    expl, max_display=max_disp, ax=ax, show=False, plot_size=None
                ))
            elif viz == "Summary bar (native)":
                expl, _ = build_shap_explanation(
                    bundle.shap_values,
                    bundle.X_arr,
                    bundle.feat_names,
                    bundle.expected_value,
                    ci_for_expl,
                )
                self._plot_on_canvas(lambda ax: shap.plots.bar(
                    expl, max_display=max_disp, ax=ax, show=False
                ))
            elif viz == "Heatmap (samples × features)":
                expl, _ = build_shap_explanation(
                    bundle.shap_values,
                    bundle.X_arr,
                    bundle.feat_names,
                    bundle.expected_value,
                    ci_for_expl,
                )
                self._plot_on_canvas(
                    lambda ax: shap.plots.heatmap(
                        expl, max_display=max_disp, ax=ax, show=False, plot_width=10
                    ),
                    figsize=(10, 6),
                )
            elif viz == "Dependence scatter":
                expl, _ = build_shap_explanation(
                    bundle.shap_values,
                    bundle.X_arr,
                    bundle.feat_names,
                    bundle.expected_value,
                    ci_for_expl,
                )
                fi = self.cmb_dep_feature.currentIndex()
                self._plot_on_canvas(
                    lambda ax: shap.plots.scatter(expl[:, fi], ax=ax, show=False)
                )
            elif viz == "Dependence + interaction color":
                expl, _ = build_shap_explanation(
                    bundle.shap_values,
                    bundle.X_arr,
                    bundle.feat_names,
                    bundle.expected_value,
                    ci_for_expl,
                )
                fi = self.cmb_dep_feature.currentIndex()
                ci_feat = self.cmb_color_feature.currentIndex()
                nfeat = expl.values.shape[1]
                if ci_feat == fi and nfeat > 1:
                    ci_feat = (fi + 1) % nfeat
                self._plot_on_canvas(
                    lambda ax, fi=fi, ci_feat=ci_feat: shap.plots.scatter(
                        expl[:, fi], color=expl[:, ci_feat], ax=ax, show=False
                    )
                )
            elif viz == "Dependence (legacy, auto interaction)":
                sv, _ = pick_class_shap_matrix(bundle.shap_values, ci_for_expl)
                fi = self.cmb_dep_feature.currentIndex()
                self._plot_with_pyplot_current_figure(
                    lambda: shap.dependence_plot(
                        fi,
                        sv,
                        bundle.X_arr,
                        feature_names=bundle.feat_names,
                        interaction_index="auto",
                        show=False,
                    ),
                    figsize=(8, 5),
                )
            elif viz == "Interaction matrix (tree, subset)":
                self._plot_interaction_matrix(bundle, class_idx, max_disp)
            elif viz == "Waterfall (one instance)":
                expl, _ = build_shap_explanation(
                    bundle.shap_values,
                    bundle.X_arr,
                    bundle.feat_names,
                    bundle.expected_value,
                    ci_for_expl,
                )
                i = min(int(self.sp_instance_index.value()), expl.values.shape[0] - 1)
                self._plot_with_pyplot_current_figure(
                    lambda: shap.plots.waterfall(expl[i], max_display=max_disp, show=False),
                    figsize=(8, 5),
                )
            elif viz == "Force plot (matplotlib)":
                expl, _ = build_shap_explanation(
                    bundle.shap_values,
                    bundle.X_arr,
                    bundle.feat_names,
                    bundle.expected_value,
                    ci_for_expl,
                )
                i = min(int(self.sp_instance_index.value()), expl.values.shape[0] - 1)
                bv = np.asarray(expl.base_values, dtype=float).ravel()
                base = float(bv[i] if i < len(bv) else bv[0])
                self._plot_with_pyplot_current_figure(
                    lambda: shap.plots.force(
                        base,
                        shap_values=expl.values[i],
                        features=expl.data[i],
                        feature_names=list(expl.feature_names),
                        matplotlib=True,
                        show=False,
                        figsize=(14, 3),
                    ),
                    figsize=(14, 3),
                )
            else:
                self._plot_shap_bar_custom(bundle, class_idx, max_disp)
        except Exception as e:
            self.txt_info.setPlainText(f"Plot failed: {e}")
            QMessageBox.warning(self, "SHAP plot", str(e))
            self.canvas.draw()
            return

        self.txt_info.setPlainText(
            f"SHAP plot: {viz}\n"
            f"Samples in explanation: {bundle.X_arr.shape[0]}, features: {bundle.X_arr.shape[1]}.\n"
            "Export this figure from the Report tab if needed."
        )
        self.state.last_results = self.state.last_results or {}
        self.state.last_results["figure_explain"] = self.canvas.figure

    def _plot_on_canvas(self, draw_ax, figsize: tuple[float, float] | None = None) -> None:
        fig = self.canvas.figure
        fig.clear()
        if figsize:
            fig.set_size_inches(figsize[0], figsize[1])
        else:
            fig.set_size_inches(9, 6)
        ax = fig.add_subplot(111)
        draw_ax(ax)
        fig.tight_layout()
        self.canvas.draw()

    def _plot_with_pyplot_current_figure(
        self, draw_fn: Any, figsize: tuple[float, float] = (8, 5)
    ) -> None:
        """SHAP plots that target plt.gcf() instead of taking ax=."""
        fig = self.canvas.figure
        fig.set_size_inches(figsize[0], figsize[1])
        plt.figure(fig.number)
        plt.clf()
        draw_fn()
        fig = self.canvas.figure
        fig.tight_layout()
        self.canvas.draw()

    def _plot_shap_bar_custom(
        self, bundle: _ShapBundle, class_idx: int | None, max_disp: int
    ) -> None:
        importance = mean_abs_shap_per_feature(bundle.shap_values)
        fn = bundle.feat_names
        if len(fn) != importance.size:
            fn = [f"f{i}" for i in range(importance.size)]

        max_disp = min(max_disp, importance.size)
        order = np.argsort(importance)[::-1][:max_disp]
        order = order[::-1]

        fig = self.canvas.figure
        fig.clear()
        fig.set_size_inches(7.5, max(4.5, 0.28 * max_disp + 2))
        ax = fig.add_subplot(111)
        vals = importance[order]
        labels = [str(fn[i])[:50] for i in order]
        ax.barh(np.arange(len(vals)), vals, color="#4C78A8")
        ax.set_yticks(np.arange(len(vals)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("mean |SHAP value|")
        ax.set_title("SHAP feature importance (mean |SHAP|)")
        fig.subplots_adjust(left=0.28, right=0.97, bottom=0.10, top=0.92)
        self.canvas.draw()

    def _plot_interaction_matrix(
        self, bundle: _ShapBundle, class_idx: int | None, max_disp: int
    ) -> None:
        """Mean |interaction| over samples, top features by main-effect |SHAP|."""
        try:
            inter = bundle.explainer.shap_interaction_values(bundle.X_arr)
        except Exception as e:
            raise RuntimeError(
                "Interaction values need TreeExplainer-compatible models (e.g. RF, GB). "
                f"Detail: {e}"
            ) from e

        if isinstance(inter, list):
            if class_idx is None:
                class_idx = 0
            inter_arr = np.asarray(inter[int(class_idx) % len(inter)], dtype=float)
        else:
            inter_arr = np.asarray(inter, dtype=float)

        if inter_arr.ndim != 3:
            raise RuntimeError(f"Unexpected interaction shape: {inter_arr.shape}")

        inter_mean = np.abs(inter_arr).mean(axis=0)  # (F, F)
        sv, _ = pick_class_shap_matrix(bundle.shap_values, class_idx)
        imp = np.abs(sv).mean(axis=0)
        top = np.argsort(imp)[::-1][:max_disp]
        sub = inter_mean[np.ix_(top, top)]
        labels = [str(bundle.feat_names[j])[:14] for j in top]

        fig = self.canvas.figure
        fig.clear()
        fig.set_size_inches(max(6, 0.45 * max_disp), max(5, 0.45 * max_disp))
        ax = fig.add_subplot(111)
        im = ax.imshow(sub, cmap="viridis", aspect="auto")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Mean |interaction|")
        ax.set_xticks(range(len(top)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_title("SHAP interaction matrix (top features, subset)")
        fig.tight_layout()
        self.canvas.draw()
