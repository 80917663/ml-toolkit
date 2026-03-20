from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class EdaTab(QWidget):
    def __init__(self, state: Any):
        super().__init__()
        self.state = state

        self.canvas = FigureCanvas(Figure(figsize=(7, 5)))
        self.canvas.figure.tight_layout()

        self.txt_summary = QTextEdit()
        self.txt_summary.setReadOnly(True)
        self.txt_summary.setMinimumHeight(140)

        self.cmb_corr_method = QComboBox()
        self.cmb_corr_method.addItems(["pearson", "spearman"])
        self.cmb_corr_method.setCurrentText("pearson")

        self.sp_max_corr_features = QSpinBox()
        self.sp_max_corr_features.setRange(5, 60)
        self.sp_max_corr_features.setValue(25)

        self.chk_show_missing_heatmap = QLabel("")  # placeholder for layout simplicity
        self.sp_heatmap_rows = QSpinBox()
        self.sp_heatmap_rows.setRange(10, 80)
        self.sp_heatmap_rows.setValue(30)

        self.btn_run = QPushButton("运行 EDA（图表 + 统计摘要）")
        self.btn_run.clicked.connect(self._on_run_clicked)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.addRow("相关性计算", self.cmb_corr_method)
        form.addRow("相关矩阵最大特征数", self.sp_max_corr_features)
        layout.addLayout(form)

        layout.addWidget(self.btn_run)
        layout.addWidget(QLabel("统计摘要（可复制到论文）"))
        layout.addWidget(self.txt_summary)
        layout.addWidget(QLabel("EDA 图表"))
        layout.addWidget(self.canvas, 1)

    def refresh_from_state(self) -> None:
        if self.state.X is None or self.state.y is None:
            self.btn_run.setEnabled(False)
            self.txt_summary.setPlainText("请先在 Data 页导入数据并准备目标 y。")
            self.canvas.figure.clear()
            self.canvas.draw()
            return

        self.btn_run.setEnabled(True)
        self.txt_summary.setPlainText("可点击按钮运行 EDA。")
        self.canvas.figure.clear()
        self.canvas.draw()

    def _on_run_clicked(self) -> None:
        if self.state.X is None or self.state.y is None:
            QMessageBox.warning(self, "提示", "请先加载数据。")
            return

        X = self.state.X
        y = self.state.y
        task_type = self.state.task_type

        df = X.copy()
        df["__target__"] = y.values

        n_samples, n_features = X.shape
        summary_lines = [
            f"样本数 n={n_samples}",
            f"特征数 p={n_features}",
        ]

        # Missing statistics
        missing_ratio = X.isna().mean().sort_values(ascending=False)
        missing_top = missing_ratio.head(10)
        summary_lines.append("")
        summary_lines.append("缺失值（Top 10）:")
        for col, r in missing_top.items():
            summary_lines.append(f"- {col}: {r:.3%}")

        # Target distribution
        summary_lines.append("")
        summary_lines.append("目标分布（y）:")
        if task_type == "classification":
            vc = y.value_counts(dropna=True)
            summary_lines.append(f"类别数：{len(vc)}")
            for k, v in vc.head(10).items():
                summary_lines.append(f"- {k}: {v}（{v/len(y):.2%}）")
        else:
            y_num = pd.to_numeric(y, errors="coerce").dropna()
            summary_lines.append(f"连续值范围：[{y_num.min():.6g}, {y_num.max():.6g}]")
            summary_lines.append("分位数（0,25,50,75,100）：")
            qs = np.quantile(y_num.to_numpy(), [0, 0.25, 0.5, 0.75, 1.0])
            summary_lines.append(f"- {qs[0]:.6g}, {qs[1]:.6g}, {qs[2]:.6g}, {qs[3]:.6g}, {qs[4]:.6g}")

        # Draw plots (missing heatmap + correlation + PCA)
        self.canvas.figure.clear()
        fig = self.canvas.figure

        # Layout: 3 subplots when possible
        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 3)

        ax1.set_title("Missing rate (top features)")
        top_k = min(30, len(missing_top))
        ax1.barh(range(top_k), missing_top.head(top_k).to_numpy()[::-1], color="#4C78A8")
        ax1.set_yticks(range(top_k))
        ax1.set_yticklabels(list(missing_top.head(top_k).index)[::-1], fontsize=8)
        ax1.set_xlabel("Missing ratio")

        # Correlation heatmap (numeric only, top by variance)
        corr_method = self.cmb_corr_method.currentText()
        max_corr_features = int(self.sp_max_corr_features.value())

        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            variances = X[numeric_cols].var(numeric_only=True).sort_values(ascending=False)
            use_cols = variances.head(max_corr_features).index.tolist()
            corr = X[use_cols].corr(method=corr_method)
            sns.heatmap(corr, ax=ax2, cmap="RdBu_r", center=0, cbar=False)
            ax2.set_title(f"Correlation ({corr_method})")
            # PCA plot below
            ax3.set_title("PCA (2D)")
            self._plot_pca(ax3, X, y)
        else:
            ax2.text(
                0.5,
                0.5,
                "Need >= 2 numeric features\nfor correlation / PCA",
                ha="center",
                va="center",
                transform=ax2.transAxes,
            )
            ax3.text(
                0.5,
                0.5,
                "Need >= 2 numeric features\nfor PCA",
                ha="center",
                va="center",
                transform=ax3.transAxes,
            )

        fig.tight_layout()
        self.canvas.draw()

        self.txt_summary.setPlainText("\n".join(summary_lines))

        # expose figure for export
        self.state.last_results = self.state.last_results or {}
        self.state.last_results["figure_eda"] = self.canvas.figure

    def _plot_pca(self, ax, X: pd.DataFrame, y: pd.Series) -> None:
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            return

        X_num = X[numeric_cols]
        # Impute + scale for PCA robustness
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        X_imp = imputer.fit_transform(X_num)
        X_scaled = scaler.fit_transform(X_imp)

        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        pc1_var = pca.explained_variance_ratio_[0]
        pc2_var = pca.explained_variance_ratio_[1]

        task_type = getattr(self.state, "task_type", None)
        if task_type == "classification":
            # Color by class if few classes
            uniq = y.dropna().unique()
            if len(uniq) <= 10:
                for cls in uniq:
                    mask = y == cls
                    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], s=18, alpha=0.7, label=str(cls))
                ax.legend(fontsize=7, loc="best")
            else:
                ax.scatter(X_pca[:, 0], X_pca[:, 1], s=18, alpha=0.7)
        else:
            ax.scatter(X_pca[:, 0], X_pca[:, 1], s=18, alpha=0.7)

        ax.set_xlabel(f"PC1 ({pc1_var:.1%})")
        ax.set_ylabel(f"PC2 ({pc2_var:.1%})")

