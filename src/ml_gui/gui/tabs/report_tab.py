from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Optional

from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class ReportTab(QWidget):
    def __init__(self, state: Any, run_root: Optional[str] = None):
        super().__init__()
        self.state = state
        if run_root is not None:
            self.run_root = run_root
        else:
            # report_tab.py -> tabs -> gui -> ml_gui -> src -> repo_root
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
            self.run_root = os.path.join(repo_root, "runs")

        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)

        self.btn_export = QPushButton("导出本次结果（图表 + 指标 + 报告）")
        self.btn_export.clicked.connect(self._on_export_clicked)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Report（v1）：生成 Markdown 报告并保存图表/指标 JSON"))
        layout.addWidget(self.btn_export)
        layout.addWidget(QLabel("导出日志"))
        layout.addWidget(self.txt_log, 1)

        self.refresh_from_state()

    def refresh_from_state(self) -> None:
        if self.state.last_results is None or self.state.last_results.get("metrics") is None:
            self.btn_export.setEnabled(False)
            self.txt_log.setPlainText("请先在 Model 页完成一次训练。")
        else:
            self.btn_export.setEnabled(True)
            self.txt_log.setPlainText("可以导出。")

    def _on_export_clicked(self) -> None:
        if self.state.last_results is None or self.state.last_results.get("metrics") is None:
            QMessageBox.warning(self, "提示", "请先完成一次训练。")
            return

        os.makedirs(self.run_root, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(self.run_root, ts)
        os.makedirs(run_dir, exist_ok=True)
        plots_dir = os.path.join(run_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # Save metrics
        last = self.state.last_results
        metrics = last.get("metrics", {})

        metrics_path = os.path.join(run_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        # Save metrics.csv for easy copy/paste
        metrics_csv_path = os.path.join(run_dir, "metrics.csv")
        try:
            import csv

            with open(metrics_csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["metric", "value"])
                for k, v in (metrics or {}).items():
                    writer.writerow([k, v])
        except Exception:
            metrics_csv_path = ""

        # Save figures if available
        saved_plots = []
        figures = {
            "model": last.get("figure_model"),
            "eda": last.get("figure_eda"),
            "explain": last.get("figure_explain"),
        }
        for name, fig in figures.items():
            if fig is None:
                continue
            try:
                out_path = os.path.join(plots_dir, f"{name}.png")
                fig.savefig(out_path, dpi=160, bbox_inches="tight")
                saved_plots.append(out_path)
            except Exception:
                # best-effort
                continue

        # Save config
        config = {
            "target_name": getattr(self.state, "target_name", None),
            "task_type": getattr(self.state, "task_type", None),
            "feature_names": getattr(self.state, "feature_names", None),
            "timestamp": ts,
            "model_name": last.get("model_name"),
            "n_classes": getattr(self.state, "n_classes", None),
            "classes": getattr(self.state, "classes", None),
            "test_size": last.get("test_size"),
            "random_state": last.get("random_state"),
        }
        cfg_path = os.path.join(run_dir, "config.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        # Build report markdown (paper-friendly)
        def _fmt(v: Any) -> str:
            if v is None:
                return "-"
            if isinstance(v, float):
                return f"{v:.6g}"
            return str(v)

        y = getattr(self.state, "y", None)
        X = getattr(self.state, "X", None)
        n_samples = int(len(y)) if y is not None else None
        n_features = int(X.shape[1]) if X is not None else None

        # Target distribution summary
        target_lines: list[str] = []
        task_type = config.get("task_type")
        if y is not None:
            try:
                import numpy as np
                import pandas as pd

                y_ser = pd.Series(y)
                if task_type == "classification":
                    vc = y_ser.value_counts(dropna=True)
                    target_lines.append(f"类别数：{len(vc)}")
                    for k, v in vc.head(10).items():
                        target_lines.append(f"- {k}: {int(v)}（{v/len(y_ser):.2%}）")
                else:
                    y_num = pd.to_numeric(y_ser, errors="coerce").dropna()
                    qs = np.quantile(y_num.to_numpy(), [0, 0.25, 0.5, 0.75, 1.0])
                    target_lines.append(
                        "分位数（0/25/50/75/100）："
                        f"{_fmt(qs[0])}, {_fmt(qs[1])}, {_fmt(qs[2])}, {_fmt(qs[3])}, {_fmt(qs[4])}"
                    )
            except Exception:
                target_lines.append("目标分布：无法解析（已忽略）。")

        # Save relative plot paths for markdown embedding
        rel_plots: list[tuple[str, str]] = []
        for abs_path in saved_plots:
            rel = os.path.relpath(abs_path, run_dir).replace("\\", "/")
            # rel looks like plots/model.png, keep label by filename
            label = os.path.splitext(os.path.basename(abs_path))[0]
            rel_plots.append((label, rel))

        lines = [
            f"# ML 科研分析报告（{ts}）",
            "",
            "## 1. 数据与任务",
            f"- 目标列 `y`：{config['target_name']}",
            f"- 任务类型：{config['task_type']}",
            f"- 模型算法：{config['model_name']}",
            f"- 样本数 `n`：{_fmt(n_samples)}",
            f"- 特征数 `p`：{_fmt(n_features)}",
        ]
        if config.get("task_type") == "classification" and config.get("n_classes") is not None:
            lines.append(f"- 类别数：{_fmt(config.get('n_classes'))}")
        if config.get("test_size") is not None:
            lines.append(f"- 数据划分：test_size={_fmt(config.get('test_size'))}, random_state={_fmt(config.get('random_state'))}")

        lines.append("")
        lines.append("### 目标分布（y）")
        lines.extend(target_lines if target_lines else ["- （无）"])

        lines.append("")
        lines.append("## 2. 评估指标（test 集）")
        if metrics:
            lines.append("")
            lines.append("| 指标 | 数值 |")
            lines.append("|---|---|")
            for k, v in metrics.items():
                lines.append(f"| {k} | {_fmt(v)} |")
        else:
            lines.append("- （无）")

        lines.append("")
        lines.append("## 3. 图表")
        if rel_plots:
            for label, rel in rel_plots:
                # Markdown image reference
                lines.append(f"### {label}")
                lines.append(f"![{label}]({rel})")
                lines.append("")
        else:
            lines.append("- （无可导出图表）")

        if metrics_csv_path:
            lines.append("## 4. 附件（便于复用）")
            lines.append(f"- `metrics.csv`: {os.path.basename(metrics_csv_path)}")

        report_path = os.path.join(run_dir, "report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        self.txt_log.setPlainText(
            "导出完成。\n"
            f"- 运行目录：{run_dir}\n"
            f"- 指标：{metrics_path}\n"
            f"- 指标表（CSV）：{metrics_csv_path or '（未生成）'}\n"
            f"- 报告：{report_path}\n"
        )

        self.refresh_from_state()

