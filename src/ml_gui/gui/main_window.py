from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ml_gui.gui.tabs.data_tab import DataTab
from ml_gui.gui.tabs.model_tab import ModelTab
from ml_gui.gui.tabs.eda_tab import EdaTab
from ml_gui.gui.tabs.explain_tab import ExplainTab
from ml_gui.gui.tabs.report_tab import ReportTab


class MainWindow(QMainWindow):
    def __init__(self, state: Any):
        super().__init__()
        self.state = state
        self.setWindowTitle("ML 科研数据分析助手（V1）")
        self.setMinimumSize(1100, 750)

        self.tabs = QTabWidget()
        self.data_tab = DataTab(state=self.state)
        self.model_tab = ModelTab(state=self.state)
        self.eda_tab = EdaTab(state=self.state)
        self.explain_tab = ExplainTab(state=self.state)
        self.report_tab = ReportTab(state=self.state)

        self.tabs.addTab(self.data_tab, "Data")
        self.tabs.addTab(self.model_tab, "Model")
        self.tabs.addTab(self.eda_tab, "EDA")
        self.tabs.addTab(self.explain_tab, "Explain")
        self.tabs.addTab(self.report_tab, "Report")

        # Global status/quick actions
        status_row = QWidget()
        status_layout = QHBoxLayout(status_row)
        status_layout.setContentsMargins(12, 8, 12, 8)
        self.task_label = QLabel("未加载数据")
        self.task_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        status_layout.addWidget(self.task_label, 1)

        self.btn_refresh = QPushButton("刷新任务类型")
        self.btn_refresh.clicked.connect(self._sync_task_label)
        status_layout.addWidget(self.btn_refresh, 0)

        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.addWidget(self.tabs, 1)
        root_layout.addWidget(status_row, 0)
        self.setCentralWidget(root)

        # Hook: after successful load, update model tab UI
        self.data_tab.on_data_loaded = self._on_data_loaded
        self.model_tab.on_trained = self._on_trained
        self._sync_task_label()

    def _on_trained(self) -> None:
        self.explain_tab.refresh_from_state()
        self.report_tab.refresh_from_state()

    def _sync_task_label(self) -> None:
        if self.state.task_type is None:
            self.task_label.setText("未加载数据")
            return

        if self.state.task_type == "classification":
            if self.state.n_classes is None:
                extra = ""
            elif self.state.n_classes == 2:
                extra = "（二分类）"
            else:
                extra = f"（多分类：{self.state.n_classes} 类）"
        else:
            extra = "（回归）"

        target = self.state.target_name or "y"
        self.task_label.setText(f"任务类型：{self.state.task_type} {extra}，目标：{target}")

    def _on_data_loaded(self) -> None:
        self._sync_task_label()
        self.model_tab.refresh_from_state()
        self.eda_tab.refresh_from_state()
        self.explain_tab.refresh_from_state()
        self.report_tab.refresh_from_state()

    def closeEvent(self, event) -> None:
        if self.state.pipeline is not None and self.state.last_results:
            QMessageBox.information(
                self,
                "提示",
                "你已完成一次训练。可在 Model 页继续查看指标与图表。",
            )
        event.accept()

