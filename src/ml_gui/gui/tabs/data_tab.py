from __future__ import annotations

from typing import Any, Optional

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QMessageBox,
)

from ml_gui.core.data_loader import load_dataset, normalize_columns_if_no_header, split_features_target
from ml_gui.core.task_inference import infer_task_type


class DataTab(QWidget):
    def __init__(self, state: Any):
        super().__init__()
        self.state = state
        self.on_data_loaded = None  # callback
        self._df: Optional[pd.DataFrame] = None

        layout = QVBoxLayout(self)

        # Load controls
        controls_row = QHBoxLayout()
        self.btn_load = QPushButton("导入数据文件")
        self.btn_load.clicked.connect(self._on_load_clicked)
        controls_row.addWidget(self.btn_load)

        self.lbl_path = QLabel("未选择文件")
        self.lbl_path.setTextInteractionFlags(Qt.TextSelectableByMouse)
        controls_row.addWidget(self.lbl_path, 1)
        layout.addLayout(controls_row)

        form = QFormLayout()
        self.chk_header = QCheckBox("CSV/Excel 存在表头")
        self.chk_header.setChecked(True)
        form.addRow("表头(header)", self.chk_header)

        self.cmb_delimiter = QComboBox()
        self.cmb_delimiter.addItems([",", ";", "\\t", "|"])
        self.cmb_delimiter.setCurrentIndex(0)
        form.addRow("CSV 分隔符", self.cmb_delimiter)

        layout.addLayout(form)

        # Preview table
        layout.addWidget(QLabel("数据预览（前 50 行）"))
        self.table_preview = QTableWidget()
        self.table_preview.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table_preview.setColumnCount(0)
        self.table_preview.setRowCount(0)
        layout.addWidget(self.table_preview, 1)

        # Target/task selection
        task_box = QWidget()
        task_layout = QVBoxLayout(task_box)
        task_layout.setContentsMargins(10, 10, 10, 10)

        self.cmb_target = QComboBox()
        self.cmb_task_override = QComboBox()
        self.cmb_task_override.addItems(["Auto", "Classification", "Regression"])

        form2 = QFormLayout()
        form2.addRow("目标列 y", self.cmb_target)
        form2.addRow("任务类型", self.cmb_task_override)
        task_layout.addLayout(form2)

        self.btn_apply = QPushButton("识别任务并准备数据")
        self.btn_apply.clicked.connect(self._apply_target_and_infer)
        task_layout.addWidget(self.btn_apply)

        self.txt_info = QTextEdit()
        self.txt_info.setReadOnly(True)
        self.txt_info.setMinimumHeight(120)
        task_layout.addWidget(self.txt_info)

        layout.addWidget(task_box, 0)

        self.cmb_target.currentIndexChanged.connect(self._on_target_changed)
        self.cmb_task_override.currentIndexChanged.connect(self._on_target_changed)

    def _on_target_changed(self) -> None:
        # In v1, we keep it simple: we only refresh info on Apply.
        pass

    def _populate_preview(self, df: pd.DataFrame, max_rows: int = 50, max_cols: int = 30) -> None:
        df_show = df.iloc[:max_rows, :max_cols]
        self.table_preview.clear()
        self.table_preview.setRowCount(df_show.shape[0])
        self.table_preview.setColumnCount(df_show.shape[1])

        self.table_preview.setHorizontalHeaderLabels([str(c) for c in df_show.columns])

        for r in range(df_show.shape[0]):
            for c in range(df_show.shape[1]):
                val = df_show.iat[r, c]
                item = QTableWidgetItem("" if pd.isna(val) else str(val))
                item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                self.table_preview.setItem(r, c, item)

    def _on_load_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择数据文件",
            "",
            "数据文件 (*.csv *.xlsx *.xls);;所有文件 (*)",
        )
        if not path:
            return

        try:
            has_header = self.chk_header.isChecked()
            delimiter_map = {",": ",", ";": ";", "\\t": "\t", "|": "|"}
            delim = delimiter_map.get(self.cmb_delimiter.currentText(), ",")

            df = load_dataset(path, has_header=has_header, csv_delimiter=delim)
            df = normalize_columns_if_no_header(df)
            self._df = df

            self.lbl_path.setText(path)
            self._populate_preview(df)

            self.cmb_target.clear()
            self.cmb_target.addItems([str(c) for c in df.columns])
            self.cmb_target.setCurrentIndex(max(0, df.shape[1] - 1))  # default last column

            self.txt_info.setPlainText(f"已加载：{path}\n行数：{df.shape[0]}，列数：{df.shape[1]}")

        except Exception as e:
            QMessageBox.critical(self, "读取失败", str(e))

    def _apply_target_and_infer(self) -> None:
        if self._df is None:
            QMessageBox.warning(self, "提示", "请先导入数据。")
            return

        target_col = self.cmb_target.currentText()
        if target_col not in self._df.columns:
            QMessageBox.critical(self, "错误", f"目标列不存在：{target_col}")
            return

        task_override = self.cmb_task_override.currentText()
        inferred = infer_task_type(self._df[target_col])

        # Prepare X/y by dropping rows with missing y
        X, y = split_features_target(self._df, target_col=target_col)

        # Choose final task type
        final_task: Optional[str]
        n_classes: Optional[int] = None
        classes: Optional[list[Any]] = None

        if task_override == "Auto":
            final_task = inferred.get("task_type")
            n_classes = inferred.get("n_classes")
            classes = inferred.get("classes")
        elif task_override == "Classification":
            final_task = "classification"
            n_classes = inferred.get("n_classes")
            classes = inferred.get("classes")
        else:
            final_task = "regression"
            n_classes = None
            classes = None

        # Update session state
        self.state.df = self._df
        self.state.X = X
        self.state.y = y
        self.state.feature_names = list(X.columns)
        self.state.target_name = target_col
        self.state.task_type = final_task
        self.state.n_classes = n_classes
        self.state.classes = classes
        self.state.pipeline = None
        self.state.last_results = None

        info = []
        info.append(f"目标列：{target_col}")
        info.append(f"已丢弃缺失 y 的行：{len(self._df) - len(y)}")
        info.append(f"推断结果（Auto）：{inferred.get('task_type')}")
        info.append(f"使用任务类型：{final_task}")
        if final_task == "classification":
            if classes is not None:
                info.append(f"类别数：{len(classes)}")
            else:
                info.append("类别数：未知（或推断失败）")
        self.txt_info.setPlainText("\n".join(info))

        if self.on_data_loaded is not None:
            self.on_data_loaded()

