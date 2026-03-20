from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Optional

from PySide6.QtWidgets import QApplication

from ml_gui.core.plot_config import configure_matplotlib_app_style


@dataclass
class SessionState:
    df: Optional[Any] = None
    X: Optional[Any] = None
    y: Optional[Any] = None
    feature_names: Optional[list[str]] = None
    target_name: Optional[str] = None
    task_type: Optional[str] = None  # classification | regression
    n_classes: Optional[int] = None
    classes: Optional[list[Any]] = None
    pipeline: Optional[Any] = None
    last_results: Optional[dict[str, Any]] = None


def main() -> None:
    # Matplotlib: publication-friendly sans-serif (figures use English labels only).
    configure_matplotlib_app_style()

    # Import GUI after configuring Matplotlib, to ensure tabs using Matplotlib pick it up.
    from ml_gui.gui.main_window import MainWindow

    app = QApplication(sys.argv)
    state = SessionState()
    window = MainWindow(state=state)
    window.show()
    sys.exit(app.exec())

