import os
import sys


def _ensure_src_on_path() -> None:
    """Allow running without installing the package."""
    repo_root = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(repo_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def main() -> None:
    _ensure_src_on_path()
    from ml_gui.app import main as gui_main

    gui_main()


if __name__ == "__main__":
    main()

