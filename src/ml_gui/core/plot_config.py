from __future__ import annotations

"""
Matplotlib global style for figures that may be exported to SCI papers:
- Latin-only axis/title text in plotting code (see tabs)
- Sans-serif fonts that render reliably in PDF/PNG
"""


def configure_matplotlib_app_style() -> None:
    """Call once at app startup (before importing tabs that use matplotlib)."""
    try:
        from matplotlib import rcParams

        # Publication-friendly: no reliance on CJK fonts in exported figures
        rcParams["font.family"] = "sans-serif"
        rcParams["font.sans-serif"] = [
            "DejaVu Sans",
            "Arial",
            "Helvetica",
            "Liberation Sans",
        ]
        rcParams["axes.unicode_minus"] = False
        rcParams["font.size"] = 10
        rcParams["axes.titlesize"] = 11
        rcParams["axes.labelsize"] = 10
        rcParams["xtick.labelsize"] = 9
        rcParams["ytick.labelsize"] = 9
        rcParams["figure.dpi"] = 100
        rcParams["savefig.dpi"] = 300
        rcParams["savefig.bbox"] = "tight"
    except Exception:
        return
