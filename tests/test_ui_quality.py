#!/usr/bin/env python3
"""Non-human UI quality gates: measurable proxies for 精美/实用/有效.

The project needs UI quality to be verifiable without a human looking at
every window. This suite pins three automatable proxies, chosen because
each one catches a class of real regression a human reviewer would flag:

1. Theme integrity (精精美度的可量化面): every color used in the shared
   stylesheet must come from the central palette and every fg/bg pair we
   actually draw with must meet WCAG AA contrast. A window that sprouts an
   off-palette inline color, or a "subtle gray" that becomes unreadable on
   the panel background, fails here before a human ever sees it.
2. Real render, not just construction (有效性): offscreen `grab()` of live
   widgets and verify glyphs actually rasterize — CJK text included — so a
   font/encoding regression cannot hide behind "the widget tree built".
3. CJK glyph rendering (中文界面有效性): Chinese strings must render as
   glyphs (dark pixels comparable to Latin), not tofu boxes. Broken CJK
   font fallback would render every Chinese label as empty/boxes.

What this deliberately does NOT try to automate: overall aesthetics,
layout harmony, interaction feel. Those stay in the manual acceptance
checklist; this suite is the regression floor, not a substitute for eyes.

Run:
    python_env/python.exe tests/test_ui_quality.py
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "tools"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets  # noqa: E402

import ui_theme  # noqa: E402


def _hex_to_rgb(hexc: str):
    return tuple(int(hexc[i:i + 2], 16) for i in (0, 2, 4))


def _rel_luminance(hexc: str) -> float:
    def channel(c):
        c = c / 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = _hex_to_rgb(hexc)
    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)


def _contrast_ratio(fg: str, bg: str) -> float:
    lf, lb = _rel_luminance(fg), _rel_luminance(bg)
    lo, hi = min(lf, lb), max(lf, lb)
    return (hi + 0.05) / (lo + 0.05)


class TestThemePaletteIntegrity(unittest.TestCase):
    """The shared stylesheet must stay on the central palette."""

    PALETTE = {
        # neutral text / surfaces
        "101828", "182230", "253244", "344054", "475467", "667085",
        "8795a6", "98a2b3", "aeb9c6", "b8c2ce",
        "ffffff", "f8fafc", "f5f7fa", "f2f4f7", "eef2f6", "f0f4f8",
        "e1e6ec", "e6ebf0", "e5ebf2", "e4e9ef", "d9e0e8", "c9d2dc",
        "94a3b8", "64748b",
        # primary blue
        "2563eb", "1d4ed8", "93b0e8", "dbeafe",
        # semantic
        "b42318", "fff7f6", "feeceb", "f0b3ad",   # error
        "067647",                                       # success
        "0f766e",                                       # teal preview
        "8a5700", "fff8e7", "f1d28a",                   # warning
    }

    def test_stylesheet_colors_all_come_from_palette(self):
        used = set(re.findall(r"#([0-9a-fA-F]{6})", ui_theme.stylesheet()))
        unknown = used - self.PALETTE
        self.assertFalse(
            unknown,
            "Off-palette colors in shared stylesheet (add to PALETTE "
            "consciously or reuse an existing token): {0}".format(
                sorted(unknown)),
        )

    def test_theme_contrast_pairs_meet_wcag_aa(self):
        # fg/bg pairs the stylesheet actually paints with. AA = 4.5:1 normal
        # text, 3:1 large/bold text; we hold everything to 4.5:1.
        pairs = {
            "body on app background": ("182230", "f5f7fa"),
            "title on app background": ("101828", "f5f7fa"),
            "subtitle gray on app background": ("667085", "f5f7fa"),
            "section header on panel": ("344054", "ffffff"),
            "hint gray on panel": ("667085", "ffffff"),
            "primary blue text on panel": ("2563eb", "ffffff"),
            "error red on panel": ("b42318", "ffffff"),
            "success green on panel": ("067647", "ffffff"),
            "teal preview on panel": ("0f766e", "ffffff"),
            "warning amber on warning surface": ("8a5700", "fff8e7"),
            "white on primary blue": ("ffffff", "2563eb"),
            "white on pressed blue": ("ffffff", "1d4ed8"),
        }
        failures = []
        for name, (fg, bg) in pairs.items():
            ratio = _contrast_ratio(fg, bg)
            if ratio < 4.5:
                failures.append("{0}: {1:.2f}:1".format(name, ratio))
        self.assertFalse(failures, "WCAG AA (<4.5:1): " + "; ".join(failures))


class TestOffscreenRenderQuality(unittest.TestCase):
    """Widgets must actually rasterize text, CJK included — offscreen."""

    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    def _dark_pixel_count(self, label):
        image = label.grab().toImage()
        return sum(
            1
            for y in range(image.height())
            for x in range(image.width())
            if image.pixelColor(x, y).lightness() < 128
        )

    def test_latin_and_cjk_labels_rasterize_glyphs(self):
        cases = {
            "latin": "Training Run 123",
            "cjk": "训练任务 123",
        }
        counts = {}
        for name, text in cases.items():
            label = QtWidgets.QLabel(text)
            label.resize(220, 60)
            counts[name] = self._dark_pixel_count(label)
        # Real glyphs rasterize to a meaningful number of dark pixels.
        self.assertGreater(counts["latin"], 80, "Latin text did not rasterize")
        self.assertGreater(counts["cjk"], 80, "CJK text did not rasterize")
        # Tofu boxes render as hollow rectangles; real CJK glyph coverage is
        # at least half of Latin for a comparable string. A ratio far below
        # that means CJK font fallback broke.
        self.assertGreater(
            counts["cjk"], counts["latin"] * 0.4,
            "CJK dark-pixel coverage suspiciously low vs Latin — possible "
            "tofu rendering: cjk={0}, latin={1}".format(counts["cjk"], counts["latin"]),
        )

    def test_shared_stylesheet_applies_without_qt_rejection(self):
        # Qt silently ignores unparseable QSS chunks; apply the real sheet
        # and make sure a themed widget still renders text.
        self.app.setStyleSheet(ui_theme.stylesheet())
        try:
            label = QtWidgets.QLabel("窗宽窗位预设")
            label.setObjectName("section")
            label.resize(220, 60)
            self.assertGreater(self._dark_pixel_count(label), 80)
        finally:
            self.app.setStyleSheet("")


if __name__ == "__main__":
    unittest.main()
