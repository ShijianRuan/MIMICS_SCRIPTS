#!/usr/bin/env python3
r"""Render and validate the real PySide6 training UI on Windows.

Run from the project root with the packaged environment:
  nninteractive_env\python.exe tools\validate_fewshot_ui_windows.py --scale 1.25
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=float, default=1.0, choices=(1.0, 1.25, 1.5))
    parser.add_argument("--tab", choices=("data", "model"), default="data")
    parser.add_argument("--output")
    args = parser.parse_args(argv)

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ["QT_SCALE_FACTOR"] = str(args.scale)
    from PySide6 import QtCore, QtGui, QtWidgets
    from tools.fewshot_training_setup_ui import QtTrainingSetupApp
    from tools.ui_theme import configure_application

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    configure_application(app, "DINOv3 Few-Shot Training")
    temp = Path(tempfile.mkdtemp(prefix="mimics_fewshot_ui_"))
    dataset = temp / "dataset"
    dataset.mkdir()
    context = {
        "organ": "liver",
        "ts_root": str(dataset),
        "mcs_output_dir": str(dataset / "mcs_output"),
        "workspace": str(dataset / "fewshot_models"),
        "project_root": str(ROOT),
        "dinov3_root": str(ROOT / "external" / "dinov3-medical-seg"),
        "python_exe": sys.executable,
        "pipeline_script": str(ROOT / "tools" / "fewshot_pipeline.py"),
        "case_ids": ["case_{:03d}".format(index) for index in range(24)],
        "config": json.loads((ROOT / "fewshot_config.json").read_text(encoding="utf-8")),
    }
    window = QtWidgets.QMainWindow()
    ui = QtTrainingSetupApp(window, context, (QtCore, QtGui, QtWidgets))
    ui.tabs.setCurrentIndex(1 if args.tab == "model" else 0)
    window.show()
    app.processEvents()

    errors = []
    for name, widget in (("Start Training", ui.start_button), ("Cancel", ui.close_button)):
        if widget is None or not widget.isVisible() or widget.width() < widget.sizeHint().width():
            errors.append("{} button is hidden or clipped".format(name))
    central = window.centralWidget().rect()
    for widget in (ui.start_button, ui.close_button, ui.status_label):
        top_left = widget.mapTo(window.centralWidget(), QtCore.QPoint(0, 0))
        rect = QtCore.QRect(top_left, widget.size())
        if not central.intersects(rect):
            errors.append("{} is outside the visible central area".format(widget.objectName() or widget.__class__.__name__))
    if args.tab == "model":
        model_selector = ui.widgets.get("model_selection")
        if model_selector is None or model_selector.count() != len(ui.model_records):
            errors.append("pretrained weights selector does not match the fixed model repository")
        if "model_path" in ui.widgets:
            errors.append("arbitrary custom model path is still exposed in the setup UI")

    output = Path(args.output) if args.output else ROOT / "docs" / "images" / "fewshot_windows_{}_scale_{}.png".format(args.tab, str(args.scale).replace(".", "_"))
    output.parent.mkdir(parents=True, exist_ok=True)
    if not window.grab().save(str(output)):
        errors.append("could not save screenshot")
    print(json.dumps({"scale": args.scale, "screenshot": str(output), "errors": errors}, indent=2))
    window.close()
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
