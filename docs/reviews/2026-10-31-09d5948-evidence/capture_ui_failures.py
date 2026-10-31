"""Capture actual widgets at two unmodified upstream assertion failures."""
from pathlib import Path
import json
import os
import sys
import time

ROOT = Path(os.environ.get("MIMICS_REVIEW_ROOT", str(Path.cwd()))).resolve()
sys.path[:0] = [str(ROOT / "tests"), str(ROOT / "tools"), str(ROOT)]
import test_ui_workflow_regressions as tests

OUT = Path(__file__).resolve().parent
results = {}
for name in ["test_task_model_chooser_full_id_readable_and_copyable", "test_completion_long_copy_wraps_within_viewport"]:
    case = tests.TestUiWorkflow(name)
    instances = []
    original_keep = case.keep
    def keep(instance):
        instances.append(instance)
        return original_keep(instance)
    case.keep = keep
    case.setUp()
    try:
        error = ""
        try:
            getattr(case, name)()
        except AssertionError as exc:
            error = str(exc)
        widget = instances[-1]
        tests.APP.processEvents()
        widget.window.grab().save(str(OUT / (name + ".png")))
        obs = dict(assertion=error, window=[widget.window.width(), widget.window.height()],
                   font_family=widget.window.font().family(), platform=tests.APP.platformName(),
                   boundary="Unmodified test stops at failure; screenshot does not turn it into a passing regression")
        if hasattr(widget, "summary"):
            obs.update(summary_text=widget.summary.toPlainText(), viewport_width=widget.summary.viewport().width(),
                       horizontal_scroll_max=widget.summary.horizontalScrollBar().maximum(),
                       wrap_mode=str(widget.summary.wordWrapMode()))
        else:
            obs.update(detail_text=widget.progress_detail.text(), detail_height=widget.progress_detail.height(),
                       elapsed_height=widget.elapsed_label.height(), word_wrap=widget.progress_detail.wordWrap(),
                       horizontal_scroll_max=[s.horizontalScrollBar().maximum() for s in widget.window.findChildren(tests.QtWidgets.QScrollArea)])
        deadline = time.monotonic() + .6
        while time.monotonic() < deadline:
            tests.APP.processEvents()
            time.sleep(.01)
        widget.window.grab().save(str(OUT / (name + "-settled.png")))
        if hasattr(widget, "progress_detail"):
            obs["settled_detail_height"] = widget.progress_detail.height()
            obs["settled_detail_text"] = widget.progress_detail.text()
            obs["settled_detail_size_hint"] = widget.progress_detail.sizeHint().height()
        results[name] = obs
    finally:
        case.doCleanups()
(OUT / "ui-failure-observations.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(results, ensure_ascii=False, indent=2))
