"""Render real Qt widgets with synthetic job/registry data; no model runs."""
from pathlib import Path
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'tests'), str(ROOT / 'tools'), str(ROOT)]
import test_ui_workflow_regressions as tests

OUT = Path(__file__).resolve().parent
observations = {}
for name in ('test_completion_long_copy_wraps_within_viewport',
             'test_task_model_chooser_full_id_readable_and_copyable'):
    case = tests.TestUiWorkflow(name)
    instances = []
    keep = case.keep
    def capture(widget):
        instances.append(widget)
        return keep(widget)
    case.keep = capture
    case.setUp()
    try:
        getattr(case, name)()
        widget = instances[-1]
        deadline = time.monotonic() + .6
        while time.monotonic() < deadline:
            tests.APP.processEvents()
            time.sleep(.01)
        widget.window.grab().save(str(OUT / (name + '.png')))
        row = dict(assertions='passed', window=[widget.window.width(), widget.window.height()],
                   platform=tests.APP.platformName(),
                   scope='Real macOS PySide6 widgets; synthetic status, registry and dependency stubs')
        if hasattr(widget, 'progress_detail'):
            label = widget.progress_detail
            row.update(height=label.height(), required_height=label.heightForWidth(label.width()),
                       width=label.width(), minimum_height=label.minimumHeight(), word_wrap=label.wordWrap())
        else:
            row.update(horizontal_scroll_max=widget.summary.horizontalScrollBar().maximum(),
                       viewport_width=widget.summary.viewport().width(),
                       text_selectable=bool(widget.summary.textInteractionFlags() & tests.QtCore.Qt.TextSelectableByKeyboard))
        observations[name] = row
    finally:
        case.doCleanups()
(OUT / 'qt-observations.json').write_text(json.dumps(observations, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(observations, ensure_ascii=False, indent=2))
