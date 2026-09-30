"""Read-only audit probes. No Mimics, GPU, real patient data, or pip mutations.

Run with numpy/pydicom/PySide6 installed. AST probes execute the repository's
actual function bodies with boundary dependencies replaced by controlled stubs.
These demonstrate control/data flow, not Windows/Mimics end-to-end behavior.
"""
import ast
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import types
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
for path in (ROOT, ROOT / "tools", ROOT / "runtime_py35"):
    sys.path.insert(0, str(path))


def extract(relative, names, namespace):
    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ROOT / relative), "exec"), namespace)


results = {}
import setup_env as setup
seen = []
check = {"packages": {"yaml": False, "onnxruntime": False},
         "gui_backends": {"pyside6": True}, "nnunet_version_compatible": True}
with mock.patch.object(setup, "check_result_dict", return_value=check), \
     mock.patch.object(setup, "_write_state"), mock.patch.object(setup, "_log"), \
     mock.patch.object(setup, "_pip_install", side_effect=lambda p: (seen.extend(p) or False, "probe")):
    setup.install()
results["repair_pip_arguments"] = seen


def fake_script(lines, **kwargs):
    source = "\n".join(lines)
    if "find_spec" in source:
        return 0, json.dumps({p: True for p in setup.REQUIRED_IMPORTS})
    if "__version__" in source:
        return 0, json.dumps({p: "error" if p == "torch" else "1" for p in setup.REQUIRED_IMPORTS})
    return 0, json.dumps({"is_available": False, "probe_error": "DLL load failed"})


states = []
with mock.patch.object(setup, "_run_python", return_value=(0, "3.13.7")), \
     mock.patch.object(setup, "_run_python_script", side_effect=fake_script), \
     mock.patch.object(setup, "_probe_gui_backends", return_value={"pyside6": True}), \
     mock.patch.object(setup, "_nnunet_version_supported", return_value=(True, "2.8.1")), \
     mock.patch.object(setup, "_find_python", return_value="test/python.exe"), \
     mock.patch.object(setup, "_log"), \
     mock.patch.object(setup, "_write_state", side_effect=lambda s, **kw: states.append({"status": s, **kw})):
    setup.check()
results["broken_torch_health"] = {
    "torch_import": states[-1]["detail"]["package_versions"]["torch"],
    "status": states[-1]["status"], "all_ok": states[-1]["all_ok"],
}

from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid, CTImageStorage
import mimics_bridge


def dicom_header(path, series, z, frames=None):
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.SeriesInstanceUID = series
    ds.Rows = ds.Columns = 4
    ds.PixelSpacing = [1, 1]
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.ImagePositionPatient = [0, 0, z]
    ds.SliceThickness = 10
    ds.Modality = "CT"
    if frames:
        ds.NumberOfFrames = frames
    ds.save_as(str(path), enforce_file_format=True)


with tempfile.TemporaryDirectory() as raw:
    root = Path(raw)
    for kind in ("mixed_series", "multi_frame"):
        folder = root / kind
        folder.mkdir()
        if kind == "mixed_series":
            for s in range(2):
                uid = generate_uid()
                for z in range(3):
                    dicom_header(folder / f"{s}_{z}.dcm", uid, z * 10)
        else:
            dicom_header(folder / "frames.dcm", generate_uid(), 0, frames=60)
        prepared = mimics_bridge.do_prepare({"image_path": str(folder), "buffers_out": str(root / (kind + "_out"))})
        results[kind] = {k: prepared.get(k) for k in ("status", "source_image_shape")}

# Execute the production callback path, but never kill any real process.
event = threading.Event()
threads = []
reaper = {"threading": threading, "time": time, "os": os, "process_exists": lambda pid: False}
extract("runtime_py35/runtime_common.py", {"terminate_process_async"}, reaper)
fake_runtime = types.SimpleNamespace(terminate_process_async=reaper["terminate_process_async"],
                                     release_local_operation=lambda *a: None)


def message(*args, **kwargs):
    threads.append(threading.current_thread().name)
    event.set()


namespace = {"runtime_common": fake_runtime, "_stop_mask_import_monitor": lambda *a: None,
             "_cleanup_work_dir": lambda *a: None, "_safe_message": message}
extract("runtime_py35/mask_import.py", {"_finish_mask_import", "_finish_mask_import_after_process"}, namespace)
process = types.SimpleNamespace(pid=99999999, poll=lambda: None,
                                terminate=lambda: None, wait=lambda **k: None)
namespace["_finish_mask_import_after_process"]({"process": process}, "simulated timeout")
event.wait(3)
results["mask_import_finish_message_threads"] = threads

# Network construction requires a separate backbone before trained weights load.
def missing_checkpoint(path):
    results["flexict_inference_backbone_dependency"] = str(path)
    raise FileNotFoundError(str(path))


namespace = {"os": os, "__file__": str(ROOT / "integrations/flexict-finetune/trainers/flexict_trainer.py"),
             "_build_backbone": lambda: object()}
extract("integrations/flexict-finetune/trainers/flexict_trainer.py", {"_load_flexict2d"}, namespace)
with mock.patch.dict(sys.modules, {
    "flexict.flexict_primus": types.SimpleNamespace(FlexiCTPrimus2D=object),
    "safetensors.torch": types.SimpleNamespace(load_file=missing_checkpoint),
}):
    try:
        namespace["_load_flexict2d"]()
    except FileNotFoundError:
        pass

# Real Qt widgets, synthetic registry rows, no background scans or file writes.
os.environ["QT_QPA_PLATFORM"] = "offscreen"
from PySide6 import QtCore, QtGui, QtWidgets
import model_manager_ui as manager
import flexict_active_learning_ui as al
from ui_theme import configure_application

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
configure_application(app, "Audit")
with mock.patch.object(manager.ModelManagerWindow, "_reload"):
    window = manager.ModelManagerWindow({"workspaces": {"nninteractive": "/tmp/audit-no-model"}}, (QtCore, QtGui, QtWidgets))
    window.rows = [dict(family="nnU-Net", target="Whole-body organs", task_id="", model_id="whole_body_v1",
                       configuration="3d_fullres", created_at_epoch=1790640000, imported=True, current=False, usable=True)]
    window._fill_table()
    window.table.selectRow(0)
    results["model_manager_selection"] = {"row": window.table.currentRow(), "use_enabled": window.use_button.isEnabled(),
                                           "remove_enabled": window.remove_button.isEnabled()}
    window.window.close()
with mock.patch.object(al, "find_al_jobs", return_value=[]):
    window = al.ActiveLearningWindow({}, (QtCore, QtGui, QtWidgets))
    results["al_empty_primary_enabled"] = window.open_button.isEnabled()
    with mock.patch.object(al, "load_ranking", return_value=[{"case": "synthetic01", "integrated": 1, "uncertain_vol": 10}]), \
         mock.patch.object(al, "load_annotation_state", return_value={"cases": {}}):
        window.job_combo.blockSignals(True)
        window.job_combo.addItem("audit")
        window._jobs = [{"job_dir": "/tmp/audit-nonexistent"}]
        try:
            window._job_selected()
        except Exception as exc:
            results["al_nonempty_table"] = type(exc).__name__ + ": " + str(exc)
    window.window.close()
print(json.dumps(results, indent=2, ensure_ascii=False))
