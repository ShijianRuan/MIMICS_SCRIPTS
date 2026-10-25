"""AL review contracts using real grid conversion and isolated Qt/files.

No model, clinical case, environment repair or native Mimics is executed.
"""
from __future__ import annotations

import json
import importlib
from concurrent.futures import ThreadPoolExecutor
import os
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "tools", ROOT / "runtime_py35"):
    sys.path.insert(0, str(path))

import numpy as np
import nibabel as nib
import active_learning_apply as apply
import mimics_bridge as bridge
import active_learning_common as common
import flexict_active_learning_ui as ui


class ConsensusContracts(unittest.TestCase):
    def _transaction(self, folder, backend):
        folder = Path(folder)
        source = np.zeros((3, 4, 5), dtype=np.uint8)
        source[0, :, :] = 1
        source[2, :, :] = 2
        path = folder / "consensus.nii.gz"
        nib.save(nib.Nifti1Image(source, np.eye(4)), path)
        work = folder / "conversion"
        result = bridge.do_prepare_masks_for_grid(dict(
            masks=[dict(name="al_0", mask_path=str(path))],
            buffers_out=str(work), target_shape=list(source.shape),
            target_voxel_to_ras_matrix=np.eye(4).tolist(),
            axes=[0, 1, 2], flips=[False, False, False]))
        work.mkdir(exist_ok=True)
        (work / "result.json").write_text(json.dumps(result))
        (folder / "status.json").write_text(json.dumps({"labels": ["liver", "spleen"]}))
        host = types.SimpleNamespace(_MONITORS={})
        host._read_json = lambda p, default=None: json.loads(Path(p).read_text()) if Path(p).exists() else default
        host._write_json = lambda p, v: Path(p).write_text(json.dumps(v))
        host._log = mock.Mock()
        created = []
        def new(name):
            mask = types.SimpleNamespace(name=name)
            created.append(mask)
            return mask
        def write(mask, path, shape, undo):
            mask.array = np.fromfile(path, dtype=np.uint8).reshape(shape)
        host.mimics_mask_apply = types.SimpleNamespace(_new_prediction_mask=new, _set_mask_from_u8=write)
        host.mimics = types.SimpleNamespace(data=types.SimpleNamespace(masks=types.SimpleNamespace(delete=mock.Mock())),
                                            dialogs=types.SimpleNamespace(message_box=mock.Mock()))
        apply.bind(host, backend + "_al", backend)
        host._al_target_still_matches = mock.Mock(return_value="")
        host._al_mark_viewed = mock.Mock()
        host._al_mark_request = mock.Mock()
        transaction = dict(request=dict(_request_path=str(folder / "request.json"),
                                       _job_dir=str(folder), case_id="case", what="consensus"),
                           result_path=str(work / "result.json"), bridge_root=str(work),
                           masks=[(backend + " Consensus", str(path))])
        return host, transaction, result, source, created

    def test_multiorgan_consensus_reaches_both_hosts_with_correct_names_and_voxels(self):
        for backend in ("nnU-Net", "FlexiCT"):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as folder:
                host, transaction, result, source, created = self._transaction(folder, backend)
                host._al_finish_conversion(transaction)
                self.assertEqual([m.name for m in created], [backend + " Consensus · liver", backend + " Consensus · spleen"])
                for label, mask in enumerate(created, 1):
                    np.testing.assert_array_equal(mask.array, source == label)
                self.assertEqual(host._al_mark_request.call_args.args[1], "applied")
                self.assertFalse(Path(transaction["bridge_root"]).exists())

    def test_truncated_second_organ_is_rejected_before_any_mask_creation(self):
        with tempfile.TemporaryDirectory() as folder:
            host, transaction, result, source, created = self._transaction(folder, "nnU-Net")
            Path(result["masks"][1]["output_path"]).write_bytes(b"x")
            host._al_finish_conversion(transaction)
            self.assertEqual(created, [])
            self.assertIn("truncated", host._al_mark_request.call_args.args[2])

    def test_unknown_or_duplicate_conversion_buffers_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            host, transaction, result, source, created = self._transaction(folder, "nnU-Net")
            result["masks"].append(dict(result["masks"][0]))
            self.assertIn("duplicate", host._al_verify_conversion_buffers(result, transaction["masks"])[1])
            result["masks"][-1]["name"] = "foreign_label1"
            self.assertIn("unexpected", host._al_verify_conversion_buffers(result, transaction["masks"])[1])

    def test_nonfinite_geometry_is_not_a_match(self):
        for value in (float("nan"), float("inf")):
            matrix = np.eye(4).tolist()
            matrix[0][0] = value
            self.assertFalse(apply._matrix_close_payload(matrix, matrix))

    def test_same_image_guid_with_changed_physical_grid_is_rejected(self):
        host = types.SimpleNamespace(_MONITORS={})
        apply.bind(host, "nnunet_al", "nnU-Net")
        original = dict(target_shape=[3, 4, 5], target_voxel_to_ras_matrix=np.eye(4).tolist())
        changed = np.eye(4)
        changed[0, 3] = 10
        host.mimics_mask_apply = types.SimpleNamespace(
            _current_project_path=lambda: "case.mcs", _same_path=lambda a, b: a == b,
            _mask_identity=lambda image: "same-guid", _active_live_grid_payload=lambda: dict(
                target_shape=[3, 4, 5], target_voxel_to_ras_matrix=changed.tolist()))
        host.mimics = types.SimpleNamespace(data=types.SimpleNamespace(images=types.SimpleNamespace(get_active=lambda: object())))
        transaction = dict(target_identity=dict(project_path="case.mcs", image_guid="same-guid"),
                           target_grid=original, request=dict(case_id="case"))
        self.assertIn("grid changed", host._al_target_still_matches(transaction))

    def test_zero_disagreement_is_valid_view_but_missing_nonzero_bands_are_not(self):
        geometry = dict(source_shape=[3, 4, 5], source_voxel_to_ras_matrix=np.eye(4).tolist(),
                        source_image_path="case.nii.gz")
        for maximum, expected_state in ((0, "applied"), (5, "failed")):
            with self.subTest(maximum=maximum):
                host = types.SimpleNamespace(_MONITORS={}, _read_json=lambda *args: dict(
                    ranking=[dict(case="case", max=maximum)]))
                apply.bind(host, "al", "nnU-Net")
                host._al_mask_paths = lambda *args: []
                host._al_case_source_geometry = lambda *args: geometry
                host._al_mark_request, host._al_mark_viewed = mock.Mock(), mock.Mock()
                host.mimics_mask_apply = types.SimpleNamespace(
                    _active_source_geometry_payload=lambda: geometry, _same_path=lambda a, b: a == b,
                    _active_live_grid_payload=lambda: dict(target_shape=[3, 4, 5], target_voxel_to_ras_matrix=np.eye(4).tolist()),
                    _current_project_path=lambda: "case.mcs", _mask_identity=lambda x: "image")
                host.mimics = types.SimpleNamespace(data=types.SimpleNamespace(images=types.SimpleNamespace(get_active=lambda: object())))
                host._al_apply_request(dict(_request_path="request.json", _job_dir="job", case_id="case", what="bands"))
                self.assertEqual(host._al_mark_request.call_args.args[1], expected_state)
                if maximum == 0:
                    self.assertIn("不代表", host._al_mark_request.call_args.args[2])


class BandContracts(unittest.TestCase):
    def test_file_counts_do_not_prove_same_voxels(self):
        with tempfile.TemporaryDirectory() as folder:
            values = np.array([3, 5, 0], np.uint8).reshape(3, 1, 1)
            nib.save(nib.Nifti1Image(values, np.eye(4)), Path(folder) / "case_uncertainty_disagreement.nii.gz")
            bands = common.write_uncertainty_bands(Path(folder), ["case"], "disagreement", 3, 5)
            self.assertEqual(bands["written"], dict(moderate=1, high=1))
            self.assertFalse(bands["coincident"])
            self.assertFalse(ui.bands_degenerate(dict(uncertainty_bands=bands)))
            self.assertFalse(ui.bands_degenerate(dict(uncertainty_bands=dict(
                written=dict(moderate=1, high=1), moderate_threshold=3, high_threshold=5))))

    def test_two_model_single_level_has_one_overlay(self):
        with tempfile.TemporaryDirectory() as folder:
            values = np.array([5, 5, 0], np.uint8).reshape(3, 1, 1)
            nib.save(nib.Nifti1Image(values, np.eye(4)), Path(folder) / "case_uncertainty_disagreement.nii.gz")
            bands = common.write_uncertainty_bands(Path(folder), ["case"], "disagreement", 3, 6)
            self.assertTrue(bands["coincident"])
            host = types.SimpleNamespace(_MONITORS={}, _read_json=lambda *args: dict(
                uncertainty_dir=folder, uncertainty_bands=bands))
            apply.bind(host, "al", "FlexiCT")
            self.assertEqual(len(host._al_mask_paths(folder, "case", "bands")), 1)


class PublicationContracts(unittest.TestCase):
    def test_concurrent_publish_has_one_commit_and_preserves_losing_candidate(self):
        import nnunet_pipeline as pipeline
        from nnunet_common import FINGERPRINT_ABSENT
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            target = root / "prediction.nii.gz"
            candidates = []
            for value in (1, 2):
                path = root / (str(value) + ".nii.gz")
                nib.save(nib.Nifti1Image(np.full((3, 4, 5), value, np.uint8), np.eye(4)), path)
                candidates.append(path)
            barrier = threading.Barrier(2)
            real_copy = pipeline.shutil.copy2
            def copy(src, dst, *args, **kwargs):
                result = real_copy(src, dst, *args, **kwargs)
                barrier.wait(timeout=3)
                return result
            def publish(path):
                try:
                    output, replaced = pipeline.publish_prediction_output(path, target, None, FINGERPRINT_ABSENT)
                    return "committed", output
                except pipeline.PredictionOutputConflict as exc:
                    return "conflict", exc.candidate_path
            with mock.patch.object(pipeline.shutil, "copy2", side_effect=copy), ThreadPoolExecutor(max_workers=2) as workers:
                results = list(workers.map(publish, candidates))
            self.assertCountEqual([kind for kind, path in results], ["committed", "conflict"])
            observed = [int(np.asarray(nib.load(path).dataobj)[0, 0, 0]) for kind, path in results]
            self.assertCountEqual(observed, [1, 2])

    def test_restore_holds_the_publication_guard_until_status_is_saved(self):
        import nnunet_common as contracts
        import nnunet_jobs as jobs
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            current, previous = root / "current.nii.gz", root / "old.nii.gz"
            for path, value in ((current, 2), (previous, 1)):
                nib.save(nib.Nifti1Image(np.full((3, 4, 5), value, np.uint8), np.eye(4)), path)
            status = root / "status.json"
            status.write_text(json.dumps(dict(output_path=str(current), replaced_previous_output=str(previous),
                                              published_output_fingerprint=jobs.file_fingerprint(current))))
            entered = threading.Event()
            def restore():
                entered.set()
                return jobs.restore_replaced_output(status)
            with ThreadPoolExecutor(max_workers=1) as workers:
                with contracts.prediction_output_guard(current):
                    future = workers.submit(restore)
                    self.assertTrue(entered.wait(1))
                    time.sleep(.1)
                    self.assertFalse(future.done())
                    np.testing.assert_array_equal(np.asarray(nib.load(current).dataobj), 2)
                result = future.result(timeout=3)
            np.testing.assert_array_equal(np.asarray(nib.load(current).dataobj), 1)
            self.assertEqual(result["published_output_fingerprint"], jobs.file_fingerprint(current))


class ReceiptContracts(unittest.TestCase):
    def module(self):
        spec = importlib.util.spec_from_file_location("receipt_review", ROOT / "runtime_py35/create_mcs_batch.py")
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, {"mimics": types.ModuleType("mimics")}):
            spec.loader.exec_module(module)
        return module

    def test_clock_rollback_uses_current_project_identity_and_keeps_previous_receipt(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as folder:
            project = Path(folder) / "case.mcs"
            project.write_bytes(b"old project")
            old = module._import_receipt_payload("case", str(project), ["old"], {})
            old["created_at_epoch"] = 200
            receipt = module._receipt_path_for(str(project), "case")
            module.write_json_atomic(receipt, old)
            project.write_bytes(b"new project")
            new = module._import_receipt_payload("case", str(project), ["new"], {})
            new["created_at_epoch"] = 100
            module._write_pending_receipts(folder, dict(case=dict(receipt_path=receipt, payload=new)))
            self.assertEqual(module.flush_pending_receipts(folder), ["case"])
            self.assertEqual(json.loads(Path(receipt).read_text()), new)
            history = list(Path(folder).glob(".case.previous-*.import_receipt.json"))
            self.assertEqual(len(history), 1)
            self.assertEqual(json.loads(history[0].read_text()), old)

    def test_ambiguous_transactions_remain_pending_despite_newer_timestamp(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as folder:
            project = Path(folder) / "case.mcs"
            project.write_bytes(b"same bytes cannot decide different mask ownership")
            existing = module._import_receipt_payload("case", str(project), ["old"], {})
            existing["created_at_epoch"] = 200
            queued = module._import_receipt_payload("case", str(project), ["new"], {})
            queued["created_at_epoch"] = 100
            receipt = module._receipt_path_for(str(project), "case")
            module.write_json_atomic(receipt, existing)
            module._write_pending_receipts(folder, dict(case=dict(receipt_path=receipt, payload=queued)))
            self.assertEqual(module.flush_pending_receipts(folder), [])
            self.assertEqual(json.loads(Path(receipt).read_text()), existing)
            self.assertIn("事务身份冲突", module._read_pending_receipts(folder)["case"]["last_error"])

    def test_archived_receipt_never_shadows_current_receipt_for_undo(self):
        spec = importlib.util.spec_from_file_location(
            "undo_review", ROOT / "runtime_py35/import_undo_mimics.py")
        import_undo_mimics = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, {"mimics": types.ModuleType("mimics")}):
            spec.loader.exec_module(import_undo_mimics)

        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)

            def receipt(case, created):
                return {
                    "schema_version": "mimics_import_receipt.v1",
                    "case_id": case,
                    "mcs_path": str(out / (case + ".mcs")),
                    "mcs_fingerprint": "sha256:" + case,
                    "created_masks": [],
                    "created_at_epoch": created,
                }

            # Clock rollback (S01): the archived older import carries the
            # newest created_at_epoch in the folder. Undo discovery must
            # still find the current receipt, not the archived evidence.
            (out / ".case.previous-deadbeef.import_receipt.json").write_text(
                json.dumps(receipt("old", 200.0)))
            (out / ".case.import_receipt.json").write_text(
                json.dumps(receipt("new", 100.0)))
            with mock.patch.object(import_undo_mimics, "_candidate_receipt_dirs",
                                   lambda: [str(out)]):
                path, payload = import_undo_mimics.find_latest_receipt()
            self.assertEqual(payload["case_id"], "new")
            self.assertTrue(str(path).endswith(".case.import_receipt.json"))


class SingleSliceGeometry(unittest.TestCase):
    def test_single_slice_rejects_skew_and_nonunit_iop_through_public_helpers(self):
        from pydicom.dataset import FileDataset, FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian, CTImageStorage, generate_uid
        with tempfile.TemporaryDirectory() as folder:
            r = 2 ** -0.5
            for index, iop in enumerate(([1, 0, 0, r, r, 0], [2, 0, 0, 0, 2, 0], [r, r, 0, -r, r, 0])):
                case = Path(folder) / str(index)
                case.mkdir()
                meta = FileMetaDataset()
                meta.TransferSyntaxUID = ExplicitVRLittleEndian
                meta.MediaStorageSOPClassUID = CTImageStorage
                meta.MediaStorageSOPInstanceUID = generate_uid()
                ds = FileDataset(str(case / "one.dcm"), {}, file_meta=meta, preamble=b"\0" * 128)
                ds.SOPClassUID = CTImageStorage
                ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
                ds.SeriesInstanceUID = generate_uid()
                ds.Rows = ds.Columns = 4
                ds.PixelSpacing = [.5, .75]
                ds.SliceThickness = 1
                ds.ImagePositionPatient = [0, 0, 0]
                ds.ImageOrientationPatient = iop
                ds.save_as(ds.filename, enforce_file_format=True)
                for helper in (bridge.get_image_shape_from_dicom, bridge.get_image_affine_from_dicom):
                    if index < 2:
                        with self.assertRaises(ValueError):
                            helper(str(case))
                    else:
                        self.assertTrue(np.isfinite(np.asarray(helper(str(case)))).all())


class QtContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6 import QtCore, QtGui, QtWidgets
        cls.qt = (QtCore, QtGui, QtWidgets)
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        cls.app.setQuitOnLastWindowClosed(False)

    def pump(self, predicate, timeout=3):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.app.processEvents()
            if predicate():
                return True
            time.sleep(.01)
        return False

    def window(self, folder, backend):
        root = Path(folder)
        job = root / "jobs" / "al1"
        job.mkdir(parents=True)
        status = dict(kind="active_learning", status="completed", job_id="al1", job_dir=str(job),
                      created_at_epoch=1, ranking=[dict(case=c, integrated=30-i, uncertain_vol=3)
                                                  for i, c in enumerate(("caseA", "caseB", "caseC"))])
        (job / "status.json").write_text(json.dumps(status))
        (job / "input_geometries.json").write_text(json.dumps(dict(cases={
            c: dict(case_name=name, source_image_path=str(root / name / "ct.nii.gz"))
            for c, name in zip(("caseA", "caseB", "caseC"), ("患者甲_动脉期", "患者乙", "患者丙"))})))
        with mock.patch.object(ui.ActiveLearningWindow, "_start_pair_loading"), mock.patch.object(ui.ActiveLearningWindow, "_start_model_loading"):
            w = ui.ActiveLearningWindow(dict(workspace=str(root), mode=backend), self.qt)
        w._loading_pairs = w._loading_models = False
        w.window.show()
        self.assertTrue(self.pump(lambda: w.table.rowCount() == 3))
        return w

    def test_filter_keeps_internal_case_identity_and_next_skips_completed(self):
        for backend in ("nnunet", "flexict"):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as folder:
                w = self.window(folder, backend)
                try:
                    ui.set_case_state(w._job_dir, "caseB", "annotated")
                    w._job_selected()
                    w.search_edit.setText("动脉")
                    self.assertEqual(w.table.rowCount(), 1)
                    w.table.selectRow(0)
                    self.assertEqual(w._selected_case(), "caseA")
                    self.assertIn("caseA", w.table.item(0, 1).toolTip())
                    w.search_edit.clear()
                    self.assertEqual(w._selected_case(), "caseA")
                    w._next_pending()
                    self.assertEqual(w._selected_case(), "caseC")
                    w.pending_only.setChecked(True)
                    self.assertEqual(w.table.rowCount(), 2)
                    self.assertEqual(w._selected_case(), "caseC")
                finally:
                    w.window.close()

    def test_failed_marking_close_keeps_window_and_retry_persists(self):
        with tempfile.TemporaryDirectory() as folder:
            w = self.window(folder, "nnunet")
            entered, release = threading.Event(), threading.Event()
            real_write = ui.set_case_state
            calls = []
            def write(*args):
                calls.append(args)
                if len(calls) == 1:
                    entered.set()
                    release.wait(3)
                    raise OSError("synthetic storage denial")
                return real_write(*args)
            try:
                with mock.patch.object(ui, "set_case_state", side_effect=write):
                    w.table.selectRow(0)
                    w._mark_annotated()
                    self.assertTrue(entered.wait(1))
                    self.assertFalse(w.window.close())
                    self.assertTrue(w.window.isVisible())
                    release.set()
                    self.assertTrue(self.pump(lambda: bool(w._state_write_failures)))
                    w._poll_background_results()
                    self.assertTrue(w.window.isVisible())
                    self.assertEqual(w.wait_for_state_writes(.1), 1)
                    self.assertNotEqual((ui.load_annotation_state(w._job_dir).get("cases", {}).get("caseA") or {}).get("state"), "annotated")
                    w._retry_failed_writes()
                    self.assertTrue(self.pump(lambda: w.wait_for_state_writes(.01) == 0))
                    w._poll_background_results()
                    self.assertEqual(ui.load_annotation_state(w._job_dir)["cases"]["caseA"]["state"], "annotated")
            finally:
                release.set()
                with mock.patch.object(w.QtWidgets.QMessageBox, "warning", return_value=w.QtWidgets.QMessageBox.Discard):
                    w.window.close()

    def test_close_during_actual_restore_waits_for_completed_file_exchange(self):
        import nnunet_jobs as jobs
        import shiboken6
        for module_name in ("nnunet_status_viewer", "flexict_status_viewer"):
            with self.subTest(backend=module_name), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                output, previous = root / "current.nii.gz", root / "previous.nii.gz"
                for path, value in ((output, 2), (previous, 1)):
                    nib.save(nib.Nifti1Image(np.full((3, 4, 5), value, np.uint8), np.eye(4)), path)
                status = root / "status.json"
                status.write_text(json.dumps(dict(output_path=str(output), replaced_previous_output=str(previous),
                                                  published_output_fingerprint=jobs.file_fingerprint(output))))
                module = importlib.import_module(module_name)
                w = module.StatusWindow(dict(workspace=str(root / "workspace")), self.qt)
                w.refresher.stop()
                w.refresher = mock.Mock()
                w.window.show()
                entered, release = threading.Event(), threading.Event()
                real_replace = os.replace
                def replace(src, dst):
                    result = real_replace(src, dst)
                    if str(src) == str(output):
                        entered.set()
                        release.wait(3)
                    return result
                try:
                    with mock.patch.object(jobs.os, "replace", side_effect=replace):
                        w._restore_previous(dict(status_path=str(status)))
                        self.assertTrue(entered.wait(1))
                        self.assertFalse(output.exists())
                        self.assertFalse(w.window.close())
                        self.assertTrue(w.window.isVisible())
                        self.assertTrue(w._restore_timer.isActive())
                        release.set()
                        self.assertTrue(self.pump(lambda: not w._restore_busy))
                    self.assertTrue(output.exists())
                    np.testing.assert_array_equal(np.asarray(nib.load(output).dataobj), 1)
                    result = json.loads(status.read_text())
                    np.testing.assert_array_equal(np.asarray(nib.load(result["replaced_previous_output"]).dataobj), 2)
                    self.assertEqual(result["published_output_fingerprint"], jobs.file_fingerprint(output))
                finally:
                    release.set()
                    if shiboken6.isValid(w.window):
                        w.window.close()


if __name__ == "__main__":
    unittest.main()
