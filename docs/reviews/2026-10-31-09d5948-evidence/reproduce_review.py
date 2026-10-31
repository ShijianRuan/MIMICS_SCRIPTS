"""Synthetic counterexamples against reviewed source; no models or process kills.

Assertions certify the observed defect, not a passing safety contract.
"""
from pathlib import Path
import contextlib
import copy
import hashlib
import importlib
import json
import os
import subprocess
import sys
import tempfile
from unittest import mock

ROOT = Path(os.environ.get("MIMICS_REVIEW_ROOT", str(Path.cwd()))).resolve()
SOURCE = "09d5948988c4efd481ddb15d162551c644aebaef"
sys.path[:0] = [str(ROOT / "tests"), str(ROOT / "runtime_py35"), str(ROOT / "tools"), str(ROOT)]
import fake_mimics_flow_test as flow
fake = flow.install_fake_mimics()
runtime = importlib.import_module("nninteractive_mimics")
observations = {}


def worker(root, image, name="image_worker_fixture", status="ready", created=1, **extra):
    folder = root / "async_jobs" / name
    for child in ["commands", "results", "inputs"]:
        (folder / child).mkdir(parents=True, exist_ok=True)
    record = dict(worker_dir=str(folder), pid=os.getpid(), cache_key=runtime._object_id(image),
                  image_guid=runtime._object_id(image), image_name=image.name,
                  model_identity="official", created_at_epoch=created,
                  shape=[2, 2, 2], image_source="mimics_buffer", python=sys.executable,
                  python_version=[3, 13], folds=["0"], runtime_log=str(root / "runtime.log"),
                  worker_log=str(folder / "worker.log"))
    record.update(extra)
    (folder / "image_worker.json").write_text(json.dumps(record))
    (folder / "worker_status.json").write_text(json.dumps(dict(status=status, pid=os.getpid())))
    (folder / "initialize.json").write_text(json.dumps(dict(parent_pid=987654321)))
    return folder, record


def main():
    files = ["runtime_py35/nninteractive_mimics.py", "tools/nnunet_pipeline.py", "nninteractive_bridge.py", "runtime_py35/import_undo_mimics.py"]
    for file in files:
        pinned = subprocess.check_output(["git", "show", SOURCE + ":" + file], cwd=ROOT)
        assert pinned == (ROOT / file).read_bytes(), "Source changed; preserve old evidence: " + file
    with tempfile.TemporaryDirectory(prefix="mimics_review_synthetic_") as temporary:
        base = Path(temporary)
        with mock.patch.dict(os.environ, {"MIMICS_RESOURCE_LOCK_DIR": str(base / "locks")}):
            image = fake.reset_scene(image_shape=(2, 2, 2))
            root = base / "image_identity"
            folder, record = worker(root, image)
            with mock.patch.dict(os.environ, {"NNINTERACTIVE_RUNTIME_DIR": str(root)}):
                runtime._ASYNC_IMAGE_WORKERS.clear()
                assert runtime._adopt_disk_image_workers({}) == 1
                with mock.patch.object(runtime, "_runtime_paths", side_effect=AssertionError("must not spawn real worker")):
                    normal = runtime._get_or_start_image_worker({}, image, str(root / "async_jobs"))
                    assert normal["worker_dir"] == str(folder)
                    # Same GUID, shape unchanged; live physical grid changed.
                    image.get_voxel_center = lambda index: [-float(index[0]) + 100.0, -float(index[1]), float(index[2])]
                    image.minimum_value = -700
                    stale = runtime._get_or_start_image_worker({}, image, str(root / "async_jobs"))
                assert stale is normal
                observations["same_guid_changed_grid"] = dict(
                    outcome="reproduced", same_worker_reused=True, current_voxel_origin_lps=image.get_voxel_center([0, 0, 0]),
                    runtime_paths_or_new_export_called=False,
                    boundary="actual disk adoption/cache consumer, synthetic image; no prediction or accuracy measurement")
            runtime._ASYNC_IMAGE_WORKERS.clear()

            # Entry point checks empty in-memory registries before adoption.
            current = fake.reset_scene(image_shape=(2, 2, 2))
            other = flow.FakeImage(current.get_voxel_buffer(), name="other image")
            root = base / "busy_order"
            folder, record = worker(root, other, status="running")
            target = flow.FakeMask("target", image=current)
            attempts = []
            def prewarm(config, active):
                attempts.append(dict(busy_after_adoption=runtime._same_model_busy_workers(config, active) is not None))
                return None
            runtime._ASYNC_IMAGE_WORKERS.clear()
            runtime._ASYNC_MONITORS.clear()
            with contextlib.ExitStack() as stack:
                stack.enter_context(mock.patch.dict(os.environ, {"NNINTERACTIVE_RUNTIME_DIR": str(root)}))
                stack.enter_context(mock.patch.object(runtime.runtime_common, "active_local_operation", return_value=None))
                stack.enter_context(mock.patch.object(runtime.runtime_common, "active_resource_lock", return_value=None))
                stack.enter_context(mock.patch.object(runtime, "_select_session_masks", return_value=dict(source=target, target=target, write_mode="in_place")))
                stack.enter_context(mock.patch.object(runtime, "_async_prompt_menu", return_value=runtime.BUTTON_POINT))
                stack.enter_context(mock.patch.object(runtime, "_prewarm_async_image_worker", side_effect=prewarm))
                stack.enter_context(mock.patch.object(runtime, "_capture_prompt", return_value=None))
                stack.enter_context(mock.patch.object(runtime.runtime_common, "terminate_process_async", side_effect=AssertionError("must not terminate a process")))
                code = runtime._run_with_config({})
            assert code == 0 and attempts == [dict(busy_after_adoption=True)]
            assert not (folder / "close.json").exists()
            observations["busy_guard_before_adoption"] = dict(outcome="reproduced", entry_return=code,
                prewarm_attempted_with_other_image_running=True, attempts=attempts,
                boundary="actual runtime entry and guards; prewarm intercepted before model/server access; no actual server restart")
            runtime._ASYNC_IMAGE_WORKERS.clear()

            # Deliberately point a stale record at this diagnostic process.
            # PID existence is real; termination is intercepted, never sent.
            image = fake.reset_scene(image_shape=(2, 2, 2))
            root = base / "unowned_pid"
            folder, record = worker(root, image, image_guid="old-image", cache_key="old-image")
            terminates = []
            with mock.patch.dict(os.environ, {"NNINTERACTIVE_RUNTIME_DIR": str(root)}), \
                 mock.patch.object(runtime.runtime_common, "terminate_process_async", side_effect=lambda **kw: terminates.append(kw)):
                count = runtime._adopt_disk_image_workers({})
                retired = runtime._retire_idle_image_workers({}, keep_model_identity="official", keep_image_guid=runtime._object_id(image))
            assert count == retired == 1 and terminates[0]["pid"] == os.getpid()
            observations["unowned_live_pid_retirement"] = dict(outcome="reproduced", adopted=count, retired=retired,
                terminate_requested_for_diagnostic_pid=True, mismatched_initialize_parent_ignored=True,
                actual_kill_executed=False,
                boundary="real PID liveness and disk lookup; termination replaced by recorder; OS PID recycling not induced")
            runtime._ASYNC_IMAGE_WORKERS.clear()

            root = base / "malformed_epoch"
            folder, record = worker(root, image, created="invalid-clock-value")
            error = None
            with mock.patch.dict(os.environ, {"NNINTERACTIVE_RUNTIME_DIR": str(root)}):
                try:
                    runtime._adopt_disk_image_workers({})
                except Exception as exc:
                    error = type(exc).__name__ + ": " + str(exc)
            assert error and error.startswith("ValueError")
            observations["valid_json_invalid_worker_schema"] = dict(outcome="reproduced", exception=error,
                boundary="actual adoption helper; malformed but readable JSON, no model")
            runtime._ASYNC_IMAGE_WORKERS.clear()

            # Real NIfTI materialization/cache, no Mimics/model execution.
            import numpy as np
            import nibabel as nib
            import nnunet_pipeline as pipeline
            import nnunet_common as common
            root = base / "training_cache"
            root.mkdir()
            raw = root / "image.raw"
            values = np.arange(8, dtype=np.int16).reshape(2, 2, 2)
            raw.write_bytes(values.tobytes())
            sha = hashlib.sha256(raw.read_bytes()).hexdigest()
            label = root / "target.nii.gz"
            nib.save(nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.uint8), np.eye(4)), label)
            request = common.normalize_request(dict(operation="train", task_name="synthetic", workspace=str(root / "workspace"), labels=[dict(name="target", id=1)]))
            payload = dict(path=str(raw), sha256=sha, shape=[2, 2, 2], dtype="int16", mimics_voxel_to_ras_matrix=np.eye(4).tolist(), source_modality="CT", source_intensity_recovery_basis="mimics_hu_gv", buffer_to_source_slope=1.0, buffer_to_source_intercept=0.0, expected_grid=dict(source_shape=[2, 2, 2], source_voxel_to_ras_matrix=np.eye(4).tolist()))
            signature = dict(kind="project_buffer_content", sha256=sha)
            first = pipeline._materialize_case("case", raw, [[label]], request, root / "cache", image_payload=payload, image_signature=signature)
            altered = dict(payload, buffer_to_source_slope=2.0)
            second = pipeline._materialize_case("case", raw, [[label]], request, root / "cache", image_payload=altered, image_signature=signature)
            pipeline.materialize_project_image_buffer(altered, root / "fresh.nii.gz", dict(case_id="case", source_geometry_expected=payload["expected_grid"]))
            cached_values = np.asanyarray(nib.load(second["image"]).dataobj)
            fresh_values = np.asanyarray(nib.load(root / "fresh.nii.gz").dataobj)
            assert not first["cache_hit"] and second["cache_hit"]
            assert not np.array_equal(cached_values, fresh_values)
            observations["project_voxel_cache_omits_intensity_contract"] = dict(outcome="reproduced", second_cache_hit=True,
                cached_values=cached_values.ravel().tolist(), fresh_values=fresh_values.ravel().tolist(),
                raw_sha256_unchanged=True, label_bytes_unchanged=True,
                boundary="actual NIfTI materializer and cache; no training or downstream accuracy measurement")

            import create_mcs_batch as creator
            import import_undo_mimics as undo
            root = base / "receipt_names"
            root.mkdir()
            latest_path = None
            for stamp, case in [(1, "ordinary-case"), (2, "kidney.previous-study")]:
                project = root / (case + ".mcs")
                project.write_bytes(b"synthetic project")
                payload = creator._import_receipt_payload(case, str(project), [], {}, mask_objects=[])
                payload["created_at_epoch"] = stamp
                path = Path(creator._receipt_path_for(str(project), case))
                path.write_text(json.dumps(payload))
                if stamp == 2:
                    latest_path = path
            with mock.patch.object(undo, "_candidate_receipt_dirs", return_value=[str(root)]):
                found, payload = undo.find_latest_receipt()
            assert latest_path.name == ".kidney.previous-study.import_receipt.json"
            assert payload["case_id"] == "ordinary-case"
            observations["ordinary_case_name_misclassified_as_archive"] = dict(outcome="reproduced",
                ignored_latest_receipt=latest_path.name, returned_older_case=payload["case_id"],
                boundary="actual receipt producer naming/payload and discovery; no project opened or masks deleted")
    result = dict(source_commit=SOURCE, observations=observations)
    destination = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("observations.json")
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    finally:
        runtime._ASYNC_IMAGE_WORKERS.clear()
        runtime._ASYNC_MONITORS.clear()
