#!/usr/bin/env python3
"""Regression tests for tools/migrate_root.py checkout migration."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

import migrate_root


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


class TestUnderOldRoot(unittest.TestCase):
    def test_detects_paths_under_old_root_case_insensitive(self):
        old_root = migrate_root._normcase(r"E:\OldInstall")
        self.assertTrue(migrate_root._under_old_root(
            r"E:\oldinstall\external\dinov3-medical-seg", old_root
        ))
        self.assertTrue(migrate_root._under_old_root(
            r"E:\OLDINSTALL\foo", old_root
        ))
        self.assertFalse(migrate_root._under_old_root(
            r"E:\OtherPlace\foo", old_root
        ))
        self.assertFalse(migrate_root._under_old_root("relative/path", old_root))


class TestRewriteConfigPaths(unittest.TestCase):
    def _make_project(self, base: Path) -> Path:
        project = base / "new_root"
        project.mkdir(parents=True)
        return project

    def test_rewrites_old_root_paths_to_relative(self):
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            project = self._make_project(base)
            _write_json(project / "fewshot_config.json", {
                "dinov3_project": r"E:\old\external\dinov3-medical-seg",
                "default_epochs": 20,
            })
            _write_json(project / "nninteractive_config.json", {
                "workspace_dir": r"E:\old\nninteractive_task_models",
            })
            log = migrate_root._rewrite_config_paths(
                r"E:\old", absolute=False, dry_run=False, project_root=project
            )
            payload = json.loads(
                (project / "fewshot_config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                payload["dinov3_project"], "external/dinov3-medical-seg"
            )
            self.assertEqual(payload["default_epochs"], 20)
            ws = json.loads(
                (project / "nninteractive_config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(ws["workspace_dir"], "nninteractive_task_models")
            self.assertTrue(any("written" in line for line in log))

    def test_absolute_mode_writes_new_absolute_paths(self):
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            project = self._make_project(base)
            _write_json(project / "fewshot_config.json", {
                "dinov3_project": r"E:\old\external\dinov3-medical-seg",
            })
            migrate_root._rewrite_config_paths(
                r"E:\old", absolute=True, dry_run=False, project_root=project
            )
            payload = json.loads(
                (project / "fewshot_config.json").read_text(encoding="utf-8")
            )
            expected = str(
                migrate_root.PROJECT_ROOT / "external" / "dinov3-medical-seg"
            )
            self.assertEqual(payload["dinov3_project"], expected)

    def test_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            project = self._make_project(base)
            _write_json(project / "fewshot_config.json", {
                "dinov3_project": r"E:\old\external\dinov3-medical-seg",
            })
            migrate_root._rewrite_config_paths(
                r"E:\old", absolute=False, dry_run=True, project_root=project
            )
            payload = json.loads(
                (project / "fewshot_config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                payload["dinov3_project"], r"E:\old\external\dinov3-medical-seg"
            )

    def test_untouched_keys_and_foreign_paths_left_alone(self):
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            project = self._make_project(base)
            _write_json(project / "mimics_io_config.json", {
                "mimics_output_dir": r"D:\SomewhereElse\output",
                "mimics_background_exe": r"E:\old\mimics\mimics.exe",
                "unrelated_key": r"E:\old\value\but\not\a\path\key",
            })
            migrate_root._rewrite_config_paths(
                r"E:\old", absolute=False, dry_run=False, project_root=project
            )
            payload = json.loads(
                (project / "mimics_io_config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                payload["mimics_output_dir"], r"D:\SomewhereElse\output"
            )
            self.assertEqual(
                payload["unrelated_key"], r"E:\old\value\but\not\a\path\key"
            )
            self.assertEqual(
                payload["mimics_background_exe"], "mimics/mimics.exe"
            )


class TestRepairFewshotIndex(unittest.TestCase):
    def _make_manifest(self, ts_root: Path, organ: str, model_id: str) -> Path:
        manifest = (
            ts_root / "fewshot_models" / "models" / organ / model_id
            / "model_manifest.json"
        )
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("{}", encoding="utf-8")
        return manifest

    def test_repairs_dead_manifest_path_by_basename(self):
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            ts_root = base / "dataset"
            manifest = self._make_manifest(ts_root, "brain", "m1")
            index_path = base / "fewshot_model_index.json"
            _write_json(index_path, {
                "models": [{
                    "model_id": "m1",
                    "manifest_path": str(base / "old_machine" / "fewshot_models"
                                         / "models" / "brain" / "m1"
                                         / "model_manifest.json"),
                    "ts_root": str(base / "old_machine"),
                }]
            })
            log = migrate_root._repair_fewshot_index(
                dry_run=False, index_path=index_path, project_root=base
            )
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            row = payload["models"][0]
            self.assertEqual(row["manifest_path"], str(manifest))
            self.assertEqual(row["ts_root"], str(ts_root))
            self.assertTrue(any("ts_root" in line for line in log))

    def test_dead_ts_root_is_repaired_when_manifest_found_elsewhere(self):
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            # The manifest has been copied to a *different* ts_root.
            new_ts_root = base / "moved_dataset"
            manifest = self._make_manifest(new_ts_root, "brain", "m2")
            index_path = base / "fewshot_model_index.json"
            _write_json(index_path, {
                "models": [{
                    "model_id": "m2",
                    "manifest_path": str(manifest),
                    "ts_root": str(base / "no_such_dir"),
                }]
            })
            migrate_root._repair_fewshot_index(
                dry_run=False, index_path=index_path, project_root=base
            )
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["models"][0]["ts_root"], str(new_ts_root)
            )

    def test_missing_manifest_kept_with_dead_path(self):
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            index_path = base / "fewshot_model_index.json"
            dead = str(base / "nowhere" / "model_manifest.json")
            _write_json(index_path, {"models": [{
                "model_id": "ghost",
                "manifest_path": dead,
                "ts_root": str(base / "no_such_dir"),
            }]})
            log = migrate_root._repair_fewshot_index(
                dry_run=False, index_path=index_path, project_root=base
            )
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["models"][0]["manifest_path"], dead)
            self.assertTrue(any("no replacement found" in line for line in log))

    def test_live_manifest_paths_untouched(self):
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            manifest = self._make_manifest(base, "liver", "ok1")
            index_path = base / "fewshot_model_index.json"
            _write_json(index_path, {"models": [{
                "model_id": "ok1",
                "manifest_path": str(manifest),
                "ts_root": str(base),
            }]})
            log = migrate_root._repair_fewshot_index(
                dry_run=False, index_path=index_path, project_root=base
            )
            self.assertTrue(
                any("nothing to do" in line for line in log), log
            )


class TestResetRuntime(unittest.TestCase):
    def test_clears_runtime_state_keeping_empty_dirs(self):
        with tempfile.TemporaryDirectory() as raw:
            import shutil

            project = Path(raw) / "new_root"
            runtime = project / ".mimics_runtime"
            (runtime / "locks").mkdir(parents=True)
            (runtime / "processes").mkdir(parents=True)
            (runtime / "locks" / "gpu.lock").write_text("{}", encoding="utf-8")
            (runtime / "processes" / "p1.json").write_text("{}", encoding="utf-8")
            # Point PROJECT_ROOT at a scratch copy so we never clear the real
            # checkout's runtime state from a test.
            scratch = Path(raw) / "scratch_root"
            shutil.copytree(project, scratch)
            original = migrate_root.PROJECT_ROOT
            migrate_root.PROJECT_ROOT = scratch
            try:
                log = migrate_root._reset_runtime(dry_run=False)
            finally:
                migrate_root.PROJECT_ROOT = original
            scratch_runtime = scratch / ".mimics_runtime"
            self.assertTrue((scratch_runtime / "locks").is_dir())
            self.assertTrue((scratch_runtime / "processes").is_dir())
            self.assertFalse((scratch_runtime / "locks" / "gpu.lock").exists())
            self.assertFalse(
                (scratch_runtime / "processes" / "p1.json").exists()
            )
            # The real checkout (and the un-migrated original tree) is never
            # touched by the scratch run.
            self.assertTrue((runtime / "locks" / "gpu.lock").exists())
            self.assertTrue(any("cleared" in line for line in log))

    def test_dry_run_keeps_everything(self):
        with tempfile.TemporaryDirectory() as raw:
            import shutil

            project = Path(raw) / "new_root"
            runtime = project / ".mimics_runtime"
            (runtime / "locks").mkdir(parents=True)
            (runtime / "processes").mkdir(parents=True)
            (runtime / "locks" / "gpu.lock").write_text("{}", encoding="utf-8")
            scratch = Path(raw) / "scratch_root"
            shutil.copytree(project, scratch)
            original = migrate_root.PROJECT_ROOT
            migrate_root.PROJECT_ROOT = scratch
            try:
                migrate_root._reset_runtime(dry_run=True)
            finally:
                migrate_root.PROJECT_ROOT = original
            self.assertTrue((scratch / ".mimics_runtime" / "locks" / "gpu.lock").exists())
            self.assertTrue((runtime / "locks").is_dir())
            self.assertTrue((runtime / "processes").is_dir())


class TestEndToEndSimulation(unittest.TestCase):
    """Cross-machine simulation: build an old-root tree, migrate it."""

    def test_full_migration_flow(self):
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            old_root = base / "old_install"
            project = base / "new_root"
            project.mkdir()
            # Old machine configs with absolute paths.
            _write_json(old_root / "fewshot_config.json", {
                "dinov3_project": str(old_root / "external" / "dinov3-medical-seg"),
            })
            _write_json(old_root / "nninteractive_config.json", {
                "workspace_dir": str(old_root / "nninteractive_task_models"),
            })
            _write_json(old_root / "nninteractive_finetune_config.json", {
                "official_model_dir": str(
                    old_root / "nninteractive_env" / "models" / "nnInteractive_v1.0"
                ),
            })
            # The "copied tree" in the new root starts as the same configs
            # (simulating a naive whole-tree copy that keeps stale paths).
            for name in ("fewshot_config.json", "nninteractive_config.json",
                         "nninteractive_finetune_config.json"):
                payload = json.loads(
                    (old_root / name).read_text(encoding="utf-8")
                )
                _write_json(project / name, payload)
            log = migrate_root._rewrite_config_paths(
                str(old_root), absolute=False, dry_run=False, project_root=project
            )
            fewshot = json.loads(
                (project / "fewshot_config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                fewshot["dinov3_project"], "external/dinov3-medical-seg"
            )
            nni = json.loads(
                (project / "nninteractive_config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(nni["workspace_dir"], "nninteractive_task_models")
            finetune = json.loads(
                (project / "nninteractive_finetune_config.json")
                .read_text(encoding="utf-8")
            )
            self.assertEqual(
                finetune["official_model_dir"],
                "nninteractive_env/models/nnInteractive_v1.0",
            )
            self.assertTrue(any("written" in line for line in log))


class TestNninteractiveTaskModelsRegistryFormat(unittest.TestCase):
    """The registry format the migration relies on (v2 model_relpath)."""

    def test_registry_uses_model_relpath(self):
        registry_path = ROOT / "nninteractive_task_models" / "registry.json"
        if not registry_path.is_file():
            self.skipTest("registry.json not yet copied into this checkout")
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
        self.assertEqual(
            payload.get("schema_version"), "nninteractive_task_registry.v2"
        )
        for task in payload.get("tasks") or []:
            for model in task.get("models") or []:
                relpath = model.get("model_relpath")
                self.assertTrue(
                    relpath and not Path(relpath).is_absolute(),
                    "model_relpath must stay relative: {!r}".format(relpath),
                )
                resolved = registry_path.parent / relpath
                if model.get("state") == "validated":
                    self.assertTrue(
                        resolved.is_dir(),
                        "validated model missing after copy: {}".format(relpath),
                    )


if __name__ == "__main__":
    unittest.main()
