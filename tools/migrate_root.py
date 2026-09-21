"""One-shot checkout migration: rewrite machine-specific absolute paths.

Move this checkout to a new machine or a new root directory and run::

    python tools/migrate_root.py --old-root E:\\old_install_root

The command rewrites:
  * absolute path values in the six root config JSONs that point under the
    old root into root-relative paths (or new absolute paths with
    ``--absolute``),
  * the global fewshot model index (``~/.mimics_script/fewshot_model_index.json``)
    whose recorded absolute manifest paths are repaired by basename match
    against the new fewshot workspaces,
  * the root ``setup_offline.bat`` when it still targets the legacy
    ``nninteractive_env`` directory name.

``--dry-run`` prints the planned changes without writing anything.
``--reset-runtime`` additionally clears ``.mimics_runtime`` runtime state
(locks, process records, queue markers - all machine-local and
regenerable).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG_FILES = [
    "fewshot_config.json",
    "nninteractive_config.json",
    "nninteractive_finetune_config.json",
    "mimics_io_config.json",
    "interactive_algorithms_config.json",
    "dataset_profiles.json",
]

# Keys inside config files whose values are known to be paths. Everything
# else is left untouched, so unrelated absolute-looking strings are safe.
PATH_KEYS = {
    "dinov3_project",
    "workspace_dir",
    "official_model_dir",
    "mimics_output_dir",
    "mimics_background_exe",
}

HOME_MARKER = "~/.mimics_script"


def _read_json(path: Path):
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle), True
    except FileNotFoundError:
        return None, False
    except Exception:
        return None, False


def _write_json(path: Path, payload) -> bool:
    try:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(str(tmp), str(path))
        return True
    except Exception:
        return False


def _normcase(path_text: str) -> str:
    return os.path.normcase(os.path.normpath(path_text))


def _under_old_root(value: str, old_root: str) -> bool:
    try:
        probe = os.path.normcase(os.path.abspath(str(value)))
        return probe.startswith(os.path.normcase(os.path.abspath(old_root)))
    except Exception:
        return False


def _rewrite_config_paths(
    old_root: str,
    absolute: bool,
    dry_run: bool,
    project_root: Path | None = None,
) -> list[str]:
    """Rewrite PATH_KEYS values pointing under old_root. Returns log lines."""
    log = []
    root = project_root or PROJECT_ROOT
    for name in CONFIG_FILES:
        path = root / name
        payload, ok = _read_json(path)
        if not ok or not isinstance(payload, dict):
            continue
        changed = False
        for key in PATH_KEYS:
            value = payload.get(key)
            if not isinstance(value, str) or not value.strip():
                continue
            if not _under_old_root(value, old_root):
                continue
            new_value = _new_path_value(value, old_root, absolute)
            if new_value != value:
                log.append(
                    "{}: {} = {} -> {}".format(name, key, value, new_value)
                )
                payload[key] = new_value
                changed = True
        if changed and not dry_run:
            if _write_json(path, payload):
                log.append("{}: written".format(name))
            else:
                log.append("{}: WRITE FAILED".format(name))
    return log


def _rewrite_project_configs(
    old_root: str, absolute: bool, dry_run: bool, project_root: Path
) -> list[str]:
    """Rewrite PATH_KEYS values in a specific project tree (testable form)."""
    return _rewrite_config_paths(old_root, absolute, dry_run, project_root)


def _new_path_value(value: str, old_root: str, absolute: bool) -> str:
    # value is under old_root; keep only the sub-path below it.
    rel = os.path.relpath(os.path.abspath(str(value)), os.path.abspath(old_root))
    if absolute:
        return str((PROJECT_ROOT / rel).resolve())
    return Path(rel).as_posix()


def _repair_fewshot_index(
    dry_run: bool,
    index_path: Path | None = None,
    project_root: Path | None = None,
) -> list[str]:
    """Repair dead absolute manifest paths in the global fewshot index.

    The index lives in ~/.mimics_script/fewshot_model_index.json and records
    absolute manifest paths from every machine that ever trained a model.
    When a manifest path is dead, look for a manifest with the same basename
    under any <ts_root>/fewshot_models/models/**/ tree reachable from the
    index rows' ts_roots and this checkout, preferring the new checkout's
    root.
    """
    root = project_root or PROJECT_ROOT
    if index_path is None:
        index_path = Path.home() / ".mimics_script" / "fewshot_model_index.json"
    payload, ok = _read_json(index_path)
    if not ok or not isinstance(payload, dict):
        return ["fewshot index: not present or unreadable (nothing to do)"]
    rows = payload.get("models")
    if not isinstance(rows, list):
        return ["fewshot index: no models key (nothing to do)"]

    # Candidate roots under which manifests may live after the move.
    candidate_roots = [root]
    for row in rows:
        if isinstance(row, dict):
            ts_root = str(row.get("ts_root") or "").strip()
            if ts_root and os.path.isdir(ts_root):
                candidate_roots.append(Path(ts_root))

    def find_manifest(basename: str) -> Path | None:
        for search_root in candidate_roots:
            for workspace in search_root.glob("fewshot_models/models"):
                # Two levels of organ/model dirs at most, plus version dirs.
                for candidate in workspace.rglob(basename):
                    if candidate.is_file():
                        return candidate
            for candidate in search_root.rglob(
                "fewshot_models/models/**/" + basename
            ):
                if candidate.is_file():
                    return candidate
        return None

    log = []
    changed = False
    for row in rows:
        if not isinstance(row, dict):
            continue
        manifest_path = str(row.get("manifest_path") or "").strip()
        reference = None
        if manifest_path and not os.path.isfile(manifest_path):
            basename = os.path.basename(manifest_path.replace("\\", "/"))
            found = find_manifest(basename)
            if found is None:
                log.append(
                    "fewshot index: {} (model {}) - no replacement found; "
                    "row kept with dead path".format(
                        manifest_path, row.get("model_id") or "?"
                    )
                )
            else:
                log.append(
                    "fewshot index: {} (model {}) -> {}".format(
                        manifest_path, row.get("model_id") or "?", found
                    )
                )
                row["manifest_path"] = str(found)
                reference = found
                changed = True
        if reference is None and manifest_path and os.path.isfile(manifest_path):
            reference = Path(manifest_path)
        ts_root = row.get("ts_root")
        if (
            isinstance(ts_root, str)
            and ts_root.strip()
            and not os.path.isdir(ts_root)
            and reference is not None
        ):
            # Point the workspace root at the tree that now hosts the model.
            new_root = _find_ts_root_for(reference)
            if new_root:
                row["ts_root"] = new_root
                log.append(
                    "fewshot index: ts_root (model {}) -> {}".format(
                        row.get("model_id") or "?", new_root
                    )
                )
                changed = True
    if changed and not dry_run:
        if _write_json(index_path, payload):
            log.append("fewshot index: written")
        else:
            log.append("fewshot index: WRITE FAILED")
    if not log:
        log.append("fewshot index: all manifest paths resolve (nothing to do)")
    return log


def _find_ts_root_for(manifest: Path) -> str | None:
    # <ts_root>/fewshot_models/models/<organ>/<model>/model_manifest.json
    for parent in manifest.parents:
        if parent.name == "fewshot_models":
            return str(parent.parent.resolve())
    return None


def _regenerate_setup_bat(dry_run: bool) -> list[str]:
    """Regenerate setup_offline.bat when it targets the legacy env name."""
    bat_path = PROJECT_ROOT / "setup_offline.bat"
    if not bat_path.is_file():
        return ["setup_offline.bat: not present (nothing to do)"]
    try:
        text = bat_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ["setup_offline.bat: unreadable (skipped)"]
    if "nninteractive_env" not in text and "mimics_script_offline" not in text:
        return ["setup_offline.bat: already targets python_env (nothing to do)"]
    log = ["setup_offline.bat: regenerating (legacy env references found)"]
    if dry_run:
        return log
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "tools"))
        import package_portable

        bat = package_portable._generate_offline_bat("3.13.7", "python313")
        bat_path.write_text(bat, encoding="utf-8")
        log.append("setup_offline.bat: written")
    except Exception as exc:
        log.append("setup_offline.bat: regeneration failed: {}".format(exc))
    return log


def _reset_runtime(dry_run: bool) -> list[str]:
    runtime_dir = PROJECT_ROOT / ".mimics_runtime"
    if not runtime_dir.is_dir():
        return [".mimics_runtime: not present (nothing to do)"]
    log = []
    keep_dirs = set()
    for entry in runtime_dir.iterdir():
        if entry.is_dir():
            log.append(".mimics_runtime/{}: cleared".format(entry.name))
            if not dry_run:
                import shutil

                shutil.rmtree(entry, ignore_errors=True)
                keep_dirs.add(entry.name)
        else:
            log.append(".mimics_runtime/{}: removed".format(entry.name))
            if not dry_run:
                try:
                    entry.unlink()
                except OSError:
                    pass
    if not dry_run:
        for name in keep_dirs:
            (runtime_dir / name).mkdir(parents=True, exist_ok=True)
    log.append(".mimics_runtime: state cleared (empty dirs kept)")
    return log


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Rewrite machine-specific absolute paths after moving this checkout."
    )
    parser.add_argument(
        "--old-root",
        required=True,
        help="Previous install root (e.g. E:\\mimics_script_offline)",
    )
    parser.add_argument(
        "--absolute",
        action="store_true",
        help="Write new absolute paths instead of root-relative paths",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print planned changes only"
    )
    parser.add_argument(
        "--reset-runtime",
        action="store_true",
        help="Also clear .mimics_runtime machine-local state",
    )
    args = parser.parse_args(argv)

    old_root = _normcase(os.path.abspath(args.old_root))
    new_root = _normcase(str(PROJECT_ROOT))
    if old_root == new_root:
        print("old-root is the current root; nothing to migrate.")
        return 0

    print("Migrating checkout at {}".format(PROJECT_ROOT))
    print("Old root: {}".format(args.old_root))
    print("Mode: {}{}".format(
        "absolute paths" if args.absolute else "root-relative paths",
        " (dry run)" if args.dry_run else "",
    ))
    print()

    log = []
    log += _rewrite_config_paths(old_root, args.absolute, args.dry_run)
    log += _repair_fewshot_index(args.dry_run)
    log += _regenerate_setup_bat(args.dry_run)
    if args.reset_runtime:
        log += _reset_runtime(args.dry_run)

    for line in log:
        print("  " + line)

    print()
    if args.dry_run:
        print("Dry run: no files were modified.")
    else:
        print("Migration complete.")
        print("Remaining manual steps (not automatable):")
        print("  * Copy nninteractive_task_models/tasks/ (weights) if not already here.")
        print("  * Copy integrations/dinov3-medical-seg code + models/ if not already here.")
        print("  * Copy your dataset roots (ts_root folders) if they moved too.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
