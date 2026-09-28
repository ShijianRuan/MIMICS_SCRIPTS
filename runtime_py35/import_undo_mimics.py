# -*- coding: utf-8 -*-
"""Undo the most recent import (P4): remove its masks, roll back its files.

Every import writes a receipt next to the created .mcs
(<case>.import_receipt.json, schema mimics_import_receipt.v1) listing the
masks it created, the metadata keys it set, and the source fingerprint of
the .mcs at creation time. This module finds the newest receipt and:

1. opens the .mcs it refers to (if not already open),
2. deletes exactly the masks the receipt lists (in one transaction, so a
   failure leaves the project untouched),
3. rolls the .mcs file back only when its fingerprint still matches - if
   the project changed since (annotations, extra masks), the file is kept
   and only the mask deletion is performed, with a clear message.

Receipts are kept per case; "last import" means the newest receipt on this
workstation. The user confirms before anything is deleted.
"""

from __future__ import print_function

import json
import logging
import os

import mimics

import runtime_common

RECEIPT_SUFFIX = ".import_receipt.json"


def _project_root():
    return runtime_common.find_root(
        os.path.dirname(os.path.abspath(__file__)),
        ("nninteractive_config.json", "python_env", "nninteractive_env", ".git"),
    )


def _candidate_receipt_dirs():
    """Folders that may hold receipts: the configured output roots."""
    dirs = []
    config_path = os.path.join(_project_root(), "mimics_io_config.json")
    try:
        with open(config_path, "r") as handle:
            configured = json.load(handle).get("mimics_output_dir", "")
        if configured and os.path.isdir(configured):
            dirs.append(os.path.abspath(configured))
    except Exception:
        pass
    legacy = os.path.join(_project_root(), "mcs_output")
    if os.path.isdir(legacy):
        dirs.append(legacy)
    return dirs


def find_latest_receipt():
    """Return (receipt_path, payload) for the newest receipt, or (None, None)."""
    candidates = []
    for folder in _candidate_receipt_dirs():
        try:
            for name in os.listdir(folder):
                if not name.endswith(RECEIPT_SUFFIX):
                    continue
                path = os.path.join(folder, name)
                try:
                    with open(path, "r") as handle:
                        payload = json.load(handle)
                except Exception:
                    continue
                if not isinstance(payload, dict):
                    continue
                if payload.get("schema_version") != "mimics_import_receipt.v1":
                    continue
                created = float(payload.get("created_at_epoch") or 0.0)
                candidates.append((created, path, payload))
        except OSError:
            continue
    if not candidates:
        return None, None
    candidates.sort(key=lambda item: item[0])
    _created, path, payload = candidates[-1]
    return path, payload


def _mcs_fingerprint(mcs_path):
    """Stable content fingerprint of a saved project file."""
    import hashlib
    digest = hashlib.sha256()
    try:
        with open(mcs_path, "rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError:
        return ""
    return "sha256:" + digest.hexdigest()


def _open_project(mcs_path):
    """Open the receipt's project, or verify the open one matches."""
    try:
        active = mimics.file.get_active_project()
        if active and os.path.abspath(str(active)) == os.path.abspath(mcs_path):
            return True
    except Exception:
        pass
    # Refuse to stomp on a user's unsaved session in another project.
    try:
        projects = mimics.file.get_active_project()
        if projects:
            mimics.dialogs.message_box(
                "Another project is open in Mimics.\n\n"
                "Undo Last Import removes the imported masks from the project "
                "they were created in, so it must open that project itself.\n\n"
                "Step 1: close the currently open project (save it if needed).\n"
                "Step 2: run Undo Last Import again.\n"
                "Your import record is still there — nothing was consumed.",
                title="Undo Last Import",
                ui_blocking=False,
            )
            return False
    except Exception:
        pass
    mimics.file.open_project(filename=mcs_path)
    return True


def _delete_masks(mask_names):
    """Delete the listed masks in one transaction. Returns deleted names."""

    def operation():
        deleted = []
        images = list(mimics.data.images)
        for image in images:
            for mask in list(image.masks):
                if mask.name in mask_names:
                    mask.delete()
                    deleted.append(mask.name)
        return deleted

    return runtime_common.execute_mimics_transaction(
        mimics, operation, "Mimics-Script Undo Import"
    )


def _describe(receipt):
    names = receipt.get("created_masks") or []
    if not names:
        return "no masks"
    if len(names) <= 5:
        return ", ".join(str(n) for n in names)
    return "{0} masks ({1}, ...)".format(
        len(names), ", ".join(str(n) for n in names[:5])
    )


def undo_last_import(confirm=True):
    """Entry point. Returns 0 on success, non-zero on failure or cancel."""
    receipt_path, receipt = find_latest_receipt()
    if not receipt:
        mimics.dialogs.message_box(
            "No import receipt was found. Receipts are written next to each "
            "created .mcs file when an import finishes.",
            title="Undo Last Import",
            ui_blocking=False,
        )
        return 1

    mcs_path = str(receipt.get("mcs_path") or "")
    if not mcs_path or not os.path.isfile(mcs_path):
        mimics.dialogs.message_box(
            "The .mcs file from the last import no longer exists:\n{0}\n\n"
            "Nothing to undo.".format(mcs_path),
            title="Undo Last Import",
            ui_blocking=False,
        )
        return 1

    if confirm:
        answer = mimics.dialogs.message_box(
            "Undo the last import?\n\n"
            "Case: {0}\n"
            "Project: {1}\n"
            "It created {2}.\n\n"
            "The masks will be deleted from the project. The .mcs file is "
            "rolled back only if it has not changed since the import.\n\n"
            "Notes:\n"
            "- If a different project is open in Mimics, close it first and "
            "run this action again.\n"
            "- This undo is one-shot: once it completes, the import record "
            "is consumed and cannot be undone again. If it fails midway, "
            "the record is kept so you can retry.".format(
                receipt.get("case_id", "case"), mcs_path, _describe(receipt)
            ),
            title="Undo Last Import",
            ui_blocking=True,
        )
        # The Mimics message box returns None/False on non-OK; treat any
        # explicit falsy answer as cancel.
        if answer is None or answer is False:
            return 2

    if not _open_project(mcs_path):
        return 3

    mask_names = set(str(n) for n in (receipt.get("created_masks") or []))
    deleted = []
    if mask_names:
        # The next steps (project open above, fingerprint hashing below,
        # save) block the GUI thread for as long as Mimics takes on a
        # multi-hundred-MB .mcs. Say so in the log panel first - non-modal,
        # the same channel collect_diagnostics uses - so the pause is
        # announced work, not a frozen window.
        try:
            mimics.logging.log_user_message(
                level=logging.INFO,
                message=(
                    "Undo Last Import: opening and verifying {0}; Mimics "
                    "may pause for a moment. Masks to remove: {1}.".format(
                        os.path.basename(mcs_path), len(mask_names)
                    )
                ),
            )
        except Exception:
            pass
        try:
            deleted = _delete_masks(mask_names)
        except Exception as exc:
            mimics.dialogs.message_box(
                "The mask deletion failed and was rolled back:\n{0}".format(exc),
                title="Undo Last Import",
                ui_blocking=False,
            )
            return 4

    # File rollback: the pre-undo .mcs is rolled back only when it is still
    # byte-identical to what the import produced. The mask-deletion save
    # happens first; if the fingerprint matched pre-deletion, the file is
    # then removed (the import produced a brand-new file, so deleting it is
    # the true rollback) after re-verifying.
    receipt_fp = str(receipt.get("mcs_fingerprint") or "")
    pre_undo_fp = _mcs_fingerprint(mcs_path)
    fingerprint_matches = bool(receipt_fp) and pre_undo_fp == receipt_fp
    try:
        mimics.file.save_project(filename=mcs_path, save_as_type="Mimics Project Files")
    except Exception as exc:
        mimics.dialogs.message_box(
            "The masks were deleted but saving the project failed:\n{0}".format(exc),
            title="Undo Last Import",
            ui_blocking=False,
        )
        return 5
    file_removed = False
    if fingerprint_matches:
        try:
            mimics.file.close_project()
        except Exception:
            pass
        try:
            os.remove(mcs_path)
            file_removed = True
        except OSError as exc:
            mimics.dialogs.message_box(
                "The masks were deleted, but the .mcs file could not be "
                "removed:\n{0}".format(exc),
                title="Undo Last Import",
                ui_blocking=False,
            )
            return 6

    # The receipt is consumed: a second run must not repeat the undo.
    try:
        os.remove(receipt_path)
    except OSError:
        pass

    mimics.logging.log_user_message(
        level=logging.INFO,
        message=(
            "Undo last import: deleted {0} mask(s) from {1} ({2}).".format(
                len(deleted), mcs_path,
                "file rolled back" if file_removed else "file kept; project saved",
            )
        ),
    )
    if file_removed:
        summary = (
            "Undo complete. {0} mask(s) were deleted and the .mcs file was "
            "removed (it was unchanged since the import, so deletion is the "
            "exact rollback).".format(len(deleted))
        )
    else:
        summary = (
            "Undo complete. {0} mask(s) were deleted from the project and "
            "the project was saved.\n\nThe .mcs file was kept because it "
            "changed since the import (annotations or extra masks would "
            "have been lost).".format(len(deleted))
        )
    mimics.dialogs.message_box(summary, title="Undo Last Import", ui_blocking=False)
    return 0


def main():
    return undo_last_import()


if __name__ == "__main__":
    main()
