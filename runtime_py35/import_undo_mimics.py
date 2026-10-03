# -*- coding: utf-8 -*-
"""Undo the most recent import (P4): remove its masks, roll back its files.

Every import writes a receipt next to the created .mcs
(<case>.import_receipt.json, schema mimics_import_receipt.v2) listing the
masks it created (name + guid + provenance token), the metadata keys it
set, and the source fingerprint of the .mcs at creation time. This module
finds the newest receipt and:

1. opens the .mcs it refers to (if not already open),
2. deletes exactly the masks the receipt can prove it created - by guid
   where recorded, so a same-name mask on another image is never touched
   (F19), stopping on ambiguity instead of guessing,
3. rolls the .mcs file back only when its fingerprint still matches AND
   the open session holds nothing beyond what the import created (F18) -
   if the project changed since (annotations, extra masks, unsaved
   manual work), the file is kept and only the mask deletion is
   performed, with a clear message.

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
RECEIPT_SCHEMA_VERSION = "mimics_import_receipt.v2"


def _object_guid(obj):
    """Stable identity of a live Mimics object, or "" when unavailable."""
    return str(getattr(obj, "guid", "") or "")


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
                if payload.get("schema_version") not in (
                    "mimics_import_receipt.v1", RECEIPT_SCHEMA_VERSION,
                ):
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
    """Open the receipt's project, or verify the open one matches.

    F17: "cannot tell which project is open" must never be treated as
    "no project is open". The state comes from the documented
    get_project_information/is_project_loaded pair (via
    runtime_common.current_project_state); an unknown state stops here
    instead of opening another project over a possibly-unsaved session.
    """
    state, active = runtime_common.current_project_state(mimics)
    if state == "path" and active and os.path.abspath(str(active)) == os.path.abspath(mcs_path):
        return True
    if state == "unknown":
        mimics.dialogs.message_box(
            "无法判断 Mimics 中是否打开了其他工程"
            "（工程查询失败，或当前版本不支持）。\n\n"
            "当前会话状态未知时，撤销上次导入拒绝打开其他工程，"
            "因此未做任何改动。\n\n"
            "第一步：关闭当前打开的工程（如需保存请先保存）。\n"
            "第二步：重新运行撤销上次导入。\n"
            "导入记录仍然保留——没有被消费。",
            title="撤销上次导入",
            ui_blocking=False,
        )
        return False
    if state in ("path", "unnamed"):
        mimics.dialogs.message_box(
            "Mimics 中打开了其他工程。\n\n"
            "撤销上次导入要从掩膜所在的工程中移除导入的掩膜，"
            "因此必须亲自打开该工程。\n\n"
            "第一步：关闭当前打开的工程（如需保存请先保存）。\n"
            "第二步：重新运行撤销上次导入。\n"
            "导入记录仍然保留——没有被消费。",
            title="撤销上次导入",
            ui_blocking=False,
        )
        return False
    mimics.file.open_project(filename=mcs_path)
    return True


def _mask_has_token(mask, token):
    """True when the mask carries this import's provenance token (T19).

    The token is written into mask metadata at import time; it survives
    renames and, unlike a guid, cannot collide with a hand-made mask.
    """
    if not token:
        return False
    try:
        item = mask.metadata.find(token)
        return item is not None
    except Exception:
        return False


def _resolve_owned_masks(receipt):
    """Map receipt entries to live masks by ownership, not by name alone.

    Returns (matches, ambiguous) where matches maps receipt index -> live
    mask. A receipt entry matches when its recorded guid is present (T19:
    survives renames), or - when the guid no longer resolves, because the
    host regenerates guids across reopen - when exactly one live mask with
    the recorded name carries this import's provenance token. A recorded
    object that resolves to nothing was deleted and recreated by hand: not
    ambiguous, nothing to delete, and the session check will keep the file.
    """
    entries = receipt.get("created_masks_v2") or []
    token = str(receipt.get("provenance_token") or "")
    all_masks = []
    for image in list(mimics.data.images):
        for mask in list(image.masks):
            all_masks.append(mask)

    matches = {}
    ambiguous = []
    for index, entry in enumerate(entries):
        guid = str(entry.get("guid") or "")
        name = str(entry.get("name") or "")
        if guid:
            found = [m for m in all_masks if _object_guid(m) == guid]
            if len(found) == 1:
                matches[index] = found[0]
                continue
        # Guid drift or v2 entry without guid: token + name decides.
        found = [
            m
            for m in all_masks
            if str(getattr(m, "name", "") or "") == name
            and _mask_has_token(m, token)
        ]
        if len(found) == 1:
            matches[index] = found[0]
        elif len(found) > 1:
            ambiguous.append(name)
    return matches, ambiguous


def _delete_masks(mask_names):
    """Delete the listed masks in one transaction. Returns deleted names.

    Kept for v1 receipts (name-based). v2 receipts go through
    _delete_owned_masks, which never deletes a same-name object that
    belongs to another image (F19).
    """

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


def _delete_owned_masks(receipt):
    """Delete only masks the receipt can prove it created (F19).

    Returns (deleted_names, ambiguous_names). Ambiguous entries are
    skipped and reported - the user resolves them by hand; the undo never
    guesses.
    """
    matches, ambiguous = _resolve_owned_masks(receipt)
    if ambiguous:
        return [], ambiguous

    def operation():
        deleted = []
        for index in sorted(matches):
            mask = matches[index]
            try:
                name = str(getattr(mask, "name", "") or "")
                mask.delete()
                deleted.append(name)
            except Exception:
                pass
        return deleted

    return runtime_common.execute_mimics_transaction(
        mimics, operation, "Mimics-Script Undo Import"
    ), ambiguous


def _session_diverged(receipt):
    """True when the open session holds work the receipt does not account
    for (F18). The disk fingerprint alone cannot see an in-memory change;
    comparing the session's live mask inventory against the receipt can.
    Returns (diverged, reason).
    """
    entries = receipt.get("created_masks_v2")
    token = str(receipt.get("provenance_token") or "")
    images = list(mimics.data.images)
    live_masks = []
    for image in images:
        for mask in list(image.masks):
            live_masks.append(mask)

    if entries is None:
        # v1 receipt: no guids, no token. Conservative name check - any
        # live mask the receipt does not list means manual work in the
        # session, and the delete-file rollback must not run (W02: 旧
        # receipt 保守处理).
        listed = set(str(n) for n in (receipt.get("created_masks") or []))
        extra = [
            str(getattr(m, "name", "") or "")
            for m in live_masks
            if str(getattr(m, "name", "") or "") not in listed
        ]
        if extra:
            return True, "session has mask(s) the import did not create: {0}".format(
                ", ".join(extra[:5])
            )
        return False, ""

    entries = list(entries)
    expected_guids = set(str(e.get("guid") or "") for e in entries)
    matched = 0
    extra = []
    for mask in live_masks:
        guid = _object_guid(mask)
        if guid and guid in expected_guids:
            matched += 1
        elif _mask_has_token(mask, token):
            # Token present: import-owned, guid drifted across reopen.
            matched += 1
        else:
            extra.append(str(getattr(mask, "name", "") or ""))

    if extra:
        return True, "session has mask(s) the import did not create: {0}".format(
            ", ".join(extra[:5])
        )
    if matched < len(expected_guids):
        # A receipt object is missing from the session: it was deleted and
        # possibly recreated by hand. Either way the delete-file rollback
        # is unsafe (F18/T19: 删除后重建同名).
        return True, "imported mask(s) were deleted or changed since the import"
    # Content edits of the imported masks themselves are invisible here;
    # the post-save fingerprint re-check below covers them.
    return False, ""


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
            "未找到导入回执。每次导入完成时，回执会写在生成的 .mcs 文件旁边。",
            title="撤销上次导入",
            ui_blocking=False,
        )
        return 1

    mcs_path = str(receipt.get("mcs_path") or "")
    if not mcs_path or not os.path.isfile(mcs_path):
        mimics.dialogs.message_box(
            "上次导入的 .mcs 文件已不存在：\n{0}\n\n"
            "没有可撤销的内容。".format(mcs_path),
            title="撤销上次导入",
            ui_blocking=False,
        )
        return 1

    if confirm:
        answer = mimics.dialogs.message_box(
            "撤销上次导入吗？\n\n"
            "病例：{0}\n"
            "工程：{1}\n"
            "该次导入创建了 {2}。\n\n"
            "掩膜将从工程中删除。.mcs 文件只有在导入后未发生改动时才会回滚。\n\n"
            "注意：\n"
            "- 如果 Mimics 中打开了其他工程，请先关闭再重新运行本操作。\n"
            "- 本撤销是一次性的：完成后导入记录即被消费，无法再次撤销。"
            "如果中途失败，记录会保留，可以重试。".format(
                receipt.get("case_id", "case"), mcs_path, _describe(receipt)
            ),
            title="撤销上次导入",
            ui_blocking=True,
        )
        # The Mimics message box returns None/False on non-OK; treat any
        # explicit falsy answer as cancel.
        if answer is None or answer is False:
            return 2

    if not _open_project(mcs_path):
        return 3

    is_v2 = bool(receipt.get("created_masks_v2")) and \
        receipt.get("schema_version") == RECEIPT_SCHEMA_VERSION
    mask_names = set(str(n) for n in (receipt.get("created_masks") or []))
    deleted = []
    ambiguous = []
    if is_v2 or mask_names:
        # The next steps (project open above, mask resolution, fingerprint
        # hashing below, save) block the GUI thread for as long as Mimics
        # takes on a multi-hundred-MB .mcs. Say so in the log panel first -
        # non-modal, the same channel collect_diagnostics uses - so the
        # pause is announced work, not a frozen window.
        try:
            mimics.logging.log_user_message(
                level=logging.INFO,
                message=(
                    "Undo Last Import: opening and verifying {0}; Mimics "
                    "may pause for a moment. Masks to remove: {1}.".format(
                        os.path.basename(mcs_path),
                        len(receipt.get("created_masks_v2") or mask_names or []),
                    )
                ),
            )
        except Exception:
            pass
    if is_v2:
        try:
            deleted, ambiguous = _delete_owned_masks(receipt)
        except Exception as exc:
            mimics.dialogs.message_box(
                "掩膜删除失败，已回滚：\n{0}".format(exc),
                title="撤销上次导入",
                ui_blocking=False,
            )
            return 4
        if ambiguous:
            # Same-name masks exist on multiple images and the receipt
            # cannot prove which one it created (F19). Never guess: the
            # receipt is kept, the user deletes the right one by hand.
            mimics.dialogs.message_box(
                "导入记录中的掩膜现在存在于多个影像上：\n{0}\n\n"
                "撤销无法证明哪一个是由该次导入创建的，因此没有自动删除任何掩膜。\n\n"
                "请手动删除正确的掩膜；如果还想回滚文件，再重新运行撤销上次导入。"
                "导入记录已保留。".format(", ".join(sorted(set(ambiguous))[:5])),
                title="撤销上次导入",
                ui_blocking=False,
            )
            return 7
    elif mask_names:
        try:
            deleted = _delete_masks(mask_names)
        except Exception as exc:
            mimics.dialogs.message_box(
                "掩膜删除失败，已回滚：\n{0}".format(exc),
                title="撤销上次导入",
                ui_blocking=False,
            )
            return 4

    # File rollback (F18): the pre-undo .mcs is removed only when BOTH the
    # disk fingerprint still matches the import AND the open session holds
    # nothing the import did not create. The disk check alone approves
    # deletion of a file whose in-memory project contains unsaved manual
    # masks: save_project would flush that manual work into the file, and
    # deleting it would destroy work the user never agreed to lose. The
    # session check runs BEFORE any save, on the untouched in-memory state.
    receipt_fp = str(receipt.get("mcs_fingerprint") or "")
    pre_undo_fp = _mcs_fingerprint(mcs_path)
    disk_matches = bool(receipt_fp) and pre_undo_fp == receipt_fp
    session_diverged, divergence_reason = (False, "")
    if disk_matches:
        session_diverged, divergence_reason = _session_diverged(receipt)
    try:
        mimics.file.save_project(filename=mcs_path, save_as_type="Mimics Project Files")
    except Exception as exc:
        mimics.dialogs.message_box(
            "掩膜已删除，但保存工程失败：\n{0}".format(exc),
            title="撤销上次导入",
            ui_blocking=False,
        )
        return 5
    file_removed = False
    if disk_matches and not session_diverged:
        # Re-verify the fingerprint after the save (T18 "已保存新增"): the
        # save of a clean session is deterministic enough to still match;
        # the save of a dirty session writes new content and will not.
        if _mcs_fingerprint(mcs_path) != receipt_fp:
            session_diverged = True
        else:
            try:
                mimics.file.close_project()
            except Exception:
                pass
            try:
                os.remove(mcs_path)
                file_removed = True
            except OSError as exc:
                mimics.dialogs.message_box(
                    "掩膜已删除，但 .mcs 文件无法移除：\n{0}".format(exc),
                    title="撤销上次导入",
                    ui_blocking=False,
                )
                return 6
    if session_diverged:
        try:
            mimics.logging.log_user_message(
                level=logging.INFO,
                message=(
                    "Undo Last Import: the .mcs file was kept because {0}.".format(
                        divergence_reason or "the project changed since the import"
                    )
                ),
            )
        except Exception:
            pass

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
            "撤销完成。已删除 {0} 个掩膜，.mcs 文件已被移除"
            "（导入后未发生改动，删除即精确回滚）。".format(len(deleted))
        )
    elif session_diverged:
        summary = (
            "撤销完成。已从工程中删除 {0} 个掩膜，工程已保存。\n\n"
            ".mcs 文件被保留，因为工程中含有该次导入之外的工作（{0}）。"
            "删除文件会破坏这些工作——确认不需要时请手动删除。".format(
                len(deleted),
                divergence_reason or "导入后发生了改动",
            )
        )
    else:
        summary = (
            "撤销完成。已从工程中删除 {0} 个掩膜，工程已保存。\n\n"
            ".mcs 文件被保留，因为它在导入后发生了改动"
            "（直接删除会丢失标注或额外掩膜）。".format(len(deleted))
        )
    mimics.dialogs.message_box(summary, title="撤销上次导入", ui_blocking=False)
    return 0


def main():
    return undo_last_import()


if __name__ == "__main__":
    main()
