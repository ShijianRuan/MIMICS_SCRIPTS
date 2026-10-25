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
   (F19), stopping on ambiguity instead of guessing - and only when
   their content still matches the import-time snapshot (R17: an
   equal-volume hand edit changes the digest without touching
   guid/name/count, so ownership alone is not enough; edited masks are
   kept and reported),
3. rolls the .mcs file back only when its fingerprint still matches AND
   the open session holds nothing beyond what the import created (F18) -
   if the project changed since (annotations, extra masks, unsaved
   manual work), the file is kept and only the mask deletion is
   performed, with a clear message.

v1 receipts (names only, no ownership proof, no content snapshot) are
refused outright (R17): the user deletes those masks by hand.

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
    # UI28: the drop window lets imports target any output folder. The
    # worker's status file records it, so the receipt written next to that
    # import's .mcs stays discoverable by undo. Most recent first.
    for extra in _recent_drop_output_dirs():
        if extra not in dirs:
            dirs.append(extra)
    return dirs


def _recent_drop_output_dirs(limit=5):
    """Output folders used by recent drop imports (newest first)."""
    # Same fixed path the drop window writes to (import_drop_window.py
    # submit_import): project .mimics_runtime/drop_import, no env override.
    status_dir = os.path.join(_project_root(), ".mimics_runtime", "drop_import")
    entries = []
    try:
        for name in os.listdir(status_dir):
            if not name.endswith("_status.json"):
                continue
            path = os.path.join(status_dir, name)
            try:
                with open(path, "r") as handle:
                    payload = json.load(handle)
                output = str(payload.get("output_path") or "").strip()
                # R03: worker rewrites (pre-worker-fix statuses) replaced the
                # bootstrap record and only kept output_mcs — derive the
                # output folder from it so those completed imports stay
                # discoverable by undo.
                if not output:
                    output = str(payload.get("output_mcs") or "").strip()
                    if output:
                        output = os.path.dirname(output)
                if output and os.path.isdir(output):
                    entries.append(
                        (float(payload.get("updated_at_epoch") or 0.0), output)
                    )
            except Exception:
                continue
    except OSError:
        return []
    entries.sort(key=lambda item: item[0], reverse=True)
    # Deduplicate before truncating (R03): several cases of one drop share
    # the same output folder; duplicates must not crowd other folders out
    # of the limit.
    seen = set()
    unique = []
    for _ts, output in entries:
        key = os.path.normcase(os.path.abspath(output))
        if key in seen:
            continue
        seen.add(key)
        unique.append(output)
    return unique[:limit]


def find_latest_receipt():
    """Return (receipt_path, payload) for the newest receipt, or (None, None)."""
    candidates = []
    for folder in _candidate_receipt_dirs():
        try:
            for name in os.listdir(folder):
                if not name.endswith(RECEIPT_SUFFIX):
                    continue
                if ".previous-" in name:
                    # Archived by flush_pending_receipts (S01): an older
                    # import's receipt kept as evidence, not an undo
                    # candidate. With a rolled-back clock its
                    # created_at_epoch is the newest in the folder and
                    # would shadow the current receipt.
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
    all_masks = list(mimics.data.masks)

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


def _mask_content_digest(mask):
    """SHA-256 of a live mask's voxel content, or "" when unreadable.

    R17: delegates to the F21 chunked implementation (buffer_byte_view +
    stream_buffer) so a multi-hundred-MB CT mask cannot freeze Mimics, and
    so the digest matches the one captured at import time (same code
    path, same byte view).
    """
    try:
        view = mask.get_voxel_buffer()
        raw = runtime_common.buffer_byte_view(view)
        return runtime_common.stream_buffer(raw, compute_sha=True)
    except Exception:
        return ""


def _content_edited(entry, mask):
    """True when a live mask's content can no longer be PROVEN unedited
    against the import-time snapshot recorded in the receipt entry
    (R17/S04).

    Ownership (guid/token) proves who created the mask, not that its
    content is unchanged. An equal-volume hand edit - move a boundary
    voxel, delete one and add another - keeps guid, name and pixel_count
    identical; only the content digest catches it.

    S04 fail-closed on missing proof: an entry without a recorded digest
    (receipt written before R17, or the import-time buffer was
    unreadable and sha256 was recorded as "") cannot prove the mask is
    unedited, so it is treated as edited - the mask is kept and reported
    for manual resolution. The old pixel-count fallback let an
    equal-volume edit through and deleted hand-corrected masks. A live
    digest that cannot be computed is likewise treated as edited.
    """
    recorded_sha = str(entry.get("sha256") or "")
    if not recorded_sha:
        return True
    return _mask_content_digest(mask) != recorded_sha


def _delete_owned_masks(receipt):
    """Delete only masks the receipt can prove it created AND left
    unedited (F19 ownership + R17 content snapshot).

    Returns (deleted_names, ambiguous_names, edited_names). Ambiguous or
    content-edited entries are skipped and reported - the user resolves
    them by hand; the undo never guesses, and it never deletes a mask
    that holds manual corrections.

    A failing per-mask delete raises out of the transaction (F31): the
    receipt is kept so the undo can be retried; a half-completed delete
    never masquerades as success.
    """
    matches, ambiguous = _resolve_owned_masks(receipt)
    if ambiguous:
        return [], ambiguous, []

    entries = list(receipt.get("created_masks_v2") or [])
    edited = []
    doomed = {}
    for index, mask in matches.items():
        entry = entries[index] if index < len(entries) else {}
        if _content_edited(entry, mask):
            edited.append(str(getattr(mask, "name", "") or ""))
        else:
            doomed[index] = mask
    if edited:
        return [], [], edited

    def operation():
        deleted = []
        for index in sorted(doomed):
            mask = doomed[index]
            name = str(getattr(mask, "name", "") or "")
            mimics.data.masks.delete(mask)
            deleted.append(name)
        return deleted

    return runtime_common.execute_mimics_transaction(
        mimics, operation, "Mimics-Script Undo Import"
    ), ambiguous, []


def _session_diverged(receipt):
    """True when the open session holds work the receipt does not account
    for (F18). The disk fingerprint alone cannot see an in-memory change;
    comparing the session's live mask inventory against the receipt can.
    Returns (diverged, reason).
    """
    entries = receipt.get("created_masks_v2")
    token = str(receipt.get("provenance_token") or "")
    live_masks = list(mimics.data.masks)

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
        # UI28: name the places that were searched so a receipt that sits
        # next to a manually chosen output folder is findable, not a dead end.
        searched = "、".join(_candidate_receipt_dirs()) or "（无）"
        mimics.dialogs.message_box(
            "未找到导入回执。\n\n"
            "每次导入完成时，回执会写在生成的 .mcs 文件旁边。\n"
            "已搜索：{0}\n\n"
            "如果导入时使用的是其他输出文件夹，请到该文件夹下查找 "
            "<病例名>.import_receipt.json。".format(searched),
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

    is_v2 = bool(receipt.get("created_masks_v2")) and \
        receipt.get("schema_version") == RECEIPT_SCHEMA_VERSION
    mask_names = set(str(n) for n in (receipt.get("created_masks") or []))
    if not is_v2 and mask_names:
        # R17: a v1 receipt records names only - no guid, no token, no
        # content snapshot. That is not strong ownership proof: a hand-
        # made same-name mask (possibly on another image) or a hand-
        # edited imported mask is indistinguishable from the original.
        # Undo refuses it outright and the receipt is kept; the user
        # deletes the masks by hand. The old name-based delete (F19's
        # ambiguity guard caught only the multi-holder case) deleted
        # single-name matches blindly. Checked BEFORE the confirm box
        # (which promises masks will be deleted) and before the project
        # is opened, so a v1 receipt is refused without side effects.
        mimics.dialogs.message_box(
            "导入记录来自旧版本，只记录了掩膜名，无法证明这些掩膜"
            "仍然属于那次导入（也没有内容快照可以排除人工修正）。\n\n"
            "为避免误删，撤销不会自动删除任何掩膜。\n\n"
            "请手动删除这些掩膜：{0}\n"
            "如需回滚 .mcs 文件，请手动使用导入前的备份。\n"
            "导入记录已保留。".format(
                ", ".join(sorted(mask_names)[:10])
            ),
            title="撤销上次导入",
            ui_blocking=False,
        )
        return 8

    if confirm:
        answer = mimics.dialogs.question_box(
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
            buttons="撤销;取消",
            ui_blocking=True,
        )
        # message_box is informational and has no documented answer.
        # Only the explicit affirmative question-box button authorizes undo.
        if answer != "撤销":
            return 2

    if not _open_project(mcs_path):
        return 3

    # F31: judge the session against its pre-undo state. The old order
    # (delete first, then check) marked every clean rollback "diverged"
    # because the imported masks were missing - they had just been deleted.
    receipt_fp = str(receipt.get("mcs_fingerprint") or "")
    pre_undo_fp = _mcs_fingerprint(mcs_path)
    disk_matches = bool(receipt_fp) and pre_undo_fp == receipt_fp
    session_diverged, divergence_reason = (False, "")
    if disk_matches:
        session_diverged, divergence_reason = _session_diverged(receipt)

    deleted = []
    ambiguous = []
    edited = []
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
            deleted, ambiguous, edited = _delete_owned_masks(receipt)
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
        if edited:
            # R17: the imported masks still exist and are still provably
            # owned, but their content no longer matches the import-time
            # snapshot - the annotator corrected them by hand (possibly
            # an equal-volume edit invisible to guid/name/count). Deleting
            # them would destroy that manual work. Keep every mask and
            # the receipt; name what was kept so the user can decide.
            mimics.dialogs.message_box(
                "以下导入的掩膜在导入后被修改过（或导入记录缺少内容快照、"
                "无法证明未被修改），可能包含人工修正，"
                "撤销没有删除它们：\n{0}\n\n"
                "如确认不需要这些修正，请手动删除这些掩膜。\n"
                "导入记录已保留，可稍后重新运行撤销。".format(
                    ", ".join(sorted(set(edited))[:10])
                ),
                title="撤销上次导入",
                ui_blocking=False,
            )
            return 9

    # File rollback (F18): the pre-undo .mcs is removed only when BOTH the
    # disk fingerprint still matches the import AND the open session holds
    # nothing the import did not create. The disk check alone approves
    # deletion of a file whose in-memory project contains unsaved manual
    # masks: save_project would flush that manual work into the file, and
    # deleting it would destroy work the user never agreed to lose. The
    # session check ran BEFORE the deletes, on the untouched in-memory state.
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
            ".mcs 文件被保留，因为工程中含有该次导入之外的工作（{1}）。"
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
