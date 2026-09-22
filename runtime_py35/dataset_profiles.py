#!/usr/bin/env python3
"""Dataset layout profiles shared by every TS-like discovery site.

Before this module existed, the same preferred image names (``ct.nii.gz``,
...), the ``segmentations/`` mask directory, and the ``mcs_output``
exclusion list were hardcoded in four different files (mimics_import,
mimics_export, mimics_label_export, io_path_setup_ui) plus the bridge's
candidate ordering.  Changing the layout meant synchronising five places.

This module is the single home.  ``dataset_profiles.json`` at the project
root holds the definitions; consumers call :func:`load_profile` and get a
fully-populated dict with defaults filled in.

Python 3.5 compatible: it runs inside Mimics as well as in the external
tools environment, so no f-strings or modern syntax.
"""

from __future__ import print_function

import json
import os

_SCHEMA = "mimics_dataset_profiles.v1"
_DEFAULT_PROFILE = "ts-like"

_CACHE = {"value": None, "path": None}

# Hard fallback used when dataset_profiles.json is missing or unreadable.
# It mirrors the ts-like profile byte for byte so behaviour never changes
# because of a lost or corrupted config file.
_FALLBACK = {
    "schema_version": _SCHEMA,
    "default_profile": _DEFAULT_PROFILE,
    "profiles": {
        "ts-like": {
            "display_name": "TotalSegmentator-like dataset",
            "description": "One folder per case with a preferred image name and masks in segmentations/.",
            "image_candidates": [
                "ct.nii.gz", "mri.nii.gz", "ct.nii", "mri.nii",
                "ct.mhd", "mri.mhd", "ct.mha", "mri.mha",
            ],
            "fallback_image_suffixes": [
                ".nii.gz", ".nrrd.gz", ".nii", ".mha", ".mhd", ".nrrd",
            ],
            "mask_dirs": ["segmentations"],
            "mask_suffixes": [
                ".seg.nii.gz", ".seg.nii", ".nii.gz", ".nrrd.gz",
                ".nii", ".mha", ".mhd", ".nrrd",
            ],
            "dicom_dirs": ["dicom"],
            "exclude_dirs": ["mcs_output", "segmentations"],
        },
        "generic": {
            "display_name": "Generic scan (no layout assumptions)",
            "description": "Any medical volume file in a case folder is the image; no preferred names.",
            "image_candidates": [],
            "fallback_image_suffixes": [
                ".nii.gz", ".nrrd.gz", ".nii", ".mha", ".mhd", ".nrrd",
            ],
            "mask_dirs": [],
            "mask_suffixes": [
                ".seg.nii.gz", ".seg.nii", ".nii.gz", ".nrrd.gz",
                ".nii", ".mha", ".mhd", ".nrrd",
            ],
            "dicom_dirs": ["dicom"],
            "exclude_dirs": [],
        },
    },
}


def profiles_path():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "dataset_profiles.json")


def _load_raw():
    path = profiles_path()
    if _CACHE["value"] is not None and _CACHE["path"] == path:
        return _CACHE["value"]
    payload = None
    try:
        with open(path, "r") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict) or not isinstance(
            payload.get("profiles"), dict
        ):
            payload = None
    except Exception:
        payload = None
    if payload is None:
        payload = json.loads(json.dumps(_FALLBACK))
    _CACHE["value"] = payload
    _CACHE["path"] = path
    return payload


def _normalize(profile_id, raw):
    profile = {
        "profile_id": profile_id,
        "display_name": str(raw.get("display_name") or profile_id),
        "description": str(raw.get("description") or ""),
        "image_candidates": [
            str(name).lower()
            for name in (raw.get("image_candidates") or [])
            if str(name or "").strip()
        ],
        "fallback_image_suffixes": [
            str(s).lower()
            for s in (raw.get("fallback_image_suffixes") or [])
            if str(s or "").strip()
        ],
        "mask_dirs": [
            str(d) for d in (raw.get("mask_dirs") or []) if str(d or "").strip()
        ],
        "mask_suffixes": [
            str(s).lower()
            for s in (raw.get("mask_suffixes") or [])
            if str(s or "").strip()
        ],
        "dicom_dirs": [
            str(d) for d in (raw.get("dicom_dirs") or []) if str(d or "").strip()
        ],
        "exclude_dirs": [
            str(d) for d in (raw.get("exclude_dirs") or []) if str(d or "").strip()
        ],
    }
    return profile


def profile_ids():
    return sorted(_load_raw().get("profiles").keys())


def default_profile_id():
    raw = _load_raw()
    wanted = str(raw.get("default_profile") or "").strip()
    if wanted and wanted in raw.get("profiles"):
        return wanted
    return _DEFAULT_PROFILE


def load_profile(profile_id=None):
    """Return a normalized profile dict.

    An unknown profile id falls back to the default rather than raising:
    discovery must never fail because of a typo in a saved config.
    """
    raw = _load_raw()
    profiles = raw.get("profiles") or {}
    wanted = str(profile_id or "").strip()
    if not wanted or wanted not in profiles:
        wanted = default_profile_id()
    if wanted not in profiles:
        wanted = "ts-like"
    return _normalize(wanted, profiles[wanted])


def is_excluded_name(name, profile=None):
    profile = profile or load_profile()
    return str(name or "") in profile["exclude_dirs"]
