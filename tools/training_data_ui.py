#!/usr/bin/env python3
"""Shared data-source language for external AI training windows."""

from __future__ import annotations


LABEL_SOURCE_CHOICES = (
    ("Saved Mimics projects (.mcs)", "mcs_refresh"),
    ("Masks stored with source images", "source_dataset"),
    ("Previously exported masks", "exported_masks"),
)

LABEL_SOURCE_HINTS = {
    "mcs_refresh": (
        "Choose the original image dataset and the folder containing saved .mcs "
        "projects. The target Mask is verified in the background; projects without "
        "a matching saved Mask are skipped."
    ),
    "source_dataset": (
        "Choose the original image dataset. Each selected case must contain a "
        "matching Mask in its segmentations folder; no .mcs or separate Mask path "
        "is needed."
    ),
    "exported_masks": (
        "Choose the original image dataset and the separate folder where Masks "
        "were exported. Cases are paired by case ID and Mask name."
    ),
}


def label_source_hint(value):
    return LABEL_SOURCE_HINTS.get(
        str(value or "mcs_refresh"),
        LABEL_SOURCE_HINTS["mcs_refresh"],
    )


def normalized_source_mode(value):
    value = str(value or "mcs_refresh").strip().lower()
    if value not in {item[1] for item in LABEL_SOURCE_CHOICES}:
        return "mcs_refresh"
    return value
