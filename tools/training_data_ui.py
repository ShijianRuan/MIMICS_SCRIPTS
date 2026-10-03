#!/usr/bin/env python3
"""Shared data-source language for external AI training windows."""

from __future__ import annotations


LABEL_SOURCE_CHOICES = (
    ("已保存的 Mimics 工程 (.mcs)", "mcs_refresh"),
    ("随源影像保存的 Mask", "source_dataset"),
    ("此前导出的 Mask 文件", "exported_masks"),
)

LABEL_SOURCE_HINTS = {
    "mcs_refresh": (
        "选择原始影像数据集和保存 .mcs 工程的文件夹。目标 Mask 会在后台校验；"
        "没有匹配已保存 Mask 的工程会被跳过。"
    ),
    "source_dataset": (
        "选择原始影像数据集。每个选中的病例需在其 segmentations 文件夹内含匹配的 "
        "Mask；无需 .mcs 工程或单独的 Mask 路径。"
    ),
    "exported_masks": (
        "选择原始影像数据集和此前导出 Mask 的文件夹。病例按 ID、Mask 按名称配对。"
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
