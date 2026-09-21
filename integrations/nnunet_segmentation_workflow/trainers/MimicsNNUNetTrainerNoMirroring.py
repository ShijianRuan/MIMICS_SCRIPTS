"""No-mirroring nnU-Net Trainer with an explicit, validated epoch count."""

from __future__ import annotations

import os

from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerNoMirroring import (
    nnUNetTrainerNoMirroring,
)


class MimicsNNUNetTrainerNoMirroring(nnUNetTrainerNoMirroring):
    """Preserve official no-mirroring behavior while honoring the workflow config."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        value = str(os.environ.get("MIMICS_NNUNET_EPOCHS") or "1000").strip()
        try:
            epochs = int(value)
        except ValueError as exc:
            raise ValueError(
                "MIMICS_NNUNET_EPOCHS must be an integer, got {!r}.".format(value)
            ) from exc
        if not 1 <= epochs <= 10000:
            raise ValueError(
                "MIMICS_NNUNET_EPOCHS must be between 1 and 10000, got {}.".format(
                    epochs
                )
            )
        self.num_epochs = epochs
