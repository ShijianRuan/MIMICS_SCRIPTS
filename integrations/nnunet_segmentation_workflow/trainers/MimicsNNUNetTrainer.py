"""Custom nnU-Net trainers used by the managed Mimics integration."""

from __future__ import annotations

import inspect
import os

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class MimicsNNUNetTrainer(nnUNetTrainer):
    """Standard nnU-Net training with a validated, configurable epoch count."""

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


# nnU-Net 2.8.0 captures init kwargs by introspecting the SUBCLASS __init__
# signature and indexing the PARENT frame's locals() -- a (*args, **kwargs)
# forwarding signature crashes there with KeyError('args') (found by the
# Phase 3a convergence smoke; the FlexiCT trainers solve it the same way).
# Publish the parent's parameter names so the introspection only sees names
# that exist in the parent frame. nnU-Net 2.5.2 does not introspect and is
# unaffected.
MimicsNNUNetTrainer.__init__.__signature__ = inspect.signature(
    nnUNetTrainer.__init__
)
