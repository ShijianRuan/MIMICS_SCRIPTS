"""Slice-batch trainer for a frozen DINO encoder and cached feature maps."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .trainer import Trainer3D, TrainingCancelled


def binary_volume_dice(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction) > 0
    target = np.asarray(target) > 0
    denominator = int(prediction.sum()) + int(target.sum())
    if denominator == 0:
        return 1.0
    return float(2.0 * np.logical_and(prediction, target).sum() / denominator)


class CachedFeatureSliceTrainer(Trainer3D):
    """Train only the raw-skip decoder on a flat, shuffled slice stream."""

    def _should_validate(self, epoch: int) -> bool:
        return self.val_loader is not None and (
            epoch == self.epochs or epoch % self.validation_interval == 0
        )

    def _is_better_checkpoint(
        self,
        val_dsc,
        best_dsc,
        *,
        should_validate,
        train_loss=None,
        best_train_loss=None,
        **_kwargs,
    ) -> bool:
        if not should_validate or not np.isfinite(val_dsc):
            return False
        val_dsc = float(val_dsc)
        best_dsc = float(best_dsc)
        if val_dsc > best_dsc:
            return True
        if val_dsc == best_dsc and val_dsc == 0.0:
            if train_loss is None or best_train_loss is None:
                return True
            return float(train_loss) < float(best_train_loss)
        return False

    def _build_scheduler(self, cfg):
        self.cosine_min_lr_ratio = float(cfg.get("cosine_min_lr_ratio", 0.1))
        if not 0.0 <= self.cosine_min_lr_ratio <= 1.0:
            raise ValueError("training.cosine_min_lr_ratio must be in [0, 1]")
        self.base_lr = float(cfg.get("lr", 1e-3))
        return None

    def _set_epoch_lr(self, completed_epoch_index: int) -> None:
        scheduler = self.config["training"].get("scheduler")
        if scheduler in (None, "constant"):
            value = self.base_lr
        elif scheduler in ("cosine", "cosine_epoch"):
            progress = float(completed_epoch_index) / max(1, int(self.epochs))
            ratio = self.cosine_min_lr_ratio + (
                1.0 - self.cosine_min_lr_ratio
            ) * (1.0 + math.cos(math.pi * progress)) / 2.0
            value = self.base_lr * ratio
        else:
            raise ValueError(
                "Cached slice training supports constant or epoch cosine scheduling"
            )
        for group in self.optimizer.param_groups:
            group["lr"] = value

    @staticmethod
    def _random_flip(
        embeddings: torch.Tensor,
        images: torch.Tensor,
        labels: torch.Tensor,
    ):
        if np.random.rand() > 0.5:
            embeddings = torch.flip(embeddings, dims=(2,))
            images = torch.flip(images, dims=(2,))
            labels = torch.flip(labels, dims=(1,))
        if np.random.rand() > 0.5:
            embeddings = torch.flip(embeddings, dims=(3,))
            images = torch.flip(images, dims=(3,))
            labels = torch.flip(labels, dims=(2,))
        return embeddings.contiguous(), images.contiguous(), labels.contiguous()

    def _train_epoch(self, epoch: int, global_step_start: int):
        del global_step_start
        self.model.backbone.eval()
        self.model.decoder_3d.train()
        total_loss = 0.0
        total_ce = 0.0
        processed = 0
        progress = tqdm(self.train_loader, desc="Epoch {}/{} [Train]".format(epoch, self.epochs))
        for batch_index, batch in enumerate(progress):
            if self._cancel_requested():
                raise TrainingCancelled()
            embeddings = batch["embedding"].to(self.device, dtype=torch.float32)
            images = batch["image"].to(self.device, dtype=torch.float32)
            labels = batch["label"].to(self.device)
            embeddings, images, labels = self._random_flip(embeddings, images, labels)

            self.optimizer.zero_grad()
            logits = self.model.decode_cached_slices(embeddings, images)
            loss_values = self.criterion(logits, labels)
            loss_values["loss"].backward()
            self.optimizer.step()

            processed += 1
            loss_value = self._to_float(loss_values["loss"])
            ce_value = self._to_float(loss_values.get("ce_loss", loss_value))
            total_loss += loss_value
            total_ce += ce_value
            average_loss = total_loss / processed
            progress.set_postfix({
                "loss": "{:.4f}".format(average_loss),
                "lr": "{:.2e}".format(self.optimizer.param_groups[0]["lr"]),
            })
            self._maybe_write_training_status(
                "training",
                epoch=epoch,
                epochs=self.epochs,
                phase="train",
                batch=batch_index + 1,
                batches=len(self.train_loader),
                lr=self.optimizer.param_groups[0]["lr"],
                metrics={"loss": average_loss, "ce_loss": total_ce / processed},
                force=batch_index + 1 == len(self.train_loader),
            )

        if not processed:
            raise RuntimeError("Cached slice training produced no complete batches")
        self._set_epoch_lr(epoch - 1)
        return {
            "loss": total_loss / processed,
            "ce_loss": total_ce / processed,
            "dice_loss": 0.0,
        }

    def _tta_probabilities(
        self,
        embeddings: torch.Tensor,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        probabilities = []
        losses = []
        for dimensions in ((), (2,), (3,), (2, 3)):
            current_embeddings = (
                torch.flip(embeddings, dims=dimensions).contiguous()
                if dimensions
                else embeddings
            )
            current_images = (
                torch.flip(images, dims=dimensions).contiguous()
                if dimensions
                else images
            )
            label_dimensions = tuple(value - 1 for value in dimensions)
            current_labels = (
                torch.flip(labels, dims=label_dimensions).contiguous()
                if label_dimensions
                else labels
            )
            logits = self.model.decode_cached_slices(current_embeddings, current_images)
            losses.append(self.criterion(logits, current_labels)["loss"])
            current = F.softmax(logits, dim=1)
            if dimensions:
                current = torch.flip(current, dims=dimensions)
            probabilities.append(current)
        return (
            torch.stack(probabilities, dim=0).mean(dim=0),
            torch.stack(losses, dim=0).mean(),
        )

    def _validate_epoch(self, epoch: int):
        self.model.backbone.eval()
        self.model.decoder_3d.eval()
        cases = {}
        losses = []
        progress = tqdm(self.val_loader, desc="Epoch {}/{} [Val]".format(epoch, self.epochs))
        with torch.no_grad():
            for batch_index, batch in enumerate(progress):
                if self._cancel_requested():
                    raise TrainingCancelled()
                embeddings = batch["embedding"].to(self.device, dtype=torch.float32)
                images = batch["image"].to(self.device, dtype=torch.float32)
                labels = batch["label"].to(self.device)
                probabilities, loss = self._tta_probabilities(
                    embeddings,
                    images,
                    labels,
                )
                prediction = probabilities.argmax(dim=1)
                losses.append(float(loss.item()))
                case_ids = list(batch["case_id"])
                for item_index, case_id in enumerate(case_ids):
                    row = cases.setdefault(str(case_id), {"prediction": [], "target": []})
                    row["prediction"].append(prediction[item_index].cpu().numpy())
                    row["target"].append(labels[item_index].cpu().numpy())
                validation_loss = float(np.mean(losses))
                progress.set_postfix({
                    "loss": "{:.4f}".format(validation_loss),
                    "slices": "{}/{}".format(batch_index + 1, len(self.val_loader)),
                })
                self._maybe_write_training_status(
                    "training",
                    epoch=epoch,
                    epochs=self.epochs,
                    phase="val",
                    batch=batch_index + 1,
                    batches=len(self.val_loader),
                    metrics={"validation_loss": validation_loss},
                    force=batch_index + 1 == len(self.val_loader),
                )

        per_case = [
            binary_volume_dice(
                np.stack(row["prediction"], axis=0),
                np.stack(row["target"], axis=0),
            )
            for row in cases.values()
        ]
        return {
            "mean_dsc": float(np.mean(per_case)) if per_case else 0.0,
            "per_class_dsc": [float(value) for value in per_case],
            "validation_loss": float(np.mean(losses)) if losses else 0.0,
        }
