"""3D training loop with sub-volume support and mixed precision."""

import os
import time
import json
import uuid
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Dict
import numpy as np

from ..utils.device import get_device, get_dtype
from ..utils.checkpoint import save_checkpoint, load_checkpoint
from ..training.losses import get_loss
from ..training.metrics import dice_score


class TrainingCancelled(Exception):
    """Raised when an external cancellation request is detected."""


class Trainer3D:
    """Training loop for 3D medical image segmentation."""

    def __init__(
        self,
        model: nn.Module,
        config: Dict,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
    ):
        # ── Determinism: seed everything so runs are reproducible. ──
        # Without this, batch_size=1 + shuffle=True gave non-reproducible runs
        # (two identical kidney_left configs diverged: 0.47 vs 0.41 best DSC).
        # A seed of 0 is used unless training.seed is set in the config.
        seed = int(config.get("training", {}).get("seed", 0))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        import random as _random
        _random.seed(seed)
        # Prefer deterministic algorithms when available; this trades a little
        # speed for reproducibility and is the right default for ablations.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.seed = seed

        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        cfg = config["training"]
        self.device = get_device()
        self.dtype = get_dtype(self.device) if cfg.get("mixed_precision", True) else torch.float32
        self.epochs = cfg.get("epochs", 200)
        self.grad_accumulation = cfg.get("grad_accumulation", 1)
        self.sub_volume_cfg = cfg.get("sub_volume", {})
        self.use_sub_volume = self.sub_volume_cfg.get("enabled", False)
        runtime_cfg = config.get("runtime", {})
        self.status_path = runtime_cfg.get("status_path")
        self.cancel_path = runtime_cfg.get("cancel_path")
        self.metrics_history_path = runtime_cfg.get("metrics_history_path")
        self.status_interval_seconds = float(runtime_cfg.get("status_interval_seconds", 2.0))
        self._last_status_write = 0.0
        self._metrics_history_rows = []
        self._last_runtime_write_error = ""
        self._last_runtime_write_error_time = 0.0

        # Experiment dir
        exp_name = config.get("exp_name", f"exp_{int(time.time())}")
        self.exp_dir = os.path.join("experiments", exp_name)
        self.ckpt_dir = os.path.join(self.exp_dir, "checkpoints")
        self.log_dir = os.path.join(self.exp_dir, "logs")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # Writer
        self.writer = SummaryWriter(self.log_dir)

        # Optimizer
        self.optimizer = self._build_optimizer(cfg)

        # Scheduler
        self.scheduler = self._build_scheduler(cfg)

        # Loss
        self.criterion = get_loss(config)

        # Move model and criterion to device
        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)

        # Print trainable info
        info = self.model.get_trainable_info()
        print(f"\n{'='*60}")
        print(f"Model: {config['model']['model_path']}")
        print(f"Fine-tuning: {info['ft_method']}")
        print(f"Decoder: {info['decoder_type']}")
        print(f"Trainable: {info['trainable']:,} / {info['total']:,} ({info['trainable_pct']:.1f}%)")
        if info['lora_trainable']:
            print(f"  LoRA: {info['lora_trainable']:,}")
        print(f"Decoder: {info['decoder_trainable']:,}")
        print(f"Device: {self.device}")
        print(f"Experiment: {self.exp_dir}")
        print(f"{'='*60}\n")
        self._write_training_status(
            "initialized",
            epoch=0,
            epochs=self.epochs,
            device=str(self.device),
            experiment_dir=self.exp_dir,
            trainable=info.get("trainable"),
            trainable_pct=info.get("trainable_pct"),
        )
        self._write_metrics_history("initialized")

    def _report_runtime_write_error(self, label, path, exc):
        now = time.time()
        key = "{}:{}:{}".format(label, path, exc)
        if key == self._last_runtime_write_error and now - self._last_runtime_write_error_time < 30.0:
            return
        self._last_runtime_write_error = key
        self._last_runtime_write_error_time = now
        print("Warning: could not update {} {}: {}".format(label, path, exc), flush=True)

    @classmethod
    def _json_safe(cls, value):
        if isinstance(value, dict):
            return {str(k): cls._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._json_safe(v) for v in value]
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value

    def _write_json_atomic(self, path, payload, label, retries=12, max_sleep=0.20):
        if not path:
            return False
        try:
            text = json.dumps(self._json_safe(payload), indent=2, sort_keys=True) + "\n"
        except Exception as exc:
            self._report_runtime_write_error(label, path, exc)
            return False
        directory = os.path.dirname(path)
        if directory:
            try:
                os.makedirs(directory, exist_ok=True)
            except Exception as exc:
                self._report_runtime_write_error(label, path, exc)
                return False
        last_error = None
        for attempt in range(max(1, int(retries))):
            tmp = "{}.{}.{}.tmp".format(path, os.getpid(), uuid.uuid4().hex)
            try:
                with open(tmp, "w") as handle:
                    handle.write(text)
                    try:
                        handle.flush()
                        os.fsync(handle.fileno())
                    except Exception:
                        pass
                os.replace(tmp, path)
                return True
            except Exception as exc:
                last_error = exc
                try:
                    if os.path.isfile(tmp):
                        os.unlink(tmp)
                except Exception:
                    pass
                time.sleep(min(float(max_sleep), 0.03 * (attempt + 1)))
        self._report_runtime_write_error(label, path, last_error)
        return False

    def _write_training_status(self, status: str, **payload):
        if not self.status_path:
            return
        data = {
            "status": status,
            "updated_at_epoch": time.time(),
        }
        if self.metrics_history_path:
            data["metrics_history"] = self.metrics_history_path
        data.update(payload)
        self._write_json_atomic(self.status_path, data, "training status")

    def _write_metrics_history(self, status, **payload):
        if not self.metrics_history_path:
            return
        data = {
            "schema_version": "mimics_fewshot_metrics_history.v1",
            "status": status,
            "epoch_count": self.epochs,
            "updated_at_epoch": time.time(),
            "history": self._metrics_history_rows,
        }
        data.update(payload)
        self._write_json_atomic(self.metrics_history_path, data, "metrics history")

    def _append_metrics_history(self, row):
        self._metrics_history_rows.append(self._json_safe(row))
        self._write_metrics_history("training")

    def _cancel_requested(self) -> bool:
        return bool(self.cancel_path and os.path.isfile(self.cancel_path))

    def _maybe_write_training_status(self, status: str, *, force: bool = False, **payload):
        now = time.time()
        if force or now - self._last_status_write >= self.status_interval_seconds:
            self._write_training_status(status, **payload)
            self._last_status_write = now

    @staticmethod
    def _to_float(value, default=0.0):
        try:
            if hasattr(value, "item"):
                value = value.item()
            return float(value)
        except Exception:
            return default

    def _build_optimizer(self, cfg: Dict) -> torch.optim.Optimizer:
        opt_name = cfg.get("optimizer", "adamw")
        lr = cfg.get("lr", 1e-3)
        wd = cfg.get("weight_decay", 0.01)

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if opt_name == "adamw":
            return torch.optim.AdamW(trainable, lr=lr, weight_decay=wd)
        elif opt_name == "adam":
            return torch.optim.Adam(trainable, lr=lr, weight_decay=wd)
        elif opt_name == "sgd":
            return torch.optim.SGD(trainable, lr=lr, weight_decay=wd, momentum=0.9)
        raise ValueError(f"Unknown optimizer: {opt_name}")

    def _build_scheduler(self, cfg: Dict):
        total_steps = max(1, self.epochs * max(1, len(self.train_loader)) // max(1, self.grad_accumulation))
        warmup_epochs = min(cfg.get("warmup_epochs", 5), max(0, self.epochs // 2))
        warmup = warmup_epochs * max(1, len(self.train_loader)) // max(1, self.grad_accumulation)

        if cfg.get("scheduler") == "cosine" and total_steps > warmup:
            # LambdaLR: linear warmup → cosine decay
            decay_steps = total_steps - warmup
            base_lr = self.optimizer.param_groups[0]["lr"]

            def lr_lambda(step):
                if step < warmup:
                    return (step + 1) / max(1, warmup)
                progress = (step - warmup) / max(1, decay_steps)
                return 0.5 * (1.0 + __import__("math").cos(__import__("math").pi * progress))

            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        return None

    def train(self):
        """Full training loop."""
        best_dsc = 0.0
        global_step = 0
        current_epoch = 0
        self._write_training_status(
            "training",
            epoch=0,
            epochs=self.epochs,
            best_dsc=best_dsc,
        )

        try:
            for epoch in range(1, self.epochs + 1):
                current_epoch = epoch
                if self._cancel_requested():
                    raise TrainingCancelled()
                self._write_training_status(
                    "training",
                    epoch=epoch,
                    epochs=self.epochs,
                    best_dsc=best_dsc,
                    phase="train",
                )
                # Train
                train_metrics = self._train_epoch(epoch, global_step)
                global_step += len(self.train_loader)

                # Validate
                val_metrics = {}
                if self.val_loader is not None:
                    self._write_training_status(
                        "training",
                        epoch=epoch,
                        epochs=self.epochs,
                        best_dsc=best_dsc,
                        phase="val",
                    )
                    val_metrics = self._validate_epoch(epoch)

                # Log
                self._log_epoch(epoch, train_metrics, val_metrics)
                latest_epoch_line = "Epoch {}/{}: train_loss={:.4f}".format(
                    epoch,
                    self.epochs,
                    float(train_metrics.get("loss", 0.0)),
                )
                if val_metrics:
                    latest_epoch_line += ", val_dice={:.4f}".format(float(val_metrics.get("mean_dsc", 0.0)))

                # Save
                val_dsc = val_metrics.get("mean_dsc", train_metrics.get("mean_dsc", 0.0))
                is_best = val_dsc > best_dsc
                if is_best:
                    best_dsc = val_dsc

                metrics = {**train_metrics, **val_metrics}
                lr = self.optimizer.param_groups[0]["lr"]
                self._append_metrics_history({
                    "epoch": epoch,
                    "epochs": self.epochs,
                    "train_loss": float(train_metrics.get("loss", 0.0)),
                    "val_dice": float(val_metrics.get("mean_dsc")) if val_metrics else None,
                    "lr": float(lr),
                    "best_dsc": float(best_dsc),
                    "is_best": bool(is_best),
                    "latest_epoch_line": latest_epoch_line,
                    "metrics": metrics,
                    "updated_at_epoch": time.time(),
                })
                save_checkpoint(
                    self.model, self.optimizer, epoch,
                    metrics,
                    self.config, self.ckpt_dir,
                    filename=f"epoch_{epoch:04d}.pth",
                    is_best=is_best,
                )
                self._write_training_status(
                    "training",
                    epoch=epoch,
                    epochs=self.epochs,
                    best_dsc=best_dsc,
                    metrics=metrics,
                    phase="epoch_complete",
                    latest_epoch_line=latest_epoch_line,
                )

            self._write_training_status(
                "completed",
                epoch=self.epochs,
                epochs=self.epochs,
                best_dsc=best_dsc,
            )
            self._write_metrics_history("completed", best_dsc=best_dsc)
            print(f"\nTraining complete. Best DSC: {best_dsc:.4f}")
        except TrainingCancelled:
            self._write_training_status(
                "cancelled",
                epoch=current_epoch,
                epochs=self.epochs,
                best_dsc=best_dsc,
            )
            self._write_metrics_history("cancelled", best_dsc=best_dsc)
            print("\nTraining cancelled by external request.")
        finally:
            self.writer.close()

    def _train_epoch(self, epoch: int, global_step_start: int) -> Dict:
        self.model.train()
        total_loss = 0.0
        total_dice = 0.0
        total_ce = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs} [Train]")
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            if self._cancel_requested():
                raise TrainingCancelled()
            images = batch["image"].to(self.device, dtype=torch.float32)
            labels = batch["label"].to(self.device)

            # Sub-volume training
            if self.use_sub_volume:
                loss_dict = self._train_step_sub_volume(images, labels)
            else:
                loss_dict = self._train_step(images, labels)

            loss = loss_dict["loss"] / self.grad_accumulation
            loss.backward()

            if (batch_idx + 1) % self.grad_accumulation == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Scheduler step (per effective batch)
                if self.scheduler is not None:
                    self.scheduler.step()

            # Stats
            total_loss += self._to_float(loss_dict["loss"])
            total_dice += self._to_float(loss_dict.get("dice_loss", 0))
            total_ce += self._to_float(loss_dict.get("ce_loss", 0))
            current = batch_idx + 1
            avg_loss = total_loss / current
            avg_dice = total_dice / current
            avg_ce = total_ce / current
            lr = self.optimizer.param_groups[0]["lr"]

            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "lr": f"{lr:.2e}",
            })
            self._maybe_write_training_status(
                "training",
                epoch=epoch,
                epochs=self.epochs,
                phase="train",
                batch=current,
                batches=len(self.train_loader),
                lr=lr,
                metrics={"loss": avg_loss, "dice_loss": avg_dice, "ce_loss": avg_ce},
                force=current == len(self.train_loader),
            )

        n = len(self.train_loader)
        return {"loss": total_loss / n, "dice_loss": total_dice / n, "ce_loss": total_ce / n}

    def _train_step(self, images: torch.Tensor, labels: torch.Tensor) -> Dict:
        with torch.autocast(
            device_type=self.device.type if self.device.type != "mps" else "cpu",
            dtype=self.dtype,
            enabled=self.dtype != torch.float32,
        ):
            pred = self.model(images)
            return self.criterion(pred, labels)

    def _train_step_sub_volume(self, images: torch.Tensor, labels: torch.Tensor) -> Dict:
        """Sub-volume training: split volume, encode/decode per sub-volume, compute global loss."""
        if self.device.type == "mps":
            return self._train_step(images, labels)

        d_sub = self.sub_volume_cfg.get("size", [32, 256, 256])[0]
        B, C, D, H, W = images.shape

        if D <= d_sub:
            return self._train_step(images, labels)

        n_parts = (D + d_sub - 1) // d_sub
        all_preds = []

        for i in range(n_parts):
            d_start = i * d_sub
            d_end = min(d_start + d_sub, D)
            sub_img = images[:, :, d_start:d_end]
            sub_lbl = labels[:, d_start:d_end]

            with torch.autocast(
                device_type=self.device.type if self.device.type != "mps" else "cpu",
                dtype=self.dtype,
                enabled=self.dtype != torch.float32,
            ):
                sub_pred = self.model(sub_img)
            all_preds.append(sub_pred)

        # Concatenate predictions along depth
        full_pred = torch.cat(all_preds, dim=2)
        return self.criterion(full_pred, labels)

    def _validate_epoch(self, epoch: int) -> Dict:
        self.model.eval()
        all_dsc = []

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch}/{self.epochs} [Val]")
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                if self._cancel_requested():
                    raise TrainingCancelled()
                images = batch["image"].to(self.device, dtype=torch.float32)
                labels = batch["label"].to(self.device)

                pred = self.model(images)
                metrics = dice_score(
                    pred, labels,
                    num_classes=self.config["model"]["num_classes"],
                )
                all_dsc.append(metrics["mean"])

                pbar.set_postfix({"dsc": f"{np.mean(all_dsc):.4f}"})
                self._maybe_write_training_status(
                    "training",
                    epoch=epoch,
                    epochs=self.epochs,
                    phase="val",
                    batch=batch_idx + 1,
                    batches=len(self.val_loader),
                    metrics={"mean_dsc": float(np.mean(all_dsc))},
                    force=(batch_idx + 1) == len(self.val_loader),
                )

        mean_dsc = float(np.mean(all_dsc)) if all_dsc else 0.0
        return {"mean_dsc": mean_dsc, "per_class_dsc": [float(a) for a in all_dsc]}

    def _log_epoch(self, epoch: int, train_m: Dict, val_m: Dict):
        for k, v in train_m.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(f"train/{k}", v, epoch)
        for k, v in val_m.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(f"val/{k}", v, epoch)

        lr = self.optimizer.param_groups[0]["lr"]
        self.writer.add_scalar("train/lr", lr, epoch)
        line = "Epoch {}/{}: train_loss={:.4f}, lr={:.2e}".format(
            epoch,
            self.epochs,
            float(train_m.get("loss", 0.0)),
            lr,
        )
        if val_m:
            line += ", val_dice={:.4f}".format(float(val_m.get("mean_dsc", 0.0)))
        print(line, flush=True)

    def resume(self, checkpoint_path: str):
        state = load_checkpoint(self.model, checkpoint_path, self.optimizer, self.device)
        print(f"Resumed from {checkpoint_path} (epoch {state['epoch']})")
        return state["epoch"]
