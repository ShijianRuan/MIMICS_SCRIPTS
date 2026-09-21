"""
DINOv3 backbone via HuggingFace transformers (local weights).
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import DINOv3ViTBackbone, AutoImageProcessor


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def intermediate_layer_indices(num_hidden_layers: int) -> list[int]:
    """Return four approximately uniform semantic depths for a ViT backbone."""
    depth = int(num_hidden_layers)
    if depth < 4:
        raise ValueError("DINOv3 multi-level extraction requires at least four layers")
    indices = [
        depth // 5,
        depth // 2 - 1,
        (3 * depth) // 4 - 1,
        depth - 1,
    ]
    if indices != sorted(set(indices)) or indices[0] < 0:
        indices = [
            int(round(index * (depth - 1) / 3.0))
            for index in range(4)
        ]
    if indices != sorted(set(indices)) or indices[0] < 0:
        raise ValueError(
            "Could not derive four unique feature layers from depth {}".format(depth)
        )
    return indices


def _local_model_depth(model_path: str) -> int:
    config_path = Path(model_path) / "config.json"
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            return int((json.load(handle) or {}).get("num_hidden_layers") or 0)
    except Exception:
        return 0


def _validated_out_indices(out_indices, num_hidden_layers: int) -> list[int]:
    depth = int(num_hidden_layers)
    indices = (
        intermediate_layer_indices(depth)
        if out_indices is None
        else [int(value) for value in out_indices]
    )
    if not indices:
        raise ValueError("At least one DINOv3 feature layer is required")
    if indices != sorted(set(indices)):
        raise ValueError("DINOv3 feature layer indices must be unique and sorted")
    if indices[0] < 0 or indices[-1] >= depth:
        raise ValueError(
            "DINOv3 feature layers {} are invalid for a {}-layer backbone".format(
                indices, depth
            )
        )
    return indices


def normalize_imagenet(images_2d: torch.Tensor, mean, std) -> torch.Tensor:
    """Normalize 3-channel [0, 1] tensors with ImageNet/processor stats."""
    if images_2d.dim() != 4 or images_2d.shape[1] != 3:
        raise RuntimeError(
            "ImageNet normalization expects a 4D tensor with 3 channels; got {}".format(
                tuple(images_2d.shape)
            )
        )
    mean_tensor = torch.as_tensor(mean, dtype=images_2d.dtype, device=images_2d.device).view(1, 3, 1, 1)
    std_tensor = torch.as_tensor(std, dtype=images_2d.dtype, device=images_2d.device).view(1, 3, 1, 1)
    return (images_2d - mean_tensor) / std_tensor


class DINOv3Backbone(nn.Module):
    """DINOv3 ViT backbone loaded from local HuggingFace-format weights.

    Args:
        model_path: path to local model dir (config.json + model.safetensors)
        out_indices: which transformer layers to extract (0-indexed, 0-11 for ViT-B)
        freeze: freeze backbone params
    """

    def __init__(
        self,
        model_path: str,
        out_indices: list = None,
        freeze: bool = True,
        input_normalization: str = "none",
        image_mean=None,
        image_std=None,
        expected_sha256=None,
    ):
        super().__init__()
        # Resolve relative path to absolute so transformers doesn't
        # misinterpret it as a HuggingFace Hub repo ID.
        self.model_path = os.path.abspath(model_path)
        configured_depth = _local_model_depth(self.model_path)
        if configured_depth <= 0:
            raise RuntimeError(
                "The local DINOv3 model has no readable num_hidden_layers in "
                "{}.".format(Path(self.model_path) / "config.json")
            )
        self.out_indices = _validated_out_indices(out_indices, configured_depth)
        expected_sha256 = str(expected_sha256 or "").strip().lower()
        if expected_sha256:
            weights_path = Path(self.model_path) / "model.safetensors"
            if not weights_path.is_file():
                raise FileNotFoundError(
                    "DINOv3 PyTorch weights were not found: {}".format(weights_path)
                )
            digest = hashlib.sha256()
            with weights_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            actual_sha256 = digest.hexdigest()
            if actual_sha256 != expected_sha256:
                raise RuntimeError(
                    "The configured PyTorch encoder checksum does not match "
                    "model.safetensors. Expected {}, found {}.".format(
                        expected_sha256,
                        actual_sha256,
                    )
                )
        self.model_sha256 = expected_sha256

        # DINOv3 stage names: stem, stage1, stage2, ..., stage12 (1-indexed)
        # transformer layer i → stage{i+1}
        out_features = [f"stage{i+1}" for i in self.out_indices]

        self.backbone = DINOv3ViTBackbone.from_pretrained(
            self.model_path,
            out_features=out_features,
            reshape_hidden_states=True,
        )
        self.processor = AutoImageProcessor.from_pretrained(self.model_path)
        self.input_normalization = str(input_normalization or "none").lower()
        processor_mean = getattr(self.processor, "image_mean", None) or IMAGENET_DEFAULT_MEAN
        processor_std = getattr(self.processor, "image_std", None) or IMAGENET_DEFAULT_STD
        self.image_mean = tuple(float(v) for v in (image_mean or processor_mean))
        self.image_std = tuple(float(v) for v in (image_std or processor_std))

        cfg = self.backbone.config
        self.patch_size = cfg.patch_size
        self.embed_dim = cfg.hidden_size
        self.num_layers = cfg.num_hidden_layers
        self.num_register_tokens = getattr(cfg, "num_register_tokens", 4)
        self.img_size = getattr(cfg, "image_size", 518)
        if int(self.num_layers) != configured_depth:
            raise RuntimeError(
                "DINOv3 config depth changed while loading: expected {}, loaded {}.".format(
                    configured_depth, self.num_layers
                )
            )

        if freeze:
            self.freeze()

    def freeze(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def get_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, images_2d: torch.Tensor) -> list:
        """Extract multi-scale feature maps from 2D images.

        Args:
            images_2d: (B, 3, H, W)

        Returns:
            list of (B, C, H/p, W/p) feature maps
        """
        if self.input_normalization in ("imagenet", "processor", "dinov3"):
            images_2d = normalize_imagenet(images_2d, self.image_mean, self.image_std)
        outputs = self.backbone(images_2d)
        return list(outputs.feature_maps)

    def preprocess(self, images_2d):
        if images_2d.dim() == 3:
            images_2d = images_2d.unsqueeze(0)
        return self.processor(images=images_2d, return_tensors="pt")


class ONNXDINOv3Backbone(nn.Module):
    """Frozen ViT-S/16 feature extractor for offline-compatible parity runs."""

    def __init__(
        self,
        model_path: str,
        out_indices: list = None,
        freeze: bool = True,
        input_normalization: str = "imagenet",
        image_mean=None,
        image_std=None,
        providers=None,
        expected_sha256=None,
        input_size=None,
    ):
        super().__init__()
        if not freeze:
            raise ValueError("The ONNX DINO backend is frozen and cannot be fine-tuned")
        if list(out_indices or [11]) != [11]:
            raise ValueError("The ONNX parity backend exposes only the final DINO feature map")
        path = Path(model_path).resolve()
        if path.is_dir():
            path = path / "model.onnx"
        if not path.is_file():
            raise FileNotFoundError(
                "DINOv3 ViT-S ONNX model was not found: {}".format(path)
            )
        expected_sha256 = str(expected_sha256 or "").strip().lower()
        if expected_sha256:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            actual_sha256 = digest.hexdigest()
            if actual_sha256 != expected_sha256:
                raise RuntimeError(
                    "The configured ONNX encoder checksum does not match the verified "
                    "model. Expected {}, found {}.".format(
                        expected_sha256,
                        actual_sha256,
                    )
                )
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is required for the local ViT-S ONNX encoder. "
                "Install onnxruntime-gpu on Windows or provide HuggingFace PyTorch weights."
            ) from exc
        if os.name == "nt" and hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls()
            except Exception:
                # Session creation below provides the actionable provider/DLL error.
                pass
        available = list(ort.get_available_providers())
        requested = list(providers or [])
        if not requested:
            requested = [
                name
                for name in ("CUDAExecutionProvider", "CPUExecutionProvider")
                if name in available
            ]
        if not requested:
            raise RuntimeError(
                "ONNX Runtime has no usable execution provider; available: {}".format(
                    ", ".join(available)
                )
            )
        self.session = ort.InferenceSession(str(path), providers=requested)
        self.model_path = str(path)
        self.out_indices = [11]
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[-1].name
        self.input_normalization = str(input_normalization or "imagenet").lower()
        self.image_mean = tuple(float(v) for v in (image_mean or IMAGENET_DEFAULT_MEAN))
        self.image_std = tuple(float(v) for v in (image_std or IMAGENET_DEFAULT_STD))
        input_shape = list(self.session.get_inputs()[0].shape)
        if len(input_shape) != 4:
            raise RuntimeError(
                "The ONNX encoder input must be four-dimensional; got {}".format(
                    input_shape
                )
            )
        configured_size = (
            tuple(int(value) for value in input_size)
            if input_size is not None
            else None
        )
        if configured_size is not None and (
            len(configured_size) != 2
            or any(value <= 0 or value % 16 for value in configured_size)
        ):
            raise ValueError(
                "The configured ONNX input size must contain two positive multiples of 16"
            )
        fixed_size = (
            tuple(int(value) for value in input_shape[-2:])
            if all(
                isinstance(value, int) and value > 0
                for value in input_shape[-2:]
            )
            else None
        )
        if fixed_size is not None:
            if configured_size is not None and configured_size != fixed_size:
                raise RuntimeError(
                    "The selected ONNX encoder declares a fixed input size of {}, "
                    "but the training configuration requests {}.".format(
                        fixed_size,
                        configured_size,
                    )
                )
            resolved_size = fixed_size
        elif configured_size is not None:
            resolved_size = configured_size
        else:
            raise RuntimeError(
                "The selected ONNX encoder has dynamic in-plane dimensions. "
                "Set data.img_size explicitly to a pair of multiples of 16."
            )
        self.patch_size = 16
        self.embed_dim = 384
        self.num_layers = 12
        self.num_register_tokens = 4
        self.img_size = resolved_size
        self.execution_providers = list(self.session.get_providers())
        self.model_sha256 = expected_sha256

    def freeze(self):
        return None

    def unfreeze(self):
        raise RuntimeError("The ONNX DINO backend cannot be unfrozen")

    def get_trainable_params(self) -> int:
        return 0

    def get_total_params(self) -> int:
        return 0

    def forward(self, images_2d: torch.Tensor) -> list:
        if images_2d.dim() != 4 or images_2d.shape[1] != 3:
            raise RuntimeError(
                "ONNX DINO input must be (N,3,H,W), got {}".format(
                    tuple(images_2d.shape)
                )
            )
        if tuple(images_2d.shape[-2:]) != tuple(self.img_size):
            raise RuntimeError(
                "The configured ONNX encoder expects {}, but training prepared {}. "
                "Set Image detail to match this encoder.".format(
                    tuple(self.img_size),
                    tuple(images_2d.shape[-2:]),
                )
            )
        values = images_2d
        if self.input_normalization in ("imagenet", "processor", "dinov3"):
            values = normalize_imagenet(values, self.image_mean, self.image_std)
        device = images_2d.device
        array = values.detach().to(device="cpu", dtype=torch.float32).numpy()
        output = self.session.run([self.output_name], {self.input_name: array})[0]
        output = np.asarray(output, dtype=np.float32)
        expected = (
            images_2d.shape[0],
            self.embed_dim,
            images_2d.shape[-2] // self.patch_size,
            images_2d.shape[-1] // self.patch_size,
        )
        if tuple(output.shape) != tuple(expected):
            raise RuntimeError(
                "Unexpected ViT-S ONNX output shape {}; expected {}".format(
                    tuple(output.shape),
                    expected,
                )
            )
        return [torch.from_numpy(output).to(device=device)]
