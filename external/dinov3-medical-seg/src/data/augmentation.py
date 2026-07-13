"""Volume augmentation for few-shot medical image segmentation.

Provides intensity and spatial augmentations that are essential for
few-shot generalisation, especially with tiny organs (adrenal gland,
aorta) where the training signal is statistically fragile.

All augmentations are applied consistently to both image and label
(where applicable — intensity transforms only affect the image).
"""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F


class VolumeAugmentation:
    """Apply conservative tensor augmentations to a training item dict.

    Config keys (all under ``augmentation`` in the YAML):

        enabled: bool = False
        flip_probability: float = 0.0
        flip_axes: [1]              # model-space Y axis; never mirror X by default

        # Intensity (image only)
        intensity:
            gamma_range: [0.7, 1.5]
            brightness_range: [0.75, 1.25]
            contrast_range: [0.8, 1.2]
            noise_std: 0.05

        # Spatial (image + label)
        spatial:
            rotation_deg: 10.0        # max in-plane rotation
            scale_range: [0.9, 1.1]   # random scaling
            translation_px: 10        # max translation in pixels
    """

    def __init__(self, config: dict | None = None, seed: int = 42):
        cfg = config or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.rng = random.Random(seed)

        # Flip
        self.flip_probability = float(cfg.get("flip_probability", 0.0))
        self.flip_axes = [int(axis) for axis in (cfg.get("flip_axes", []) or [])]
        if any(axis not in (0, 1, 2) for axis in self.flip_axes):
            raise ValueError("flip_axes use model ZYX indices 0, 1, 2")

        # Intensity
        ic = cfg.get("intensity", {}) or {}
        self.gamma_range = _parse_range(ic.get("gamma_range", [0.7, 1.5]))
        self.brightness_range = _parse_range(ic.get("brightness_range", [0.75, 1.25]))
        self.contrast_range = _parse_range(ic.get("contrast_range", [0.8, 1.2]))
        self.noise_std = float(ic.get("noise_std", 0.05))

        # Spatial
        sc = cfg.get("spatial", {}) or {}
        self.rotation_deg = float(sc.get("rotation_deg", 10.0))
        self.scale_range = _parse_range(sc.get("scale_range", [0.9, 1.1]))
        self.translation_px = int(sc.get("translation_px", 10))

    def __call__(self, item: dict) -> dict:
        if not self.enabled:
            return item

        item = dict(item)
        image = item["image"]   # (1, D, H, W) or (C, D, H, W)
        label = item["label"]   # (D, H, W)

        # 1. Random flip (in-plane)
        if self.flip_axes and self.flip_probability > 0.0 and self.rng.random() < self.flip_probability:
            # Image is (C,Z,Y,X) while label is (Z,Y,X): adding one converts
            # model ZYX index to image tensor index.
            model_axis = self.rng.choice(self.flip_axes)
            axis = model_axis + 1
            image = torch.flip(image, dims=(axis,))
            label = torch.flip(label, dims=(model_axis,))

        # 2. Intensity augmentations (image only, per-channel identical)
        image = self._augment_intensity(image)

        # 3. Spatial augmentations (image + label together)
        image, label = self._augment_spatial(image, label)

        item["image"] = image
        item["label"] = label
        return item

    # ── intensity ──────────────────────────────────────────

    def _augment_intensity(self, image: torch.Tensor) -> torch.Tensor:
        """Apply random gamma, brightness, contrast, and Gaussian noise."""
        # Gamma
        if self.gamma_range[1] > self.gamma_range[0]:
            gamma = self.rng.uniform(*self.gamma_range)
            image = torch.clamp(image, min=1e-8) ** gamma

        # Brightness (additive shift)
        if self.brightness_range[1] > self.brightness_range[0]:
            shift = self.rng.uniform(*self.brightness_range) - 1.0
            image = image + shift

        # Contrast (multiplicative)
        if self.contrast_range[1] > self.contrast_range[0]:
            factor = self.rng.uniform(*self.contrast_range)
            mean_val = image.mean()
            image = mean_val + factor * (image - mean_val)

        # Gaussian noise
        if self.noise_std > 0:
            noise = torch.randn_like(image) * self.noise_std * self.rng.random()
            image = image + noise

        return torch.clamp(image, 0.0, 1.0)

    # ── spatial ────────────────────────────────────────────

    def _augment_spatial(self, image: torch.Tensor, label: torch.Tensor):
        """Apply random in-plane rotation + scaling + translation."""
        do_rotate = self.rotation_deg > 0
        do_scale = self.scale_range[1] > self.scale_range[0]
        do_translate = self.translation_px > 0

        if not (do_rotate or do_scale or do_translate):
            return image, label

        _, D, H, W = image.shape
        # Build 2x3 affine matrix: [a  b  tx]
        #                          [c  d  ty]
        a, b, c, d = 1.0, 0.0, 0.0, 1.0
        tx, ty = 0.0, 0.0

        # Rotation
        if do_rotate:
            angle = self.rng.uniform(-self.rotation_deg, self.rotation_deg)
            angle_rad = angle * 3.14159265 / 180.0
            cos_a, sin_a = torch.cos(torch.tensor(angle_rad)), torch.sin(torch.tensor(angle_rad))
            a, b = float(cos_a), float(-sin_a)
            c, d = float(sin_a), float(cos_a)

        # Scale
        if do_scale:
            scale = self.rng.uniform(*self.scale_range)
            a *= scale
            b *= scale
            c *= scale
            d *= scale

        # Translation (in normalised coords [-1, 1])
        if do_translate:
            tx = self.rng.uniform(-self.translation_px, self.translation_px) / W * 2
            ty = self.rng.uniform(-self.translation_px, self.translation_px) / H * 2

        theta = torch.tensor([[a, b, tx], [c, d, ty]], dtype=torch.float32)  # (2, 3)
        theta_batch = theta.unsqueeze(0).repeat(D, 1, 1)  # (D, 2, 3)

        # Apply to image (each slice independently with same affine). The image
        # is (C, D, H, W); ct_windows / 2.5d give C>1, so warp all C channels
        # with the shared per-slice grid instead of assuming a single channel.
        C = image.shape[0]
        img_t = image.permute(1, 0, 2, 3)  # (D, C, H, W)
        grid = F.affine_grid(theta_batch, (D, C, H, W), align_corners=False)
        img_warped = F.grid_sample(img_t, grid, mode="bilinear",
                                   align_corners=False, padding_mode="border")
        image = img_warped.permute(1, 0, 2, 3)  # back to (C, D, H, W)

        # Apply to label
        lbl_t = label.unsqueeze(1).float()  # (D, 1, H, W)
        lbl_warped = F.grid_sample(lbl_t, grid, mode="nearest",
                                    align_corners=False, padding_mode="border")
        label = lbl_warped.squeeze(1).long()

        return image, label


def _parse_range(val):
    """Parse a two-element range list/tuple."""
    if isinstance(val, (list, tuple)) and len(val) == 2:
        return (float(val[0]), float(val[1]))
    return (0.0, 0.0)
