# Copyright (c) IRCAD France
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree. (GNU GPL v3)

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchio as tio
from matplotlib.widgets import Slider


def visualize_propagation_with_slicer(
    prev_masks,
    gt,
    image,
    clicks=None,
    labels=None,
):
    """
    Interactive 3D slicer to visualize the propagation step with all slices.

    Args:
        prev_masks: Predicted masks [B, 1, D, H, W]
        gt: Ground truth masks [B, 1, D, H, W]
        image: Input image volume [B, C, D, H, W]
        clicks: List of click coordinates [[z1, x1, y1], [z2, x2, y2], ...]
        labels: List of click labels [1, 0, ...] where 1=include, 0=exclude
        object_score_logits: Classification logits [B, 1]
        target_obj: Target classification [B, 1]
        pos: Current position in propagation sequence
        step: Current training step
        threshold: Threshold for binarizing prediction (default: 0.5)
        save_dir: Directory to save visualizations
    """

    # Normalize clicks: handle tensors with batch dim [1, N, 3] → [N, 3], or convert to list
    if clicks is not None:
        if isinstance(clicks, torch.Tensor):
            while clicks.dim() > 2:
                clicks = clicks.squeeze(0)
            clicks = clicks.detach().cpu().tolist()
        if labels is not None and isinstance(labels, torch.Tensor):
            while labels.dim() > 1:
                labels = labels.squeeze(0)
            labels = labels.detach().cpu().tolist()

    # Convert to numpy
    pred = prev_masks[0, 0].detach().cpu().numpy()
    pred_binary = (pred).astype(np.float32)
    gt_np = gt[0, 0].detach().cpu().numpy()

    # Get image (take first channel if multi-channel)
    if image.dim() == 5:
        img_np = image[0, 0].detach().cpu().numpy()
    else:
        img_np = image[0].detach().cpu().numpy()

    # Normalize image to [0, 1] for display
    img_min, img_max = img_np.min(), img_np.max()
    if img_max > img_min:
        img_np = (img_np - img_min) / (img_max - img_min)

    # Get classification info

    num_slices = pred.shape[0]

    # Create figure with sliders
    fig = plt.figure(figsize=(24, 6))

    # Create grid for subplots
    gs = fig.add_gridspec(2, 4, height_ratios=[20, 1], hspace=0.3, wspace=0.3)

    # Create axes for images
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[0, 3])

    # Create axis for slider (spans all columns)
    ax_slider = fig.add_subplot(gs[1, :])

    # Initial slice (middle)
    initial_slice = num_slices // 2

    # Plot initial images
    im1 = ax1.imshow(img_np[initial_slice], cmap='gray')
    ax1.set_title('Input Volume', fontsize=12, fontweight='bold')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046)

    im2 = ax2.imshow(gt_np[initial_slice], cmap='gray', vmin=0, vmax=1)
    ax2.set_title('Ground Truth', fontsize=12, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046)

    im3 = ax3.imshow(pred_binary[initial_slice], cmap='gray', vmin=0, vmax=1)
    ax3.set_title('Prediction', fontsize=12, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046)

    # Overlay on image
    im4_base = ax4.imshow(img_np[initial_slice], cmap='gray', alpha=0.7)
    im4_gt = ax4.imshow(np.ma.masked_where(gt_np[initial_slice] < 0.0, gt_np[initial_slice]),
                        cmap='Greens',
                        alpha=0.6,
                        vmin=0,
                        vmax=1)
    im4_pred = ax4.imshow(np.ma.masked_where(pred_binary[initial_slice] < 0.0, pred_binary[initial_slice]),
                          cmap='Reds',
                          alpha=0.6,
                          vmin=0,
                          vmax=1)
    ax4.set_title('Overlay on Volume\n(GT=Green, Pred=Red)', fontsize=12, fontweight='bold')
    ax4.axis('off')

    # Plot click points if provided
    click_artists = {'ax1': [], 'ax2': [], 'ax3': [], 'ax4': []}
    if clicks is not None and labels is not None:
        for click, label in zip(clicks, labels):
            z, x, y = click
            color = 'lime' if label == 1 else 'red'
            marker_size = 100

            # Only plot on initial slice if z matches
            if z == initial_slice:
                click_artists['ax1'].append(
                    ax1.scatter(y, x, c=color, s=marker_size, marker='o', edgecolors='white', linewidths=2))
                click_artists['ax2'].append(
                    ax2.scatter(y, x, c=color, s=marker_size, marker='o', edgecolors='white', linewidths=2))
                click_artists['ax3'].append(
                    ax3.scatter(y, x, c=color, s=marker_size, marker='o', edgecolors='white', linewidths=2))
                click_artists['ax4'].append(
                    ax4.scatter(y, x, c=color, s=marker_size, marker='o', edgecolors='white', linewidths=2))

    # Create slider
    slider = Slider(ax=ax_slider,
                    label='Slice',
                    valmin=0,
                    valmax=num_slices - 1,
                    valinit=initial_slice,
                    valstep=1,
                    color='skyblue')

    # Update function for slider
    def update(val):
        slice_idx = int(slider.val)

        # Update images
        im1.set_data(img_np[slice_idx])
        im2.set_data(gt_np[slice_idx])
        im3.set_data(pred_binary[slice_idx])

        # Update overlay
        im4_base.set_data(img_np[slice_idx])
        im4_gt.set_data(np.ma.masked_where(gt_np[slice_idx] < 0.0, gt_np[slice_idx]))
        im4_pred.set_data(np.ma.masked_where(pred_binary[slice_idx] < 0.0, pred_binary[slice_idx]))

        # Update click points
        if clicks is not None and labels is not None:
            # Remove old click artists
            for ax_name in click_artists:
                for artist in click_artists[ax_name]:
                    artist.remove()
                click_artists[ax_name].clear()

            # Add new click artists for current slice
            for click, label in zip(clicks, labels):
                z, x, y = click
                if z == slice_idx:
                    color = 'lime' if label == 1 else 'red'
                    marker_size = 100
                    click_artists['ax1'].append(
                        ax1.scatter(y, x, c=color, s=marker_size, marker='o', edgecolors='white', linewidths=2))
                    click_artists['ax2'].append(
                        ax2.scatter(y, x, c=color, s=marker_size, marker='o', edgecolors='white', linewidths=2))
                    click_artists['ax3'].append(
                        ax3.scatter(y, x, c=color, s=marker_size, marker='o', edgecolors='white', linewidths=2))
                    click_artists['ax4'].append(
                        ax4.scatter(y, x, c=color, s=marker_size, marker='o', edgecolors='white', linewidths=2))

        fig.canvas.draw_idle()

    slider.on_changed(update)

    print(f"[VIZ] Interactive mode: Use slider to navigate through {num_slices} slices")

    plt.show()


# Recursively patch a dictionary with an other
def deep_patch(d, patch):
    """
    Recursively patches a dictionary with values from patch.
    Mutates the original dictionary 'd' in place.
    """
    for k, v in patch.items():
        if (k in d and isinstance(d[k], dict) and isinstance(v, dict)):
            deep_patch(d[k], v)
        else:
            d[k] = v
    return d


# Downsizing transform that maintains aspect ratio
class DownsizeIfLarge(tio.Transform):

    def __init__(self, max_size=(512, 512, 512), **kwargs):
        super().__init__(**kwargs)
        self.max_size = max_size

    def apply_transform(self, subject):
        # Get the first image to check dimensions
        first_image = list(subject.get_images_dict().values())[0]
        current_shape = first_image.shape[1:]  # Skip channel dimension

        # Check if any dimension exceeds maximum
        needs_downsizing = any(current > max_dim for current, max_dim in zip(current_shape, self.max_size))

        if not needs_downsizing:
            return subject

        # Calculate scaling factor to maintain aspect ratio
        scale_factors = [
            max_dim / current if current > max_dim else 1.0 for current, max_dim in zip(current_shape, self.max_size)
        ]

        # Use the minimum scale factor to maintain aspect ratio
        scale_factor = min(scale_factors)

        # Calculate new target shape
        new_shape = [int(current * scale_factor) for current in current_shape]

        # Apply resize to maintain aspect ratio
        return tio.Resize(target_shape=new_shape)(subject)


def calculate_dice_score(pred, gt):
    """
    Calculate Dice score between prediction and ground truth.
    Args:
        pred: Predicted mask tensor
        gt: Ground truth mask tensor
    Returns:
        dice_score: Float value between 0 and 1
    """
    pred = (pred.float() > 0).float()
    gt = (gt.float() > 0).float()

    intersection = (pred * gt).sum()
    union = pred.sum() + gt.sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    dice = (2.0 * intersection) / union
    return dice.item()
