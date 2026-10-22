# Copyright (c) IRCAD France
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree. (GNU GPL v3)

from typing import Dict, List, Tuple

import cc3d
import cupy as cp
import numpy as np
import torch
from cupyx.scipy import ndimage as ndi_gpu


class OptimizedSegmentationCombiner:
    """
    Optimized GPU-accelerated 3D segmentation combiner for merging overlapping patch segmentations.

    This class handles the combination of multiple overlapping 3D segmentation masks from different
    bounding box regions into a single coherent segmentation. It uses weighted averaging with
    Gaussian weighting favoring center voxels, similar to nnUNet's approach. The class is optimized
    for GPU computation using CUDA and CuPy for efficient 3D connected component analysis.

    Key features:
    - Weighted averaging of overlapping segmentations using Gaussian-like weights
    - GPU-accelerated 3D connected component filtering
    - Weight map caching for memory efficiency
    - Support for connected component filtering based on included/excluded points
    - Stability score computation for segmentation quality assessment

    Args:
        device: PyTorch device for computation (default: CUDA if available, else CPU)
        sigma_factor: Standard deviation factor for Gaussian weighting (default: 0.125)
        patch_size: Default patch dimensions as (D, H, W) tuple (default: (32, 192, 192))
    """

    def __init__(self, device=None, sigma_factor=0.125, patch_size=(32, 192, 192)):
        """
        Initialize the OptimizedSegmentationCombiner.

        Args:
            device: PyTorch device. Defaults to 'cuda:0' if CUDA is available, otherwise 'cpu'
            sigma_factor: Controls the spread of Gaussian weights. Lower values = sharper weighting
            patch_size: Expected patch dimensions, used for initializing temporary tensors
        """
        self.device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.sigma_factor = sigma_factor

        # Cache for precomputed weight maps: stores weight tensors indexed by (D, H, W) dimensions
        self._weight_cache: Dict[Tuple[int, int, int], torch.Tensor] = {}

        # Maximum cache size to prevent memory issues - can be adjusted based on available GPU memory
        self.max_cache_size = 300
        self.patch_size = patch_size

        # Temporary tensors for intermediate computations
        self.temp_z = torch.zeros(self.patch_size, device=self.device)
        self.lrm = torch.nn.functional.interpolate(self.temp_z.unsqueeze(0).unsqueeze(0),
                                                   size=tuple(x // 2 for x in self.patch_size),
                                                   mode="trilinear",
                                                   align_corners=False).squeeze(0).squeeze(0)

        # Reusable mask tensors to avoid repeated allocations
        self.high_res_mask = torch.zeros(self.patch_size).to(self.device)
        self.prev_mask = torch.zeros(self.patch_size).to(self.device)
        self.void = torch.tensor(0.0, device=self.device)

    def get_or_create_weight_map(self, d: int, h: int, w: int) -> torch.Tensor:
        """
        Get precomputed weight map or create and cache it if not exists.

        Args:
            d, h, w: Dimensions of the patch

        Returns:
            3D weight map tensor
        """
        cache_key = (d, h, w)

        # Return cached weight map if available
        if cache_key in self._weight_cache:
            return self._weight_cache[cache_key]

        # Check cache size and clear if necessary
        if len(self._weight_cache) >= self.max_cache_size:
            self._clear_weight_cache()

        # Create new weight map
        weight_map = self._create_center_weight_map(d, h, w)

        # Cache the weight map
        self._weight_cache[cache_key] = weight_map

        return weight_map

    def _create_center_weight_map(self, d: int, h: int, w: int) -> torch.Tensor:
        """
        Create a 3D weight map where center voxels have higher weights.

        Implements Gaussian-like weighting similar to nnUNet's patch overlap handling.
        This reduces boundary artifacts by downweighting predictions near patch edges
        and upweighting predictions toward the center.

        Args:
            d: Depth (z-axis) dimension of the weight map
            h: Height (x-axis) dimension of the weight map
            w: Width (y-axis) dimension of the weight map

        Returns:
            torch.Tensor: 3D weight map of shape (d, h, w) with values in [0.01, 1.0]
        """
        # Create coordinate grids
        z_coords = torch.linspace(0, d - 1, d, device=self.device)
        x_coords = torch.linspace(0, h - 1, h, device=self.device)
        y_coords = torch.linspace(0, w - 1, w, device=self.device)

        # Center coordinates
        z_center = (d - 1) / 2
        x_center = (h - 1) / 2
        y_center = (w - 1) / 2

        # Create meshgrid
        Z, X, Y = torch.meshgrid(z_coords, x_coords, y_coords, indexing='ij')

        # Calculate distances from center, normalized by dimensions
        z_dist = ((Z - z_center) / (d / 2))**2
        x_dist = ((X - x_center) / (h / 2))**2
        y_dist = ((Y - y_center) / (w / 2))**2

        # Combined distance
        distance_sq = z_dist + x_dist + y_dist

        # Gaussian-like weight (higher at center)
        weights = torch.exp(-distance_sq / (2 * self.sigma_factor**2))

        # Ensure minimum weight (avoid complete zeros at edges)
        min_weight = 0.01
        weights = torch.clamp(weights, min=min_weight)

        return weights

    def _clear_weight_cache(self):
        """
        Clear the weight cache to free GPU memory.

        Called automatically when cache size exceeds max_cache_size limit.
        """
        self._weight_cache.clear()

    def precompute_common_weights(self, common_sizes: list):
        """
        Precompute weight maps for commonly used patch sizes.

        This is an optimization technique to avoid creating weight maps during inference.
        Call this once during initialization with all patch sizes you'll use.

        Args:
            common_sizes: List of tuples (d, h, w) representing common patch dimensions

        Example:
            combiner.precompute_common_weights([(32, 192, 192), (64, 192, 192)])
        """

        for d, h, w in common_sizes:
            self.get_or_create_weight_map(d, h, w)

    def get_cache_stats(self):
        """
        Get statistics about the weight cache usage.

        Returns:
            dict: Dictionary containing:
                - 'cache_size': Number of cached weight maps
                - 'cached_dimensions': List of (d, h, w) tuples for cached maps
                - 'memory_usage_mb': Total GPU memory used by cache in MB
        """
        return {
            'cache_size': len(self._weight_cache),
            'cached_dimensions': list(self._weight_cache.keys()),
            'memory_usage_mb': sum(w.element_size() * w.nelement() for w in self._weight_cache.values()) / (1024 * 1024)
        }

    def calculate_stability_score(self, masks: torch.Tensor, mask_threshold: float,
                                  threshold_offset: float) -> torch.Tensor:
        """
        Compute stability score for a 3D volume mask.

        Stability score measures consistency by computing the ratio of intersection to union
        with slight threshold variations. Higher scores indicate more stable segmentations.

        Args:
            masks: 3D segmentation mask tensor
            mask_threshold: Primary threshold for binarization
            threshold_offset: Amount to vary threshold (tests threshold ± offset)

        Returns:
            torch.Tensor: Stability score (ratio between 0 and 1)
        """
        intersections = (masks > (mask_threshold + threshold_offset)).sum(dtype=torch.int32)
        unions = (masks > (mask_threshold - threshold_offset)).sum(dtype=torch.int32)
        epsilon = 1e-8
        return intersections / (unions + epsilon)

    def connected_components_3d(self,
                                binary_mask: torch.Tensor,
                                include_points: List[List[float]] = None,
                                threshold: float = 0.5) -> torch.Tensor:
        """
        Perform 3D connected component filtering entirely on GPU using CuPy.

        Keeps only connected components that contain at least one of the specified include_points.
        This is useful for removing spurious disconnected regions and keeping only components
        associated with user interactions or ground truth annotations.

        GPU implementation uses zero-copy transfers between PyTorch and CuPy for efficiency.

        Args:
            binary_mask: 3D segmentation mask tensor on GPU (D, H, W)
            include_points: List of [z, x, y] coordinates. Keep only components containing
                          at least one of these points. If None or empty, returns binarized mask.
            threshold: Probability threshold for binarizing the mask (default: 0.5)

        Returns:
            torch.Tensor: Filtered 3D binary mask on GPU with shape (D, H, W)

        Raises:
            RuntimeError: If GPU connected component analysis fails (e.g., insufficient VRAM)
        """
        if include_points is None or len(include_points) == 0:
            # No filtering needed
            return (binary_mask > threshold)

        # Binarize mask
        binary = (binary_mask > threshold)

        if (self.device == 'cpu' or self.device == torch.device('cpu')):
            # CPU path: use cc3d
            binary_np = binary.numpy()
            labeled_array, num_features = cc3d.connected_components(binary_np, return_N=True)

            if num_features == 0:
                return torch.zeros_like(binary_mask)

            # Find components to keep (containing include_points)
            components_to_keep = set()
            for point in include_points:
                z, x, y = int(round(point[0])), int(round(point[1])), int(round(point[2]))
                # Check bounds
                if 0 <= z < labeled_array.shape[0] and 0 <= x < labeled_array.shape[1] and 0 <= y < labeled_array.shape[
                        2]:
                    label = int(labeled_array[z, x, y])
                    if label > 0:  # Valid component
                        components_to_keep.add(label)

            if len(components_to_keep) == 0:
                return binary_mask

            keep_labels = np.array(list(components_to_keep))
            filtered_mask_np = np.isin(labeled_array, keep_labels)
            return torch.from_numpy(filtered_mask_np)

        else:
            # GPU path: use CuPy (zero-copy transfers)
            binary_cp = cp.from_dlpack(binary.detach())

            # Perform GPU connected component labeling
            labeled_array, num_features = ndi_gpu.label(binary_cp)

            if num_features == 0:
                return torch.zeros_like(binary_mask)

            # Find components to keep (containing include_points) - stays on GPU
            components_to_keep = set()
            for point in include_points:
                z, x, y = int(round(point[0])), int(round(point[1])), int(round(point[2]))
                # Check bounds
                if 0 <= z < labeled_array.shape[0] and 0 <= x < labeled_array.shape[1] and 0 <= y < labeled_array.shape[
                        2]:
                    # Access single element on GPU and transfer only that value
                    label = int(labeled_array[z, x, y].get())
                    if label > 0:  # Valid component
                        components_to_keep.add(label)

            if len(components_to_keep) == 0:
                return binary_mask

            # Use CuPy's isin for GPU filtering
            keep_labels = cp.array(list(components_to_keep))
            filtered_mask_cp = cp.isin(labeled_array, keep_labels)

            # Convert back to PyTorch tensor (zero-copy on GPU)
            return torch.from_dlpack(filtered_mask_cp)

    def combine_segmentations(self,
                              data_bboxes,
                              original_shape,
                              include_coords,
                              exclude_coords,
                              apply_cc_filtering=True,
                              cc_threshold=0.5):
        """
        Combine segmentations from multiple overlapping bounding boxes into a single volume.

        This is the main method of the class. It performs weighted averaging of all patch
        segmentations, handling overlaps using Gaussian-weighted blending. Optionally applies
        connected component filtering to keep only components associated with user interactions.

        Algorithm:
        1. Initialize zero tensors for combined segmentation and weight accumulation
        2. For each bounding box:
           - Extract and validate the segmentation mask
           - Get precomputed Gaussian weight map
           - Apply 2x weight boost if patch contains user clicks (ground truth emphasis)
           - Add weighted contribution to combined segmentation
           - Accumulate weights for normalization
        3. Normalize by accumulated weights
        4. Optionally filter to keep only components containing include_coords points
        5. Reset patches that lost all segmentation after filtering

        Args:
            data_bboxes: Dictionary mapping bbox keys to data dictionaries. Each data dict contains:
                - 'bbox_coords': ((z_start, z_end), (x_start, x_end), (y_start, y_end))
                - 'previous_masks': 3D segmentation tensor or None
                - 'points': User click points (used to boost weight, can be None)
                - 'classif': Classification score (optional)
                - Other metadata fields
            original_shape: Tuple (D, H, W) of the original volume dimensions
            include_coords: List of [z, x, y] points to include in connected components
            exclude_coords: List of [z, x, y] points to exclude (currently unused)
            apply_cc_filtering: If True, filter to keep only components with include_coords (default: True)
            cc_threshold: Probability threshold for binarizing during CC filtering (default: 0.5)

        Returns:
            torch.Tensor: Combined 3D segmentation mask of shape original_shape

        Note:
            - Patches with user clicks receive 2x weight bonus as ground truth
            - Patches that lose all segmentation after CC filtering have their data reset
            - Handles out-of-bounds patches and shape mismatches gracefully
        """
        device = self.device
        D, H, W = original_shape

        # Initialize combined segmentation and weight accumulator
        combined_segmentation = torch.zeros(original_shape, dtype=torch.float16, device=device)
        weight_accumulator = torch.zeros(original_shape, dtype=torch.float16, device=device)

        # Process each bounding box
        for bbox_key, data in data_bboxes.items():
            bbox_coords = data['bbox_coords']
            box_segmentation = data['previous_masks'] if data['previous_masks'] is not None else self.prev_mask

            # Extract coordinates
            z_range, x_range, y_range = bbox_coords
            z_start, z_end = int(z_range[0]), int(z_range[1])
            x_start, x_end = int(x_range[0]), int(x_range[1])
            y_start, y_end = int(y_range[0]), int(y_range[1])

            patch_d = z_end - z_start
            patch_h = x_end - x_start
            patch_w = y_end - y_start

            # Ensure box_segmentation is on correct device
            if not isinstance(box_segmentation, torch.Tensor):
                box_segmentation = torch.tensor(box_segmentation, device=device)
            else:
                box_segmentation = box_segmentation.to(device)

            # Handle shape mismatches
            if box_segmentation.shape != (patch_d, patch_h, patch_w):
                seg_d, seg_h, seg_w = box_segmentation.shape

                if (seg_d, seg_h, seg_w) != (patch_d, patch_h, patch_w):
                    extract_z_start = max(0, z_start) if z_start >= 0 else 0
                    extract_x_start = max(0, x_start) if x_start >= 0 else 0
                    extract_y_start = max(0, y_start) if y_start >= 0 else 0

                    extract_z_end = min(seg_d, z_end) if z_end <= seg_d else seg_d
                    extract_x_end = min(seg_h, x_end) if x_end <= seg_h else seg_h
                    extract_y_end = min(seg_w, y_end) if y_end <= seg_w else seg_w

                    if (extract_z_end > extract_z_start and extract_x_end > extract_x_start
                            and extract_y_end > extract_y_start):

                        box_segmentation = box_segmentation[extract_z_start:extract_z_end,
                                                            extract_x_start:extract_x_end,
                                                            extract_y_start:extract_y_end]

                        z_start, z_end = extract_z_start, extract_z_end
                        x_start, x_end = extract_x_start, extract_x_end
                        y_start, y_end = extract_y_start, extract_y_end

                        patch_d = z_end - z_start
                        patch_h = x_end - x_start
                        patch_w = y_end - y_start
                    else:
                        continue

                if box_segmentation.shape != (patch_d, patch_h, patch_w):
                    box_segmentation = torch.nn.functional.interpolate(box_segmentation.unsqueeze(0).unsqueeze(0),
                                                                       size=(patch_d, patch_h, patch_w),
                                                                       mode='trilinear',
                                                                       align_corners=False).squeeze(0).squeeze(0)

            # Use precomputed weight map
            weight_map = self.get_or_create_weight_map(patch_d, patch_h, patch_w)
            has_points = data.get('points') is not None
            if has_points:
                # Patches with clicks get 2x weight - they're the ground truth!
                weight_map = weight_map * 2.0
            # Calculate overlapping regions
            orig_z_start, orig_z_end = max(0, z_start), min(D, z_end)
            orig_x_start, orig_x_end = max(0, x_start), min(H, x_end)
            orig_y_start, orig_y_end = max(0, y_start), min(W, y_end)

            patch_z_start = max(0, -z_start)
            patch_x_start = max(0, -x_start)
            patch_y_start = max(0, -y_start)

            patch_z_end = patch_z_start + (orig_z_end - orig_z_start)
            patch_x_end = patch_x_start + (orig_x_end - orig_x_start)
            patch_y_end = patch_y_start + (orig_y_end - orig_y_start)

            # Process overlap
            if (orig_z_end > orig_z_start and orig_x_end > orig_x_start and orig_y_end > orig_y_start
                    and patch_z_end > patch_z_start and patch_x_end > patch_x_start and patch_y_end > patch_y_start):

                overlap_segmentation = box_segmentation[patch_z_start:patch_z_end, patch_x_start:patch_x_end,
                                                        patch_y_start:patch_y_end]

                overlap_weights = weight_map[patch_z_start:patch_z_end, patch_x_start:patch_x_end,
                                             patch_y_start:patch_y_end]

                combined_segmentation[orig_z_start:orig_z_end, orig_x_start:orig_x_end,
                                      orig_y_start:orig_y_end] += overlap_segmentation * overlap_weights

                weight_accumulator[orig_z_start:orig_z_end, orig_x_start:orig_x_end,
                                   orig_y_start:orig_y_end] += overlap_weights
            else:
                print(f"No overlap found for bbox {bbox_key}")

        # Normalize by accumulated weights
        combined_segmentation = (torch.where(weight_accumulator > 0, combined_segmentation / weight_accumulator,
                                             combined_segmentation) > 0)

        # Apply connected component filtering if requested
        if apply_cc_filtering and (len(include_coords) > 0 or len(exclude_coords) > 0):

            try:
                combined_segmentation = self.connected_components_3d(combined_segmentation,
                                                                     include_points=include_coords,
                                                                     threshold=cc_threshold)
            except RuntimeError as e:
                raise RuntimeError(f"Connected component filtering failed: {e}\n"
                                   "Please fix your CuPy installation. Run `cupy.show_config()` for diagnostics, "
                                   "or reinstall CuPy with the correct CUDA version: "
                                   "https://docs.cupy.dev/en/stable/install.html") from e

            # Reset patches where mask is completely removed after filtering
            for bbox_key, data in data_bboxes.items():
                bbox_coords = data['bbox_coords']
                z_range, x_range, y_range = bbox_coords
                z_start, z_end = int(z_range[0]), int(z_range[1])
                x_start, x_end = int(x_range[0]), int(x_range[1])
                y_start, y_end = int(y_range[0]), int(y_range[1])

                # Calculate overlap with volume bounds
                orig_z_start, orig_z_end = max(0, z_start), min(D, z_end)
                orig_x_start, orig_x_end = max(0, x_start), min(H, x_end)
                orig_y_start, orig_y_end = max(0, y_start), min(W, y_end)

                # Check if patch still has any mask after filtering
                patch_mask = combined_segmentation[orig_z_start:orig_z_end, orig_x_start:orig_x_end,
                                                   orig_y_start:orig_y_end]

                if patch_mask.sum() == 0:
                    # Reset this patch
                    data['points'] = None
                    data['labels'] = None
                    data['classif'] = None
                    data['auto_prompts'] = False
                    data['previous_masks'] = None
                    data['high_res_masks'] = None
                    data['low_res_masks'] = None
                    data['reprop'] = False

        return combined_segmentation
