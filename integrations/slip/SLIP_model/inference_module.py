# Copyright (c) IRCAD France
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree. (GNU GPL v3)

# Standard library imports
import gc
import time
from collections import deque
from typing import List, Tuple

# Third-party imports
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
import torchio as tio
from scipy import ndimage
from torch.nn.utils.rnn import pad_sequence

# Local imports
from SLIP_model.build_SLIP import slip_model_registry3D
from SLIP_model.seg_combiner import OptimizedSegmentationCombiner
from SLIP_model.utils import DownsizeIfLarge, deep_patch, visualize_propagation_with_slicer


class Inference_module:

    def __init__(
        self,
        device='cuda',
        checkpoint='path_to_ckpt',
        max_size=512,  # Max size of the input volume. Not designed to go over 720
        ori_size=False,
        activate_propagation=True,
        keep_largest_component=True,  # Keep only the largest 3D component
        torch_compile=True,
        vis=False,
    ):
        """
        Initialize the inference module.

        Args:
            device: Compute device ('cuda' or 'cpu').
            checkpoint: Path to the model checkpoint (.pth file).
            max_size: Maximum spatial dimension of the input volume before downsampling.
            ori_size: If True, skip axis reordering and keep the original orientation.
            activate_propagation: If True, automatically propagate segmentation to neighboring patches after each click.
            keep_largest_component: If True, retain only the largest connected component in the ground truth.
            torch_compile: If True, compile the model with torch.compile for faster inference.
            vis: If True, display an interactive visualization after each click.
        """

        # Store all configuration parameters
        self.device = device
        self.checkpoint = checkpoint
        self.max_size = max_size

        self.ori_size = ori_size
        self.activate_propagation = activate_propagation
        self.keep_largest_component = keep_largest_component
        self.torch_compile = torch_compile
        self.vis = vis
        # Initialize training state
        self.num_feature_levels = 3

        self.sigmoid_scale_for_mem_enc = 1.0  # scale factor for mask sigmoid prob
        self.sigmoid_bias_for_mem_enc = 0.0  # bias factor for mask sigmoid prob

        self.num_maskmem = 6
        self.action_history = []
        self.number_of_prop = 0
        self.tile_step = 0.9
        self.patch_size = (32, 192, 192)  # Don't change
        self.propagated = False
        self.temp_z = torch.zeros(self.patch_size, device=self.device)
        self.lrm = torch.nn.functional.interpolate(self.temp_z.unsqueeze(0).unsqueeze(0),
                                                   size=tuple(x // 2 for x in self.patch_size),
                                                   mode="trilinear",
                                                   align_corners=False).squeeze(0).squeeze(0)

        self.high_res_mask = torch.zeros(self.patch_size).to(self.device)
        self.prev_mask = torch.zeros(self.patch_size).to(self.device)
        self.void = torch.tensor(0.0, device=self.device)

        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float16
        self.build_model()

        # Store transformation parameters for coordinate conversion
        self.original_shape = None
        self.transformed_shape = None
        self.scale_factors = None
        self.axis_permutation = None
        self.inverse_permutation = None
        self.apply_cc_filtering = True

        # Parameters for propagation
        self.STABILITY_THRESHOLD = 0.80
        self.IOU_THRESHOLD = 0.5
        self.MAX_MEMORY_SIZE = 6
        self.MAX_INIT_MEMORY = 3

        # Undo
        self.bbox_reverse_patches = []

        self.update_reverse_patches(self.device)

    # ═══════════════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ═══════════════════════════════════════════════════════════════════════════

    def process_image(self, image_path, gt_path, target_label, precomputed_gt_binary=None, numpy_image=None):
        """
        Load and preprocess a volume for inference.

        Handles ground truth loading (or creation of an empty mask when gt_path
        is None), optional largest-component filtering, axis reordering so the
        smallest dimension is first, TorchIO-based clamping and Z-normalisation,
        sliding-window patch extraction, and batch embedding computation.
        After this call, self.bbox_data, self.gt3D, self.vol and the coordinate
        transformation parameters are ready for click_inference.

        Args:
            image_path: Path to the input NIfTI image (.nii.gz).
            gt_path: Path to the ground truth NIfTI label map, or None to skip GT.
            target_label: Integer label value to binarise from the GT, or None to
                          auto-select the first non-zero label.
            precomputed_gt_binary: Optional pre-binarised GT numpy array (float32)
                                   that bypasses NIfTI loading entirely.
            numpy_image: Optional numpy array (Z, H, W) to use as the input image,
                         bypassing NIfTI file loading entirely.
        """

        # Load input image
        if numpy_image is not None:
            img = numpy_image.astype(np.float32)[None]  # Shape: (1, Z, H, W)
            print(f"Using provided numpy image, shape: {img.shape}")
        else:
            input_image = sitk.ReadImage(str(image_path))
            img = sitk.GetArrayFromImage(input_image)[None]  # Shape: (1, Z, H, W)

        # ── Ground truth ──────────────────────────────────────────────────────
        if precomputed_gt_binary is not None:
            # 4D NIfTI path: caller already extracted the binary channel
            gt_mask_binary = precomputed_gt_binary.astype(np.float32)
            selected_label = target_label
            print(
                f"Using precomputed binary mask for label {target_label}, "
                f"shape: {gt_mask_binary.shape}, "
                f"voxels: {int(gt_mask_binary.sum())}",
                flush=True)
        elif gt_path is None:
            # No ground truth provided: create an empty mask the size of the image
            gt_mask_binary = np.zeros(img.shape[1:], dtype=np.float32)  # (Z, H, W)
            selected_label = target_label if target_label is not None else 0
            print(f"No ground truth provided, using empty mask of shape: {gt_mask_binary.shape}")
        else:

            # Legacy 3D path: derive binary mask from label value
            gt_image = sitk.ReadImage(str(gt_path))
            gt_mask = sitk.GetArrayFromImage(gt_image)

            unique_labels = np.unique(gt_mask)

            if target_label is None:
                non_zero_labels = unique_labels[unique_labels != 0]
                if len(non_zero_labels) == 0:
                    print(f"No non-zero labels found in {image_path.name}")
                    return [], [], None
                selected_label = non_zero_labels[0]
            else:
                selected_label = target_label

            gt_mask_binary = (gt_mask == selected_label).astype(np.float32)
            print(f"Selected label: {selected_label}")

        if self.keep_largest_component:
            print("Warning: Keeping largest component in GT, SLIP is designed for single object segmentation")
            gt_mask_binary = self.keep_largest_connected_component(gt_mask_binary)

        print(f"Selected label: {selected_label}")
        print(f"Mask shape: {gt_mask_binary.shape}")

        # Skip if no voxels with this label
        if np.sum(gt_mask_binary) == 0 and gt_path is not None:
            print(f"No voxels found for label {selected_label}, skipping...")
            return [], [], None

        # Reorder dimensions so smallest is first
        # Current shape: (1, Z, H, W) for img and (Z, H, W) for gt_mask_binary
        img_spatial_shape = img.shape[1:]  # (Z, H, W)

        # Store original shape for coordinate transformation
        self.original_shape = img_spatial_shape

        # Find the axis with smallest dimension
        smallest_axis = np.argmin(img_spatial_shape)
        # Save BEFORE any permutation — this is the true original space
        self.vol_original = torch.from_numpy(img.astype(np.float32)).float().to(self.device)  # (1, Z, H, W)
        self.gt3D_original = torch.from_numpy(gt_mask_binary[None][None].astype(np.float32)).float().to(
            self.device)  # (1, 1, Z, H, W)

        # Create permutation to move smallest axis to position 0
        # For img: need to account for batch dimension at position 0
        if smallest_axis != 0 and self.ori_size is False:
            # Create permutation for spatial dimensions
            spatial_axes = [0, 1, 2]
            spatial_axes[0], spatial_axes[smallest_axis] = spatial_axes[smallest_axis], spatial_axes[0]

            # Store axis permutation for coordinate transformation
            self.axis_permutation = np.array(spatial_axes)
            # Create inverse permutation
            self.inverse_permutation = np.argsort(self.axis_permutation)

            # Add batch dimension (0) at the beginning
            img_perm = [0] + [ax + 1 for ax in spatial_axes]  # [0, smallest+1, ..., ...]
            gt_perm = spatial_axes  # [smallest, ..., ...]

            # Transpose
            img = np.transpose(img, img_perm)
            gt_mask_binary = np.transpose(gt_mask_binary, gt_perm)

        else:
            # No permutation applied
            self.axis_permutation = None
            self.inverse_permutation = None

        # Convert to tensor
        gt_tensor = torch.from_numpy(gt_mask_binary).float().to(self.device)

        # Create TorchIO subject
        subject = tio.Subject(image=tio.ScalarImage(tensor=torch.from_numpy(img.astype(np.float32)).float()),
                              label=tio.LabelMap(tensor=torch.from_numpy(gt_mask_binary[None]).float()))

        # Store shape before transformation
        shape_before_transform = img.shape[1:]  # (Z, H, W) - skip batch dim

        # Define transformations
        transforms = tio.Compose([
            DownsizeIfLarge(max_size=(self.max_size, self.max_size, self.max_size)),
            tio.Clamp(-1000, 1000),  # Could break the ZNormalization in rare cases ...
            tio.ZNormalization(),
        ])
        # Apply transformations
        transformed = transforms(subject)

        # Extract transformed data
        img_transformed = transformed['image'].data.unsqueeze(0)  # Shape: (1, 1, Z, H, W)
        gt_transformed = transformed['label'].data.unsqueeze(0)  # Shape: (1, 1, Z, H, W)

        # Store final shape and calculate scale factors
        shape_after_transform = img_transformed.shape[2:]  # (Z, H, W) - skip batch and channel dims
        self.transformed_shape = shape_after_transform

        # Calculate scale factors for coordinate transformation
        self.scale_factors = np.array([
            shape_after_transform[0] / shape_before_transform[0], shape_after_transform[1] / shape_before_transform[1],
            shape_after_transform[2] / shape_before_transform[2]
        ],
                                      dtype=np.float32)

        # Convert to tensors on device
        img = img_transformed
        gt_tensor = gt_transformed.to(self.device)

        if img.ndim != 5:
            raise ValueError("Input image must be 5D with shape (1, 1, Z, H, W)")

        self.model.eval()
        self.number_of_prop = 0

        img = img.to(self.device).float()
        gt_tensor = gt_tensor.to(self.device).type(torch.long)
        # store unsqueezed img & GT
        self.gt3D = gt_tensor.clone()
        self.vol = img.clone()
        # Squeeze image & GT
        gt_tensor = gt_tensor.squeeze()
        img = img.squeeze()

        with torch.inference_mode(), torch.amp.autocast(self.device, dtype=self.dtype, enabled=(self.device == "cuda")):
            self.click_points = None
            self.click_labels = None
            size = gt_tensor.shape
            patch_size = self.patch_size  # Adjust as needed
            tile_step_size = self.tile_step  # Adjust as needed
            self.bboxes = self.generate_bounding_boxes(
                size,
                patch_size,
                tile_step_size,
            )
            print("len bbox", len(self.bboxes))
            start_time = time.time()
            self.bbox_data = self.process_patches_with_embeddings(img.to("cuda"), gt_tensor.to("cuda"), self.bboxes)
            print(f"⏱️ process_patches_with_embeddings took: {time.time() - start_time:.2f} seconds")
            self.model.prompt_encoder.to(self.device)
            self.model.mask_decoder.to(self.device)
            self.model.memory_attention.to(self.device)
            self.model.memory_encoder.to(self.device)
            common_patch_sizes = [
                self.patch_size,  # Your common sizes
                # Add other common sizes you use
            ]
            self.combiner = OptimizedSegmentationCombiner(device=self.device, sigma_factor=0.125)
            self.combiner.precompute_common_weights(common_patch_sizes)
        del gt_tensor
        torch.cuda.empty_cache()

    def click_inference(self, clicks, labels):
        """
        Perform interactive segmentation inference from user click inputs.

        Transforms clicks from original image space to processed space,
        runs the interaction loop, maps the predicted mask back to original
        space, accumulates clicks for visualisation, and optionally displays
        the result with an interactive slicer.

        Args:
            clicks: List of click coordinates [[z, y, x], …] in the ORIGINAL
                    image space (before any axis reordering or resizing).
            labels: List of click labels — 1 for include (positive), 0 for exclude (negative).

        Returns:
            dice_scores: Float Dice score of the current prediction against the GT.
            prev_masks_original: Boolean mask tensor in original image space,
                                 shape (1, 1, Z, H, W).
        """

        with torch.inference_mode(), torch.amp.autocast(self.device, dtype=self.dtype, enabled=(self.device == "cuda")):

            # Track interaction time
            interaction_start_time = time.time()
            click_num = len(clicks)
            operation = "include" if labels[-1] == 1 else "exclude"
            predicted_mask = self.interaction(self.gt3D, clicks, labels, click_num=click_num, operation=operation)
            interaction_time = time.time() - interaction_start_time
            print(f"Interaction took: {interaction_time:.2f} seconds")
            predicted_mask_original = self.transform_mask_to_original_space(predicted_mask.float())

            new_clicks = torch.tensor(clicks, dtype=torch.float32).unsqueeze(0)  # (1, N, 3)
            new_labels = torch.tensor(labels, dtype=torch.int32).unsqueeze(0)  # (1, N)
            if not hasattr(self, 'click_points_original') or self.click_points_original is None:
                self.click_points_original = new_clicks
                self.click_labels_original = new_labels
            else:
                self.click_points_original = torch.cat([self.click_points_original, new_clicks], dim=1)
                self.click_labels_original = torch.cat([self.click_labels_original, new_labels], dim=1)

            if self.vis:
                visualize_propagation_with_slicer(prev_masks=predicted_mask_original,
                                                  gt=self.gt3D_original,
                                                  image=self.vol_original,
                                                  clicks=self.click_points_original,
                                                  labels=self.click_labels_original)
            return predicted_mask_original

    def undo(self, action_history, apply_cc_filtering=True):
        """
        Revert the segmentation state to before the last click.

        Pops the most recent entry from the undo stack (bbox_reverse_patches),
        restores all affected patch fields via deep_patch, trims the accumulated
        click tensors (both processed-space and original-space), recomputes the
        combined segmentation from the remaining history, and optionally
        re-displays the updated visualisation.

        Args:
            action_history: The module's action_history list; the last entry is
                            popped in-place.
        """
        if len(action_history) > 1:
            action_history.pop()

        else:
            self.reset()
        if len(self.bbox_reverse_patches) > 0:
            reverse_patch = self.bbox_reverse_patches.pop()
            deep_patch(self.bbox_data, reverse_patch)
        # Pop the last accumulated click/label from tensors
        if self.click_points is not None and self.click_points.shape[1] > 1:
            self.click_points = self.click_points[:, :-1, :]
            self.click_labels = self.click_labels[:, :-1]
        else:
            self.click_points = None
            self.click_labels = None
        # Trim original-space clicks too
        if self.click_points_original is not None and self.click_points_original.shape[1] > 1:
            self.click_points_original = self.click_points_original[:, :-1, :]
            self.click_labels_original = self.click_labels_original[:, :-1]
        else:
            self.click_points_original = None
            self.click_labels_original = None
        included_points = list(filter(lambda x: x["type"] == "include", action_history))
        excluded_points = list(filter(lambda x: x["type"] == "exclude", action_history))

        # Red view (sagittal) - reorder coordinates
        included_coords = [[coord[0], coord[1], coord[2]] for p in included_points for coord in p["coords"]]
        excluded_coords = [[coord[0], coord[1], coord[2]] for p in excluded_points for coord in p["coords"]]
        size = self.gt3D.squeeze(0).squeeze(0).shape

        if len(action_history) < 1 and apply_cc_filtering is True:
            apply_cc_filtering = False
        else:
            apply_cc_filtering = apply_cc_filtering

        prev_masks = self.combiner.combine_segmentations(self.bbox_data,
                                                         size,
                                                         include_coords=included_coords,
                                                         exclude_coords=excluded_coords,
                                                         apply_cc_filtering=apply_cc_filtering)
        prev_masks_formatted = prev_masks > 0

        prev_masks_original = self.transform_mask_to_original_space(
            prev_masks_formatted.unsqueeze(0).unsqueeze(0).float())
        if self.vis:
            visualize_propagation_with_slicer(prev_masks=prev_masks_original,
                                              gt=self.gt3D_original,
                                              image=self.vol_original,
                                              clicks=self.click_points_original,
                                              labels=self.click_labels_original)
        return prev_masks_original

    def reset(self):
        """
        Reset the segmentation state completely.

        Clears all previous masks, click history, undo stack, and propagation
        counters so the module is ready for a fresh segmentation session on the
        same image without reloading embeddings.
        """
        self._reset_previous_masks()
        self.action_history = []
        self.number_of_prop = 0

    # ═══════════════════════════════════════════════════════════════════════════
    # INTERNAL —
    # ═══════════════════════════════════════════════════════════════════════════
    def _reset_previous_masks(self):
        """
        Clear all per-patch segmentation state in bbox_data without reloading embeddings.

        Resets previous_masks, high_res_masks, low_res_masks, points, labels,
        classification outputs, IoU predictions, and memory embeddings for every
        patch. Also clears the undo stack and accumulated click tensors.
        """
        if not hasattr(self, 'bbox_data') or self.bbox_data is None:
            # print("Warning: bbox_data not found or is None")
            return

        # Remove all undo steps
        self.bbox_reverse_patches = []
        self.click_points = None
        self.click_labels = None
        self.click_points_original = None
        self.click_labels_original = None
        for bbox_key, data in self.bbox_data.items():
            if 'previous_masks' in data:
                # Create zeros tensor with the target shape on CPU
                data['previous_masks'] = None
                data['high_res_masks'] = None
                data['low_res_masks'] = None

            data['points'] = None
            data['reprop'] = False
            data['labels'] = None
            data['classif'] = None
            data['iou_pred'] = None
            data['embed_with_mem'] = None

    def update_reverse_patches(self, device):
        """Move all tensors stored in bbox_reverse_patches to the given device."""
        for i in range(len(self.bbox_reverse_patches)):
            for bbkey in self.bbox_reverse_patches[i]:
                for key in self.bbox_reverse_patches[i][bbkey]:
                    value = self.bbox_reverse_patches[i][bbkey][key]
                    if torch.is_tensor(value):
                        self.bbox_reverse_patches[i][bbkey][key] = self.bbox_reverse_patches[i][bbkey][key].to(device)

    def backup_bbox_data(self, bbox_key, value_key, defaultValue):
        """
        Save the current value of a field in bbox_data to the undo stack.

        If the field exists in bbox_data, its current value is stored so it can
        be restored by undo(). If the field does not exist, defaultValue is stored instead.

        Args:
            bbox_key: Key identifying the patch in bbox_data.
            value_key: Field name within the patch dict to back up.
            defaultValue: Fallback value to store if the field is absent.
        """

        if bbox_key not in self.bbox_reverse_patches[-1]:
            self.bbox_reverse_patches[-1][bbox_key] = {}

        # If we currently have a value in bbox_data
        if value_key in self.bbox_data[bbox_key]:

            # If we have a tensor, we perform a deep copy of the current value
            self.bbox_reverse_patches[-1][bbox_key][value_key] = \
                self.bbox_data[bbox_key][value_key]

        # If no value, then revert to the default provided
        else:
            self.bbox_reverse_patches[-1][bbox_key][value_key] = defaultValue

    def transform_coordinates_to_processed_space(self, clicks):
        """
        Transform user click coordinates from original image space to processed space.
        Handles both axis reordering and resizing.

        Args:
            clicks: List of [z, x, y] coordinates in original space or tensor

        Returns:
            Transformed coordinates in processed space
        """
        if self.original_shape is None or self.transformed_shape is None:
            raise RuntimeError("Image must be processed first with process_image() before transforming coordinates")

        # Convert to numpy for easier manipulation
        if isinstance(clicks, torch.Tensor):
            clicks_np = clicks.cpu().numpy()
        else:
            clicks_np = np.array(clicks, dtype=np.float32)

        original_shape = clicks_np.shape
        if len(original_shape) == 2:  # [[z, x, y], ...]
            clicks_np = clicks_np.reshape(-1, 3)
        elif len(original_shape) == 3:  # [batch, num_points, 3]
            clicks_np = clicks_np.reshape(-1, 3)

        transformed_clicks = []

        for click in clicks_np:
            z, x, y = click

            # Step 1: Apply axis permutation if it was done
            if self.axis_permutation is not None:
                coords = np.array([z, x, y])
                # Apply the permutation
                coords = coords[self.axis_permutation]
                z, x, y = coords

            # Step 2: Apply scaling if resizing was done
            if self.scale_factors is not None:
                z = z * self.scale_factors[0]
                x = x * self.scale_factors[1]
                y = y * self.scale_factors[2]

            # Clamp to valid range
            z = np.clip(z, 0, self.transformed_shape[0] - 1)
            x = np.clip(x, 0, self.transformed_shape[1] - 1)
            y = np.clip(y, 0, self.transformed_shape[2] - 1)

            transformed_clicks.append([z, x, y])

        transformed_clicks = np.array(transformed_clicks, dtype=np.float32)

        # Restore original shape
        if len(original_shape) == 2:
            pass  # Already correct shape
        elif len(original_shape) == 3:
            transformed_clicks = transformed_clicks.reshape(original_shape)

        return transformed_clicks

    def batch_forward(self, image_embedding, low_res_masks, high_res_features, points=None):
        """
        Run a single forward pass through the prompt encoder and mask decoder.

        If no points are provided, a dummy null prompt (label -1) is used so the
        decoder still produces a valid output (useful during propagation).

        Args:
            model: The SLIP model instance (self.model is always used internally).
            image_embedding: Image feature map, shape (B, C, D, H, W).
            low_res_masks: Previous low-resolution mask logits, shape (B, 1, D/2, H/2, W/2).
            high_res_features: List of high-resolution feature maps from the encoder FPN.
            points: Optional tuple (coords, labels) where coords is (B, N, 3) and labels is (B, N).

        Returns:
            low_res_masks: Updated low-resolution mask logits (B, 1, D/2, H/2, W/2).
            ious: Predicted IoU scores (B,).
            object_score_logits: Object presence logits (B, 1).
        """
        if points is None:
            B = image_embedding.size(0)
            sam_point_coords = torch.zeros(B, 1, 3, device=self.device)
            sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=self.device)
            points = (sam_point_coords, sam_point_labels)

        sparse_embeddings, dense_embeddings, point_pe = self.model.prompt_encoder(
            embeddings=image_embedding,
            points=points,
            boxes=None,
            masks=low_res_masks,
        )
        low_res_masks, ious, sam_output_tokens, object_score_logits = self.model.mask_decoder(
            image_embeddings=image_embedding.to(self.device),
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            sparse_pe=point_pe,
            multimask_output=False,
            high_res_features=high_res_features,
        )
        is_obj_appearing = object_score_logits > 0
        low_res_masks = torch.where(is_obj_appearing[:, None, None, None], low_res_masks, -1024)
        return low_res_masks, ious, object_score_logits

    def build_model(self):
        """
        Instantiate the SLIP model, load the checkpoint weights, and optionally
        compile the model with torch.compile and run a comprehensive warm-up pass
        to pre-compile all CUDA kernels before inference.
        """

        self.model = slip_model_registry3D["slip"](192, 32, checkpoint=None).to(self.device)
        self.hidden_dim = self.model.image_encoder.neck.d_model
        # Load checkpoint for testing
        if self.checkpoint:
            checkpoint_data = torch.load(self.checkpoint, map_location=self.device, weights_only=False)

            self.model.load_state_dict(checkpoint_data['model_state_dict'], strict=False)
            print(f"Loaded checkpoint from {self.checkpoint}")

        try:
            if torch.cuda.is_available() and self.device != "cpu":
                if self.torch_compile:
                    self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=False)

                # COMPREHENSIVE WARM-UP: Exercise all inference paths
                print("Running comprehensive warm-up to compile all CUDA kernels...")
                self.warmup_model()
                print("Warm-up complete - all kernels compiled")

                print("Done")
        except Exception as e:
            print(f"torch.compile failed: {e}")

    def interaction(self, gt3D, clicks, labels, click_num, operation):
        """
        Core interaction loop: transforms clicks, runs patch inference, triggers
        propagation, combines all patch segmentations into a full-volume mask, and
        computes the Dice score against the ground truth.

        Args:
            gt3D: Ground truth tensor, shape (1, 1, Z, H, W).
            clicks: List of click coordinates [[z, x, y], …] in ORIGINAL image space.
            labels: List of click labels (1 = include, 0 = exclude).
            click_num: Current click count, used for logging in action_history.
            operation: String tag for the action ('include' or 'exclude').

        Returns:
            predicted_mask: Combined full-volume prediction, shape (1, 1, Z, H, W).
        """
        self.click_metrics = []
        self.bbox_reverse_patches.append(dict())

        # Transform clicks from original space to processed space
        clicks_transformed = self.transform_coordinates_to_processed_space(clicks)

        # Convert clicks and labels to tensors
        if not isinstance(clicks_transformed, torch.Tensor):
            clicks_tensor = torch.tensor(clicks_transformed, dtype=torch.float32, device=self.device)
        else:
            clicks_tensor = clicks_transformed.to(self.device)

        if not isinstance(labels, torch.Tensor):
            labels_tensor = torch.tensor(labels, dtype=torch.int32, device=self.device)
        else:
            labels_tensor = labels.to(self.device)

        # Ensure proper shape: [batch_size, num_points, 3] for clicks
        if clicks_tensor.dim() == 2:
            clicks_tensor = clicks_tensor.unsqueeze(0)  # Add batch dimension

        # Ensure proper shape: [batch_size, num_points] for labels
        if labels_tensor.dim() == 1:
            labels_tensor = labels_tensor.unsqueeze(0)  # Add batch dimension

        # Keep only the NEW click for bbox processing
        new_clicks_tensor = clicks_tensor
        new_labels_tensor = labels_tensor

        # Accumulate clicks for include/exclude extraction (combine_segmentations)
        if not hasattr(self, 'click_points') or self.click_points is None:
            self.click_points = new_clicks_tensor
            self.click_labels = new_labels_tensor
        else:
            self.click_points = torch.cat([self.click_points, new_clicks_tensor], dim=1)
            self.click_labels = torch.cat([self.click_labels, new_labels_tensor], dim=1)
        # Full accumulated clicks for combine_segmentations include/exclude
        all_clicks_tensor = self.click_points
        all_labels_tensor = self.click_labels
        # Extract include/exclude points from FULL history for combine_segmentations
        include_points = [[
            all_clicks_tensor[0, i, 0].item(), all_clicks_tensor[0, i, 1].item(), all_clicks_tensor[0, i, 2].item()
        ] for i in range(all_clicks_tensor.shape[1]) if all_labels_tensor[0, i].item() == 1]

        exclude_points = [[
            all_clicks_tensor[0, i, 0].item(), all_clicks_tensor[0, i, 1].item(), all_clicks_tensor[0, i, 2].item()
        ] for i in range(all_clicks_tensor.shape[1]) if all_labels_tensor[0, i].item() == 0]
        self.action_history.append({
            "name": click_num,
            "coords": clicks,
            "type": operation,
        })

        # Find bounding boxes that contain the clicks and get local coordinates
        bbox_keys_to_process = self.find_box_and_local_coordinates(new_clicks_tensor, new_labels_tensor)

        # If the click is in multiple patches, they are batched :)
        if bbox_keys_to_process:
            self._process_batch_patches(bbox_keys_to_process)
        else:
            print("No valid bbox keys found for processing")

        # Get the size of the ground truth
        size = gt3D.squeeze(0).squeeze(0).shape

        # Perform propagation if activated
        if self.activate_propagation:
            self.autom_prop()
        # Combine segmentations from all patches
        predicted_mask = self.combiner.combine_segmentations(self.bbox_data,
                                                             size,
                                                             include_coords=include_points,
                                                             exclude_coords=exclude_points,
                                                             apply_cc_filtering=self.propagated)

        predicted_mask = predicted_mask.unsqueeze(0).unsqueeze(0)
        return predicted_mask

    def find_box_and_local_coordinates(self, points, labels):
        """
        Find all bounding-box patches that contain the given click points and
        convert the global coordinates to patch-local coordinates.

        For each click, every patch whose bounding box contains that click is
        identified. The click is stored in patch-local space inside bbox_data,
        and the patch key is added to the returned list so it can be processed.

        Args:
            points: Click coordinates in processed space, shape (B, N, 3) — [z, x, y].
            labels: Click labels, shape (B, N) — 1 for include, 0 for exclude.

        Returns:
            List of bbox keys (strings) whose patches contain at least one click.
        """
        results = []
        batch_size, num_points, _ = points.shape

        for b in range(batch_size):
            for n in range(num_points):
                z, x, y = points[b, n]

                point_label = labels[b, n]
                # point_label = labels
                for key in self.bbox_data:
                    z_range, x_range, y_range = self.bbox_data[key]['bbox_coords']

                    if (x_range[0] <= x < x_range[1] and y_range[0] <= y < y_range[1] and z_range[0] <= z < z_range[1]):

                        local_x = x - x_range[0]
                        local_y = y - y_range[0]
                        local_z = z - z_range[0]
                        local_point = torch.tensor([[[local_z, local_x, local_y]]],
                                                   dtype=points.dtype,
                                                   device=points.device)
                        local_label = point_label.unsqueeze(0).unsqueeze(0)
                        # Set classif to True if this point has label 1
                        if point_label == 1:
                            self.bbox_data[key]['classif'] = True
                        if self.bbox_data[key]['points'] is None:
                            self.backup_bbox_data(key, 'points', None)
                            self.backup_bbox_data(key, 'labels', None)

                            self.bbox_data[key]['points'] = local_point
                            self.bbox_data[key]['labels'] = local_label

                        else:
                            self.backup_bbox_data(key, 'points', None)
                            self.backup_bbox_data(key, 'labels', None)

                            self.bbox_data[key]['points'] = torch.cat([self.bbox_data[key]['points'], local_point],
                                                                      dim=1)
                            self.bbox_data[key]['labels'] = torch.cat([self.bbox_data[key]['labels'], local_label],
                                                                      dim=1)

                        results.append(key)

        return results

    def _process_batch_patches(self, bbox_keys_to_process, batch_size=4):
        """
        Split the list of clicked patch keys into mini-batches and process each
        batch through _prepare_batch_data → _process_single_batch.

        Args:
            bbox_keys_to_process: List of bbox keys whose patches need inference.
            batch_size: Maximum number of patches to process in a single forward pass.
        """
        if not bbox_keys_to_process:
            return
        batch_size = min(batch_size, len(bbox_keys_to_process))
        # Process in batches
        for i in range(0, len(bbox_keys_to_process), batch_size):
            batch_keys = bbox_keys_to_process[i:i + batch_size]

            # Prepare batch data
            batch_data = self._prepare_batch_data(batch_keys)

            if batch_data:
                self._process_single_batch(batch_data, batch_keys)

    def _prepare_batch_data(self, batch_keys):
        """
        Collect and batch all tensors required for a forward pass from the given patch keys.

        Handles padding of variable-length point sequences, batching of image
        embeddings and memory-conditioned embeddings, and stacking multi-level
        high-resolution feature maps.

        Args:
            batch_keys: List of bbox keys to include in this batch.

        Returns:
            Dictionary with batched tensors ready for _process_single_batch, or
            None if batch_keys is empty.
        """
        if not batch_keys:
            return None

        # Collect data from all patches in batch
        image_embeddings = []
        image_embeddings_with_mem = []
        high_res_features_lists = []  # List of lists
        prev_masks = []
        low_res_masks = []
        points_batch = []
        labels_batch = []
        points_list = []
        labels_list = []
        for bbox_key in batch_keys:
            data = self.bbox_data[bbox_key]
            image_embeddings_patch = data['embeddings'].to(self.device)
            embeddings_with_mem = data['embed_with_mem']

            if embeddings_with_mem is not None:
                data["embed_with_mem"] = embeddings_with_mem
            else:
                data["embed_with_mem"] = image_embeddings_patch
                embeddings_with_mem = image_embeddings_patch
            # Standard tensors - can be batched normally
            image_embeddings.append(image_embeddings_patch)
            image_embeddings_with_mem.append(embeddings_with_mem)
            self.bbox_data[bbox_key]['embeddings'] = image_embeddings_patch

            prev_mask = data['previous_masks'] if data['previous_masks'] is not None else self.prev_mask
            prev_masks.append(prev_mask.unsqueeze(0).unsqueeze(0))
            prev_mask = data['low_res_masks'] if data['low_res_masks'] is not None else self.lrm
            low_res_masks.append(prev_mask.unsqueeze(0).unsqueeze(0).to(self.device))

            # High-res features - keep as list for now
            high_res_features_patch = [f.to(self.device) for f in data['high_res_features']]
            self.bbox_data[bbox_key]['high_res_features'] = high_res_features_patch
            high_res_features_lists.append(high_res_features_patch)

            # Points and labels
            points_list.append(data['points'].squeeze(0))  # -> (N, 3)
            labels_list.append(data['labels'].squeeze(0))  # -> (N,)

        # Batch standard tensors
        batched_image_embeddings = torch.cat(image_embeddings, dim=0)
        batched_image_embeddings_with_mem = torch.cat(image_embeddings_with_mem, dim=0).to(self.device)
        batched_low_res_masks = torch.cat(low_res_masks, dim=0)
        padded_points = pad_sequence(points_list, batch_first=True, padding_value=0)  # (B, max_N, 3)
        padded_labels = pad_sequence(labels_list, batch_first=True, padding_value=-1)  # (B, max_N)

        # Move to CUDA
        points_batch = padded_points.to(self.device)
        labels_batch = padded_labels.to(self.device)

        # For high-res features: convert from list of lists to list of batched tensors
        # This is the key difference - model expects list of batched tensors
        num_feature_levels = len(high_res_features_lists[0])
        batched_high_res_features = []

        for level_idx in range(num_feature_levels):
            # Collect all features at this level across the batch
            level_features = [features_list[level_idx] for features_list in high_res_features_lists]
            # Stack them into a batched tensor
            batched_level_features = torch.cat(level_features, dim=0)
            batched_high_res_features.append(batched_level_features)

        # Prepare low-res masks

        return {
            'image_embeddings': batched_image_embeddings,
            'image_embeddings_with_mem': batched_image_embeddings_with_mem,
            'high_res_features': batched_high_res_features,  # List of batched tensors
            'low_res_masks': batched_low_res_masks,
            'points': points_batch,
            'labels': labels_batch
        }

    def _update_batch_results(self, low_res_masks_batch, IoU_batch, object_score_logits_batch, batch_keys):
        """
        Write inference results back into bbox_data for each patch in the batch,
        after backing up the previous state for undo support.

        Upsamples low-resolution masks to full patch size and updates stored
        masks, IoU predictions, classification logits, and re-propagation flags.

        Args:
            low_res_masks_batch: Batched low-res mask logits, shape (B, 1, D/2, H/2, W/2).
            IoU_batch: Predicted IoU scores, shape (B,).
            object_score_logits_batch: Object presence logits, shape (B, 1).
            batch_keys: Ordered list of bbox keys corresponding to the batch dimension.
        """
        size1 = self.patch_size

        # Process each result in the batch
        for i, bbox_key in enumerate(batch_keys):

            # Record the previous state before applying the patch update
            self.backup_bbox_data(bbox_key, 'iou_pred', None)
            self.backup_bbox_data(bbox_key, 'object_score_logits', None)
            self.backup_bbox_data(bbox_key, 'obj_logits', None)
            self.backup_bbox_data(bbox_key, 'classif', None)
            self.backup_bbox_data(bbox_key, 'reprop', False)

            self.backup_bbox_data(bbox_key, 'low_res_masks', self.lrm)

            self.backup_bbox_data(bbox_key, 'previous_masks', self.prev_mask)
            self.backup_bbox_data(bbox_key, 'high_res_masks', self.high_res_mask)

            # Extract individual mask from batch
            individual_mask = low_res_masks_batch[i:i + 1]  # Keep batch dimension for interpolation
            iou_value = IoU_batch[i]
            object_score_logits_value = object_score_logits_batch[i].detach().item()
            object_score_logits_value2 = object_score_logits_batch[i]

            new_classif = object_score_logits_value2 > 0

            classif = self.bbox_data[bbox_key].get('classif', None)
            if self.propagated and classif != new_classif.detach().item():
                self.bbox_data[bbox_key]['reprop'] = True
            else:
                self.bbox_data[bbox_key]['reprop'] = False
            # Interpolate to final size
            previous_mask_updated = F.interpolate(individual_mask, size=size1, mode='trilinear')
            self.bbox_data[bbox_key]['iou_pred'] = iou_value
            self.bbox_data[bbox_key]['classif'] = new_classif.detach().item()
            # Update the stored data
            self.bbox_data[bbox_key]['low_res_masks'] = (individual_mask.squeeze(0).squeeze(0))
            self.bbox_data[bbox_key]['object_score_logits'] = (object_score_logits_value)
            self.bbox_data[bbox_key]['obj_logits'] = object_score_logits_value2
            self.bbox_data[bbox_key]['previous_masks'] = (previous_mask_updated.squeeze(0).squeeze(0))
            self.bbox_data[bbox_key]['high_res_masks'] = (previous_mask_updated.squeeze(0).squeeze(0))

    def _process_single_batch(self, batch_data, batch_keys):
        """
        Run inference on a prepared batch of patches.

        Two inference modes:
        - No propagation (CASE 1): single forward pass using raw image embeddings.
        - Post-propagation (CASE 2): dual forward pass — once without memory and
          once with memory-conditioned embeddings — keeping the result with the
          higher IoU for each patch individually.

        Args:
            batch_data: Dictionary of batched tensors from _prepare_batch_data.
            batch_keys: Ordered list of bbox keys for the batch.
        """
        # ===============================
        # CASE 1 — No propagation
        # ===============================
        if not self.propagated:

            masks, IoUs, logits = self.batch_forward(batch_data['image_embeddings'], batch_data['low_res_masks'],
                                                     batch_data['high_res_features'],
                                                     [batch_data['points'], batch_data['labels']])

            self._update_batch_results(masks, IoUs, logits, batch_keys)
            return

        # ===============================
        # CASE 2 — Propagated → double inference
        # ===============================

        # --- Inference WITHOUT memory ---
        masks_no_mem, IoU_no_mem, logits_no_mem = self.batch_forward(batch_data['image_embeddings'],
                                                                     batch_data['low_res_masks'],
                                                                     batch_data['high_res_features'],
                                                                     [batch_data['points'], batch_data['labels']])

        # --- Inference WITH memory ---
        if batch_data['image_embeddings_with_mem'] is not None:
            masks_mem, IoU_mem, logits_mem = self.batch_forward(batch_data['image_embeddings_with_mem'],
                                                                batch_data['low_res_masks'],
                                                                batch_data['high_res_features'],
                                                                [batch_data['points'], batch_data['labels']])

        # ===============================
        # Build final outputs PER PATCH
        # ===============================

        final_masks = []
        final_IoUs = []
        final_logits = []

        for i in range(len(batch_keys)):

            if IoU_mem[i] > IoU_no_mem[i]:
                final_masks.append(masks_mem[i])
                final_IoUs.append(IoU_mem[i])
                final_logits.append(logits_mem[i])
            else:
                final_masks.append(masks_no_mem[i])
                final_IoUs.append(IoU_no_mem[i])
                final_logits.append(logits_no_mem[i])

        # Stack back into batch tensors
        final_masks = torch.stack(final_masks)
        final_IoUs = torch.stack(final_IoUs)
        final_logits = torch.stack(final_logits)

        # Save results normally
        self._update_batch_results(final_masks, final_IoUs, final_logits, batch_keys)

    def autom_prop(self):
        """
        Trigger automatic BFS propagation after a click.

        On the first propagation call (number_of_prop == 0), runs with early-stop
        checking to avoid expanding into empty regions. On subsequent calls, runs
        in re-propagation mode to update patches whose classification changed.
        """
        if self.number_of_prop == 0:
            self.propagate_batched_2(batch_size=1, check=True)
        else:
            self.propagate_batched_2(batch_size=1, check=False, reprop=True)

    def compute_embeddings(self, vol):
        """
        Run the image encoder on a volume patch and return the feature maps.

        Applies the FPN neck convolutions (conv_s0, conv_s1) to the two lowest
        levels, then returns the top-level feature map as the main embedding and
        the lower levels as high-resolution features for the mask decoder.

        Args:
            vol: Input volume tensor, shape (B, 1, D, H, W).

        Returns:
            features: Top-level (lowest-resolution) feature map, shape (B, C, d, h, w).
            high_res_features: List of higher-resolution feature maps for skip connections.
        """

        slip = self.model

        out = slip.image_encoder(vol)

        out["backbone_fpn"][0] = slip.mask_decoder.conv_s0(out["backbone_fpn"][0])
        out["backbone_fpn"][1] = slip.mask_decoder.conv_s1(out["backbone_fpn"][1])

        feature_maps = out["backbone_fpn"][-self.num_feature_levels:]

        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
        feat_sizes = [feat.shape[-3:] for feat in feature_maps]
        high_res_features = [
            x.permute(1, 2, 0).view(x.size(1), x.size(2), *s) for x, s in zip(vision_feats[:-1], feat_sizes[:-1])
        ]

        features = feature_maps[-1]
        return features, high_res_features

    def _prepare_memory_conditioned_features(self, frames_in_mem, mem_pos, image_embedding):
        """
        Fuse the current frame's visual feature map with previous memory embeddings.

        Args:
            frames_in_mem (list[Tensor] or None): Previous memory feature maps, each (B, C, D, H, W)
            mem_pos (list[int] or None): Temporal positions for each memory frame
            image_embedding (Tensor): Current frame’s feature map, shape (B, C, D, H, W)

        Returns:
            Tensor: Memory-conditioned feature map, shape (B, C, D, H, W)
        """
        B, C = image_embedding.size(0), self.hidden_dim
        _, D, H, W = image_embedding.shape[1:]  # (C, D, H, W)
        curr_embeds = image_embedding.flatten(2).permute(2, 0, 1)

        if frames_in_mem is None:
            pix_feat_with_mem = curr_embeds + self.model.no_mem_pos_enc
            pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, D, H, W)
            return pix_feat_with_mem

        memory_embeddings = [f.to(self.device).flatten(2).permute(2, 0, 1) for f in frames_in_mem]

        memory = torch.cat(memory_embeddings, dim=0)

        # --- Prepare temporal encodings ---
        memory_pos_embed_list = []
        for idx, (mem_frame, pos) in enumerate(zip(memory_embeddings, mem_pos)):
            S, B, C = mem_frame.shape
            # Temporal encoding (maskmem_enc) stays as-is
            temp_enc = self.model.maskmem_enc[self.num_maskmem - pos - 1]  # (hidden_dim,)
            temp_enc = temp_enc.unsqueeze(0).expand(S, B, C)  # broadcast over sequence

            memory_pos_embed_list.append(temp_enc)
        memory_pos_embed = torch.cat(memory_pos_embed_list, dim=0)  # (total_S, B, C)
        curr_spatial_embed = torch.cat(memory_pos_embed_list, dim=0)

        # --- Apply memory attention ---
        num_obj_ptr_tokens = 0

        pix_feat_with_mem = self.model.memory_attention(
            curr=curr_embeds.to(self.device),
            memory=memory.to(self.device),
            curr_pos=curr_spatial_embed.to(self.device),
            memory_pos=memory_pos_embed.to(self.device),
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )

        # --- Reshape output back to (B, C, D, H, W) ---
        B, C, D, H, W = image_embedding.shape  # (C, D, H, W)
        pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, D, H, W)

        return pix_feat_with_mem

    def _encode_new_memory(
        self,
        image_embedding,
        feat_sizes,
        pred_masks_high_res,
        object_score_logits,
    ):
        """
        Encode the current patch's image features and predicted mask into a memory
        feature map that can be attended to by future frames/patches.

        Applies sigmoid scaling/bias, runs the memory encoder, and adds a
        no-object spatial embedding for patches predicted to be empty.

        Args:
            image_embedding: Patch image features, shape (B, C, D, H, W).
            feat_sizes: Spatial dimensions (D, H, W) of the feature map.
            pred_masks_high_res: High-res predicted mask logits, shape (B, 1, D, H, W).
            object_score_logits: Object presence logits, shape (B, 1).

        Returns:
            maskmem_features: Encoded memory feature map, shape (B, C, D, H, W).
        """
        D, H, W = feat_sizes  # top-level (lowest-resolution) feature size
        # top-level feature, (HW)BC => BCHW
        pix_feat = image_embedding
        mask_for_mem = torch.sigmoid(pred_masks_high_res)
        # apply scale and bias terms to the sigmoid probabilities
        if self.sigmoid_scale_for_mem_enc != 1.0:
            mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
        if self.sigmoid_bias_for_mem_enc != 0.0:
            mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc
        maskmem_out = self.model.memory_encoder(
            pix_feat,
            mask_for_mem,
            skip_mask_sigmoid=True  # sigmoid already applied
        )
        maskmem_features = maskmem_out
        # add a no-object embedding to the spatial memory to indicate that the patch
        # is predicted to be occluded (i.e. no object is appearing in the patch)
        is_obj_appearing = (object_score_logits > 0).float()
        maskmem_features += (1 - is_obj_appearing[..., None, None, None]
                             ) * self.model.no_obj_embed_spatial[..., None, None, None].expand(*maskmem_features.shape)

        return maskmem_features

    def _encode_memory_in_output(
        self,
        image_embedding,
        feat_sizes,
        high_res_masks,
        object_score_logits,
    ):
        """
        Encode the current prediction into memory.

        Calls _encode_new_memory and returns the
        resulting memory feature map for use in future propagation steps.

        Args:
            image_embedding: Current patch image features, shape (B, C, D, H, W).
            feat_sizes: Spatial dimensions (D, H, W) of the feature map.
            high_res_masks: Predicted high-res mask logits, shape (B, 1, D, H, W).
            object_score_logits: Object presence logits, shape (B, 1).

        Returns:
            maskmem_features: Encoded memory feature map, shape (B, C, D, H, W).
        """

        high_res_masks_for_mem_enc = high_res_masks
        maskmem_features = self._encode_new_memory(
            image_embedding=image_embedding.to(self.device),
            feat_sizes=feat_sizes,
            pred_masks_high_res=high_res_masks_for_mem_enc.to(self.device),
            object_score_logits=object_score_logits.to(self.device),
        )

        return maskmem_features

    def _get_6_neighbors(self, coord):
        """Return the 6 face-connected neighbors of a 3D grid coordinate."""
        z, y, x = coord
        return [(z + 1, y, x), (z - 1, y, x), (z, y + 1, x), (z, y - 1, x), (z, y, x + 1), (z, y, x - 1)]

    def _patch_iou_stability(self, patch, device):
        """
        Compute the IoU value and mask stability score for a single patch.

        Stability is defined as 1 minus the fraction of voxels near the decision
        boundary (logit in [-0.5, 0.5]). A higher stability means the mask is
        more confident (fewer uncertain voxels).

        Args:
            patch: Patch data dictionary containing 'high_res_masks' and 'iou_pred'.
            device: Device to move the mask tensor to for computation.

        Returns:
            iou_val: Float IoU prediction value.
            stability: Float stability score in [0, 1].
        """
        high_res = patch['high_res_masks'].to(device)
        iou_pred = patch.get('iou_pred', None)
        iou_val = iou_pred.item() if torch.is_tensor(iou_pred) else float(iou_pred or 0.0)
        uncertain = ((high_res >= -0.5) & (high_res <= 0.5)).sum().item()
        stability = 1.0 - uncertain / high_res.numel()

        return iou_val, stability

    def _passes_quality_gate(self, patch):
        """
        Return True if a patch meets the minimum IoU and stability thresholds
        required to be used as a memory seed or propagation source.
        """
        iou_val, stability = self._patch_iou_stability(patch, self.device)
        return iou_val >= self.IOU_THRESHOLD and stability >= self.STABILITY_THRESHOLD

    ### Propagation Functions ###

    # ── STEP 1 ─────────────────────────────────────────────────────────────────
    def _classify_patches(self):
        """Split bbox_data into positive_patches, prompted_only, to_prop."""
        positive_patches, prompted_only, to_prop = [], [], []

        for bbox_key, patch_data in self.bbox_data.items():
            has_points = patch_data.get('points') is not None
            labels = patch_data.get('labels')
            has_positive_label = False
            if has_points and labels is not None:
                if torch.is_tensor(labels):
                    has_positive_label = (labels == 1).any().item()
                else:
                    has_positive_label = 1 in labels
            if has_points:
                if has_positive_label:
                    positive_patches.append((bbox_key, patch_data))
                else:
                    prompted_only.append((bbox_key, patch_data))
            else:
                to_prop.append((bbox_key, patch_data))

        return positive_patches, prompted_only, to_prop

    # ── STEP 2 ─────────────────────────────────────────────────────────────────
    def _pick_best_seed(self, positive_patches):
        """Return (best_key, best_patch) or (None, None) if no patch passes quality gate."""
        best_key, best_patch, best_score = None, None, -1.0
        for k, p in positive_patches:
            iou_val, stability = self._patch_iou_stability(p, self.device)
            score = 0.6 * iou_val + 0.3 * stability
            if score > best_score:
                best_key, best_patch, best_score = k, p, score

        if best_key is None:
            return None, None

        iou_val, stability = self._patch_iou_stability(best_patch, self.device)
        if iou_val < self.IOU_THRESHOLD or stability < self.STABILITY_THRESHOLD:
            return None, None

        return best_key, best_patch

    # ── STEP 3 ─────────────────────────────────────────────────────────────────
    def _build_memory_keys(self, largest_key, positive_patches, prompted_only, reprop):
        """
        Collect all keys that should seed memory.
        Returns memory_keys list, or None to abort propagation.
        """
        memory_keys = [largest_key]

        if reprop:
            reprop_keys = [k for k, pd in self.bbox_data.items() if pd.get('reprop', False)]
            if not reprop_keys:
                return None  # nothing to re-propagate → abort

            for k in reprop_keys:
                if k == largest_key or k in memory_keys:
                    continue
                if self._passes_quality_gate(self.bbox_data[k]):
                    memory_keys.append(k)

        for k, p in positive_patches:
            if k != largest_key and k not in memory_keys and self._passes_quality_gate(p):
                memory_keys.append(k)

        for k, p in prompted_only:
            mask = p['high_res_masks'].to(self.device)
            if (mask > 0).sum().item() == 0 and k not in memory_keys:
                memory_keys.append(k)

        return memory_keys

    # ── STEP 4 ─────────────────────────────────────────────────────────────────
    def _build_initial_memory(self, memory_keys):
        """Encode up to _MAX_INIT_MEMORY prompted patches into memory structures."""
        batch_embeddings = torch.cat([self.bbox_data[k]['embeddings'].to(self.device) for k in memory_keys], dim=0)
        feat_sizes = batch_embeddings.shape[2:5]
        memory_embeds, mem_pos, mem_types, memory_coords = [], [], [], []

        for idx, k in enumerate(memory_keys):
            if idx >= self.MAX_INIT_MEMORY:
                break
            mask = self.bbox_data[k]['high_res_masks'].to(self.device)
            prev_mask_init = mask.unsqueeze(0).unsqueeze(0)
            embeddings = self.bbox_data[k]['embeddings'].to(self.device)
            object_score_logits = self.bbox_data[k]['obj_logits'].to(self.device)
            is_obj_appearing = object_score_logits > 0

            if not is_obj_appearing:
                prev_mask_init = prev_mask_init.where(is_obj_appearing[:, None, None, None],
                                                      torch.tensor(-1024., device=self.device))

            maskmem_feat = self._encode_memory_in_output(embeddings, feat_sizes, prev_mask_init, object_score_logits)
            memory_embeds.append(maskmem_feat.detach())
            mem_pos.append(0)
            mem_types.append('prompted')

        return memory_embeds, mem_pos, mem_types, memory_coords, feat_sizes

    # ── STEP 5 ─────────────────────────────────────────────────────────────────
    def _seed_bfs_queue(self, memory_keys, grid_to_key, consumed):
        """Seed BFS queue from all memory patch neighbors."""
        queue, in_queue = deque(), set()
        for k in memory_keys:
            coord = tuple(int(round(x)) for x in self.bbox_data[k]['grid_coords'])
            for nb_coord in self._get_6_neighbors(coord):
                nb_key = grid_to_key.get(nb_coord)
                if nb_key and nb_key not in consumed and nb_key not in in_queue:
                    queue.append(nb_key)
                    in_queue.add(nb_key)
        return queue, in_queue

    # ── Memory management ──────────────────────────────────────────────────────
    def _evict_memory(self, memory_embeds, mem_types):
        """Remove the latest propagation memory entry until under MAX_MEMORY_SIZE."""
        while len(memory_embeds) > self.MAX_MEMORY_SIZE:
            propagate_indices = [i for i, t in enumerate(mem_types) if t == 'propagate']
            if not propagate_indices:
                break
            idx = propagate_indices[0]
            del memory_embeds[idx]
            del mem_types[idx]

    def _rebuild_mem_pos(self, mem_types):
        pos, counter = [], 1
        for t in mem_types:
            if t == 'prompted':
                pos.append(0)
            else:
                pos.append(counter)
                counter += 1
        return pos

    # ── Reprop early-stop ──────────────────────────────────────────────────────
    def _check_reprop_classif_changed(self, batch, prior_classifs, is_obj_appearing):
        for i, (k, _) in enumerate(batch):
            if k == "__PAD__":
                continue
            prior, new_val = prior_classifs[i], is_obj_appearing[i]
            if prior is None and new_val is not None and not new_val.item():
                continue
            if prior != new_val:
                return True
        return False

    # ── Store batch results ────────────────────────────────────────────────────
    def _store_batch_results(self, batch, batch_keys_iter, embed_with_mem, prev_masks_filtered, low_res_masks_filtered,
                             ious, is_obj_appearing):
        """
        Persist propagation results for a batch of patches into bbox_data.

        Backs up the previous state of each field for undo support before
        overwriting with the new masks, memory-conditioned embeddings, IoU
        predictions, and classification outputs.

        Args:
            batch: List of (key, patch_data) tuples processed in this batch.
            batch_keys_iter: Ordered list of keys (may include '__PAD__' entries).
            embed_with_mem: Memory-conditioned embeddings, shape (B, C, D, H, W).
            prev_masks_filtered: Upsampled high-res masks, shape (B, 1, D, H, W).
            low_res_masks_filtered: Low-res mask logits, shape (B, 1, D/2, H/2, W/2).
            ious: IoU predictions, shape (B,).
            is_obj_appearing: Boolean object presence flags, shape (B, 1).
        """
        for i, k in enumerate(batch_keys_iter):
            if k == "__PAD__":
                continue
            for field, default in [('previous_masks', self.prev_mask), ('embed_with_mem', None), ('iou_pred', None),
                                   ('low_res_masks', self.lrm), ('high_res_masks', self.high_res_mask),
                                   ('classif', None)]:
                self.backup_bbox_data(k, field, default)

            self.bbox_data[k]['previous_masks'] = prev_masks_filtered[i].squeeze(0)
            self.bbox_data[k]['embed_with_mem'] = embed_with_mem[i].unsqueeze(0).contiguous()
            self.bbox_data[k]['iou_pred'] = ious[i:i + 1]
            self.bbox_data[k]['low_res_masks'] = low_res_masks_filtered[i].squeeze(0)
            self.bbox_data[k]['high_res_masks'] = prev_masks_filtered[i].squeeze(0)
            self.bbox_data[k]['classif'] = is_obj_appearing[i:i + 1]

    # ── Public entry point ─────────────────────────────────────────────────────
    def propagate_batched_2(self,
                            batch_size=1,
                            max_patches=1000,
                            max_consecutive_negatives=15,
                            check=False,
                            reprop=False):
        """
        BFS propagation using grid neighbor lookups.
        See individual _* helpers for each step.
        """
        # STEP 1 — classify patches
        positive_patches, prompted_only, to_prop = self._classify_patches()
        if not to_prop:
            return

        # STEP 2 — pick best seed
        largest_key, _ = self._pick_best_seed(positive_patches)
        if largest_key is None:
            return

        # STEP 3 — build memory keys
        memory_keys = self._build_memory_keys(largest_key, positive_patches, prompted_only, reprop)
        if memory_keys is None:
            return

        # Mark all prompted patches as consumed
        consumed = set(memory_keys)

        # STEP 4 — initial memory tensors
        memory_embeds, mem_pos, mem_types, memory_coords, feat_sizes = self._build_initial_memory(memory_keys)

        # STEP 5 — grid lookup + BFS seed
        candidate_lookup = dict(to_prop)
        grid_to_key = {tuple(int(round(x)) for x in p['grid_coords']): k for k, p in to_prop}
        queue, in_queue = self._seed_bfs_queue(memory_keys, grid_to_key, consumed)

        # STEP 6 — BFS loop
        num_positive = 0
        consecutive_negative_count = 0
        no_classif_change_count = 0
        batch_idx = 0
        processed = 0

        while queue and processed < max_patches:
            batch = []
            while queue and len(batch) < batch_size:
                k = queue.popleft()
                if k in consumed:
                    continue
                batch.append((k, candidate_lookup[k]))
                consumed.add(k)

            if not batch:
                break

            processed += len(batch)
            batch_idx += 1

            batch_keys_iter = [k for k, _ in batch]
            batch_embeddings = torch.cat([p['embeddings'] for _, p in batch], dim=0)
            num_levels = len(batch[0][1]['high_res_features'])
            batched_high_res = [
                torch.cat([p['high_res_features'][lvl] for _, p in batch], dim=0) for lvl in range(num_levels)
            ]

            prior_classifs = [self.bbox_data[k].get('classif', None) for k, _ in batch] if reprop else None

            embed_with_mem = self._prepare_memory_conditioned_features(memory_embeds, mem_pos, batch_embeddings)
            low_res_masks, ious, object_score_logits = self.batch_forward(embed_with_mem,
                                                                          self.lrm.unsqueeze(0).unsqueeze(0),
                                                                          batched_high_res,
                                                                          points=None)

            prev_masks = F.interpolate(low_res_masks,
                                       size=self.temp_z.shape[-3:],
                                       mode='trilinear',
                                       align_corners=False)

            is_obj_appearing = object_score_logits > 0

            # # Reprop early stop
            if reprop:
                changed = self._check_reprop_classif_changed(batch, prior_classifs, is_obj_appearing)
                if changed:
                    no_classif_change_count = 0
                else:
                    no_classif_change_count += 1
                if no_classif_change_count >= 4:
                    break

            # Expand BFS frontier for confirmed patches
            for i, (k, p) in enumerate(batch):
                if k == "__PAD__" or not is_obj_appearing[i].item():
                    continue
                coord = tuple(int(round(x)) for x in p['grid_coords'])
                for nb_coord in self._get_6_neighbors(coord):
                    nb_key = grid_to_key.get(nb_coord)
                    if nb_key and nb_key not in consumed and nb_key not in in_queue:
                        queue.append(nb_key)
                        in_queue.add(nb_key)

            # Consecutive-negative early stop
            num_real = sum(1 for k, _ in batch if k != "__PAD__")
            num_neg = sum(1 for i, (k, _) in enumerate(batch) if k != "__PAD__" and not is_obj_appearing[i].item())
            consecutive_negative_count = consecutive_negative_count + 1 if num_neg == num_real else 0
            if consecutive_negative_count >= max_consecutive_negatives:
                break

            # Check early stop
            if check:
                num_positive += is_obj_appearing.sum().item()
                if num_positive <= 2 and batch_idx == 4:
                    break

            if batch_idx == 6:
                self.propagated = True
                self.number_of_prop += 1

            batch_coord_tensor = torch.stack([
                torch.as_tensor(p['grid_coords'], dtype=torch.float32, device=self.device) for k, p in batch
                if k != "__PAD__"
            ]).mean(dim=0)
            memory_coords.append(batch_coord_tensor)

            # Encode into memory
            maskmem_feat = self._encode_memory_in_output(batch_embeddings, feat_sizes, prev_masks, object_score_logits)
            memory_embeds.append(maskmem_feat)
            mem_types.append('propagate')
            # Store results
            self._store_batch_results(batch, batch_keys_iter, embed_with_mem, prev_masks, low_res_masks, ious,
                                      is_obj_appearing)

            # Memory management
            self._evict_memory(memory_embeds, mem_types)
            mem_pos = self._rebuild_mem_pos(mem_types)

    def extract_3d_patch(self, image3D, bbox):
        """
        Extract a 3D patch from the image based on bounding box coordinates.

        Args:
            image3D: Input 3D image tensor
            bbox: Bounding box coordinates (z_min, z_max, x_min, x_max, y_min, y_max)

        Returns:
            3D patch tensor
        """

        [z_min, z_max], [x_min, x_max], [y_min, y_max] = bbox

        if len(image3D.shape) == 4:
            C, D, H, W = image3D.shape
        elif len(image3D.shape) == 3:
            D, H, W = image3D.shape
            C = None
        else:
            raise ValueError(f"Unsupported image3D shape: {image3D.shape}")

        # Desired patch size
        # Compute valid indices
        z0 = max(z_min, 0)
        z1 = min(z_max, D)
        x0 = max(x_min, 0)
        x1 = min(x_max, H)
        y0 = max(y_min, 0)
        y1 = min(y_max, W)

        # Extract valid slice
        if C is not None:
            patch = image3D[:, z0:z1, x0:x1, y0:y1]
        else:
            patch = image3D[z0:z1, x0:x1, y0:y1]

        # Compute required padding: (left, right) for each dimension
        pad_z = (z0 - z_min, z_max - z1)
        pad_x = (x0 - x_min, x_max - x1)
        pad_y = (y0 - y_min, y_max - y1)

        # F.pad expects (W, H, D) order and doubles (left, right) for each
        pad = [pad_y[0], pad_y[1], pad_x[0], pad_x[1], pad_z[0], pad_z[1]]

        if any(p > 0 for p in pad):
            if C is not None:
                patch = F.pad(patch, pad, mode='constant', value=0)
            else:
                patch = F.pad(patch.unsqueeze(0), pad, mode='constant', value=0)[0]

        return patch

    def get_patch_grid_coordinates(self, bboxes):
        """
        Create a mapping from patch index to 3D grid coordinates.
        This helps identify neighboring patches.
        """
        patch_coords = {}

        # Determine grid dimensions based on bounding boxes
        z_positions = sorted(set([bbox[0][0] for bbox in bboxes]))  # z_min values
        x_positions = sorted(set([bbox[1][0] for bbox in bboxes]))  # x_min values
        y_positions = sorted(set([bbox[2][0] for bbox in bboxes]))  # y_min values

        # Create mapping from position to grid index
        z_to_idx = {pos: i for i, pos in enumerate(z_positions)}
        x_to_idx = {pos: i for i, pos in enumerate(x_positions)}
        y_to_idx = {pos: i for i, pos in enumerate(y_positions)}

        for patch_idx, bbox in enumerate(bboxes):
            z_min, x_min, y_min = bbox[0][0], bbox[1][0], bbox[2][0]
            grid_z = z_to_idx[z_min]
            grid_x = x_to_idx[x_min]
            grid_y = y_to_idx[y_min]

            patch_coords[patch_idx] = (grid_z, grid_x, grid_y)

        return patch_coords

    def _compute_single_patch_embedding(self, patch_info):
        """Compute embedding for a single patch (used for CPU parallel processing)."""
        bbox_key = patch_info['bbox_key']
        patch_gpu = patch_info['patch'].unsqueeze(0).unsqueeze(0).to(self.device)
        image_embedding, high_res_features = self.compute_embeddings(patch_gpu)
        result = {
            'bbox_key': bbox_key,
            'embedding': image_embedding.detach().clone(),
            'high_res_features': [f.detach().clone() for f in high_res_features],
            'gt_patch': patch_info['gt_patch'],
            'grid_coords': patch_info['grid_coords'],
            'bbox': patch_info['bbox'],
        }
        del patch_gpu, image_embedding, high_res_features
        return result

    def process_patches_with_embeddings(self, image3D, gt3D_bool, bboxes, batch_size=4):
        """
        Process patches with optional batching for embedding computation.
        On CPU, uses parallel threads for embedding computation.
        On GPU, uses batched processing.
        """
        bbox_data = {}
        use_cpu = 'cuda' not in str(self.device)

        # Extract all patches first (CPU only)
        print("Extracting patches...")
        patch_coords = self.get_patch_grid_coordinates(bboxes)

        patch_data = []
        for i, bbox in enumerate(bboxes):
            bbox_key = f"bbox{i+1}"
            patch = self.extract_3d_patch(image3D, bbox)
            gt_patch = self.extract_3d_patch(gt3D_bool, bbox)
            grid_coords = patch_coords[i]
            patch_data.append({
                'bbox_key': bbox_key,
                'patch': patch,
                'gt_patch': gt_patch,
                'bbox': bbox,
                'grid_coords': grid_coords
            })

        # ── CPU path: parallel embedding computation ──────────────────────────
        if use_cpu:
            print(f"CPU mode: computing embeddings sequentially for {len(patch_data)} patches...")
            for patch_info in patch_data:
                res = self._compute_single_patch_embedding(patch_info)
                bbox_key = res['bbox_key']
                bbox_data[bbox_key] = {
                    'embeddings': res['embedding'],
                    'iou_pred': None,
                    'embed_with_mem': None,
                    'classif': None,
                    'high_res_features': res['high_res_features'],
                    'previous_masks': None,
                    'points': None,
                    'labels': None,
                    'reprop': False,
                    'patch': None,
                    'gt_patch': res['gt_patch'],
                    'grid_coords': res['grid_coords'],
                    'bbox_coords': res['bbox'],
                    'low_res_masks': None,
                    'high_res_masks': None,
                    'obj_logits': None,
                }
            print("All patches processed successfully (CPU sequential)")
            return bbox_data

        # ── GPU path: batched processing ──────────────────────────────────────
        for i in range(0, len(patch_data), batch_size):
            batch_end = min(i + batch_size, len(patch_data))
            batch_patches = patch_data[i:batch_end]

            try:
                # Stack patches into batch
                patches_gpu = torch.stack([p['patch'].unsqueeze(0).unsqueeze(0) for p in batch_patches
                                           ]).squeeze(1).to(self.device)  # [batch_size, 1, D, H, W]

                # Batch compute embeddings
                batch_embeddings, batch_high_res_features = self.compute_embeddings(patches_gpu)

                # Split batch results and move to CPU
                for j, patch_info in enumerate(batch_patches):
                    bbox_key = patch_info['bbox_key']

                    # Extract individual results from batch
                    individual_embedding = batch_embeddings[j:j + 1].to(self.device)
                    individual_high_res = [f[j:j + 1].to(self.device) for f in batch_high_res_features]
                    # Store data
                    bbox_data[bbox_key] = {
                        'embeddings': individual_embedding,
                        'iou_pred': None,
                        'embed_with_mem': None,
                        'classif': None,
                        'high_res_features': individual_high_res,
                        'previous_masks': None,
                        'points': None,
                        'labels': None,
                        'patch': patch_info['patch'],
                        'gt_patch': patch_info['gt_patch'],
                        'grid_coords': patch_info['grid_coords'],
                        'bbox_coords': patch_info['bbox'],
                        'low_res_masks': None,
                        'high_res_masks': None,
                        'obj_logits': None,
                    }

                # Clean up batch tensors
                del patches_gpu, batch_embeddings
                for f in batch_high_res_features:
                    del f
                del batch_high_res_features

                # Force cleanup after each batch
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            except RuntimeError as e:
                print(f"Batch processing failed ({e}), falling back to sequential for this batch...")

                # Fallback to sequential processing for this batch
                for patch_info in batch_patches:
                    bbox_key = patch_info['bbox_key']
                    print(f"  Processing {bbox_key} individually")
                    grid_coords = patch_info['grid_coords']
                    # Process single patch
                    patch_gpu = patch_info['patch'].unsqueeze(0).unsqueeze(0).to(self.device)

                    image_embedding, high_res_features = self.compute_embeddings(patch_gpu)

                    # Move to CPU
                    image_embedding_cpu = image_embedding.detach().clone().to(self.device)
                    high_res_features_cpu = [f.detach().clone().to(self.device) for f in high_res_features]

                    # Store data
                    bbox_data[bbox_key] = {
                        'embeddings': image_embedding_cpu,
                        'iou_pred': None,
                        'classif': None,
                        'embed_with_mem': None,
                        'high_res_features': high_res_features_cpu,
                        'previous_masks': None,
                        'points': None,
                        'labels': None,
                        'reprop': False,
                        'patch': patch_info['patch'],
                        'gt_patch': patch_info['gt_patch'],
                        'bbox_coords': patch_info['bbox'],
                        'low_res_masks': None,
                        'high_res_masks': None,
                        'grid_coords': patch_info['grid_coords'],
                        'obj_logits': None,
                    }

                    # Cleanup
                    del patch_gpu, image_embedding, high_res_features, image_embedding_cpu, high_res_features_cpu
                    torch.cuda.empty_cache()

            # Clean up patch info for this batch
            for patch_info in batch_patches:
                del patch_info['patch']
            # Force garbage collection after each batch
            gc.collect()

        print("All patches processed successfully")
        return bbox_data

    def compute_steps_for_sliding_window(self, image_size: Tuple[int, ...], tile_size: Tuple[int, ...],
                                         tile_step_size: float) -> List[List[int]]:
        assert np.all([i >= j
                       for i, j in zip(image_size, tile_size)]), "image size must be as large or larger than patch_size"
        assert 0 < tile_step_size <= 1, 'step_size must be larger than 0 and smaller or equal to 1'

        # our step width is patch_size*step_size at most, but can be narrower
        target_step_sizes_in_voxels = [i * tile_step_size for i in tile_size]
        num_steps = [
            int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_step_sizes_in_voxels, tile_size)
        ]
        steps = []

        for dim in range(len(tile_size)):
            # the highest step value for this dimension is
            max_step_value = image_size[dim] - tile_size[dim]
            if num_steps[dim] > 1:
                actual_step_size = max_step_value / (num_steps[dim] - 1)
            else:
                actual_step_size = 99999999999  # does not matter because there is only one step at 0
            steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]
            steps.append(steps_here)
        return steps

    def _run_warmup_passes(self, model, patch_size, device, dtype, hidden_dim):
        """
        Execute one full forward pass through every model component to trigger
        torch.compile kernel compilation before actual inference begins.

        Steps covered: image encoder → prompt encoder → mask decoder →
        memory encoder → memory attention → full pipeline with memory.
        Failures are caught and reported as non-critical warnings.

        Args:
            model: The compiled (or uncompiled) SLIP model.
            patch_size: Tuple (D, H, W) used to create dummy input tensors.
            device: Target device string.
            dtype: Mixed-precision dtype (bfloat16 or float16).
            hidden_dim: Channel dimension of the image encoder output.
        """
        try:
            # =====================================================
            # 1. IMAGE ENCODER WARM-UP
            # =====================================================
            print("  [1/6] Warming up image encoder...")
            dummy_image = torch.randn(1, 1, patch_size[0], patch_size[1], patch_size[2], device=device, dtype=dtype)

            # Trigger image encoder compilation
            image_embeddings, high_res_features = self.compute_embeddings(dummy_image)

            # =====================================================
            # 2. PROMPT ENCODER WARM-UP
            # =====================================================
            print("  [2/6] Warming up prompt encoder...")

            # Dummy prompts
            dummy_points = torch.zeros((1, 1, 3), device=device)
            dummy_labels = torch.ones((1, 1), dtype=torch.int32, device=device)
            dummy_low_res_mask = torch.zeros((1, 1, patch_size[0] // 2, patch_size[1] // 2, patch_size[2] // 2),
                                             device=device,
                                             dtype=dtype)

            # Trigger prompt encoder compilation
            sparse_embeddings, dense_embeddings, point_pe = model.prompt_encoder(
                embeddings=image_embeddings,
                points=[dummy_points, dummy_labels],
                boxes=None,
                masks=dummy_low_res_mask,
            )

            # =====================================================
            # 3. MASK DECODER WARM-UP
            # =====================================================
            print("  [3/6] Warming up mask decoder...")

            # Trigger mask decoder compilation
            low_res_masks, ious, sam_output_tokens, object_score_logits = model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                sparse_pe=point_pe,
                multimask_output=False,
                high_res_features=high_res_features,
            )

            # =====================================================
            # 4. MEMORY ENCODER WARM-UP
            # =====================================================
            print("  [4/6] Warming up memory encoder...")

            # Simulate mask for memory encoding
            dummy_high_res_mask = F.interpolate(low_res_masks, size=patch_size, mode='trilinear', align_corners=False)

            # Apply sigmoid for memory encoding
            mask_for_mem = torch.sigmoid(dummy_high_res_mask)

            # Trigger memory encoder compilation
            pix_feat = image_embeddings
            maskmem_features = model.memory_encoder(pix_feat, mask_for_mem, skip_mask_sigmoid=True)
            # =====================================================
            # 5. MEMORY ATTENTION WARM-UP
            # =====================================================
            print("  [5/6] Warming up memory attention...")

            # Prepare memory inputs
            B = image_embeddings.size(0)
            C = hidden_dim  # Use self.hidden_dim from your init
            _, D, H, W = image_embeddings.shape[1:]

            curr_embeds = image_embeddings.flatten(2).permute(2, 0, 1)  # (DHW, B, C)
            memory = maskmem_features.flatten(2).permute(2, 0, 1)  # (DHW, B, C)

            # FIX: maskmem_enc has shape (num_maskmem, 1, temp_enc_dim)
            # We need to extract the actual embedding and reshape correctly
            maskmem_enc_single = model.maskmem_enc[0]  # Shape: (1, 1, 64) or (1, 64) or (64,)

            # Ensure we have the right shape: (temp_enc_dim,)
            if maskmem_enc_single.dim() == 3:
                temp_enc_dim = maskmem_enc_single.shape[2]
                maskmem_enc_single = maskmem_enc_single.squeeze(0).squeeze(0)  # (64,)
            elif maskmem_enc_single.dim() == 2:
                temp_enc_dim = maskmem_enc_single.shape[1]
                maskmem_enc_single = maskmem_enc_single.squeeze(0)  # (64,)
            else:
                temp_enc_dim = maskmem_enc_single.shape[0]

            # Now expand correctly: (64,) -> (1, 1, 64) -> (DHW, B, 64)
            temp_enc = maskmem_enc_single.unsqueeze(0).unsqueeze(0).expand(memory.shape[0], B, temp_enc_dim)

            # If dimensions don't match hidden_dim, use no_mem_pos_enc instead
            if temp_enc_dim != C:
                # Use the positional encoding that matches the expected dimension
                if hasattr(model, 'no_mem_pos_enc'):
                    # no_mem_pos_enc should have the right dimension
                    temp_enc_base = model.no_mem_pos_enc  # Could be shape: (DHW, 1, C) or (1, 1, 1, 1, C) etc.

                    # Squeeze all singleton dimensions first to get to (seq_len, C) or (C,)
                    temp_enc_base = temp_enc_base.squeeze()

                    if temp_enc_base.dim() == 2:
                        # Shape is (seq_len, C)
                        seq_len, feat_dim = temp_enc_base.shape
                        if seq_len >= memory.shape[0]:
                            # Take what we need and expand batch dimension
                            temp_enc = temp_enc_base[:memory.shape[0]].unsqueeze(1).expand(-1, B, -1)
                        else:
                            # Repeat if we need more positions
                            repeat_factor = (memory.shape[0] + seq_len - 1) // seq_len
                            temp_enc_repeated = temp_enc_base.repeat(repeat_factor, 1)
                            temp_enc = temp_enc_repeated[:memory.shape[0]].unsqueeze(1).expand(-1, B, -1)
                    elif temp_enc_base.dim() == 1:
                        # Shape is (C,) - broadcast to all positions
                        temp_enc = temp_enc_base.unsqueeze(0).unsqueeze(0).expand(memory.shape[0], B, -1)
                    else:
                        # Unexpected shape - fall back to zeros
                        print(f"  ⚠️  Unexpected no_mem_pos_enc shape after squeeze: {temp_enc_base.shape}")
                        temp_enc = torch.zeros(memory.shape[0], B, C, device=device, dtype=dtype)
                else:
                    # Fallback: create zeros with correct dimension
                    print(f"  ⚠️  Dimension mismatch: maskmem_enc has {temp_enc_dim}D, but hidden_dim is {C}D")
                    temp_enc = torch.zeros(memory.shape[0], B, C, device=device, dtype=dtype)

            # Trigger memory attention compilation
            pix_feat_with_mem = model.memory_attention(
                curr=curr_embeds,
                memory=memory,
                curr_pos=temp_enc,
                memory_pos=temp_enc,
                num_obj_ptr_tokens=0,
            )

            # Reshape back to test full pipeline
            pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, D, H, W)
            # =====================================================
            # 6. FULL FORWARD PASS WITH MEMORY
            # =====================================================
            print("  [6/6] Warming up full inference pipeline with memory...")

            # One more pass with memory-conditioned features
            sparse_embeddings2, dense_embeddings2, point_pe2 = model.prompt_encoder(
                embeddings=pix_feat_with_mem,
                points=[dummy_points, dummy_labels],
                boxes=None,
                masks=dummy_low_res_mask,
            )

            _, _, _, _ = model.mask_decoder(
                image_embeddings=pix_feat_with_mem,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings2,
                dense_prompt_embeddings=dense_embeddings2,
                sparse_pe=point_pe2,
                multimask_output=False,
                high_res_features=high_res_features,
            )

            print("  ✓ All components warmed up successfully")
            self.reset()
        except Exception as e:
            print(f"  ⚠ Warm-up failed (non-critical): {e}")
            import traceback
            traceback.print_exc()

        finally:
            # Clean up dummy tensors
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    def warmup_model(self):
        """
        Warm up all model components within the correct AMP context to match
        the precision used during actual inference (bfloat16 or float16 on CUDA,
        plain float32 on CPU). Delegates to _run_warmup_passes.
        """
        # Use same AMP context as actual inference
        with torch.no_grad():
            if "cuda" in self.device:
                with torch.amp.autocast("cuda", dtype=self.dtype, enabled=True):
                    self._run_warmup_passes(self.model, self.patch_size, self.device, self.dtype, self.hidden_dim)

    def get_sliding_window_slicers(self,
                                   image_size: Tuple[int, ...],
                                   patch_size: Tuple[int, ...],
                                   tile_step_size: float,
                                   verbose: bool = True):
        """
        Generate slicers for sliding window approach.
        Args:
            image_size: Size of the input image
            patch_size: Size of each patch/tile
            tile_step_size: Step size as fraction (0 < tile_step_size <= 1)
            verbose: Print debug information
        Returns:
            List of slice tuples for extracting patches
        """
        slicers = []
        if len(patch_size) < len(image_size):
            assert len(patch_size) == len(image_size) - 1, \
                'if patch_size has less entries than image_size, len(patch_size) must be one shorter'
            steps = self.compute_steps_for_sliding_window(image_size[1:], patch_size, tile_step_size)
            if verbose:
                print(f'n_steps {image_size[0] * len(steps[0]) * len(steps[1])}, image size is'
                      f' {image_size}, patch_size {patch_size}, '
                      f'tile_step_size {tile_step_size}\nsteps:\n{steps}')
            for d in range(image_size[0]):
                for sx in steps[0]:
                    for sy in steps[1]:
                        slicers.append(
                            tuple([slice(None), d, *[slice(si, si + ti) for si, ti in zip((sx, sy), patch_size)]]))
        else:
            steps = self.compute_steps_for_sliding_window(image_size, patch_size, tile_step_size)
            if verbose:
                print(f'n_steps {np.prod([len(i) for i in steps])}, image size is {image_size}, '
                      f'patch_size {patch_size}, tile_step_size {tile_step_size}\nsteps:\n{steps}')
            for sx in steps[0]:
                for sy in steps[1]:
                    for sz in steps[2]:
                        slicers.append(
                            tuple([slice(None), *[slice(si, si + ti) for si, ti in zip((sx, sy, sz), patch_size)]]))
        return slicers

    def slicers_to_bboxes(self, slicers):
        """
        Convert slice objects to bounding boxes.
        Args:
            slicers: List of slice tuples from get_sliding_window_slicers
        Returns:
            List of bounding boxes as [(x_min, x_max), (y_min, y_max), (z_min, z_max)]
        """
        bboxes = []
        for slicer in slicers:
            bbox = []
            # Skip the first slice(None) and extract spatial dimensions
            spatial_slices = [s for s in slicer[1:] if isinstance(s, slice)]

            for s in spatial_slices:
                if s.start is not None and s.stop is not None:
                    bbox.append((s.start, s.stop))
                else:
                    # Handle cases where slice might be incomplete
                    bbox.append((0, 1))

            # Ensure we have 3D coordinates
            while len(bbox) < 3:
                bbox.append((0, 1))

            bboxes.append(bbox)
        return bboxes

    def generate_bounding_boxes(self,
                                volume_shape: Tuple[int, ...],
                                patch_size: Tuple[int, ...],
                                tile_step_size: float = 0.5,
                                verbose: bool = False):
        """
        Generate bounding boxes for sliding window patches on a volume.

        Args:
            volume_shape: Shape of the volume (e.g., (128, 256, 256))
            patch_size: Size of each patch (e.g., (64, 64, 64))
            tile_step_size: Step size as fraction (0 < tile_step_size <= 1)
            verbose: Print debug information

        Returns:
            List of bounding boxes as [(x_min, x_max), (y_min, y_max), (z_min, z_max)]
        """
        # Generate slicers
        slicers = self.get_sliding_window_slicers(volume_shape, patch_size, tile_step_size, verbose)

        # Convert to bounding boxes
        bboxes = self.slicers_to_bboxes(slicers)

        if verbose:
            print(f"Generated {len(bboxes)} bounding boxes")
            print(f"First few boxes: {bboxes[:3]}")

        return bboxes

    def keep_largest_connected_component(self, mask):
        """
        Keep only the largest connected component in a binary mask.

        Args:
            mask: Binary mask as numpy array (any shape)

        Returns:
            Binary mask with only the largest connected component
        """
        # Ensure binary mask
        binary_mask = (mask > 0).astype(np.uint8)

        # Label connected components
        labeled_mask, num_features = ndimage.label(binary_mask)

        if num_features == 0:
            print("Warning: No connected components found in mask")
            return binary_mask

        # Find the largest component
        component_sizes = np.bincount(labeled_mask.ravel())
        # Ignore background (label 0)
        component_sizes[0] = 0

        largest_component_label = component_sizes.argmax()

        # Keep only the largest component
        largest_component_mask = (labeled_mask == largest_component_label).astype(np.float32)

        print(f"Found {num_features} connected components")
        print(f"Largest component size: {component_sizes[largest_component_label]} voxels")
        print(f"Original mask size: {np.sum(binary_mask)} voxels")

        return largest_component_mask

    def transform_mask_to_original_space(self, mask):
        """
        Transform predicted mask from processed space back to original image space.
        Handles inverse resizing and inverse axis permutation.

        Args:
            mask: Tensor of shape (1, 1, Z, H, W) in processed space

        Returns:
            Tensor of shape (1, 1, Z, H, W) in original space
        """
        # Step 1: Inverse resize
        if self.scale_factors is not None and not np.allclose(self.scale_factors, 1.0):
            # If axis permutation was applied, original_shape is already permuted,
            # so we resize to shape_before_transform which corresponds to permuted original
            target_size = tuple(int(s / sf) for s, sf in zip(mask.shape[2:], self.scale_factors))
            mask = F.interpolate(mask.float(), size=target_size, mode='trilinear', align_corners=False)

        # Step 2: Inverse axis permutation
        if self.inverse_permutation is not None:
            # mask shape: (1, 1, D, H, W) — only permute spatial dims (2, 3, 4)
            perm = [0, 1] + [2 + self.inverse_permutation[i] for i in range(3)]
            mask = mask.permute(perm).contiguous()

        return mask > 0
