import os
import random
from io import BytesIO
from pathlib import Path
from typing import Literal

import comfy.utils
import cv2
import folder_paths
import numpy as np
import torch
from PIL import Image

from ..log import log
from ..utils import EASINGS, apply_easing, glob_multiple, pil2tensor
from .transform import MTB_TransformImage


def hex_to_rgb(hex_color: str, bgr: bool = False):
    hex_color = hex_color.lstrip("#")
    if bgr:
        return tuple(int(hex_color[i : i + 2], 16) for i in (4, 2, 0))

    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


class MTB_BatchFloatMath:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reverse": ("BOOLEAN", {"default": False}),
                "operation": (
                    ["add", "sub", "mul", "div", "pow", "abs"],
                    {"default": "add"},
                ),
            }
        }

    RETURN_TYPES = ("FLOATS",)
    CATEGORY = "mtb/utils"
    FUNCTION = "execute"

    def execute(self, reverse: bool, operation: str, **kwargs: list[float]):
        res: list[float] = []
        vals = list(kwargs.values())

        if reverse:
            vals = vals[::-1]

        ref_count = len(vals[0])
        for v in vals:
            if len(v) != ref_count:
                raise ValueError(
                    f"All values must have the same length (current: {len(v)}, ref: {ref_count})"
                )

        match operation:
            case "add":
                for i in range(ref_count):
                    result = sum(v[i] for v in vals)
                    res.append(result)
            case "sub":
                for i in range(ref_count):
                    result = vals[0][i] - sum(v[i] for v in vals[1:])
                    res.append(result)
            case "mul":
                for i in range(ref_count):
                    result = vals[0][i] * vals[1][i]
                    res.append(result)
            case "div":
                for i in range(ref_count):
                    result = vals[0][i] / vals[1][i]
                    res.append(result)
            case "pow":
                for i in range(ref_count):
                    result: float = vals[0][i] ** vals[1][i]
                    res.append(result)
            case "abs":
                for i in range(ref_count):
                    result = abs(vals[0][i])
                    res.append(result)
            case _:
                log.info(f"For now this mode ({operation}) is not implemented")

        return (res,)


class MTB_BatchFloatNormalize:
    """Normalize the values in the list of floats"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"floats": ("FLOATS",)},
        }

    RETURN_TYPES = ("FLOATS",)
    RETURN_NAMES = ("normalized_floats",)
    CATEGORY = "mtb/batch"
    FUNCTION = "execute"

    def execute(
        self,
        floats: list[float],
    ):
        min_value = min(floats)
        max_value = max(floats)

        normalized_floats = [
            (x - min_value) / (max_value - min_value) for x in floats
        ]
        log.debug(f"Floats: {floats}")
        log.debug(f"Normalized Floats: {normalized_floats}")

        return (normalized_floats,)


class MTB_BatchTimeWrap:
    """Remap a batch using a time curve (FLOATS)"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_count": ("INT", {"default": 25, "min": 2}),
                "frames": ("IMAGE",),
                "curve": ("FLOATS",),
            },
        }

    RETURN_TYPES = ("IMAGE", "FLOATS")
    RETURN_NAMES = ("image", "interpolated_floats")
    CATEGORY = "mtb/batch"
    FUNCTION = "execute"

    def execute(
        self, target_count: int, frames: torch.Tensor, curve: list[float]
    ):
        """Apply time warping to a list of video frames based on a curve."""
        log.debug(f"Input frames shape: {frames.shape}")
        log.debug(f"Curve: {curve}")

        total_duration = sum(curve)

        log.debug(f"Total duration: {total_duration}")

        B, H, W, C = frames.shape

        log.debug(f"Batch Size: {B}")

        normalized_times = np.linspace(0, 1, target_count)
        interpolated_curve = np.interp(
            normalized_times, np.linspace(0, 1, len(curve)), curve
        ).tolist()
        log.debug(f"Interpolated curve: {interpolated_curve}")

        interpolated_frame_indices = [
            (B - 1) * value for value in interpolated_curve
        ]
        log.debug(f"Interpolated frame indices: {interpolated_frame_indices}")

        rounded_indices = [
            int(round(idx)) for idx in interpolated_frame_indices
        ]
        rounded_indices = np.clip(rounded_indices, 0, B - 1)

        # Gather frames based on interpolated indices
        warped_frames = []
        for index in rounded_indices:
            warped_frames.append(frames[index].unsqueeze(0))

        warped_tensor = torch.cat(warped_frames, dim=0)
        log.debug(f"Warped frames shape: {warped_tensor.shape}")
        return (warped_tensor, interpolated_curve)


class MTB_ImageBatchToSublist:
    """
    # Image Batch To Sublist 🔄

    Splits a large batched tensor into smaller sub-batches for memory-efficient processing.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sub_batch_size": (
                    "INT",
                    {"default": 1, "min": 1, "max": 1000, "step": 1},
                ),
            },
            "optional": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT")
    RETURN_NAMES = ("image_list", "mask_list", "item_count")

    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "split_batch"
    CATEGORY = "batch_processing"

    def split_batch(
        self,
        sub_batch_size: int,
        image: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ):
        if image is None and mask is None:
            raise ValueError(
                "You must either pass mask or image, none received"
            )

        image_count = 0
        if image is not None:
            image_count = image.size(0)

        mask_count = 0
        if mask is not None:
            mask_count = mask.size(0)

        if image_count > 0 and mask_count > 0 and mask_count != image_count:
            raise ValueError(
                f"When providing image and mask, batch size must match (got {mask.size(0)} mask and {image.size(0)} images)"
            )

        batch_size = max(image_count, mask_count)

        num_full_batches = batch_size // sub_batch_size
        im_batches = []
        mask_batches = []

        for i in range(num_full_batches):
            start_idx = i * sub_batch_size
            end_idx = start_idx + sub_batch_size
            if image_count > 0:
                im_batches.append(image[start_idx:end_idx, ...])

            if mask_count > 0:
                mask_batches.append(mask[start_idx:end_idx, ...])

        if batch_size % sub_batch_size != 0:
            remaining_start = num_full_batches * sub_batch_size
            if image_count > 0:
                im_batches.append(image[remaining_start:, ...])

            if mask_count > 0:
                mask_batches.append(mask[remaining_start:, ...])

        return (im_batches, mask_batches, len(im_batches))


class MTB_SublistToImageBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tensors": ("IMAGE",),
            }
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "merge_batches"
    CATEGORY = "batch_processing"
    DOCUMENTATION = """# Sublist to Image Batch 🔄

Merges a list of sub-batched tensors back into a single large batch.
"""

    def merge_batches(self, tensors: list[torch.Tensor]):
        if len(tensors) <= 1:
            return (tensors[0],)

        result = tensors[0]

        for next_tensor in tensors[1:]:
            if result.shape[1:] != next_tensor.shape[1:]:
                next_tensor = comfy.utils.common_upscale(
                    next_tensor.movedim(-1, 1),
                    result.shape[2],
                    result.shape[1],
                    "lanczos",
                    "center",
                ).movedim(1, -1)

            result = torch.cat((result, next_tensor), dim=0)

        return (result,)


class MTB_BatchMake:
    """Simply duplicates the input frame as a batch"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "count": ("INT", {"default": 1}),
            },
            "optional": {"mask": ("MASK",)},
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_batch"
    CATEGORY = "mtb/batch"

    def generate_batch(self, image: torch.Tensor, count, mask=None):
        if len(image.shape) == 3:
            image = image.unsqueeze(0)

        return (
            image.repeat(count, 1, 1, 1),
            mask.repeat(count, 1, 1) if mask else mask,
        )


class MTB_BatchShape:
    """Generates a batch of 2D shapes with optional shading (experimental)"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "count": ("INT", {"default": 1}),
                "shape": (
                    ["Box", "Circle", "Diamond", "Tube"],
                    {"default": "Circle"},
                ),
                "image_width": ("INT", {"default": 512}),
                "image_height": ("INT", {"default": 512}),
                "shape_size": ("INT", {"default": 100}),
                "color": ("COLOR", {"default": "#ffffff"}),
                "bg_color": ("COLOR", {"default": "#000000"}),
                "shade_color": ("COLOR", {"default": "#000000"}),
                "thickness": ("INT", {"default": 5}),
                "shadex": ("FLOAT", {"default": 0.0}),
                "shadey": ("FLOAT", {"default": 0.0}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_shapes"
    CATEGORY = "mtb/batch"

    def generate_shapes(
        self,
        count,
        shape,
        image_width,
        image_height,
        shape_size,
        color,
        bg_color,
        shade_color,
        thickness,
        shadex,
        shadey,
    ):
        log.debug(f"COLOR: {color}")
        log.debug(f"BG_COLOR: {bg_color}")
        log.debug(f"SHADE_COLOR: {shade_color}")

        # Parse color input to BGR tuple for OpenCV
        color = hex_to_rgb(color)
        bg_color = hex_to_rgb(bg_color)
        shade_color = hex_to_rgb(shade_color)
        res = []
        for x in range(count):
            # Initialize an image canvas
            canvas = np.full(
                (image_height, image_width, 3), bg_color, dtype=np.uint8
            )
            mask = np.zeros((image_height, image_width), dtype=np.uint8)

            # Compute the center point of the shape
            center = (image_width // 2, image_height // 2)

            if shape == "Box":
                half_size = shape_size // 2
                top_left = (center[0] - half_size, center[1] - half_size)
                bottom_right = (center[0] + half_size, center[1] + half_size)
                cv2.rectangle(mask, top_left, bottom_right, 255, -1)
            elif shape == "Circle":
                cv2.circle(mask, center, shape_size // 2, 255, -1)
            elif shape == "Diamond":
                pts = np.array(
                    [
                        [center[0], center[1] - shape_size // 2],
                        [center[0] + shape_size // 2, center[1]],
                        [center[0], center[1] + shape_size // 2],
                        [center[0] - shape_size // 2, center[1]],
                    ]
                )
                cv2.fillPoly(mask, [pts], 255)

            elif shape == "Tube":
                cv2.ellipse(
                    mask,
                    center,
                    (shape_size // 2, shape_size // 2),
                    0,
                    0,
                    360,
                    255,
                    thickness,
                )

            # Color the shape
            canvas[mask == 255] = color

            # Apply shading effects to a separate shading canvas
            shading = np.zeros_like(canvas, dtype=np.float32)
            shading[:, :, 0] = shadex * np.linspace(0, 1, image_width)
            shading[:, :, 1] = shadey * np.linspace(
                0, 1, image_height
            ).reshape(-1, 1)
            shading_canvas = cv2.addWeighted(
                canvas.astype(np.float32), 1, shading, 1, 0
            ).astype(np.uint8)

            # Apply shading only to the shape area using the mask
            canvas[mask == 255] = shading_canvas[mask == 255]
            res.append(canvas)

        return (pil2tensor(res),)


class MTB_BatchFloatFill:
    """Fills a batch float with a single value until it reaches the target length"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "floats": ("FLOATS",),
                "direction": (["head", "tail"], {"default": "tail"}),
                "value": ("FLOAT", {"default": 0.0}),
                "count": ("INT", {"default": 1}),
            }
        }

    FUNCTION = "fill_floats"
    RETURN_TYPES = ("FLOATS",)
    CATEGORY = "mtb/batch"

    def fill_floats(self, floats, direction, value, count):
        size = len(floats)
        if size > count:
            raise ValueError(
                f"Size ({size}) is less then target count ({count})"
            )

        rem = count - size
        if direction == "tail":
            floats = floats + [value] * rem
        else:
            floats = [value] * rem + floats
        return (floats,)


class MTB_BatchFloatAssemble:
    """Assembles mutiple batches of floats into a single stream (batch)"""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"reverse": ("BOOLEAN", {"default": False})}}

    RETURN_TYPES = ("FLOATS",)
    CATEGORY = "mtb/batch"
    FUNCTION = "assemble_floats"

    def assemble_floats(self, reverse: bool, **kwargs: list[float]):
        res: list[float] = []

        if reverse:
            for x in reversed(kwargs.values()):
                if x:
                    res += x
        else:
            for x in kwargs.values():
                if x:
                    res += x

        return (res,)


class MTB_BatchFloat:
    """Generates a batch of float values with interpolation"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (
                    ["Single", "Steps"],
                    {"default": "Steps"},
                ),
                "count": ("INT", {"default": 2}),
                "min": (
                    "FLOAT",
                    {"default": 0.0, "min": -1e4, "max": 1e4, "step": 0.001},
                ),
                "max": (
                    "FLOAT",
                    {"default": 1.0, "min": -1e4, "max": 1e4, "step": 0.001},
                ),
                "easing": (
                    [
                        "Linear",
                        "Sine In",
                        "Sine Out",
                        "Sine In/Out",
                        "Quart In",
                        "Quart Out",
                        "Quart In/Out",
                        "Cubic In",
                        "Cubic Out",
                        "Cubic In/Out",
                        "Circ In",
                        "Circ Out",
                        "Circ In/Out",
                        "Back In",
                        "Back Out",
                        "Back In/Out",
                        "Elastic In",
                        "Elastic Out",
                        "Elastic In/Out",
                        "Bounce In",
                        "Bounce Out",
                        "Bounce In/Out",
                    ],
                    {"default": "Linear"},
                ),
            }
        }

    FUNCTION = "set_floats"
    RETURN_TYPES = ("FLOATS",)
    CATEGORY = "mtb/batch"

    def set_floats(
        self,
        mode: Literal["Steps"] | Literal["Single"] = "Steps",
        count: int = 1,
        min: float = 0.0,  # noqa: A002
        max: float = 1.0,  # noqa: A002
        easing: str = "Linear",
    ):
        if mode == "Steps" and count == 1:
            raise ValueError(
                "Steps mode requires at least a count of 2 values"
            )
        keyframes = []
        if mode == "Single":
            keyframes = [min] * count
            return (keyframes,)

        for i in range(count):
            normalized_step = i / (count - 1)
            eased_step = apply_easing(normalized_step, easing)
            eased_value = min + (max - min) * eased_step
            keyframes.append(eased_value)

        return (keyframes,)


class MTB_BatchSequencePlus:
    """Sequences multiple image batches with transition effects."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "transition": (
                    [
                        "none",
                        "crossfade",
                        "slide_left",
                        "slide_right",
                        "slide_up",
                        "slide_down",
                        "wipe_left",
                        "wipe_right",
                        "wipe_up",
                        "wipe_down",
                        "band_wipe_h",
                        "band_wipe_v",
                    ],
                    {"default": "none"},
                ),
                "overlap_frames": (
                    "INT",
                    {"default": 0, "min": 0, "max": 120, "step": 1},
                ),
                "reverse": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "sequence_batches"
    CATEGORY = "mtb/batch"

    def apply_transition(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor,
        transition: str,
        progress: float,
    ):
        """Apply transition effect between two frames."""
        if transition == "none":
            return frame1 if progress < 0.5 else frame2

        elif transition == "crossfade":
            return frame1 * (1 - progress) + frame2 * progress

        elif transition.startswith("slide_"):
            h, w = frame1.shape[1:3]
            if transition == "slide_left":
                offset = int(w * progress)
                frame2 = torch.roll(frame2, shifts=-offset, dims=2)
            elif transition == "slide_right":
                offset = int(w * progress)
                frame2 = torch.roll(frame2, shifts=offset, dims=2)
            elif transition == "slide_up":
                offset = int(h * progress)
                frame2 = torch.roll(frame2, shifts=-offset, dims=1)
            elif transition == "slide_down":
                offset = int(h * progress)
                frame2 = torch.roll(frame2, shifts=offset, dims=1)
            return frame1 * (1 - progress) + frame2 * progress

        elif transition.startswith("wipe_"):
            h, w = frame1.shape[1:3]
            mask = torch.zeros_like(frame1)
            if transition == "wipe_left":
                edge = int(w * progress)
                mask[:, :, :edge, :] = 1
            elif transition == "wipe_right":
                edge = int(w * (1 - progress))
                mask[:, :, edge:, :] = 1
            elif transition == "wipe_up":
                edge = int(h * progress)
                mask[:, :edge, :, :] = 1
            elif transition == "wipe_down":
                edge = int(h * (1 - progress))
                mask[:, edge:, :, :] = 1
            return frame1 * (1 - mask) + frame2 * mask

        elif transition.startswith("band_wipe_"):
            h, w = frame1.shape[1:3]
            mask = torch.zeros_like(frame1)
            num_bands = 10  # Number of bands

            if transition == "band_wipe_h":
                band_width = w / num_bands
                for i in range(num_bands):
                    edge = int((w * progress) - (i * band_width))
                    start = int(i * band_width)
                    end = int(min(start + edge, (i + 1) * band_width))
                    if end > start:
                        mask[:, :, start:end, :] = 1
            else:  # band_wipe_v
                band_height = h / num_bands
                for i in range(num_bands):
                    edge = int((h * progress) - (i * band_height))
                    start = int(i * band_height)
                    end = int(min(start + edge, (i + 1) * band_height))
                    if end > start:
                        mask[:, start:end, :, :] = 1

            return frame1 * (1 - mask) + frame2 * mask

        return frame1

    def sequence_batches(
        self, transition: str, overlap_frames: int, reverse: bool, **kwargs
    ):
        images: list[torch.Tensor] = list(kwargs.values())

        if reverse:
            images = images[::-1]

        processed_images: list[torch.Tensor] = []
        for img in images:
            if len(img.shape) == 3:
                img = img.unsqueeze(0)
            processed_images.append(img)

        if overlap_frames == 0 or transition == "none":
            return (torch.cat(processed_images, dim=0),)

        result_frames: list[torch.Tensor] = []

        if len(processed_images) > 0:
            result_frames.extend(
                list(processed_images[0][: -overlap_frames // 2])
            )

        for i in range(1, len(processed_images)):
            prev_batch = processed_images[i - 1]
            curr_batch = processed_images[i]

            prev_frames = min(overlap_frames // 2, len(prev_batch))
            next_frames = min(overlap_frames // 2, len(curr_batch))
            total_overlap = prev_frames + next_frames

            if total_overlap < 2:
                # when not enough frames for transition, just concatenate
                result_frames.extend(list(prev_batch[-prev_frames:]))
                result_frames.extend(list(curr_batch[:next_frames]))
                continue

            for t in range(total_overlap):
                progress = t / (total_overlap - 1)

                prev_idx = (
                    len(prev_batch) - prev_frames + min(t, prev_frames - 1)
                )
                next_idx = max(0, t - prev_frames)

                transition_frame = self.apply_transition(
                    prev_batch[prev_idx : prev_idx + 1],
                    curr_batch[next_idx : next_idx + 1],
                    transition,
                    progress,
                )
                result_frames.append(transition_frame[0])

            if i < len(processed_images) - 1:
                result_frames.extend(
                    list(curr_batch[next_frames : -overlap_frames // 2])
                )
            else:
                result_frames.extend(list(curr_batch[next_frames:]))

        result = torch.stack(result_frames, dim=0)

        return (result,)


class MTB_BatchSequence:
    """Sequences multiple image batches one after another"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reverse": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "sequence_batches"
    CATEGORY = "mtb/batch"

    def sequence_batches(self, reverse: bool, **kwargs):
        images = list(kwargs.values())
        if reverse:
            images = images[::-1]

        processed = []
        for img in images:
            if len(img.shape) == 3:
                img = img.unsqueeze(0)
            processed.append(img)

        return (torch.cat(processed, dim=0),)


class MTB_BatchMerge:
    """Merges multiple image batches with different frame counts"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fusion_mode": (
                    ["add", "multiply", "average"],
                    {"default": "average"},
                ),
                "fill": (["head", "tail"], {"default": "tail"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "merge_batches"
    CATEGORY = "mtb/batch"

    def merge_batches(self, fusion_mode: str, fill: str, **kwargs):
        images = kwargs.values()
        max_frames = max(img.shape[0] for img in images)

        adjusted_images = []
        for img in images:
            frame_count = img.shape[0]
            if frame_count < max_frames:
                fill_frame = img[0] if fill == "head" else img[-1]
                fill_frames = fill_frame.repeat(
                    max_frames - frame_count, 1, 1, 1
                )
                adjusted_batch = (
                    torch.cat((fill_frames, img), dim=0)
                    if fill == "head"
                    else torch.cat((img, fill_frames), dim=0)
                )
            else:
                adjusted_batch = img
            adjusted_images.append(adjusted_batch)

        # Merge the adjusted batches
        merged_image = None
        for img in adjusted_images:
            if merged_image is None:
                merged_image = img
            else:
                if fusion_mode == "add":
                    merged_image += img
                elif fusion_mode == "multiply":
                    merged_image *= img
                elif fusion_mode == "average":
                    merged_image = (merged_image + img) / 2

        return (merged_image,)


class MTB_Batch2dTransform:
    """Transform a batch of images using a batch of keyframes"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "border_handling": (
                    ["edge", "constant", "reflect", "symmetric"],
                    {"default": "edge"},
                ),
                "constant_color": ("COLOR", {"default": "#000000"}),
            },
            "optional": {
                "x": ("FLOATS",),
                "y": ("FLOATS",),
                "zoom": ("FLOATS",),
                "angle": ("FLOATS",),
                "shear": ("FLOATS",),
                "use_normalized": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "If true, transform values will be scaled to image dimensions.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "transform_batch"
    CATEGORY = "mtb/batch"

    def get_num_elements(
        self, param: None | torch.Tensor | list[torch.Tensor] | list[float]
    ) -> int:
        if isinstance(param, torch.Tensor):
            return torch.numel(param)

        elif isinstance(param, list):
            return len(param)

        return 0

    def transform_batch(
        self,
        image: torch.Tensor,
        border_handling: str,
        constant_color: str,
        x: list[float] | None = None,
        y: list[float] | None = None,
        zoom: list[float] | None = None,
        angle: list[float] | None = None,
        shear: list[float] | None = None,
        use_normalized: bool = False,
    ):
        if all(
            self.get_num_elements(param) <= 0
            for param in [x, y, zoom, angle, shear]
        ):
            raise ValueError(
                "At least one transform parameter must be provided"
            )

        keyframes: dict[str, list[float]] = {
            "x": [],
            "y": [],
            "zoom": [],
            "angle": [],
            "shear": [],
        }

        default_vals = {"x": 0, "y": 0, "zoom": 1.0, "angle": 0, "shear": 0}

        if x and self.get_num_elements(x) > 0:
            keyframes["x"] = x
        if y and self.get_num_elements(y) > 0:
            keyframes["y"] = y
        if zoom and self.get_num_elements(zoom) > 0:
            # some easing types like elastic can pull back... maybe it should abs the value?
            keyframes["zoom"] = [max(x, 0.00001) for x in zoom]
        if angle and self.get_num_elements(angle) > 0:
            keyframes["angle"] = angle
        if shear and self.get_num_elements(shear) > 0:
            keyframes["shear"] = shear

        for name, values in keyframes.items():
            count = len(values)
            if count > 0 and count != image.shape[0]:
                raise ValueError(
                    f"Length of {name} values ({count}) must match number of images ({image.shape[0]})"
                )
            if count == 0:
                keyframes[name] = [default_vals[name]] * image.shape[0]

        transformer = MTB_TransformImage()
        res = [
            transformer.transform(
                image[i].unsqueeze(0),
                keyframes["x"][i],
                keyframes["y"][i],
                keyframes["zoom"][i],
                keyframes["angle"][i],
                keyframes["shear"][i],
                border_handling,
                constant_color,
                use_normalized=use_normalized,
            )[0]
            for i in range(image.shape[0])
        ]
        return (torch.cat(res, dim=0),)


class MTB_BatchFloatFit:
    """Fit a list of floats using a source and target range"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "values": ("FLOATS", {"forceInput": True}),
                "clamp": ("BOOLEAN", {"default": False}),
                "auto_compute_source": ("BOOLEAN", {"default": False}),
                "source_min": ("FLOAT", {"default": 0.0, "step": 0.01}),
                "source_max": ("FLOAT", {"default": 1.0, "step": 0.01}),
                "target_min": ("FLOAT", {"default": 0.0, "step": 0.01}),
                "target_max": ("FLOAT", {"default": 1.0, "step": 0.01}),
                "easing": (
                    EASINGS,
                    {"default": "Linear"},
                ),
            }
        }

    FUNCTION = "fit_range"
    RETURN_TYPES = ("FLOATS",)
    CATEGORY = "mtb/batch"
    DESCRIPTION = "Fit a list of floats using a source and target range"

    def fit_range(
        self,
        values: list[float],
        clamp: bool,
        auto_compute_source: bool,
        source_min: float,
        source_max: float,
        target_min: float,
        target_max: float,
        easing: str,
    ):
        if auto_compute_source:
            source_min = min(values)
            source_max = max(values)

        from .graph_utils import MTB_FitNumber

        res = []
        fit_number = MTB_FitNumber()
        for value in values:
            (transformed_value,) = fit_number.set_range(
                value,
                clamp,
                source_min,
                source_max,
                target_min,
                target_max,
                easing,
            )
            res.append(transformed_value)

        return (res,)


class MTB_PlotBatchFloat:
    """Plot floats"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 768}),
                "height": ("INT", {"default": 768}),
                "point_size": ("INT", {"default": 4}),
                "seed": ("INT", {"default": 1}),
                "start_at_zero": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("plot",)
    FUNCTION = "plot"
    CATEGORY = "mtb/batch"

    def plot(
        self,
        width: int,
        height: int,
        point_size: int,
        seed: int,
        start_at_zero: bool,
        interactive_backend: bool = False,
        **kwargs,
    ):
        import matplotlib

        # NOTE: This is for notebook usage or tests, i.e not exposed to comfy that should always use Agg
        if not interactive_backend:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
        fig.set_edgecolor("black")
        fig.patch.set_facecolor("#2e2e2e")
        # Setting background color and grid
        ax.set_facecolor("#2e2e2e")  # Dark gray background
        ax.grid(color="gray", linestyle="-", linewidth=0.5, alpha=0.5)

        # Finding global min and max across all lists for scaling the plot
        all_values = [value for values in kwargs.values() for value in values]
        global_min = min(all_values)
        global_max = max(all_values)

        y_padding = 0.05 * (global_max - global_min)
        ax.set_ylim(global_min - y_padding, global_max + y_padding)

        max_length = max(len(values) for values in kwargs.values())
        if start_at_zero:
            x_values = np.linspace(0, max_length - 1, max_length)
        else:
            x_values = np.linspace(1, max_length, max_length)

        ax.set_xlim(1, max_length)  # Set X-axis limits
        np.random.seed(seed)
        colors = np.random.rand(len(kwargs), 3)  # Generate random RGB values
        for color, (label, values) in zip(
            colors, kwargs.items(), strict=False
        ):
            ax.plot(x_values[: len(values)], values, label=label, color=color)
        ax.legend(
            title="Legend",
            title_fontsize="large",
            fontsize="medium",
            edgecolor="black",
            loc="best",
        )

        # Setting labels and title
        ax.set_xlabel("Time", fontsize="large", color="white")
        ax.set_ylabel("Value", fontsize="large", color="white")
        ax.set_title(
            "Plot of Values over Time", fontsize="x-large", color="white"
        )

        # Adjusting tick colors to be visible on dark background
        ax.tick_params(colors="white")

        # Changing color of the axes border
        for _, spine in ax.spines.items():
            spine.set_edgecolor("white")

        # Rendering the plot into a NumPy array
        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        image = Image.open(buf)
        plt.close(fig)  # Closing the figure to free up memory

        return (pil2tensor(image),)

    def draw_point(self, image, point, color, point_size):
        x, y = point
        y = image.shape[0] - 1 - y  # Invert Y-coordinate
        half_size = point_size // 2
        x_start, x_end = (
            max(0, x - half_size),
            min(image.shape[1], x + half_size + 1),
        )
        y_start, y_end = (
            max(0, y - half_size),
            min(image.shape[0], y + half_size + 1),
        )
        image[y_start:y_end, x_start:x_end] = color

    def draw_line(self, image, start, end, color):
        x1, y1 = start
        x2, y2 = end

        # Invert Y-coordinate
        y1 = image.shape[0] - 1 - y1
        y2 = image.shape[0] - 1 - y2

        dx = x2 - x1
        dy = y2 - y1
        is_steep = abs(dy) > abs(dx)
        if is_steep:
            x1, y1 = y1, x1
            x2, y2 = y2, x2
        swapped = False
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
            swapped = True
        dx = x2 - x1
        dy = y2 - y1
        error = int(dx / 2.0)
        y = y1
        ystep = None
        if y1 < y2:
            ystep = 1
        else:
            ystep = -1
        for x in range(x1, x2 + 1):
            coord = (y, x) if is_steep else (x, y)
            image[coord] = color
            error -= abs(dy)
            if error < 0:
                y += ystep
                error += dx
        if swapped:
            image[(x1, y1)] = color
            image[(x2, y2)] = color


DEFAULT_INTERPOLANT = lambda t: t * t * t * (t * (t * 6 - 15) + 10)


class MTB_BatchShake:
    """Applies a shaking effect to batches of images."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "position_amount_x": ("FLOAT", {"default": 1.0}),
                "position_amount_y": ("FLOAT", {"default": 1.0}),
                "rotation_amount": ("FLOAT", {"default": 10.0}),
                "frequency": ("FLOAT", {"default": 1.0, "min": 0.005}),
                "frequency_divider": ("FLOAT", {"default": 1.0, "min": 0.005}),
                "octaves": ("INT", {"default": 1, "min": 1}),
                "seed": ("INT", {"default": 0}),
            },
        }

    RETURN_TYPES = ("IMAGE", "FLOATS", "FLOATS", "FLOATS")
    RETURN_NAMES = ("image", "pos_x", "pos_y", "rot")
    FUNCTION = "apply_shake"
    CATEGORY = "mtb/batch"

    # def interpolant(self, t):
    # return t * t * t * (t * (t * 6 - 15) + 10)

    def generate_perlin_noise_2d(
        self, shape, res, tileable=(False, False), interpolant=None
    ):
        """Generate a 2D numpy array of perlin noise.

        Args:
            shape: The shape of the generated array (tuple of two ints).
                This must be a multple of res.
            res: The number of periods of noise to generate along each
                axis (tuple of two ints). Note shape must be a multiple of
                res.
            tileable: If the noise should be tileable along each axis
                (tuple of two bools). Defaults to (False, False).
            interpolant: The interpolation function, defaults to
                t*t*t*(t*(t*6 - 15) + 10).

        Returns
        -------
            A numpy array of shape shape with the generated noise.

        Raises
        ------
            ValueError: If shape is not a multiple of res.
        """
        interpolant = interpolant or DEFAULT_INTERPOLANT
        delta = (res[0] / shape[0], res[1] / shape[1])
        d = (shape[0] // res[0], shape[1] // res[1])
        grid = (
            np.mgrid[0 : res[0] : delta[0], 0 : res[1] : delta[1]].transpose(
                1, 2, 0
            )
            % 1
        )
        # Gradients
        angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
        gradients = np.dstack((np.cos(angles), np.sin(angles)))
        if tileable[0]:
            gradients[-1, :] = gradients[0, :]
        if tileable[1]:
            gradients[:, -1] = gradients[:, 0]
        gradients = gradients.repeat(d[0], 0).repeat(d[1], 1)
        g00 = gradients[: -d[0], : -d[1]]
        g10 = gradients[d[0] :, : -d[1]]
        g01 = gradients[: -d[0], d[1] :]
        g11 = gradients[d[0] :, d[1] :]
        # Ramps
        n00 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1])) * g00, 2)
        n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
        n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
        n11 = np.sum(
            np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2
        )
        # Interpolation
        t = interpolant(grid)
        n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
        n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
        return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)

    def generate_fractal_noise_2d(
        self,
        shape,
        res,
        octaves=1,
        persistence=0.5,
        lacunarity=2,
        tileable=(True, True),
        interpolant=None,
    ):
        """Generate a 2D numpy array of fractal noise.

        Args:
            shape: The shape of the generated array (tuple of two ints).
                This must be a multiple of lacunarity**(octaves-1)*res.
            res: The number of periods of noise to generate along each
                axis (tuple of two ints). Note shape must be a multiple of
                (lacunarity**(octaves-1)*res).
            octaves: The number of octaves in the noise. Defaults to 1.
            persistence: The scaling factor between two octaves.
            lacunarity: The frequency factor between two octaves.
            tileable: If the noise should be tileable along each axis
                (tuple of two bools). Defaults to (True,True).
            interpolant: The, interpolation function, defaults to
                t*t*t*(t*(t*6 - 15) + 10).

        Returns
        -------
            A numpy array of fractal noise and of shape shape generated by
            combining several octaves of perlin noise.

        Raises
        ------
            ValueError: If shape is not a multiple of
                (lacunarity**(octaves-1)*res).
        """
        interpolant = interpolant or DEFAULT_INTERPOLANT

        noise = np.zeros(shape)
        frequency = 1
        amplitude = 1
        for _ in range(octaves):
            noise += amplitude * self.generate_perlin_noise_2d(
                shape,
                (frequency * res[0], frequency * res[1]),
                tileable,
                interpolant,
            )
            frequency *= lacunarity
            amplitude *= persistence
        return noise

    def fbm(self, x, y, octaves):
        # noise_2d = self.generate_fractal_noise_2d((256, 256), (8, 8), octaves)
        # Now, extract a single noise value based on x and y, wrapping indices if necessary
        x_idx = int(x) % 256
        y_idx = int(y) % 256
        return self.noise_pattern[x_idx, y_idx]

    def apply_shake(
        self,
        images,
        position_amount_x,
        position_amount_y,
        rotation_amount,
        frequency,
        frequency_divider,
        octaves,
        seed,
    ):
        # Rehash
        np.random.seed(seed)
        self.position_offset = np.random.uniform(-1e3, 1e3, 3)
        self.rotation_offset = np.random.uniform(-1e3, 1e3, 3)
        self.noise_pattern = self.generate_perlin_noise_2d(
            (512, 512), (32, 32), (True, True)
        )

        # Assuming frame count is derived from the first dimension of images tensor
        frame_count = images.shape[0]

        frequency = frequency / frequency_divider

        # Generate shaking parameters for each frame
        x_translations = []
        y_translations = []
        rotations = []

        for frame_num in range(frame_count):
            time = frame_num * frequency
            x_idx = (self.position_offset[0] + frame_num) % 256
            y_idx = (self.position_offset[1] + frame_num) % 256

            np_position = np.array(
                [
                    self.fbm(x_idx, time, octaves),
                    self.fbm(y_idx, time, octaves),
                ]
            )

            # np_position = np.array(
            #     [
            #         self.fbm(self.position_offset[0] + frame_num, time, octaves),
            #         self.fbm(self.position_offset[1] + frame_num, time, octaves),
            #     ]
            # )
            # np_rotation = self.fbm(self.rotation_offset[2] + frame_num, time, octaves)

            rot_idx = (self.rotation_offset[2] + frame_num) % 256
            np_rotation = self.fbm(rot_idx, time, octaves)

            x_translations.append(np_position[0] * position_amount_x)
            y_translations.append(np_position[1] * position_amount_y)
            rotations.append(np_rotation * rotation_amount)

        # Convert lists to tensors
        # x_translations = torch.tensor(x_translations, dtype=torch.float32)
        # y_translations = torch.tensor(y_translations, dtype=torch.float32)
        # rotations = torch.tensor(rotations, dtype=torch.float32)

        # Create an instance of Batch2dTransform
        transform = MTB_Batch2dTransform()

        log.debug(
            f"Applying shaking with parameters: \nposition {position_amount_x}, {position_amount_y}\nrotation {rotation_amount}\nfrequency {frequency}\noctaves {octaves}"
        )

        # Apply shaking transformations to images
        shaken_images = transform.transform_batch(
            images,
            border_handling="edge",  # Assuming edge handling as default
            constant_color="#000000",  # Assuming black as default constant color
            x=x_translations,
            y=y_translations,
            angle=rotations,
        )[0]

        return (shaken_images, x_translations, y_translations, rotations)


class MTB_BatchFromFolder:
    """Load images from a folder with options for latest, oldest, or random selection."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enable": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Enable or disable the node. If disabled, returns passthrough_image or an empty tensor.",
                    },
                ),
                "folder_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Path to the folder containing images. Relative paths are resolved to the ComfyUI output directory.",
                    },
                ),
                "mode": (
                    ["latest", "oldest", "random"],
                    {
                        "default": "latest",
                        "tooltip": "How to select images: latest, oldest, or random.",
                    },
                ),
                "count": (
                    "INT",
                    {
                        "default": 10,
                        "min": 1,
                        "max": 1000,
                        "tooltip": "Number of images to load from the folder.",
                    },
                ),
                "filter": (
                    "STRING",
                    {
                        "default": "*",
                        "tooltip": "Glob filter for image filenames (e.g. *.png).",
                    },
                ),
            },
            "optional": {
                "passthrough_image": (
                    "IMAGE",
                    {
                        "tooltip": "If provided and node is disabled, this image is passed through instead of returning an empty tensor."
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    CATEGORY = "mtb/batch"
    FUNCTION = "load_from_folder"

    def load_from_folder(
        self,
        enable: bool,
        folder_path: str,
        mode: str,
        count: int,
        filter: str,
        passthrough_image=None,
    ):
        """Load images from a folder with the specified selection mode."""
        if not enable:
            if passthrough_image is not None:
                log.debug(
                    "MTB_BatchFromFolder: Using passthrough image (disabled)"
                )
                return (passthrough_image,)
            log.debug(
                "MTB_BatchFromFolder: Disabled and no passthrough_image provided, returning empty tensor"
            )
            return (torch.zeros(0, 0, 0, 3),)

        path_obj = Path(folder_path)
        if not path_obj.is_absolute():
            output_dir = Path(folder_paths.get_output_directory())
            path_obj = output_dir / folder_path
            path_obj = path_obj.resolve()

        if not path_obj.exists():
            log.error(f"Folder path does not exist: {path_obj}")
            return (torch.zeros(0, 0, 0, 3),)

        if not path_obj.is_dir():
            log.error(f"Path is not a directory: {path_obj}")
            return (torch.zeros(0, 0, 0, 3),)

        patterns = [filter] if filter else ["*"]
        files = glob_multiple(path_obj, patterns)

        image_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"]
        image_files = [
            f for f in files if f.suffix.lower() in image_extensions
        ]

        if not image_files:
            log.warning(
                f"No image files found in {path_obj} with filter {filter}"
            )
            return (torch.zeros(0, 0, 0, 3),)

        if mode == "latest":
            image_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        elif mode == "oldest":
            image_files.sort(key=lambda x: os.path.getmtime(x))
        elif mode == "random":
            random.shuffle(image_files)

        selected_files = image_files[:count]

        if len(selected_files) < count:
            log.warning(
                f"Requested {count} images but only found {len(selected_files)}"
            )

        loaded_images = []
        for file_path in selected_files:
            try:
                img = Image.open(file_path)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                loaded_images.append(img)
            except Exception as e:
                log.error(f"Error loading image {file_path}: {e}")

        if not loaded_images:
            log.error("Failed to load any images")
            return (torch.zeros(0, 0, 0, 3),)

        return (pil2tensor(loaded_images),)


__nodes__ = [
    MTB_Batch2dTransform,
    MTB_BatchFloat,
    MTB_BatchFloatAssemble,
    MTB_BatchFloatFill,
    MTB_BatchFloatFit,
    MTB_BatchFloatMath,
    MTB_BatchFloatNormalize,
    MTB_BatchFromFolder,
    MTB_BatchMake,
    MTB_BatchMerge,
    MTB_BatchSequence,
    MTB_BatchSequencePlus,
    MTB_BatchShake,
    MTB_BatchShape,
    MTB_BatchTimeWrap,
    MTB_PlotBatchFloat,
    MTB_SublistToImageBatch,
    MTB_ImageBatchToSublist,
]
