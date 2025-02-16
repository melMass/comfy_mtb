import json
import os

import numpy as np
import torch
from comfy.cli_args import args
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from ..log import log


class MTB_StackImages:
    """Stack the input images horizontally or vertically."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"vertical": ("BOOLEAN", {"default": False})},
            "optional": {
                "match_method": (
                    ["error", "smallest", "largest"],
                    {"default": "error"},
                )
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "stack"
    CATEGORY = "mtb/image utils"

    def stack(self, vertical, match_method="error", **kwargs):
        if not kwargs:
            raise ValueError("At least one tensor must be provided.")

        tensors = list(kwargs.values())
        log.debug(
            f"Stacking {len(tensors)} tensors "
            f"{'vertically' if vertical else 'horizontally'}"
        )

        target_device = tensors[0].device

        normalized_tensors = [
            self.normalize_to_rgba(tensor.to(target_device))
            for tensor in tensors
        ]

        max_batch_size = max(tensor.shape[0] for tensor in normalized_tensors)
        normalized_tensors = [
            self.duplicate_frames(tensor, max_batch_size)
            for tensor in normalized_tensors
        ]
        if match_method != "error":
            if vertical:
                # match widths
                widths = [tensor.shape[2] for tensor in normalized_tensors]
                target_width = (
                    min(widths) if match_method == "smallest" else max(widths)
                )
                normalized_tensors = [
                    self.resize_tensor(tensor, width=target_width)
                    for tensor in normalized_tensors
                ]
            else:
                # match heights
                heights = [tensor.shape[1] for tensor in normalized_tensors]
                target_height = (
                    min(heights)
                    if match_method == "smallest"
                    else max(heights)
                )
                normalized_tensors = [
                    self.resize_tensor(tensor, height=target_height)
                    for tensor in normalized_tensors
                ]
        else:
            if vertical:
                width = normalized_tensors[0].shape[2]
                if any(
                    tensor.shape[2] != width for tensor in normalized_tensors
                ):
                    raise ValueError(
                        "All tensors must have the same width "
                        "for vertical stacking."
                    )
            else:
                height = normalized_tensors[0].shape[1]
                if any(
                    tensor.shape[1] != height for tensor in normalized_tensors
                ):
                    raise ValueError(
                        "All tensors must have the same height "
                        "for horizontal stacking."
                    )

        dim = 1 if vertical else 2

        stacked_tensor = torch.cat(normalized_tensors, dim=dim)

        return (stacked_tensor,)

    def normalize_to_rgba(self, tensor):
        """Normalize tensor to have 4 channels (RGBA)."""
        _, _, _, channels = tensor.shape
        # already RGBA
        if channels == 4:
            return tensor
        # RGB to RGBA
        elif channels == 3:
            alpha_channel = torch.ones(
                tensor.shape[:-1] + (1,), device=tensor.device
            )
            return torch.cat((tensor, alpha_channel), dim=-1)
        else:
            raise ValueError(
                "Tensor has an unsupported number of channels: "
                "expected 3 (RGB) or 4 (RGBA)."
            )

    def duplicate_frames(self, tensor, target_batch_size):
        """Duplicate frames in tensor to match the target batch size."""
        current_batch_size = tensor.shape[0]
        if current_batch_size < target_batch_size:
            duplication_factors: int = target_batch_size // current_batch_size
            duplicated_tensor = tensor.repeat(duplication_factors, 1, 1, 1)
            remaining_frames = target_batch_size % current_batch_size
            if remaining_frames > 0:
                duplicated_tensor = torch.cat(
                    (duplicated_tensor, tensor[:remaining_frames]), dim=0
                )
            return duplicated_tensor
        else:
            return tensor

    def resize_tensor(self, tensor, width=None, height=None):
        """Resize tensor to specified width or height while maintaining aspect ratio."""
        current_height, current_width = tensor.shape[1:3]

        if width is not None and width != current_width:
            scale_factor = width / current_width
            new_height = int(current_height * scale_factor)
            new_width = width
        elif height is not None and height != current_height:
            scale_factor = height / current_height
            new_width = int(current_width * scale_factor)
            new_height = height
        else:
            return tensor

        resized = torch.nn.functional.interpolate(
            tensor.permute(0, 3, 1, 2),
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=False,
        )

        return resized.permute(0, 2, 3, 1)


class MTB_PickFromBatch:
    """Pick a specific number of images from a batch.

    either from the start or end.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "from_direction": (["end", "start"], {"default": "start"}),
                "count": ("INT", {"default": 1}),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "pick_from_batch"
    CATEGORY = "mtb/image utils"

    def pick_from_batch(self, image, from_direction, count, mask=None):
        batch_size = image.size(0)

        # Limit count to the available number of images in the batch
        count = min(count, batch_size)

        selected_masks = None

        if from_direction == "end":
            selected_tensors = image[-count:]
            if mask is not None:
                selected_masks = mask[-count:]
        else:
            selected_tensors = image[:count]
            if mask is not None:
                selected_masks = mask[:count]

        return (selected_tensors, selected_masks)


import folder_paths


class MTB_SaveImage:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                "filename_prefix": (
                    "STRING",
                    {
                        "default": "ComfyUI",
                        "tooltip": "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes.",
                    },
                ),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "save_images"

    # OUTPUT_NODE = True

    CATEGORY = "mtb/image utils"
    DESCRIPTION = """Saves the input images to your ComfyUI output directory.
    This behaves exactly like the native SaveImage node but isn't an output node.
    The reason I made this is to allow 'inlining' image save in loops for instance,
    using the native node there wouldn't run for each iteration of the loop."""

    def save_images(
        self,
        images,
        filename_prefix="ComfyUI",
        prompt=None,
        extra_pnginfo=None,
    ):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = (
            folder_paths.get_save_image_path(
                filename_prefix,
                self.output_dir,
                images[0].shape[1],
                images[0].shape[0],
            )
        )
        results = list()
        for batch_number, image in enumerate(images):
            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            if not args.disable_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            filename_with_batch_num = filename.replace(
                "%batch_num%", str(batch_number)
            )
            file = f"{filename_with_batch_num}_{counter:05}_.png"
            img.save(
                os.path.join(full_output_folder, file),
                pnginfo=metadata,
                compress_level=self.compress_level,
            )
            results.append(
                {"filename": file, "subfolder": subfolder, "type": self.type}
            )
            counter += 1

        return {"ui": {"images": results}, "result": (images,)}


__nodes__ = [MTB_StackImages, MTB_PickFromBatch, MTB_SaveImage]
