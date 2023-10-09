import os
from pathlib import Path
from typing import List

import cv2
import folder_paths
import numpy as np
import torch

from ..utils import apply_easing, pil2tensor
from .transform import TransformImage


def hex_to_rgb(hex_color, bgr=False):
    hex_color = hex_color.lstrip("#")
    if bgr:
        return tuple(int(hex_color[i : i + 2], 16) for i in (4, 2, 0))

    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


class BatchMake:
    """Simply duplicates the input frame as a batch"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "count": ("INT", {"default": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_batch"
    CATEGORY = "mtb/batch"

    def generate_batch(self, image: torch.Tensor, count):
        if len(image.shape) == 3:
            image = image.unsqueeze(0)

        return (image.repeat(count, 1, 1, 1),)


class BatchShape:
    """Generates a batch of 2D shapes with optional shading (experimental)"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "count": ("INT", {"default": 1}),
                "shape": (
                    ["Box", "Circle", "Diamond"],
                    {"default": "Box"},
                ),
                "image_width": ("INT", {"default": 512}),
                "image_height": ("INT", {"default": 512}),
                "shape_size": ("INT", {"default": 100}),
                "color": ("COLOR", {"default": "#ffffff"}),
                "bg_color": ("COLOR", {"default": "#000000"}),
                "shade_color": ("COLOR", {"default": "#000000"}),
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
        shadex,
        shadey,
    ):
        print(f"COLOR: {color}")
        print(f"BG_COLOR: {bg_color}")
        print(f"SHADE_COLOR: {shade_color}")

        # Parse color input to BGR tuple for OpenCV
        color = hex_to_rgb(color)
        bg_color = hex_to_rgb(bg_color)
        shade_color = hex_to_rgb(shade_color)
        res = []
        for x in range(count):
            # Initialize an image canvas
            canvas = np.full((image_height, image_width, 3), bg_color, dtype=np.uint8)
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

            # Color the shape
            canvas[mask == 255] = color

            # Apply shading effects to a separate shading canvas
            shading = np.zeros_like(canvas, dtype=np.float32)
            shading[:, :, 0] = shadex * np.linspace(0, 1, image_width)
            shading[:, :, 1] = shadey * np.linspace(0, 1, image_height).reshape(-1, 1)
            shading_canvas = cv2.addWeighted(
                canvas.astype(np.float32), 1, shading, 1, 0
            ).astype(np.uint8)

            # Apply shading only to the shape area using the mask
            canvas[mask == 255] = shading_canvas[mask == 255]
            res.append(canvas)

        return (pil2tensor(res),)


class BatchFloatFill:
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
            raise ValueError(f"Size ({size}) is less then target count ({count})")

        rem = count - size
        if direction == "tail":
            floats = floats + [value] * rem
        else:
            floats = [value] * rem + floats
        return (floats,)


class BatchFloatAssemble:
    """Assembles mutiple batches of floats into a single stream (batch)"""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"reverse": ("BOOLEAN", {"default": False})}}

    FUNCTION = "assemble_floats"
    RETURN_TYPES = ("FLOATS",)
    CATEGORY = "mtb/batch"

    def assemble_floats(self, reverse, **kwargs):
        res = []
        if reverse:
            for x in reversed(kwargs.values()):
                res += x
        else:
            for x in kwargs.values():
                res += x

        return (res,)


class BatchFloat:
    """Generates a batch of float values with interpolation"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (
                    ["Single", "Steps"],
                    {"default": "Single"},
                ),
                "count": ("INT", {"default": 1}),
                "min": ("FLOAT", {"default": 0.0}),
                "max": ("FLOAT", {"default": 1.0}),
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

    def set_floats(self, mode, count, min, max, easing):
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


class Batch2dTransform:
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
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "transform_batch"
    CATEGORY = "mtb/batch"

    def transform_batch(
        self,
        image: torch.Tensor,
        border_handling,
        constant_color,
        x=None,
        y=None,
        zoom=None,
        angle=None,
        shear=None,
    ):
        if not any([x, y, zoom, angle]):
            raise ValueError("At least one transform parameter must be provided")

        keyframes = {"x": [], "y": [], "zoom": [], "angle": [], "shear": []}

        default_vals = {"x": 0, "y": 0, "zoom": 1.0, "angle": 0, "shear": 0}

        if x:
            keyframes["x"] = x
        if y:
            keyframes["y"] = y
        if zoom:
            keyframes["zoom"] = zoom
        if angle:
            keyframes["angle"] = angle
        if shear:
            keyframes["shear"] = shear

        for name, values in keyframes.items():
            count = len(values)
            if count > 0 and count != image.shape[0]:
                raise ValueError(
                    f"Length of {name} values ({count}) must match number of images ({image.shape[0]})"
                )
            if count == 0:
                keyframes[name] = [default_vals[name]] * image.shape[0]

        transformer = TransformImage()
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
            )[0]
            for i in range(image.shape[0])
        ]
        return (torch.cat(res, dim=0),)


__nodes__ = [
    BatchFloat,
    Batch2dTransform,
    BatchShape,
    BatchMake,
    BatchFloatAssemble,
    BatchFloatFill,
]
