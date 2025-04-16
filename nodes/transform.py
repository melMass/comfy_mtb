from math import ceil, sqrt
from typing import cast

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from ..utils import hex_to_rgb, log, pil2tensor, tensor2pil


class MTB_TransformImage:
    """Save torch tensors (image, mask or latent) to disk, useful to debug things outside comfy

    it return a tensor representing the transformed images with the same shape as the input tensor
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "x": (
                    "FLOAT",
                    {"default": 0, "step": 1, "min": -4096, "max": 4096},
                ),
                "y": (
                    "FLOAT",
                    {"default": 0, "step": 1, "min": -4096, "max": 4096},
                ),
                "zoom": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.001, "step": 0.01},
                ),
                "angle": (
                    "FLOAT",
                    {"default": 0, "step": 1, "min": -360, "max": 360},
                ),
                "shear": (
                    "FLOAT",
                    {"default": 0, "step": 1, "min": -4096, "max": 4096},
                ),
                "border_handling": (
                    ["edge", "constant", "reflect", "symmetric"],
                    {"default": "edge"},
                ),
                "constant_color": ("COLOR", {"default": "#000000"}),
            },
            "optional": {
                "filter_type": (
                    [
                        "nearest",
                        "box",
                        "bilinear",
                        "hamming",
                        "bicubic",
                        "lanczos",
                    ],
                    {"default": "bilinear"},
                ),
                "stretch_x": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.001, "max": 10.0, "step": 0.01},
                ),
                "stretch_y": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.001, "max": 10.0, "step": 0.01},
                ),
                "use_normalized": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "If true, transform values are scaled to image dimensions.",
                    },
                ),
            },
        }

    FUNCTION = "transform"
    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "mtb/transform"

    def transform(
        self,
        image: torch.Tensor,
        x: float,
        y: float,
        zoom: float,
        angle: float,
        shear: float,
        border_handling="edge",
        constant_color=None,
        filter_type="nearest",
        stretch_x=1.0,
        stretch_y=1.0,
        use_normalized: bool = False,
    ):
        filter_map = {
            "nearest": Image.NEAREST,
            "box": Image.BOX,
            "bilinear": Image.BILINEAR,
            "hamming": Image.HAMMING,
            "bicubic": Image.BICUBIC,
            "lanczos": Image.LANCZOS,
        }
        resampling_filter = filter_map[filter_type]

        _, frame_height, frame_width, _ = image.size()
        if use_normalized:
            x = float(x) * frame_width
            y = float(y) * frame_height
        x = int(x)
        y = int(y)
        angle = int(angle)

        log.debug(
            f"Zoom: {zoom} | x: {x}, y: {y}, angle: {angle}, shear: {shear} | stretch_x: {stretch_x}, stretch_y: {stretch_y}"
        )

        if image.size(0) == 0:
            return (torch.zeros(0),)
        transformed_images = []

        new_height, new_width = (
            int(frame_height * zoom),
            int(frame_width * zoom),
        )

        log.debug(f"New height: {new_height}, New width: {new_width}")

        # - Calculate diagonal of the original image
        diagonal = sqrt(frame_width**2 + frame_height**2)
        max_padding = ceil(diagonal * zoom - min(frame_width, frame_height))
        # Calculate padding for zoom
        pw = int(frame_width - new_width)
        ph = int(frame_height - new_height)

        pw += abs(max_padding)
        ph += abs(max_padding)

        padding = [
            max(0, pw + x),
            max(0, ph + y),
            max(0, pw - x),
            max(0, ph - y),
        ]

        constant_color = hex_to_rgb(constant_color)
        log.debug(f"Fill Tuple: {constant_color}")

        for img in tensor2pil(image):
            img = TF.pad(
                img,
                padding=padding,
                padding_mode=border_handling,
                fill=constant_color or 0,
            )

            if stretch_x != 1.0 or stretch_y != 1.0:
                img = cast(
                    Image.Image,
                    TF.affine(
                        img,
                        angle=angle,
                        scale=zoom,
                        translate=[x, y],
                        shear=shear,
                        interpolation=resampling_filter,
                    ),
                )

                width, height = img.size
                center = (width // 2, height // 2)

                stretch_x_factor = 1.0 / stretch_x
                stretch_y_factor = 1.0 / stretch_y

                matrix = [
                    stretch_x_factor,
                    0,
                    center[0] - center[0] * stretch_x_factor,
                    0,
                    stretch_y_factor,
                    center[1] - center[1] * stretch_y_factor,
                ]

                img = img.transform(
                    img.size, Image.AFFINE, matrix, resampling_filter
                )
            else:
                img = cast(
                    Image.Image,
                    TF.affine(
                        img,
                        angle=angle,
                        scale=zoom,
                        translate=[x, y],
                        shear=shear,
                        interpolation=resampling_filter,
                    ),
                )

            left = abs(padding[0])
            upper = abs(padding[1])
            right = img.width - abs(padding[2])
            bottom = img.height - abs(padding[3])

            # log.debug("crop is [:,top:bottom, left:right] for tensors")
            log.debug("crop is [left, top, right, bottom] for PIL")
            log.debug(f"crop is {left}, {upper}, {right}, {bottom}")
            img = img.crop((left, upper, right, bottom))

            transformed_images.append(img)

        return (pil2tensor(transformed_images),)


__nodes__ = [MTB_TransformImage]
