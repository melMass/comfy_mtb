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
    ):
        x = int(x)
        y = int(y)
        angle = int(angle)

        log.debug(
            f"Zoom: {zoom} | x: {x}, y: {y}, angle: {angle}, shear: {shear}"
        )

        if image.size(0) == 0:
            return (torch.zeros(0),)
        transformed_images = []
        frames_count, frame_height, frame_width, frame_channel_count = (
            image.size()
        )

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
                img,  # transformed_frame,
                padding=padding,
                padding_mode=border_handling,
                fill=constant_color or 0,
            )

            img = cast(
                Image.Image,
                TF.affine(
                    img, angle=angle, scale=zoom, translate=[x, y], shear=shear
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
