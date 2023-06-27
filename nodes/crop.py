import torch
from ..utils import tensor2pil, pil2tensor
from PIL import Image, ImageFilter, ImageDraw
import numpy as np


class BoundingBox:
    """The bounding box (BBOX) custom type used by other nodes"""
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "x": ("INT", {"default": 0, "max": 10000000, "min": 0, "step": 1}),
                "y": ("INT", {"default": 0, "max": 10000000, "min": 0, "step": 1}),
                "width": (
                    "INT",
                    {"default": 256, "max": 10000000, "min": 0, "step": 1},
                ),
                "height": (
                    "INT",
                    {"default": 256, "max": 10000000, "min": 0, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("BBOX",)
    FUNCTION = "do_crop"
    CATEGORY = "image/crop"

    def do_crop(self, x, y, width, height):
        return (x, y, width, height)


class BBoxFromMask:
    """From a mask extract the bounding box"""
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = (
        "BBOX",
        "IMAGE",
    )
    RETURN_NAMES = (
        "bbox",
        "image (optional)",
    )
    FUNCTION = "extract_bounding_box"
    CATEGORY = "image/crop"

    def extract_bounding_box(self, mask: torch.Tensor, image=None):

        mask = tensor2pil(mask)

        alpha_channel = np.array(mask)
        non_zero_indices = np.nonzero(alpha_channel)

        min_x, max_x = np.min(non_zero_indices[1]), np.max(non_zero_indices[1])
        min_y, max_y = np.min(non_zero_indices[0]), np.max(non_zero_indices[0])

        # Create a bounding box tuple
        if image != None:
            # Convert the image to a NumPy array
            image = image.numpy()
            # Crop the image from the bounding box
            image = image[:, min_y:max_y, min_x:max_x]
            image = torch.from_numpy(image)

        bounding_box = (min_x, min_y, max_x - min_x, max_y - min_y)
        return (
            bounding_box,
            image,
        )


class Crop:
    """Crops an image and an optional mask to a given bounding box

    The bounding box can be given as a tuple of (x, y, width, height) or as a BBOX type
    The BBOX input takes precedence over the tuple input
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "mask": ("MASK",),
                "x": ("INT", {"default": 0, "max": 10000000, "min": 0, "step": 1}),
                "y": ("INT", {"default": 0, "max": 10000000, "min": 0, "step": 1}),
                "width": (
                    "INT",
                    {"default": 256, "max": 10000000, "min": 0, "step": 1},
                ),
                "height": (
                    "INT",
                    {"default": 256, "max": 10000000, "min": 0, "step": 1},
                ),
                "bbox": ("BBOX",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "BBOX")
    FUNCTION = "do_crop"

    CATEGORY = "image/crop"

    def do_crop(
        self, image: torch.Tensor, mask=None, x=0, y=0, width=256, height=256, bbox=None
    ):

        image = image.numpy()
        if mask:
            mask = mask.numpy()

        if bbox != None:
            x, y, width, height = bbox

        cropped_image = image[:, y : y + height, x : x + width, :]
        cropped_mask = mask[y : y + height, x : x + width] if mask != None else None
        crop_data = (x, y, width, height)

        return (
            torch.from_numpy(cropped_image),
            torch.from_numpy(cropped_mask) if mask != None else None,
            crop_data,
        )


class Uncrop:
    """Uncrops an image to a given bounding box

    The bounding box can be given as a tuple of (x, y, width, height) or as a BBOX type
    The BBOX input takes precedence over the tuple input"""
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "crop_image": ("IMAGE",),
                "bbox": ("BBOX",),
                "border_blending": (
                    "FLOAT",
                    {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "do_crop"

    CATEGORY = "image/crop"

    def do_crop(self, image, crop_image, bbox, border_blending):
        def inset_border(image, border_width=20, border_color=(0)):
            width, height = image.size
            bordered_image = Image.new(image.mode, (width, height), border_color)
            bordered_image.paste(image, (0, 0))
            draw = ImageDraw.Draw(bordered_image)
            draw.rectangle(
                (0, 0, width - 1, height - 1), outline=border_color, width=border_width
            )
            return bordered_image

        image = tensor2pil(image)
        crop_img = tensor2pil(crop_image)
        crop_img = crop_img.convert("RGB")

        # uncrop the image based on the bounding box
        bb_x, bb_y, bb_width, bb_height = bbox

        if border_blending > 1.0:
            border_blending = 1.0
        elif border_blending < 0.0:
            border_blending = 0.0

        blend_ratio = (max(crop_img.size) / 2) * float(border_blending)

        blend = image.convert("RGBA")
        mask = Image.new("L", image.size, 0)

        mask_block = Image.new("L", (bb_width, bb_height), 255)
        mask_block = inset_border(mask_block, int(blend_ratio / 2), (0))

        mask.paste(mask_block, (bb_x, bb_y, bb_x + bb_width, bb_y + bb_height))
        blend.paste(crop_img, (bb_x, bb_y, bb_x + bb_width, bb_y + bb_height))

        mask = mask.filter(ImageFilter.BoxBlur(radius=blend_ratio / 4))
        mask = mask.filter(ImageFilter.GaussianBlur(radius=blend_ratio / 4))

        blend.putalpha(mask)
        image = Image.alpha_composite(image.convert("RGBA"), blend)

        return (pil2tensor(image.convert("RGB")),)


__nodes__ = [
    BBoxFromMask,
    BoundingBox,
    Crop,
    Uncrop
]