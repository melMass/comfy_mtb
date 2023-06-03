import torch
from ..utils import tensor2pil, pil2tensor
from PIL import Image, ImageFilter, ImageDraw


class BoundingBox:
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


class Crop:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
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
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "BBOX")
    FUNCTION = "do_crop"

    CATEGORY = "image/crop"

    def do_crop(self, image: torch.Tensor, mask, x, y, width, height):

        image = image.numpy()
        mask = mask.numpy()
        cropped_image = image[:, y : y + height, x : x + width, :]
        cropped_mask = mask[y : y + height, x : x + width]
        crop_data = (x, y, width, height)

        return (
            torch.from_numpy(cropped_image),
            torch.from_numpy(cropped_mask),
            crop_data,
        )


class Uncrop:
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

        print(bbox)
        mask.paste(mask_block, (bb_x, bb_y, bb_x + bb_width, bb_y + bb_height))
        blend.paste(crop_img, (bb_x, bb_y, bb_x + bb_width, bb_y + bb_height))

        mask = mask.filter(ImageFilter.BoxBlur(radius=blend_ratio / 4))
        mask = mask.filter(ImageFilter.GaussianBlur(radius=blend_ratio / 4))

        blend.putalpha(mask)
        image = Image.alpha_composite(image.convert("RGBA"), blend)

        return (pil2tensor(image.convert("RGB")),)
