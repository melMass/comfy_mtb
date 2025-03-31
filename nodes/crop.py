import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

from ..log import log
from ..utils import np2tensor, pil2tensor, tensor2np, tensor2pil


class MTB_Bbox:
    """The bounding box (BBOX) custom type used by other nodes"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # "bbox": ("BBOX",),
                "x": (
                    "INT",
                    {"default": 0, "max": 10000000, "min": 0, "step": 1},
                ),
                "y": (
                    "INT",
                    {"default": 0, "max": 10000000, "min": 0, "step": 1},
                ),
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
    CATEGORY = "mtb/crop"

    def do_crop(self, x: int, y: int, width: int, height: int):  # bbox
        return ((x, y, width, height),)


class MTB_SplitBbox:
    """Split the components of a bbox"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"bbox": ("BBOX",)},
        }

    CATEGORY = "mtb/crop"
    FUNCTION = "split_bbox"
    RETURN_TYPES = ("INT", "INT", "INT", "INT")
    RETURN_NAMES = ("x", "y", "width", "height")

    def split_bbox(self, bbox):
        return (bbox[0], bbox[1], bbox[2], bbox[3])


class MTB_UpscaleBboxBy:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bbox": ("BBOX",),
                "scale": ("FLOAT", {"default": 1.0}),
            },
        }

    CATEGORY = "mtb/crop"
    RETURN_TYPES = ("BBOX",)

    FUNCTION = "upscale"

    def upscale(
        self, bbox: tuple[int, int, int, int], scale: float
    ) -> tuple[tuple[int, int, int, int]]:
        x, y, width, height = bbox

        center_x = x + width // 2
        center_y = y + height // 2

        new_width = int(width * scale)
        new_height = int(height * scale)

        new_x = center_x - new_width // 2
        new_y = center_y - new_height // 2

        scaled = (new_x, new_y, new_width, new_height)
        return (scaled,)


class MTB_BboxFromMask:
    """From a mask extract the bounding box"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "invert": ("BOOLEAN", {"default": False}),
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
    CATEGORY = "mtb/crop"

    def extract_bounding_box(
        self, mask: torch.Tensor, invert: bool, image=None
    ):
        # if image != None:
        #     if mask.size(0) != image.size(0):
        #         if mask.size(0) != 1:
        #             log.error(
        #                 f"Batch count mismatch for mask and image, it can either be 1 mask for X images, or X masks for X images (mask: {mask.shape} | image: {image.shape})"
        #             )

        #             raise Exception(
        #                 f"Batch count mismatch for mask and image, it can either be 1 mask for X images, or X masks for X images (mask: {mask.shape} | image: {image.shape})"
        #             )

        # we invert it
        _mask = tensor2pil(1.0 - mask)[0] if invert else tensor2pil(mask)[0]
        alpha_channel = np.array(_mask)

        non_zero_indices = np.nonzero(alpha_channel)

        min_x, max_x = np.min(non_zero_indices[1]), np.max(non_zero_indices[1])
        min_y, max_y = np.min(non_zero_indices[0]), np.max(non_zero_indices[0])

        # Create a bounding box tuple
        if image != None:
            # Convert the image to a NumPy array
            imgs = tensor2np(image)
            out = []
            for img in imgs:
                # Crop the image from the bounding box
                img = img[min_y:max_y, min_x:max_x, :]
                log.debug(f"Cropped image to shape {img.shape}")
                out.append(img)

            image = np2tensor(out)
            log.debug(f"Cropped images shape: {image.shape}")
        bounding_box = (min_x, min_y, max_x - min_x, max_y - min_y)
        return (
            bounding_box,
            image,
        )


class MTB_Crop:
    """Crops an image and an optional mask to a given bounding box

    The bounding box can be given as a tuple of (x, y, width, height) or as a BBOX type
    The BBOX input takes precedence over the tuple input
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "mask": ("MASK",),
                "x": (
                    "INT",
                    {"default": 0, "max": 10000000, "min": 0, "step": 1},
                ),
                "y": (
                    "INT",
                    {"default": 0, "max": 10000000, "min": 0, "step": 1},
                ),
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

    CATEGORY = "mtb/crop"

    def do_crop(
        self,
        image: torch.Tensor,
        mask=None,
        x=0,
        y=0,
        width=256,
        height=256,
        bbox=None,
    ):
        image = image.numpy()
        if mask is not None:
            mask = mask.numpy()

        if bbox is not None:
            x, y, width, height = bbox

        cropped_image = image[:, y : y + height, x : x + width, :]
        cropped_mask = None
        if mask is not None:
            cropped_mask = (
                mask[:, y : y + height, x : x + width]
                if mask is not None
                else None
            )
        crop_data = (x, y, width, height)

        return (
            torch.from_numpy(cropped_image),
            torch.from_numpy(cropped_mask)
            if cropped_mask is not None
            else None,
            crop_data,
        )


# def calculate_intersection(rect1, rect2):
#     x_left = max(rect1[0], rect2[0])
#     y_top = max(rect1[1], rect2[1])
#     x_right = min(rect1[2], rect2[2])
#     y_bottom = min(rect1[3], rect2[3])

#     return (x_left, y_top, x_right, y_bottom)


def bbox_check(bbox, target_size=None):
    if not target_size:
        return bbox

    new_bbox = (
        bbox[0],
        bbox[1],
        min(target_size[0] - bbox[0], bbox[2]),
        min(target_size[1] - bbox[1], bbox[3]),
    )
    if new_bbox != bbox:
        log.warn(f"BBox too big, constrained to {new_bbox}")

    return new_bbox


def bbox_to_region(bbox, target_size=None):
    bbox = bbox_check(bbox, target_size)

    # to region
    return (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])


class MTB_Uncrop:
    """Uncrops an image to a given bounding box

    The bounding box can be given as a tuple of (x, y, width, height) or as a BBOX type
    The BBOX input takes precedence over the tuple input
    """

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

    CATEGORY = "mtb/crop"

    def do_crop(self, image, crop_image, bbox, border_blending):
        def inset_border(image, border_width=20, border_color=(0)):
            width, height = image.size
            bordered_image = Image.new(
                image.mode, (width, height), border_color
            )
            bordered_image.paste(image, (0, 0))
            draw = ImageDraw.Draw(bordered_image)
            draw.rectangle(
                (0, 0, width - 1, height - 1),
                outline=border_color,
                width=border_width,
            )
            return bordered_image

        single = image.size(0) == 1
        if image.size(0) != crop_image.size(0):
            if not single:
                raise ValueError(
                    "The Image batch count is greater than 1, but doesn't match the crop_image batch count. If using batches they should either match or only crop_image must be greater than 1"
                )

        images = tensor2pil(image)
        crop_imgs = tensor2pil(crop_image)
        out_images = []
        for i, crop in enumerate(crop_imgs):
            if single:
                img = images[0]
            else:
                img = images[i]

            # uncrop the image based on the bounding box
            bb_x, bb_y, bb_width, bb_height = bbox

            paste_region = bbox_to_region(
                (bb_x, bb_y, bb_width, bb_height), img.size
            )
            # log.debug(f"Paste region: {paste_region}")
            # new_region = adjust_paste_region(img.size, paste_region)
            # log.debug(f"Adjusted paste region: {new_region}")
            # # Check if the adjusted paste region is different from the original

            crop_img = crop.convert("RGB")

            log.debug(f"Crop image size: {crop_img.size}")
            log.debug(f"Image size: {img.size}")

            if border_blending > 1.0:
                border_blending = 1.0
            elif border_blending < 0.0:
                border_blending = 0.0

            blend_ratio = (max(crop_img.size) / 2) * float(border_blending)

            blend = img.convert("RGBA")
            mask = Image.new("L", img.size, 0)

            mask_block = Image.new("L", (bb_width, bb_height), 255)
            mask_block = inset_border(mask_block, int(blend_ratio / 2), (0))

            mask.paste(mask_block, paste_region)
            log.debug(f"Blend size: {blend.size} | kind {blend.mode}")
            log.debug(
                f"Crop image size: {crop_img.size} | kind {crop_img.mode}"
            )
            log.debug(f"BBox: {paste_region}")
            blend.paste(crop_img, paste_region)

            mask = mask.filter(ImageFilter.BoxBlur(radius=blend_ratio / 4))
            mask = mask.filter(
                ImageFilter.GaussianBlur(radius=blend_ratio / 4)
            )

            blend.putalpha(mask)
            img = Image.alpha_composite(img.convert("RGBA"), blend)
            out_images.append(img.convert("RGB"))

        return (pil2tensor(out_images),)


class MTB_BBoxForceDimensions:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bbox": ("BBOX",),
                "width": ("INT", {"default": 512, "min": 1, "max": 8192}),
                "height": ("INT", {"default": 512, "min": 1, "max": 8192}),
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }

    CATEGORY = "mtb/crop"
    RETURN_TYPES = ("BBOX",)
    FUNCTION = "force_dimensions"

    def force_dimensions(
        self,
        bbox: tuple[int, int, int, int],
        width: int,
        height: int,
        image: torch.Tensor = None,
    ) -> tuple[tuple[int, int, int, int]]:
        x, y, curr_width, curr_height = bbox

        center_x = x + curr_width // 2
        center_y = y + curr_height // 2

        new_x = center_x - width // 2
        new_y = center_y - height // 2

        if image is not None:
            img_height, img_width = image.shape[1:3]
            x_overflow = max(0, new_x + width - img_width) + min(0, new_x)
            y_overflow = max(0, new_y + height - img_height) + min(0, new_y)
            if width > img_width or height > img_height:
                x_exceed = width - img_width if width > img_width else 0
                y_exceed = height - img_height if height > img_height else 0
                raise ValueError(
                    f"Target bbox dimensions ({width}x{height}) exceed image bounds ({img_width}x{img_height}) "
                    f"by {x_exceed}px horizontally and {y_exceed}px vertically"
                )

            if x_overflow > 0 or x_overflow < 0:
                new_x -= x_overflow

            if y_overflow > 0:
                new_y -= y_overflow
            elif y_overflow < 0:
                new_y -= y_overflow  # Add the negative overflow

        return ((int(new_x), int(new_y), width, height),)


__nodes__ = [
    MTB_BboxFromMask,
    MTB_Bbox,
    MTB_Crop,
    MTB_Uncrop,
    MTB_SplitBbox,
    MTB_UpscaleBboxBy,
    MTB_BBoxForceDimensions,
]
