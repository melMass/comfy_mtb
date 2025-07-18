from typing import NamedTuple

import torch
import torchvision.transforms.functional as TF

from ..log import log


class BoundingBox(NamedTuple):
    """The bounding box tuple."""

    x: int
    y: int
    width: int
    height: int


class MTB_Bbox:
    """A literal bounding box."""

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

    def do_crop(
        self, x: int, y: int, width: int, height: int
    ) -> tuple[BoundingBox]:  # bbox
        return (BoundingBox(x, y, width, height),)


class MTB_SplitBbox:
    """Split the components of a bbox."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"bbox": ("BBOX",)},
        }

    CATEGORY = "mtb/crop"
    FUNCTION = "split_bbox"
    RETURN_TYPES = ("INT", "INT", "INT", "INT")
    RETURN_NAMES = ("x", "y", "width", "height")

    def split_bbox(self, bbox: BoundingBox) -> BoundingBox:
        return bbox


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

    def upscale(self, bbox: BoundingBox, scale: float) -> tuple[BoundingBox]:
        x, y, width, height = bbox

        center_x = x + width / 2
        center_y = y + height / 2

        new_width = int(width * scale)
        new_height = int(height * scale)

        new_x = int(center_x - new_width / 2)
        new_y = int(center_y - new_height / 2)

        return (BoundingBox(new_x, new_y, new_width, new_height),)


class MTB_BboxFromMask:
    """From a mask extract the bounding box."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "invert": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Optional image"}),
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
        self,
        mask: torch.Tensor,
        *,
        invert: bool = False,
        image: torch.Tensor | None = None,
    ) -> tuple[BoundingBox, torch.Tensor | None]:
        mask = 1 - mask if invert else mask
        non_zero_indices = torch.nonzero(mask)

        if non_zero_indices.numel() == 0:
            log.warning(
                "BboxFromMask: Mask is empty. Returning a (0,0,0,0) bbox."
            )
            return (BoundingBox(0, 0, 0, 0), image)

        min_coords = torch.min(non_zero_indices, dim=0).values
        max_coords = torch.max(non_zero_indices, dim=0).values

        min_y, min_x = min_coords[1].item(), min_coords[2].item()
        max_y, max_x = max_coords[1].item(), max_coords[2].item()

        width = max_x - min_x + 1
        height = max_y - min_y + 1

        bounding_box = BoundingBox(
            int(min_x), int(min_y), int(width), int(height)
        )

        cropped_image = None
        if image is not None:
            cropped_image = image[:, min_y : max_y + 1, min_x : max_x + 1, :]

        return (bounding_box, cropped_image)


class MTB_Crop:
    """Crop an image and an optional mask to a given bounding box.

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
        *,
        mask: torch.Tensor | None = None,
        x: int = 0,
        y: int = 0,
        width: int = 256,
        height: int = 256,
        bbox: BoundingBox | None = None,
    ):
        if bbox is not None:
            x, y, width, height = bbox

        if width <= 0 or height <= 0:
            log.error(
                "Crop dimensions must be positive. Check the BBOX or widget inputs."
            )
            return (
                torch.zeros_like(image),
                torch.zeros_like(mask) if mask is not None else None,
                (x, y, width, height),
            )

        cropped_image = image[:, y : y + height, x : x + width, :]
        cropped_mask = (
            mask[:, y : y + height, x : x + width]
            if mask is not None
            else None
        )
        crop_data = BoundingBox(x, y, width, height)

        return (
            cropped_image,
            cropped_mask if cropped_mask is not None else None,
            crop_data,
        )


# def calculate_intersection(rect1, rect2):
#     x_left = max(rect1[0], rect2[0])
#     y_top = max(rect1[1], rect2[1])
#     x_right = min(rect1[2], rect2[2])
#     y_bottom = min(rect1[3], rect2[3])

#     return (x_left, y_top, x_right, y_bottom)


def bbox_check(bbox: BoundingBox, target_size: tuple[int, int] | None = None):
    if not target_size:
        return bbox

    new_bbox = BoundingBox(
        bbox.x,
        bbox.y,
        min(target_size[0] - bbox.x, bbox.width),
        min(target_size[1] - bbox.y, bbox.height),
    )
    if new_bbox != bbox:
        log.warning(f"BBox too big, constrained to {new_bbox}")

    return new_bbox


def bbox_to_region(
    bbox: BoundingBox, target_size: tuple[int, int] | None = None
):
    bbox = bbox_check(bbox, target_size)

    # to region
    return (bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height)


class MTB_Uncrop:
    """Uncrop an image to a given bounding box."""

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
    FUNCTION = "do_uncrop"
    CATEGORY = "mtb/crop"

    def do_uncrop(
        self,
        image: torch.Tensor,
        crop_image: torch.Tensor,
        bbox: BoundingBox,
        border_blending: float = 0.25,
    ):
        if len(image) > 1 and len(image) != len(crop_image):
            raise ValueError(
                "Uncrop: Batch size of background 'image' must be 1 or match the 'crop_image' batch size."
            )
        import comfy.utils

        pbar = comfy.utils.ProgressBar(4)

        device = image.device

        log.debug(f"Working on device: {device}")

        crop_image = crop_image.to(device)

        if len(image) == 1 and len(crop_image) > 1:
            image = image.repeat(len(crop_image), 1, 1, 1)

        batch_size, bg_h, bg_w, _ = image.shape
        _, fg_h, fg_w, _ = crop_image.shape
        x, y, width, height = bbox

        if (width, height) != (fg_w, fg_h):
            log.warning(
                f"Uncrop: crop_image size {(fg_w, fg_h)} "
                "differs from bbox {(width, height)}. Resizing to fit bbox."
            )

        resized_crop = crop_image.permute(0, 3, 1, 2)
        resized_crop = torch.nn.functional.interpolate(
            resized_crop,
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        )
        resized_crop = resized_crop.permute(0, 2, 3, 1)

        pbar.update(1)
        # paste coords
        paste_x1 = max(x, 0)
        paste_y1 = max(y, 0)
        paste_x2 = min(x + width, bg_w)
        paste_y2 = min(y + height, bg_h)

        # region from crop (bound)
        crop_x1 = max(0, -x)
        crop_y1 = max(0, -y)
        crop_x2 = crop_x1 + (paste_x2 - paste_x1)
        crop_y2 = crop_y1 + (paste_y2 - paste_y1)

        if paste_x1 >= paste_x2 or paste_y1 >= paste_y2:
            log.warning(
                "Uncrop: BBOX is entirely outside the image boundaries. Returning original image."
            )
            return (image,)

        pbar.update(1)
        source_slice = resized_crop[:, crop_y1:crop_y2, crop_x1:crop_x2, :]

        final_image = image.clone()
        final_image[:, paste_y1:paste_y2, paste_x1:paste_x2, :] = source_slice

        pbar.update(1)

        blend_radius = int(max(width, height) * border_blending * 0.5)
        if blend_radius > 0:
            _device = device
            if torch.cuda.is_available():
                _device = torch.device("cuda")

            log.debug("Processing blending")
            alpha_mask = torch.zeros((batch_size, bg_h, bg_w), device=_device)
            alpha_mask[:, paste_y1:paste_y2, paste_x1:paste_x2] = 1.0

            kernel_size = 2 * blend_radius + 1

            log.debug("Gaussian blur...")
            alpha_mask = TF.gaussian_blur(
                alpha_mask.unsqueeze(1), kernel_size=[kernel_size, kernel_size]
            ).squeeze(1)
            alpha_mask = alpha_mask.unsqueeze(-1)

            log.debug("Applying blending")
            final_image = final_image.to(_device) * alpha_mask + image.to(
                _device
            ) * (1.0 - alpha_mask)

        pbar.update(1)
        return (final_image.to(device),)


class MTB_BBoxForceDimensions:
    """
    Resize a BBOX to new dimensions while keeping its center.

    Optionally constrains the BBOX to stay within image boundaries.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bbox": ("BBOX",),
                "width": ("INT", {"default": 512, "min": 1, "max": 8192}),
                "height": ("INT", {"default": 512, "min": 1, "max": 8192}),
                "constrain_to_image": ("BOOLEAN", {"default": True}),
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
        *,
        bbox: tuple[int, int, int, int],
        width: int,
        height: int,
        constrain_to_image: bool = True,
        image: torch.Tensor | None = None,
    ) -> tuple[tuple[int, int, int, int]]:
        x, y, curr_width, curr_height = bbox

        center_x = x + curr_width // 2
        center_y = y + curr_height // 2

        new_x = center_x - width // 2
        new_y = center_y - height // 2

        if constrain_to_image and image is not None:
            img_height, img_width = image.shape[1:3]
            new_x = max(0, min(new_x, img_width - width))
            new_y = max(0, min(new_y, img_height - height))
            width = min(width, img_width)
            height = min(height, img_height)

        return ((new_x, new_y, width, height),)


__nodes__ = [
    MTB_BboxFromMask,
    MTB_Bbox,
    MTB_Crop,
    MTB_Uncrop,
    MTB_SplitBbox,
    MTB_UpscaleBboxBy,
    MTB_BBoxForceDimensions,
]
