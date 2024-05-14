import itertools
import json
import math
import os

import folder_paths
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo
from skimage.filters import gaussian
from skimage.util import compare_images

from ..log import log
from ..utils import np2tensor, pil2tensor, tensor2np, tensor2pil

# try:
#     from cv2.ximgproc import guidedFilter
# except ImportError:
#     log.warning("cv2.ximgproc.guidedFilter not found, use opencv-contrib-python")


def gaussian_kernel(
    kernel_size: int, sigma_x: float, sigma_y: float, device=None
):
    x, y = torch.meshgrid(
        torch.linspace(-1, 1, kernel_size, device=device),
        torch.linspace(-1, 1, kernel_size, device=device),
        indexing="ij",
    )
    d_x = x * x / (2.0 * sigma_x * sigma_x)
    d_y = y * y / (2.0 * sigma_y * sigma_y)
    g = torch.exp(-(d_x + d_y))
    return g / g.sum()


class MTB_ColorCorrect:
    """Various color correction methods"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "clamp": ([True, False], {"default": True}),
                "gamma": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01},
                ),
                "contrast": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01},
                ),
                "exposure": (
                    "FLOAT",
                    {"default": 0.0, "min": -5.0, "max": 5.0, "step": 0.01},
                ),
                "offset": (
                    "FLOAT",
                    {"default": 0.0, "min": -5.0, "max": 5.0, "step": 0.01},
                ),
                "hue": (
                    "FLOAT",
                    {"default": 0.0, "min": -0.5, "max": 0.5, "step": 0.01},
                ),
                "saturation": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01},
                ),
                "value": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "correct"
    CATEGORY = "mtb/image processing"

    @staticmethod
    def gamma_correction_tensor(image, gamma):
        gamma_inv = 1.0 / gamma
        return image.pow(gamma_inv)

    @staticmethod
    def contrast_adjustment_tensor(image, contrast):
        contrasted = (image - 0.5) * contrast + 0.5
        return torch.clamp(contrasted, 0.0, 1.0)

    @staticmethod
    def exposure_adjustment_tensor(image, exposure):
        return image * (2.0**exposure)

    @staticmethod
    def offset_adjustment_tensor(image, offset):
        return image + offset

    @staticmethod
    def hsv_adjustment(image: torch.Tensor, hue, saturation, value):
        images = tensor2pil(image)
        out = []
        for img in images:
            hsv_image = img.convert("HSV")

            h, s, v = hsv_image.split()

            h = h.point(lambda x: (x + hue * 255) % 256)
            s = s.point(lambda x: int(x * saturation))
            v = v.point(lambda x: int(x * value))

            hsv_image = Image.merge("HSV", (h, s, v))
            rgb_image = hsv_image.convert("RGB")
            out.append(rgb_image)
        return pil2tensor(out)

    @staticmethod
    def hsv_adjustment_tensor_not_working(
        image: torch.Tensor, hue, saturation, value
    ):
        """Abandonning for now"""
        image = image.squeeze(0).permute(2, 0, 1)

        max_val, _ = image.max(dim=0, keepdim=True)
        min_val, _ = image.min(dim=0, keepdim=True)
        delta = max_val - min_val

        hue_image = torch.zeros_like(max_val)
        mask = delta != 0.0

        r, g, b = image[0], image[1], image[2]
        hue_image[mask & (max_val == r)] = ((g - b) / delta)[
            mask & (max_val == r)
        ] % 6.0
        hue_image[mask & (max_val == g)] = ((b - r) / delta)[
            mask & (max_val == g)
        ] + 2.0
        hue_image[mask & (max_val == b)] = ((r - g) / delta)[
            mask & (max_val == b)
        ] + 4.0

        saturation_image = delta / (max_val + 1e-7)
        value_image = max_val

        hue_image = (hue_image + hue) % 1.0
        saturation_image = torch.where(
            mask, saturation * saturation_image, saturation_image
        )
        value_image = value * value_image

        c = value_image * saturation_image
        x = c * (1 - torch.abs((hue_image % 2) - 1))
        m = value_image - c

        prime_image = torch.zeros_like(image)
        prime_image[0] = torch.where(
            max_val == r, c, torch.where(max_val == g, x, prime_image[0])
        )
        prime_image[1] = torch.where(
            max_val == r, x, torch.where(max_val == g, c, prime_image[1])
        )
        prime_image[2] = torch.where(
            max_val == g, x, torch.where(max_val == b, c, prime_image[2])
        )

        rgb_image = prime_image + m

        rgb_image = rgb_image.permute(1, 2, 0).unsqueeze(0)

        return rgb_image

    def correct(
        self,
        image: torch.Tensor,
        clamp: bool,
        gamma: float = 1.0,
        contrast: float = 1.0,
        exposure: float = 0.0,
        offset: float = 0.0,
        hue: float = 0.0,
        saturation: float = 1.0,
        value: float = 1.0,
    ):
        # Apply color correction operations
        image = self.gamma_correction_tensor(image, gamma)
        image = self.contrast_adjustment_tensor(image, contrast)
        image = self.exposure_adjustment_tensor(image, exposure)
        image = self.offset_adjustment_tensor(image, offset)
        image = self.hsv_adjustment(image, hue, saturation, value)

        if clamp:
            image = torch.clamp(image, 0.0, 1.0)

        return (image,)


class MTB_ImageCompare:
    """Compare two images and return a difference image"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "imageA": ("IMAGE",),
                "imageB": ("IMAGE",),
                "mode": (
                    ["checkerboard", "diff", "blend"],
                    {"default": "checkerboard"},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "compare"
    CATEGORY = "mtb/image"

    def compare(self, imageA: torch.Tensor, imageB: torch.Tensor, mode):
        if imageA.dim() == 4:
            batch_count = imageA.size(0)
            return (
                torch.cat(
                    tuple(
                        self.compare(imageA[i], imageB[i], mode)[0]
                        for i in range(batch_count)
                    ),
                    dim=0,
                ),
            )

        num_channels_A = imageA.size(2)
        num_channels_B = imageB.size(2)

        # handle RGBA/RGB mismatch
        if num_channels_A == 3 and num_channels_B == 4:
            imageA = torch.cat(
                (imageA, torch.ones_like(imageA[:, :, 0:1])), dim=2
            )
        elif num_channels_B == 3 and num_channels_A == 4:
            imageB = torch.cat(
                (imageB, torch.ones_like(imageB[:, :, 0:1])), dim=2
            )
        match mode:
            case "diff":
                compare_image = torch.abs(imageA - imageB)
            case "blend":
                compare_image = 0.5 * (imageA + imageB)
            case "checkerboard":
                imageA = imageA.numpy()
                imageB = imageB.numpy()
                compared_channels = [
                    torch.from_numpy(
                        compare_images(
                            imageA[:, :, i], imageB[:, :, i], method=mode
                        )
                    )
                    for i in range(imageA.shape[2])
                ]

                compare_image = torch.stack(compared_channels, dim=2)
            case _:
                compare_image = None
                raise ValueError(f"Unknown mode {mode}")

        compare_image = compare_image.unsqueeze(0)

        return (compare_image,)


import requests


class MTB_LoadImageFromUrl:
    """Load an image from the given URL"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": (
                    "STRING",
                    {
                        "default": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Example.jpg/800px-Example.jpg"
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load"
    CATEGORY = "mtb/IO"

    def load(self, url):
        # get the image from the url
        image = Image.open(requests.get(url, stream=True).raw)
        image = ImageOps.exif_transpose(image)
        return (pil2tensor(image),)


class MTB_Blur:
    """Blur an image using a Gaussian filter."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "sigmaX": (
                    "FLOAT",
                    {"default": 3.0, "min": 0.0, "max": 200.0, "step": 0.01},
                ),
                "sigmaY": (
                    "FLOAT",
                    {"default": 3.0, "min": 0.0, "max": 200.0, "step": 0.01},
                ),
            },
            "optional": {"sigmasX": ("FLOATS",), "sigmasY": ("FLOATS",)},
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "blur"
    CATEGORY = "mtb/image processing"

    def blur(
        self, image: torch.Tensor, sigmaX, sigmaY, sigmasX=None, sigmasY=None
    ):
        image_np = image.numpy() * 255

        blurred_images = []
        if sigmasX is not None:
            if sigmasY is None:
                sigmasY = sigmasX
            if len(sigmasX) != image.size(0):
                raise ValueError(
                    f"SigmasX must have same length as image, sigmasX is {len(sigmasX)} but the batch size is {image.size(0)}"
                )

            for i in range(image.size(0)):
                blurred = gaussian(
                    image_np[i],
                    sigma=(sigmasX[i], sigmasY[i], 0),
                    channel_axis=2,
                )
                blurred_images.append(blurred)

            image_np = np.array(blurred_images)
        else:
            for i in range(image.size(0)):
                blurred = gaussian(
                    image_np[i], sigma=(sigmaX, sigmaY, 0), channel_axis=2
                )
                blurred_images.append(blurred)

            image_np = np.array(blurred_images)
        return (np2tensor(image_np).squeeze(0),)


class MTB_Sharpen:
    """Sharpens an image using a Gaussian kernel."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "sharpen_radius": (
                    "INT",
                    {"default": 1, "min": 1, "max": 31, "step": 1},
                ),
                "sigma_x": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1},
                ),
                "sigma_y": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1},
                ),
                "alpha": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "do_sharp"
    CATEGORY = "mtb/image processing"

    def do_sharp(
        self,
        image: torch.Tensor,
        sharpen_radius: int,
        sigma_x: float,
        sigma_y: float,
        alpha: float,
    ):
        if sharpen_radius == 0:
            return (image,)

        channels = image.shape[3]

        kernel_size = 2 * sharpen_radius + 1
        kernel = gaussian_kernel(kernel_size, sigma_x, sigma_y) * -(alpha * 10)

        # Modify center of kernel to make it a sharpening kernel
        center = kernel_size // 2
        kernel[center, center] = kernel[center, center] - kernel.sum() + 1.0

        kernel = kernel.repeat(channels, 1, 1).unsqueeze(1)
        tensor_image = image.permute(0, 3, 1, 2)

        tensor_image = F.pad(
            tensor_image,
            (sharpen_radius, sharpen_radius, sharpen_radius, sharpen_radius),
            "reflect",
        )
        sharpened = F.conv2d(
            tensor_image, kernel, padding=center, groups=channels
        )

        # Remove padding
        sharpened = sharpened[
            :,
            :,
            sharpen_radius:-sharpen_radius,
            sharpen_radius:-sharpen_radius,
        ]

        sharpened = sharpened.permute(0, 2, 3, 1)
        result = torch.clamp(sharpened, 0, 1)

        return (result,)


# https://github.com/lllyasviel/AdverseCleaner/blob/main/clean.py
# def deglaze_np_img(np_img):
#     y = np_img.copy()
#     for _ in range(64):
#         y = cv2.bilateralFilter(y, 5, 8, 8)
#     for _ in range(4):
#         y = guidedFilter(np_img, y, 4, 16)
#     return y


# class DeglazeImage:
#     """Remove adversarial noise from images"""

#     @classmethod
#     def INPUT_TYPES(cls):
#         return {"required": {"image": ("IMAGE",)}}

#     CATEGORY = "mtb/image processing"

#     RETURN_TYPES = ("IMAGE",)
#     FUNCTION = "deglaze_image"

#     def deglaze_image(self, image):
#         return (np2tensor(deglaze_np_img(tensor2np(image))),)


class MTB_MaskToImage:
    """Converts a mask (alpha) to an RGB image with a color and background"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "color": ("COLOR",),
                "background": ("COLOR", {"default": "#000000"}),
            }
        }

    CATEGORY = "mtb/generate"

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "render_mask"

    def render_mask(self, mask, color, background):
        masks = tensor2np(mask)[0]
        images = []
        for m in masks:
            _mask = Image.fromarray(m).convert("L")

            log.debug(
                f"Converted mask to PIL Image format, size: {_mask.size}"
            )

            image = Image.new("RGBA", _mask.size, color=color)
            # apply the mask
            image = Image.composite(
                image, Image.new("RGBA", _mask.size, color=background), _mask
            )

            # image = ImageChops.multiply(image, mask)
            # apply over background
            # image = Image.alpha_composite(Image.new("RGBA", image.size, color=background), image)

            images.append(image.convert("RGB"))

        return (pil2tensor(images),)


class MTB_ColoredImage:
    """Constant color image of given size."""

    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "color": ("COLOR",),
                "width": ("INT", {"default": 512, "min": 16, "max": 8160}),
                "height": ("INT", {"default": 512, "min": 16, "max": 8160}),
            },
            "optional": {
                "foreground_image": ("IMAGE",),
                "foreground_mask": ("MASK",),
            },
        }

    CATEGORY = "mtb/generate"

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "render_img"

    def resize_and_crop(self, img, target_size):
        # Calculate scaling factors for both dimensions
        scale_x = target_size[0] / img.width
        scale_y = target_size[1] / img.height

        # Use the smaller scaling factor to maintain aspect ratio
        scale = max(scale_x, scale_y)

        # Resize the image based on calculated scale
        new_size = (int(img.width * scale), int(img.height * scale))
        img = img.resize(new_size, Image.LANCZOS)

        # Calculate cropping coordinates
        left = (img.width - target_size[0]) / 2
        top = (img.height - target_size[1]) / 2
        right = (img.width + target_size[0]) / 2
        bottom = (img.height + target_size[1]) / 2

        # Crop and return the image
        return img.crop((left, top, right, bottom))

    def resize_and_crop_thumbnails(self, img, target_size):
        img.thumbnail(target_size, Image.LANCZOS)
        left = (img.width - target_size[0]) / 2
        top = (img.height - target_size[1]) / 2
        right = (img.width + target_size[0]) / 2
        bottom = (img.height + target_size[1]) / 2
        return img.crop((left, top, right, bottom))

    def render_img(
        self,
        color,
        width,
        height,
        foreground_image: torch.Tensor | None = None,
        foreground_mask: torch.Tensor | None = None,
    ):
        image = Image.new("RGBA", (width, height), color=color)
        output = []
        if foreground_image is not None:
            fg_masks = [None] * foreground_image.size()[0]

            if foreground_mask is not None:
                fg_size = foreground_image.size()[0]
                mask_size = foreground_mask.size()[0]

                if fg_size == 1 and mask_size > fg_size:
                    foreground_image = foreground_image.repeat(
                        mask_size, 1, 1, 1
                    )

                if foreground_image.size()[0] != foreground_mask.size()[0]:
                    raise ValueError(
                        "Foreground image and mask must have same batch size"
                    )
                fg_masks = tensor2pil(foreground_mask.unsqueeze(-1))

            fg_images = tensor2pil(foreground_image)

            for fg_image, fg_mask in zip(fg_images, fg_masks):
                # Resize and crop if dimensions mismatch
                if fg_image.size != image.size:
                    fg_image = self.resize_and_crop(fg_image, image.size)
                    if fg_mask:
                        fg_mask = self.resize_and_crop(fg_mask, image.size)

                if fg_mask:
                    output.append(
                        Image.composite(
                            fg_image.convert("RGBA"),
                            image,
                            fg_mask,
                        ).convert("RGB")
                    )
                else:
                    if fg_image.mode != "RGBA":
                        raise ValueError(
                            "Foreground image must be in 'RGBA' mode "
                            f"when no mask is provided, got {fg_image.mode}"
                        )
                    output.append(
                        Image.alpha_composite(image, fg_image).convert("RGB")
                    )

        else:
            if foreground_mask is not None:
                log.warn("Mask ignored because no foreground image is given")
            output.append(image.convert("RGB"))

        output = pil2tensor(output)

        return (output,)


class MTB_ImagePremultiply:
    """Premultiply image with mask"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "invert": ("BOOLEAN", {"default": False}),
            }
        }

    CATEGORY = "mtb/image"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("RGBA",)
    FUNCTION = "premultiply"

    def premultiply(self, image, mask, invert):
        images = tensor2pil(image)
        masks = tensor2pil(mask) if invert else tensor2pil(1.0 - mask)
        single = len(mask) == 1
        masks = [x.convert("L") for x in masks]

        out = []
        for i, img in enumerate(images):
            cur_mask = masks[0] if single else masks[i]

            img.putalpha(cur_mask)
            out.append(img)

        # if invert:
        #     image = Image.composite(image,Image.new("RGBA", image.size, color=(0,0,0,0)), mask)
        # else:
        #     image = Image.composite(Image.new("RGBA", image.size, color=(0,0,0,0)), image, mask)

        return (pil2tensor(out),)


class MTB_ImageResizeFactor:
    """Extracted mostly from WAS Node Suite, with a few edits (most notably multiple image support) and less features."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "factor": (
                    "FLOAT",
                    {"default": 2, "min": 0.01, "max": 16.0, "step": 0.01},
                ),
                "supersample": ("BOOLEAN", {"default": True}),
                "resampling": (
                    [
                        "nearest",
                        "linear",
                        "bilinear",
                        "bicubic",
                        "trilinear",
                        "area",
                        "nearest-exact",
                    ],
                    {"default": "nearest"},
                ),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    CATEGORY = "mtb/image"
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "resize"

    def resize(
        self,
        image: torch.Tensor,
        factor: float,
        supersample: bool,
        resampling: str,
        mask=None,
    ):
        # Check if the tensor has the correct dimension
        if len(image.shape) not in [3, 4]:  # HxWxC or BxHxWxC
            raise ValueError(
                "Expected image tensor of shape (H, W, C) or (B, H, W, C)"
            )

        # Transpose to CxHxW or BxCxHxW for PyTorch
        if len(image.shape) == 3:
            image = image.permute(2, 0, 1).unsqueeze(0)  # CxHxW
        else:
            image = image.permute(0, 3, 1, 2)  # BxCxHxW

        # Compute new dimensions
        B, C, H, W = image.shape
        new_H, new_W = int(H * factor), int(W * factor)

        align_corner_filters = ("linear", "bilinear", "bicubic", "trilinear")
        # Resize the image
        resized_image = F.interpolate(
            image,
            size=(new_H, new_W),
            mode=resampling,
            align_corners=resampling in align_corner_filters,
        )

        # Optionally supersample
        if supersample:
            resized_image = F.interpolate(
                resized_image,
                scale_factor=2,
                mode=resampling,
                align_corners=resampling in align_corner_filters,
            )

        # Transpose back to the original format: BxHxWxC or HxWxC
        if len(image.shape) == 4:
            resized_image = resized_image.permute(0, 2, 3, 1)
        else:
            resized_image = resized_image.squeeze(0).permute(1, 2, 0)

        # Apply mask if provided
        if mask is not None:
            if len(mask.shape) != len(resized_image.shape):
                raise ValueError(
                    "Mask tensor should have the same dimensions as the image tensor"
                )
            resized_image = resized_image * mask

        return (resized_image,)


class MTB_SaveImageGrid:
    """Save all the images in the input batch as a grid of images."""

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                "save_intermediate": ("BOOLEAN", {"default": False}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "mtb/IO"

    def create_image_grid(self, image_list):
        total_images = len(image_list)

        # Calculate the grid size based on the square root of the total number of images
        grid_size = (
            int(math.sqrt(total_images)),
            int(math.ceil(math.sqrt(total_images))),
        )

        # Get the size of the first image to determine the grid size
        image_width, image_height = image_list[0].size

        # Create a new blank image to hold the grid
        grid_width = grid_size[0] * image_width
        grid_height = grid_size[1] * image_height
        grid_image = Image.new("RGB", (grid_width, grid_height))

        # Iterate over the images and paste them onto the grid
        for i, image in enumerate(image_list):
            x = (i % grid_size[0]) * image_width
            y = (i // grid_size[0]) * image_height
            grid_image.paste(image, (x, y, x + image_width, y + image_height))

        return grid_image

    def save_images(
        self,
        images,
        filename_prefix="Grid",
        save_intermediate=False,
        prompt=None,
        extra_pnginfo=None,
    ):
        (
            full_output_folder,
            filename,
            counter,
            subfolder,
            filename_prefix,
        ) = folder_paths.get_save_image_path(
            filename_prefix,
            self.output_dir,
            images[0].shape[1],
            images[0].shape[0],
        )
        image_list = []
        batch_counter = counter

        metadata = PngInfo()
        if prompt is not None:
            metadata.add_text("prompt", json.dumps(prompt))
        if extra_pnginfo is not None:
            for x in extra_pnginfo:
                metadata.add_text(x, json.dumps(extra_pnginfo[x]))

        for idx, image in enumerate(images):
            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            image_list.append(img)

            if save_intermediate:
                file = f"{filename}_batch-{idx:03}_{batch_counter:05}_.png"
                img.save(
                    os.path.join(full_output_folder, file),
                    pnginfo=metadata,
                    compress_level=4,
                )

            batch_counter += 1

        file = f"{filename}_{counter:05}_.png"
        grid = self.create_image_grid(image_list)
        grid.save(
            os.path.join(full_output_folder, file),
            pnginfo=metadata,
            compress_level=4,
        )

        results = [
            {"filename": file, "subfolder": subfolder, "type": self.type}
        ]
        return {"ui": {"images": results}}


class MTB_ImageTileOffset:
    """Mimics an old photoshop technique to check for seamless textures"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "tilesX": ("INT", {"default": 2, "min": 1}),
                "tilesY": ("INT", {"default": 2, "min": 1}),
            }
        }

    CATEGORY = "mtb/generate"

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "tile_image"

    def tile_image(
        self, image: torch.Tensor, tilesX: int = 2, tilesY: int = 2
    ):
        if tilesX < 1 or tilesY < 1:
            raise ValueError("The number of tiles must be at least 1.")

        batch_size, height, width, channels = image.shape
        tile_height = height // tilesY
        tile_width = width // tilesX

        output_image = torch.zeros_like(image)

        for i, j in itertools.product(range(tilesY), range(tilesX)):
            start_h = i * tile_height
            end_h = start_h + tile_height
            start_w = j * tile_width
            end_w = start_w + tile_width

            tile = image[:, start_h:end_h, start_w:end_w, :]

            output_start_h = (i + 1) % tilesY * tile_height
            output_start_w = (j + 1) % tilesX * tile_width
            output_end_h = output_start_h + tile_height
            output_end_w = output_start_w + tile_width

            output_image[
                :, output_start_h:output_end_h, output_start_w:output_end_w, :
            ] = tile

        return (output_image,)


__nodes__ = [
    MTB_ColorCorrect,
    MTB_ImageCompare,
    MTB_ImageTileOffset,
    MTB_Blur,
    # DeglazeImage,
    MTB_MaskToImage,
    MTB_ColoredImage,
    MTB_ImagePremultiply,
    MTB_ImageResizeFactor,
    MTB_SaveImageGrid,
    MTB_LoadImageFromUrl,
    MTB_Sharpen,
]
