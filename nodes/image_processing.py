import torch
from skimage.filters import gaussian
from skimage.restoration import denoise_tv_chambolle
from skimage.util import compare_images
from skimage.color import rgb2hsv, hsv2rgb
import numpy as np
import torchvision.transforms.functional as F
from PIL import Image, ImageChops
from ..utils import tensor2pil, pil2tensor, img_np_to_tensor, img_tensor_to_np
import cv2
import torch
from ..log import log
import folder_paths
from PIL.PngImagePlugin import PngInfo
import json
import os

try:
    from cv2.ximgproc import guidedFilter
except ImportError:
    log.error("guidedFilter not found, use opencv-contrib-python")


class ColorCorrect:
    """Various color correction methods"""

    def __init__(self):
        pass

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
    CATEGORY = "image/postprocessing"

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
        image = tensor2pil(image)
        hsv_image = image.convert("HSV")

        h, s, v = hsv_image.split()

        h = h.point(lambda x: (x + hue * 255) % 256)
        s = s.point(lambda x: int(x * saturation))
        v = v.point(lambda x: int(x * value))

        hsv_image = Image.merge("HSV", (h, s, v))
        rgb_image = hsv_image.convert("RGB")

        return pil2tensor(rgb_image)

    @staticmethod
    def hsv_adjustment_tensor_not_working(image: torch.Tensor, hue, saturation, value):
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


class HsvToRgb:
    """Convert HSV image to RGB"""

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "convert"
    CATEGORY = "image/postprocessing"

    def convert(self, image):
        image = image.numpy()

        image = image.squeeze()
        # image = image.transpose(1,2,3,0)
        image = hsv2rgb(image)
        image = np.expand_dims(image, axis=0)

        # image = image.transpose(3,0,1,2)
        return (torch.from_numpy(image),)


class RgbToHsv:
    """Convert RGB image to HSV"""

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "convert"
    CATEGORY = "image/postprocessing"

    def convert(self, image):
        image = image.numpy()

        image = np.squeeze(image)
        image = rgb2hsv(image)
        image = np.expand_dims(image, axis=0)

        return (torch.from_numpy(image),)


class ImageCompare:
    """Compare two images and return a difference image"""

    def __init__(self):
        pass

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
    CATEGORY = "image"

    def compare(self, imageA: torch.Tensor, imageB: torch.Tensor, mode):
        imageA = imageA.numpy()
        imageB = imageB.numpy()

        imageA = imageA.squeeze()
        imageB = imageB.squeeze()

        image = compare_images(imageA, imageB, method=mode)

        image = np.expand_dims(image, axis=0)
        return (torch.from_numpy(image),)


class Denoise:
    """Denoise an image using total variation minimization."""

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "weight": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "denoise"
    CATEGORY = "image/postprocessing"

    def denoise(self, image: torch.Tensor, weight):
        image = image.numpy()
        image = image.squeeze()
        image = denoise_tv_chambolle(image, weight=weight)

        image = np.expand_dims(image, axis=0)
        return (torch.from_numpy(image),)


class Blur:
    """Blur an image using a Gaussian filter."""

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "sigmaX": (
                    "FLOAT",
                    {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.01},
                ),
                "sigmaY": (
                    "FLOAT",
                    {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "blur"
    CATEGORY = "image/postprocessing"

    def blur(self, image: torch.Tensor, sigmaX, sigmaY):
        image = image.numpy()
        image = image.transpose(1, 2, 3, 0)
        image = gaussian(image, sigma=(sigmaX, sigmaY, 0, 0))
        image = image.transpose(3, 0, 1, 2)
        return (torch.from_numpy(image),)


# https://github.com/lllyasviel/AdverseCleaner/blob/main/clean.py
def deglaze_np_img(np_img):
    y = np_img.copy()
    for _ in range(64):
        y = cv2.bilateralFilter(y, 5, 8, 8)
    for _ in range(4):
        y = guidedFilter(np_img, y, 4, 16)
    return y


class DeglazeImage:
    """Remove adversarial noise from images"""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    CATEGORY = "image"

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "deglaze_image"

    def deglaze_image(self, image):
        return (img_np_to_tensor(deglaze_np_img(img_tensor_to_np(image))),)


class MaskToImage:
    """Converts a mask (alpha) to an RGB image with a color and background"""

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "color": ("COLOR",),
                "background": ("COLOR", {"default": "#000000"}),
            }
        }

    CATEGORY = "image/mask"

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "render_mask"

    def render_mask(self, mask, color, background):
        mask = img_tensor_to_np(mask)
        mask = Image.fromarray(mask).convert("L")

        image = Image.new("RGBA", mask.size, color=color)
        # apply the mask
        image = Image.composite(
            image, Image.new("RGBA", mask.size, color=background), mask
        )

        # image = ImageChops.multiply(image, mask)
        # apply over background
        # image = Image.alpha_composite(Image.new("RGBA", image.size, color=background), image)

        image = pil2tensor(image.convert("RGB"))

        return (image,)


class ColoredImage:
    """Constant color image of given size"""

    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "color": ("COLOR",),
                "width": ("INT", {"default": 512, "min": 16, "max": 8160}),
                "height": ("INT", {"default": 512, "min": 16, "max": 8160}),
            }
        }

    CATEGORY = "image"

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "render_img"

    def render_img(self, color, width, height):
        image = Image.new("RGB", (width, height), color=color)

        image = pil2tensor(image)

        return (image,)


class ImagePremultiply:
    """Premultiply image with mask"""

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "invert": (["True", "False"], {"default": "False"}),
            }
        }

    CATEGORY = "image"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "premultiply"

    def premultiply(self, image, mask, invert):
        invert = invert == "True"
        image = tensor2pil(image)
        mask = tensor2pil(mask).convert("L")

        if invert:
            mask = ImageChops.invert(mask)

        image.putalpha(mask)

        # if invert:
        #     image = Image.composite(image,Image.new("RGBA", image.size, color=(0,0,0,0)), mask)
        # else:
        #     image = Image.composite(Image.new("RGBA", image.size, color=(0,0,0,0)), image, mask)

        return (pil2tensor(image),)


class ImageResizeFactor:
    """
    Extracted mostly from WAS Node Suite, with a few edits (most notably multiple image support) and less features.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "factor": (
                    "FLOAT",
                    {"default": 2, "min": 0.01, "max": 16.0, "step": 0.01},
                ),
                "supersample": (["true", "false"], {"default": "true"}),
                "resampling": (
                    ["lanczos", "nearest", "bilinear", "bicubic"],
                    {"default": "lanczos"},
                ),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    CATEGORY = "image"
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "resize"

    def resize_image(
        self,
        image: torch.Tensor,
        factor: float = 0.5,
        supersample=False,
        resample="lanczos",
        mask=None,
    ) -> torch.Tensor:
        batch_count = 1
        img = tensor2pil(image)

        if isinstance(img, list):
            log.debug("Multiple images detected (list)")
            out = []
            for im in img:
                im = self.resize_image(
                    pil2tensor(im), factor, supersample, resample, mask
                )
                out.append(im)
            return torch.cat(out, dim=0)
        elif isinstance(img, torch.Tensor):
            if len(image.shape) > 3:
                batch_count = image.size(0)

        if batch_count > 1:
            log.debug("Multiple images detected (batch count)")
            out = [
                self.resize_image(image[i], factor, supersample, resample, mask)
                for i in range(batch_count)
            ]
            return torch.cat(out, dim=0)

        log.debug("Resizing image")
        # Get the current width and height of the image
        current_width, current_height = img.size

        log.debug(f"Current width: {current_width}, Current height: {current_height}")

        # Calculate the new width and height based on the given mode and parameters
        new_width, new_height = int(factor * current_width), int(
            factor * current_height
        )

        log.debug(f"New width: {new_width}, New height: {new_height}")

        # Define a dictionary of resampling filters
        resample_filters = {"nearest": 0, "bilinear": 2, "bicubic": 3, "lanczos": 1}

        # Apply supersample
        if supersample == "true":
            super_size = (new_width * 8, new_height * 8)
            log.debug(f"Applying supersample: {super_size}")
            img = img.resize(
                super_size, resample=Image.Resampling(resample_filters[resample])
            )

        # Resize the image using the given resampling filter
        resized_image = img.resize(
            (new_width, new_height),
            resample=Image.Resampling(resample_filters[resample]),
        )

        return pil2tensor(resized_image)

    def resize(
        self,
        image: torch.Tensor,
        factor: float,
        supersample: str,
        resampling: str,
        mask=None,
    ):
        log.debug(f"Resizing image with factor {factor} and resampling {resampling}")
        supersample = supersample == "true"
        batch_count = image.size(0)
        log.debug(f"Batch count: {batch_count}")
        if batch_count == 1:
            log.debug("Batch count is 1, returning single image")
            return (self.resize_image(image, factor, supersample, resampling),)
        else:
            log.debug("Batch count is greater than 1, returning multiple images")
            images = [
                self.resize_image(image[i], factor, supersample, resampling)
                for i in range(batch_count)
            ]
            images = torch.cat(images, dim=0)
            return (images,)


import math


class SaveImageGrid:
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
                "save_intermediate": (["true", "false"], {"default": "false"}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "image"

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
        save_intermediate="false",
        prompt=None,
        extra_pnginfo=None,
    ):
        save_intermediate = save_intermediate == "true"
        (
            full_output_folder,
            filename,
            counter,
            subfolder,
            filename_prefix,
        ) = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0]
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
            os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=4
        )

        results = [{"filename": file, "subfolder": subfolder, "type": self.type}]
        return {"ui": {"images": results}}


__nodes__ = [
    ColorCorrect,
    HsvToRgb,
    RgbToHsv,
    ImageCompare,
    Denoise,
    Blur,
    DeglazeImage,
    MaskToImage,
    ColoredImage,
    ImagePremultiply,
    ImageResizeFactor,
    SaveImageGrid,
]
