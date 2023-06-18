import torch
from skimage.filters import gaussian
from skimage.restoration import denoise_tv_chambolle
from skimage.util import compare_images
from skimage.color import rgb2hsv, hsv2rgb
import numpy as np
import torchvision.transforms.functional as F
from PIL import Image
from ..utils import tensor2pil, pil2tensor
import cv2
import torch

try:
    from cv2.ximgproc import guidedFilter
except ImportError:
    print("guidedFilter not found")


class ColorCorrect:
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


class HSVtoRGB:
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


class RGBtoHSV:
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

        # image = image.transpose(1,2,3,0)
        image = np.squeeze(image)
        image = rgb2hsv(image)
        image = np.expand_dims(image, axis=0)

        # image = image.transpose(3,0,1,2)
        return (torch.from_numpy(image),)


class ImageCompare:
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
        # image = image.transpose(1,2,3,0)
        image = image.squeeze()
        image = denoise_tv_chambolle(image, weight=weight)

        # image = image.transpose(3,0,1,2)
        image = np.expand_dims(image, axis=0)
        return (torch.from_numpy(image),)


class Blur:
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
        # image = ndimage.gaussian_filter(image, sigma)
        image = gaussian(image, sigma=(sigmaX, sigmaY, 0, 0))
        # (image, sigma=sigma, multichannel=True)
        image = image.transpose(3, 0, 1, 2)
        return (torch.from_numpy(image),)



def img_np_to_tensor(img_np):
    return torch.from_numpy(img_np / 255.0)[None,]
def img_tensor_to_np(img_tensor):
    img_tensor = img_tensor.clone()
    img_tensor = img_tensor * 255.0
    return img_tensor.squeeze(0).numpy().astype(np.float32)

#https://github.com/lllyasviel/AdverseCleaner/blob/main/clean.py
def deglaze_np_img(np_img):
    y = np_img.copy()
    for _ in range(64):
        y = cv2.bilateralFilter(y, 5, 8, 8)
    for _ in range(4):
        y = guidedFilter(np_img, y, 4, 16)
    return y

class DeglazeImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE", ) } }

    CATEGORY = "image"

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "deglaze_image"
    def deglaze_image(self, image):
        return (img_np_to_tensor(deglaze_np_img(img_tensor_to_np(image))),)

