import qrcode
from ..utils import pil2tensor, here, tensor2np
from PIL import Image, ImageOps
from functools import lru_cache
import numpy as np
import torch
import hashlib
from pathlib import Path
from ..log import log
import uuid
import folder_paths
import comfy.model_management as model_management
import subprocess


class SaveGif:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "fps": ("INT", {"default": 12, "min": 1, "max": 120}),
                "resize_by": ("FLOAT", {"default": 1.0, "min": 0.1}),
                "pingpong": ("BOOL", {"default": False}),
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    CATEGORY = "mtb/IO"
    FUNCTION = "save_gif"

    def save_gif(self, image, fps=12, resize_by=1.0, pingpong=False):
        if image.size(0) == 0:
            return ("",)

        images = tensor2np(image)
        images = [frame.astype(np.uint8) for frame in images]
        if pingpong:
            reversed_frames = images[::-1]
            images.extend(reversed_frames)

        height, width, _ = image[0].shape

        ruuid = uuid.uuid4()
        out_path = f"{folder_paths.output_directory}/{ruuid}.gif"

        log.debug(f"Saving a gif file {width}x{height} as {ruuid}.gif")

        # Prepare the FFmpeg command
        command = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{width}x{height}",
            "-pix_fmt",
            "rgb24",  # GIF only supports rgb24
            "-r",
            str(fps),
            "-i",
            "-",
            "-vf",
            f"fps={fps},scale={width * resize_by}:-1",  # Set frame rate and resize if necessary
            "-y",
            out_path,
        ]

        process = subprocess.Popen(command, stdin=subprocess.PIPE)

        for frame in images:
            model_management.throw_exception_if_processing_interrupted()
            process.stdin.write(frame.tobytes())

        process.stdin.close()
        process.wait()
        results = []
        results.append({"filename": f"{ruuid}.gif", "subfolder": "", "type": "output"})
        return {"ui": {"images": results}}



class QrCode:
    """Basic QR Code generator"""

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"default": "https://www.github.com"}),
                "width": (
                    "INT",
                    {"default": 256, "max": 8096, "min": 0, "step": 1},
                ),
                "height": (
                    "INT",
                    {"default": 256, "max": 8096, "min": 0, "step": 1},
                ),
                "error_correct": (("L", "M", "Q", "H"), {"default": "L"}),
                "box_size": ("INT", {"default": 10, "max": 8096, "min": 0, "step": 1}),
                "border": ("INT", {"default": 4, "max": 8096, "min": 0, "step": 1}),
                "invert": (("True", "False"), {"default": "False"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "do_qr"
    CATEGORY = "mtb/generate"

    def do_qr(self, url, width, height, error_correct, box_size, border, invert):
        if error_correct == "L" or error_correct not in ["M", "Q", "H"]:
            error_correct = qrcode.constants.ERROR_CORRECT_L
        elif error_correct == "M":
            error_correct = qrcode.constants.ERROR_CORRECT_M
        elif error_correct == "Q":
            error_correct = qrcode.constants.ERROR_CORRECT_Q
        else:
            error_correct = qrcode.constants.ERROR_CORRECT_H

        qr = qrcode.QRCode(
            version=1,
            error_correction=error_correct,
            box_size=box_size,
            border=border,
        )
        qr.add_data(url)
        qr.make(fit=True)

        back_color = (255, 255, 255) if invert == "True" else (0, 0, 0)
        fill_color = (0, 0, 0) if invert == "True" else (255, 255, 255)

        code = img = qr.make_image(back_color=back_color, fill_color=fill_color)

        # that we now resize without filtering
        code = code.resize((width, height), Image.NEAREST)

        return (pil2tensor(code),)


__nodes__ = [
    QrCode,
    SaveGif,
]
