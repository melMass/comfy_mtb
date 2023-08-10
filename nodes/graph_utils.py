from ..log import log
from PIL import Image
import urllib.request
import urllib.parse
import torch
import json
from comfy.cli_args import args
from ..utils import pil2tensor
import io


def get_image(filename, subfolder, folder_type):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen(
        f"http://{args.listen}:{args.port}/view?{url_values}"
    ) as response:
        return io.BytesIO(response.read())


class GetBatchFromHistory:
    """Very experimental node to load images from the history of the server.

    Queue items without output are ignored in the count."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enable": ("BOOLEAN", {"default": True}),
                "count": ("INT", {"default": 1, "min": 0}),
                "offset": ("INT", {"default": 0, "min": -1e9, "max": 1e9}),
                "internal_count": ("INT", {"default": 0}),
            },
            "optional": {
                "passthrough_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = "images"
    CATEGORY = "mtb/animation"
    FUNCTION = "load_from_history"

    def load_from_history(
        self,
        enable=True,
        count=0,
        offset=0,
        internal_count=0,  # hacky way to invalidate the node
        passthrough_image=None,
    ):
        if not enable or count == 0:
            if passthrough_image is not None:
                log.debug("Using passthrough image")
                return (passthrough_image,)
            log.debug("Load from history is disabled for this iteration")
            return (torch.zeros(0),)
        frames = []

        with urllib.request.urlopen(
            f"http://{args.listen}:{args.port}/history"
        ) as response:
            return self.load_batch_frames(response, offset, count, frames)

    def load_batch_frames(self, response, offset, count, frames):
        history = json.loads(response.read())

        output_images = []
        for k, run in history.items():
            for o in run["outputs"]:
                for node_id in run["outputs"]:
                    node_output = run["outputs"][node_id]
                    if "images" in node_output:
                        images_output = []
                        for image in node_output["images"]:
                            image_data = get_image(
                                image["filename"], image["subfolder"], image["type"]
                            )
                            images_output.append(image_data)
                        output_images.extend(images_output)
        if not output_images:
            return (torch.zeros(0),)
        for i, image in enumerate(list(reversed(output_images))):
            if i < offset:
                continue
            if i >= offset + count:
                break
            # Decode image as tensor
            img = Image.open(image)
            log.debug(f"Image from history {i} of shape {img.size}")
            frames.append(img)

            # Display the shape of the tensor
            # print("Tensor shape:", image_tensor.shape)

            # return (output_images,)
        if not frames:
            return (torch.zeros(0),)
        elif len(frames) != count:
            log.warning(f"Expected {count} images, got {len(frames)} instead")
        output = pil2tensor(
            list(reversed(frames)),
        )

        return (output,)


class StringReplace:
    """Basic string replacement"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string": ("STRING", {"forceInput": True}),
                "old": ("STRING", {"default": ""}),
                "new": ("STRING", {"default": ""}),
            }
        }

    FUNCTION = "replace_str"
    RETURN_TYPES = ("STRING",)
    CATEGORY = "mtb/string"

    def replace_str(self, string: str, old: str, new: str):
        log.debug(f"Current string: {string}")
        log.debug(f"Find string: {old}")
        log.debug(f"Replace string: {new}")

        string = string.replace(old, new)

        log.debug(f"New string: {string}")

        return (string,)


class FitNumber:
    """Fit the input float using a source and target range"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("FLOAT", {"default": 0, "forceInput": True}),
                "clamp": ("BOOLEAN", {"default": False}),
                "source_min": ("FLOAT", {"default": 0.0}),
                "source_max": ("FLOAT", {"default": 1.0}),
                "target_min": ("FLOAT", {"default": 0.0}),
                "target_max": ("FLOAT", {"default": 1.0}),
            }
        }

    FUNCTION = "set_range"
    RETURN_TYPES = ("FLOAT",)
    CATEGORY = "mtb/math"

    def set_range(
        self,
        value: float,
        clamp: bool,
        source_min: float,
        source_max: float,
        target_min: float,
        target_max: float,
    ):
        res = target_min + (target_max - target_min) * (value - source_min) / (
            source_max - source_min
        )

        if clamp:
            if target_min > target_max:
                res = max(min(res, target_min), target_max)
            else:
                res = max(min(res, target_max), target_min)

        return (res,)


__nodes__ = [StringReplace, FitNumber, GetBatchFromHistory]
