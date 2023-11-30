import io, json, urllib.parse, urllib.request

import numpy as np
import torch
from PIL import Image

from ..log import log
from ..utils import apply_easing, get_server_info, pil2tensor


def get_image(filename, subfolder, folder_type):
    log.debug(
        f"Getting image {filename} from foldertype {folder_type} {f'in subfolder: {subfolder}' if subfolder else ''}"
    )
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    base_url, port = get_server_info()

    url_values = urllib.parse.urlencode(data)
    url = f"http://{base_url}:{port}/view?{url_values}"
    log.debug(f"Fetching image from {url}")
    with urllib.request.urlopen(url) as response:
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
    RETURN_NAMES = ("images",)
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

        base_url, port = get_server_info()

        history_url = f"http://{base_url}:{port}/history"
        log.debug(f"Fetching history from {history_url}")
        output = torch.zeros(0)
        with urllib.request.urlopen(history_url) as response:
            output = self.load_batch_frames(response, offset, count, frames)

        if output.size(0) == 0:
            log.warn("No output found in history")

        return (output,)

    def load_batch_frames(self, response, offset, count, frames):
        history = json.loads(response.read())

        output_images = []

        for run in history.values():
            for node_output in run["outputs"].values():
                if "images" in node_output:
                    for image in node_output["images"]:
                        image_data = get_image(
                            image["filename"], image["subfolder"], image["type"]
                        )
                        output_images.append(image_data)

        if not output_images:
            return torch.zeros(0)

        # Directly get desired range of images
        start_index = max(len(output_images) - offset - count, 0)
        end_index = len(output_images) - offset
        selected_images = output_images[start_index:end_index]

        frames = [Image.open(image) for image in selected_images]

        if not frames:
            return torch.zeros(0)
        elif len(frames) != count:
            log.warning(f"Expected {count} images, got {len(frames)} instead")

        return pil2tensor(frames)


class AnyToString:
    """Tries to take any input and convert it to a string"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"input": ("*")},
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "do_str"
    CATEGORY = "mtb/converters"

    def do_str(self, input):
        if isinstance(input, str):
            return (input,)
        elif isinstance(input, torch.Tensor):
            return (f"Tensor of shape {input.shape} and dtype {input.dtype}",)
        elif isinstance(input, Image.Image):
            return (f"PIL Image of size {input.size} and mode {input.mode}",)
        elif isinstance(input, np.ndarray):
            return (f"Numpy array of shape {input.shape} and dtype {input.dtype}",)

        elif isinstance(input, dict):
            return (f"Dictionary of {len(input)} items, with keys {input.keys()}",)

        else:
            log.debug(f"Falling back to string conversion of {input}")
            return (str(input),)


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


class MTB_MathExpression:
    """Node to evaluate a simple math expression string"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "expression": ("STRING", {"default": "", "multiline": True}),
            }
        }

    FUNCTION = "eval_expression"
    RETURN_TYPES = ("FLOAT", "INT")
    RETURN_NAMES = ("result (float)", "result (int)")
    CATEGORY = "mtb/math"
    DESCRIPTION = "evaluate a simple math expression string (!! Fallsback to eval)"

    def eval_expression(self, expression, **kwargs):
        import math
        from ast import literal_eval

        for key, value in kwargs.items():
            print(f"Replacing placeholder <{key}> with value {value}")
            expression = expression.replace(f"<{key}>", str(value))

        result = -1
        try:
            result = literal_eval(expression)
        except SyntaxError as e:
            raise ValueError(
                f"The expression syntax is wrong '{expression}': {e}"
            ) from e

        except ValueError:
            try:
                expression = expression.replace("^", "**")
                result = eval(expression)
            except Exception as e:
                # Handle any other exceptions and provide a meaningful error message
                raise ValueError(
                    f"Error evaluating expression '{expression}': {e}"
                ) from e

        return (result, int(result))


class FitNumber:
    """Fit the input float using a source and target range"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("FLOAT", {"default": 0, "forceInput": True}),
                "clamp": ("BOOLEAN", {"default": False}),
                "source_min": ("FLOAT", {"default": 0.0, "step": 0.01}),
                "source_max": ("FLOAT", {"default": 1.0, "step": 0.01}),
                "target_min": ("FLOAT", {"default": 0.0, "step": 0.01}),
                "target_max": ("FLOAT", {"default": 1.0, "step": 0.01}),
                "easing": (
                    [
                        "Linear",
                        "Sine In",
                        "Sine Out",
                        "Sine In/Out",
                        "Quart In",
                        "Quart Out",
                        "Quart In/Out",
                        "Cubic In",
                        "Cubic Out",
                        "Cubic In/Out",
                        "Circ In",
                        "Circ Out",
                        "Circ In/Out",
                        "Back In",
                        "Back Out",
                        "Back In/Out",
                        "Elastic In",
                        "Elastic Out",
                        "Elastic In/Out",
                        "Bounce In",
                        "Bounce Out",
                        "Bounce In/Out",
                    ],
                    {"default": "Linear"},
                ),
            }
        }

    FUNCTION = "set_range"
    RETURN_TYPES = ("FLOAT",)
    CATEGORY = "mtb/math"
    DESCRIPTION = "Fit the input float using a source and target range"

    def set_range(
        self,
        value: float,
        clamp: bool,
        source_min: float,
        source_max: float,
        target_min: float,
        target_max: float,
        easing: str,
    ):
        if source_min == source_max:
            normalized_value = 0
        else:
            normalized_value = (value - source_min) / (source_max - source_min)
        if clamp:
            normalized_value = max(min(normalized_value, 1), 0)

        eased_value = apply_easing(normalized_value, easing)

        # - Convert the eased value to the target range
        res = target_min + (target_max - target_min) * eased_value

        return (res,)


class ConcatImages:
    """Add images to batch"""

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "concatenate_tensors"
    CATEGORY = "mtb/image"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"reverse": ("BOOLEAN", {"default": False})},
        }

    def concatenate_tensors(self, reverse, **kwargs):
        tensors = tuple(kwargs.values())
        batch_sizes = [tensor.size(0) for tensor in tensors]

        concatenated = torch.cat(tensors, dim=0)

        # Update the batch size in the concatenated tensor
        concatenated_size = list(concatenated.size())
        concatenated_size[0] = sum(batch_sizes)
        concatenated = concatenated.view(*concatenated_size)

        return (concatenated,)


__nodes__ = [
    StringReplace,
    FitNumber,
    GetBatchFromHistory,
    AnyToString,
    ConcatImages,
    MTB_MathExpression,
]
