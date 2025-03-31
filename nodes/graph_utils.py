import io
import json
import re
import urllib.parse
import urllib.request
from math import pi

import comfy.model_management as model_management
import comfy.utils
import numpy as np
import torch
from PIL import Image

from ..log import log
from ..utils import (
    EASINGS,
    apply_easing,
    get_server_info,
    numpy_NFOV,
    pil2tensor,
    tensor2np,
)


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


class MTB_ToDevice:
    """Send a image or mask tensor to the given device."""

    @classmethod
    def INPUT_TYPES(cls):
        devices = ["cpu"]
        if torch.backends.mps.is_available():
            devices.append("mps")
        if torch.cuda.is_available():
            devices.append("cuda:0")
            for i in range(1, torch.cuda.device_count()):
                devices.append(f"cuda:{i}")
            devices.append("cuda")

        return {
            "required": {
                "ignore_errors": ("BOOLEAN", {"default": False}),
                "device": (
                    devices,
                    {
                        "default": "cuda"
                        if torch.cuda.is_available()
                        else "cpu"
                    },
                ),
            },
            "optional": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("images", "masks")
    CATEGORY = "mtb/utils"
    FUNCTION = "to_device"

    def to_device(
        self,
        *,
        ignore_errors: bool = False,
        device: str = "cuda",
        image: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ):
        if not ignore_errors and image is None and mask is None:
            raise ValueError(
                "You must either provide an image or a mask,"
                + " use ignore_error to passthrough"
            )
        if (
            device.startswith("cuda")
            and ":" not in device
            and device != "cuda"
        ):
            device = f"cuda:{device[4:]}"

        try:
            if image is not None:
                image = image.to(device)
            if mask is not None:
                mask = mask.to(device)
        except RuntimeError as e:
            if not ignore_errors:
                raise RuntimeError(
                    f"Failed to move tensor to device {device}: {str(e)}"
                ) from e
            log.warning(
                f"Failed to move tensor to device {device}, ignoring: {str(e)}"
            )
        return (image, mask)


# class MTB_ApplyTextTemplate:
class MTB_ApplyTextTemplate:
    """
    Experimental node to interpolate strings from inputs.

    Interpolation just requires {}, for instance:

    Some string {var_1} and {var_2}
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "template": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    CATEGORY = "mtb/utils"
    FUNCTION = "execute"

    def execute(self, *, template: str, **kwargs):
        res = f"{template}"
        for k, v in kwargs.items():
            res = res.replace(f"{{{k}}}", f"{v}")

        return (res,)


class MTB_MatchDimensions:
    """Match images dimensions along the given dimension, preserving aspect ratio."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": ("IMAGE",),
                "reference": ("IMAGE",),
                "match": (["height", "width"], {"default": "height"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "new_width", "new_height")
    CATEGORY = "mtb/utils"
    FUNCTION = "execute"

    def execute(
        self, source: torch.Tensor, reference: torch.Tensor, match: str
    ):
        import torchvision.transforms.functional as VF

        _batch_size, height, width, _channels = source.shape
        _rbatch_size, rheight, rwidth, _rchannels = reference.shape

        source_aspect_ratio = width / height
        # reference_aspect_ratio = rwidth / rheight

        source = source.permute(0, 3, 1, 2)
        reference = reference.permute(0, 3, 1, 2)

        if match == "height":
            new_height = rheight
            new_width = int(rheight * source_aspect_ratio)
        else:
            new_width = rwidth
            new_height = int(rwidth / source_aspect_ratio)

        resized_images = [
            VF.resize(
                source[i],
                (new_height, new_width),
                antialias=True,
                interpolation=Image.BICUBIC,
            )
            for i in range(_batch_size)
        ]
        resized_source = torch.stack(resized_images, dim=0)
        resized_source = resized_source.permute(0, 2, 3, 1)

        return (resized_source, new_width, new_height)


class MTB_FloatToFloats:
    """Conversion utility for compatibility with other extensions (AD, IPA, Fitz are using FLOAT to represent list of floats.)"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "float": ("FLOAT", {"default": 0.0, "forceInput": True}),
            }
        }

    RETURN_TYPES = ("FLOATS",)
    RETURN_NAMES = ("floats",)
    CATEGORY = "mtb/utils"
    FUNCTION = "convert"

    def convert(self, float: float):
        return (float,)


class MTB_FloatsToInts:
    """Conversion utility for compatibility with frame interpolation."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "floats": ("FLOATS", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("INTS", "INT")
    CATEGORY = "mtb/utils"
    FUNCTION = "convert"

    def convert(self, floats: list[float]):
        vals = [int(x) for x in floats]
        return (vals, vals)


class MTB_FloatsToFloat:
    """Conversion utility for compatibility with other extensions (AD, IPA, Fitz are using FLOAT to represent list of floats.)"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "floats": ("FLOATS",),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("float",)
    CATEGORY = "mtb/utils"
    FUNCTION = "convert"

    def convert(self, floats):
        return (floats,)


class MTB_AutoPanEquilateral:
    """Generate a 360 panning video from an equilateral image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "equilateral_image": ("IMAGE",),
                "fovX": ("FLOAT", {"default": 45.0}),
                "fovY": ("FLOAT", {"default": 45.0}),
                "elevation": ("FLOAT", {"default": 0.5}),
                "frame_count": ("INT", {"default": 100}),
                "width": ("INT", {"default": 768}),
                "height": ("INT", {"default": 512}),
            },
            "optional": {
                "floats_fovX": ("FLOATS",),
                "floats_fovY": ("FLOATS",),
                "floats_elevation": ("FLOATS",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    CATEGORY = "mtb/utils"
    FUNCTION = "generate_frames"

    def check_floats(self, f: list[float] | None, expected_count: int):
        if f:
            if len(f) == expected_count:
                return True
            return False
        return True

    def generate_frames(
        self,
        equilateral_image: torch.Tensor,
        fovX: float,
        fovY: float,
        elevation: float,
        frame_count: int,
        width: int,
        height: int,
        floats_fovX: list[float] | None = None,
        floats_fovY: list[float] | None = None,
        floats_elevation: list[float] | None = None,
    ):
        source = tensor2np(equilateral_image)

        if len(source) > 1:
            log.warn(
                "You provided more than one image in the equilateral_image input, only the first will be used."
            )
        if not all(
            [
                self.check_floats(x, frame_count)
                for x in [floats_fovX, floats_fovY, floats_elevation]
            ]
        ):
            raise ValueError(
                "You provided less than the expected number of fovX, fovY, or elevation values."
            )

        source = source[0]
        frames = []

        pbar = comfy.utils.ProgressBar(frame_count)
        for i in range(frame_count):
            rotation_angle = (i / frame_count) * 2 * pi

            if floats_elevation:
                elevation = floats_elevation[i]

            if floats_fovX:
                fovX = floats_fovX[i]

            if floats_fovY:
                fovY = floats_fovY[i]

            fov = [fovX / 100, fovY / 100]
            center_point = [rotation_angle / (2 * pi), elevation]

            nfov = numpy_NFOV(fov, height, width)
            frame = nfov.to_nfov(source, center_point=center_point)

            frames.append(frame)

            model_management.throw_exception_if_processing_interrupted()
            pbar.update(1)

        return (pil2tensor(frames),)


class MTB_GetBatchFromHistory:
    """Very experimental node to load images from the history of the server.

    Queue items without output are ignored in the count.
    """

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
        *,
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
                            image["filename"],
                            image["subfolder"],
                            image["type"],
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


class MTB_AnyToString:
    """Tries to take any input and convert it to a string."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"input": ("*",)},
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
            return (
                f"Numpy array of shape {input.shape} and dtype {input.dtype}",
            )

        elif isinstance(input, dict):
            return (
                f"Dictionary of {len(input)} items, with keys {input.keys()}",
            )

        else:
            log.debug(f"Falling back to string conversion of {input}")
            return (str(input),)


class MTB_StringReplace:
    """Basic string replacement with regex support."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string": ("STRING", {"forceInput": True}),
                "old": ("STRING", {"default": ""}),
                "new": ("STRING", {"default": ""}),
                "use_regex": ("BOOLEAN", {"default": False}),
            }
        }

    FUNCTION = "replace_str"
    RETURN_TYPES = ("STRING",)
    CATEGORY = "mtb/string"

    def replace_str(self, string: str, old: str, new: str, use_regex: bool):
        log.debug(f"Current string: {string}")
        log.debug(f"Find string: {old}")
        log.debug(f"Replace string: {new}")
        log.debug(f"Use regex: {use_regex}")

        if use_regex:
            try:
                string = re.sub(old, new, string)
            except re.error as e:
                raise ValueError(f"Regex error: {e}") from e
        else:
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
    DESCRIPTION = (
        "evaluate a simple math expression string, only supports literal_eval"
    )

    def eval_expression(self, expression: str, **kwargs):
        from ast import literal_eval

        for key, value in kwargs.items():
            log.debug(f"Replacing placeholder <{key}> with value {value}")
            expression = expression.replace(f"<{key}>", str(value))

        result = -1
        try:
            result = literal_eval(expression)
        except SyntaxError as e:
            raise ValueError(
                f"The expression syntax is wrong '{expression}': {e}"
            ) from e

        except Exception as e:
            raise ValueError(
                f"Math expression only support literal_eval now: {e}"
            )

        return (result, int(result))


class MTB_FitNumber:
    """Fit the input float using a source and target range"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("FLOAT", {"default": 0, "forceInput": True}),
                "clamp": ("BOOLEAN", {"default": False}),
                "source_min": (
                    "FLOAT",
                    {"default": 0.0, "step": 0.01, "min": -1e5},
                ),
                "source_max": (
                    "FLOAT",
                    {"default": 1.0, "step": 0.01, "min": -1e5},
                ),
                "target_min": (
                    "FLOAT",
                    {"default": 0.0, "step": 0.01, "min": -1e5},
                ),
                "target_max": (
                    "FLOAT",
                    {"default": 1.0, "step": 0.01, "min": -1e5},
                ),
                "easing": (
                    EASINGS,
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


class MTB_ConcatImages:
    """Add images to batch."""

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "concatenate_tensors"
    CATEGORY = "mtb/image"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"reverse": ("BOOLEAN", {"default": False})},
            "optional": {
                "on_mismatch": (
                    ["Error", "Smallest", "Largest"],
                    {"default": "Smallest"},
                )
            },
        }

    def concatenate_tensors(
        self,
        reverse: bool,
        on_mismatch: str = "Smallest",
        **kwargs: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        tensors = list(kwargs.values())

        if on_mismatch == "Error":
            shapes = [tensor.shape for tensor in tensors]
            if not all(shape == shapes[0] for shape in shapes):
                raise ValueError(
                    "All input tensors must have the same shape when on_mismatch is 'Error'."
                )

        else:
            import torch.nn.functional as F

            if on_mismatch == "Smallest":
                target_shape = min(
                    (tensor.shape for tensor in tensors),
                    key=lambda s: (s[1], s[2]),
                )
            else:  # on_mismatch == "Largest"
                target_shape = max(
                    (tensor.shape for tensor in tensors),
                    key=lambda s: (s[1], s[2]),
                )

            target_height, target_width = target_shape[1], target_shape[2]

            resized_tensors = []
            for tensor in tensors:
                if (
                    tensor.shape[1] != target_height
                    or tensor.shape[2] != target_width
                ):
                    resized_tensor = F.interpolate(
                        tensor.permute(0, 3, 1, 2),
                        size=(target_height, target_width),
                        mode="bilinear",
                        align_corners=False,
                    )
                    resized_tensor = resized_tensor.permute(0, 2, 3, 1)
                    resized_tensors.append(resized_tensor)
                else:
                    resized_tensors.append(tensor)

            tensors = resized_tensors

        concatenated = torch.cat(tensors, dim=0)

        return (concatenated,)


class MTB_TensorOps:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tensor": ("IMAGE",),
                "operation": (
                    [
                        "multiply",
                        "divide",
                        "add",
                        "subtract",
                        "power",
                        "clamp",
                        "abs",
                        "log",
                        "exp",
                        "convert_dtype",
                        "normalize_range",
                        "normalize_per_channel",
                    ],
                    {"default": "multiply"},
                ),
                "value": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -1000000.0,
                        "max": 1000000.0,
                        "step": 0.01,
                    },
                ),
                "source_min": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -1000000.0,
                        "max": 1000000.0,
                        "step": 0.01,
                    },
                ),
                "source_max": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -1000000.0,
                        "max": 1000000.0,
                        "step": 0.01,
                    },
                ),
                "target_min": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -1000000.0,
                        "max": 1000000.0,
                        "step": 0.01,
                    },
                ),
                "target_max": (
                    "FLOAT",
                    {
                        "default": 16.0,
                        "min": -1000000.0,
                        "max": 1000000.0,
                        "step": 0.01,
                    },
                ),
                "dtype": (
                    ["uint8", "float32", "float16", "bfloat16"],
                    {"default": "float32"},
                ),
                "use_mean": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "target_tensor": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "mtb/tensor_ops"

    def apply(
        self,
        tensor,
        operation="multiply",
        value=1.0,
        source_min=0.0,
        source_max=1.0,
        target_min=0.0,
        target_max=1.0,
        dtype="float32",
        use_mean=False,
        target_tensor=None,
    ):
        log.debug(
            f"Input tensor stats: shape={tensor.shape}, dtype={tensor.dtype}, range=[{tensor.min().item():.6f}, {tensor.max().item():.6f}]"
        )
        if operation == "normalize_per_channel":
            if target_tensor is None:
                raise ValueError(
                    "Target tensor required for per-channel normalization"
                )

            result = tensor.clone()
            for c in range(tensor.shape[-1]):
                if use_mean:
                    source_mean = tensor[..., c].mean()
                    target_mean = target_tensor[..., c].mean()
                    scale = target_mean / source_mean
                    result[..., c] = tensor[..., c] * scale
                else:
                    source_min = tensor[..., c].min()
                    source_max = tensor[..., c].max()
                    target_min = target_tensor[..., c].min()
                    target_max = target_tensor[..., c].max()

                    normalized = (tensor[..., c] - source_min) / (
                        source_max - source_min
                    )
                    result[..., c] = (
                        normalized * (target_max - target_min) + target_min
                    )

                log.debug(
                    f"Channel {c} - Scale: source=[{source_min:.6f}, {source_max:.6f}], target=[{target_min:.6f}, {target_max:.6f}]"
                )

        elif operation == "normalize_range":
            if target_tensor is not None:
                target_min = target_tensor.min().item()
                target_max = target_tensor.max().item()
                log.debug(
                    f"Using target tensor range: [{target_min:.6f}, {target_max:.6f}]"
                )

            normalized = (tensor - source_min) / (source_max - source_min)
            result = normalized * (target_max - target_min) + target_min
        elif operation == "convert_dtype":
            if dtype == "float32":
                result = tensor.float()
            elif dtype == "float16":
                result = tensor.half()
            elif dtype == "bfloat16":
                result = tensor.bfloat16()

        else:
            result = tensor
            if operation == "multiply":
                result = tensor * value
            elif operation == "divide":
                result = tensor / value if value != 0 else tensor
            elif operation == "add":
                result = tensor + value
            elif operation == "subtract":
                result = tensor - value
            elif operation == "power":
                result = torch.pow(tensor, value)
            elif operation == "clamp":
                if target_tensor is not None:
                    result = torch.clamp(
                        tensor,
                        target_tensor.min().item(),
                        target_tensor.max().item(),
                    )
                else:
                    result = torch.clamp(tensor, source_min, source_max)
            elif operation == "abs":
                result = torch.abs(tensor)
            elif operation == "log":
                result = torch.log(tensor.clamp(min=1e-10))
            elif operation == "exp":
                result = torch.exp(tensor)

        log.debug(
            f"Output tensor stats: shape={result.shape}, dtype={result.dtype}, range=[{result.min().item():.6f}, {result.max().item():.6f}]"
        )
        return (result,)


__nodes__ = [
    MTB_StringReplace,
    MTB_FitNumber,
    MTB_GetBatchFromHistory,
    MTB_AnyToString,
    MTB_ConcatImages,
    MTB_MathExpression,
    MTB_ToDevice,
    MTB_ApplyTextTemplate,
    MTB_MatchDimensions,
    MTB_AutoPanEquilateral,
    MTB_FloatsToFloat,
    MTB_FloatToFloats,
    MTB_FloatsToInts,
    MTB_TensorOps,
]
