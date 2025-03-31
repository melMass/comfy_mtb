import base64
import io
import json
from pathlib import Path

import folder_paths
import torch

from ..log import log
from ..utils import tensor2pil


def get_detailed_type_info(obj):
    type_info = []

    type_name = type(obj).__name__
    type_info.append(f"Type: {type_name}")

    if isinstance(obj, torch.Tensor):
        type_info.extend(
            [
                f"Shape: {obj.shape}",
                f"Dtype: {obj.dtype}",
                f"Device: {obj.device}",
                f"Requires grad: {obj.requires_grad}",
                f"Stride: {obj.stride()}",
                f"Contiguous: {obj.is_contiguous()}",
            ]
        )
    elif isinstance(obj, (list, tuple)):
        type_info.extend(
            [
                f"Length: {len(obj)}",
                f"Container type: {type_name}",
            ]
        )
        if obj:
            type_info.append(f"Element type: {type(obj[0]).__name__}")
    elif isinstance(obj, dict):
        type_info.extend(
            [
                f"Length: {len(obj)}",
                f"Keys: {list(obj.keys())}",
            ]
        )
    elif hasattr(obj, "__dict__"):
        attributes = [attr for attr in dir(obj) if not attr.startswith("_")]
        type_info.append(f"Attributes: {attributes}")

    return type_info


# region processors
def process_tensor(tensor: torch.Tensor, as_type=False):
    log.debug(f"Tensor: {tensor.shape}")

    if as_type:
        return {
            "text": [f"Tensor of shape {tensor.shape} of type {tensor.dtype}"]
        }

    is_mask = len(tensor.shape) == 3

    if is_mask:
        tensor = tensor.unsqueeze(-1).repeat(1, 1, 1, 3)

    image = tensor2pil(tensor)
    b64_imgs = []
    for im in image:
        if is_mask:
            im = im.convert("L")

        buffered = io.BytesIO()
        im.save(buffered, format="PNG")
        b64_imgs.append(
            "data:image/png;base64,"
            + base64.b64encode(buffered.getvalue()).decode("utf-8")
        )

    return {"b64_images": b64_imgs}


def process_list(anything, as_type=False):
    text = []
    if not anything:
        return {"text": []}

    if as_type:
        type_info = get_detailed_type_info(anything)
        type_info.extend(get_detailed_type_info(anything[0]))
        return {"text": type_info}

    first_element = anything[0]
    if (
        isinstance(first_element, list)
        and first_element
        and isinstance(first_element[0], torch.Tensor)
    ):
        text.append(
            "List of List of Tensors: "
            f"{first_element[0].shape} (x{len(anything)})"
        )

    elif isinstance(first_element, torch.Tensor):
        text.append(
            f"List of Tensors: {first_element.shape} (x{len(anything)})"
        )
    else:
        text.append(f"Array ({len(anything)}): {anything}")

    return {"text": text}


def process_dict(anything, as_type=False):
    text = []
    if as_type:
        return {"text": get_detailed_type_info(anything)}

    if "samples" in anything:
        is_empty = (
            "(empty)" if torch.count_nonzero(anything["samples"]) == 0 else ""
        )
        text.append(f"Latent Samples: {anything['samples'].shape} {is_empty}")

    elif "waveform" in anything:
        is_empty = (
            "(empty) " if torch.count_nonzero(anything["samples"]) == 0 else ""
        )

        text.append(
            f"Audio Samples: {anything['waveform'].shape}{is_empty} | sample rate {anything['sample_rate']}"
        )

    else:
        log.debug(f"Unhandled dict: {anything.keys()}")
        text.append(json.dumps(anything, indent=2))

    return {"text": text}


def process_bool(anything, as_type=False):
    return {"text": ["True" if anything else "False"]}


def process_text(anything, as_type=False):
    if as_type:
        return {"text": get_detailed_type_info(anything)}

    return {"text": [str(anything)]}


# endregion


class MTB_Debug:
    """Experimental node to debug any Comfy values.

    support for more types and widgets is planned.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"output_to_console": ("BOOLEAN", {"default": False})},
            "optional": {"as_detailed_types": ("BOOLEAN", {"default": False})},
        }

    RETURN_TYPES = ()
    FUNCTION = "do_debug"
    CATEGORY = "mtb/debug"
    OUTPUT_NODE = True

    def do_debug(
        self, output_to_console: bool, as_detailed_types: bool, **kwargs
    ):
        output = {"ui": {"items": []}}

        if output_to_console:
            for k, v in kwargs.items():
                log.info(f"{k}: {v}")

        for input_name, anything in kwargs.items():
            processor = processors.get(type(anything), process_text)

            processed = processor(anything, as_detailed_types)

            item = {
                "input": input_name,
                **processed,
            }
            output["ui"]["items"].append(item)

        return output


class MTB_SaveTensors:
    """Save torch tensors (image, mask or latent) to disk.

    useful to debug things outside comfy.
    """

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "mtb/debug"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filename_prefix": ("STRING", {"default": "ComfyPickle"}),
            },
            "optional": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "latent": ("LATENT",),
            },
        }

    FUNCTION = "save"
    OUTPUT_NODE = True
    RETURN_TYPES = ()
    CATEGORY = "mtb/debug"

    def save(
        self,
        filename_prefix,
        image: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        latent: torch.Tensor | None = None,
    ):
        (
            full_output_folder,
            filename,
            counter,
            subfolder,
            filename_prefix,
        ) = folder_paths.get_save_image_path(filename_prefix, self.output_dir)
        full_output_folder = Path(full_output_folder)
        if image is not None:
            image_file = f"{filename}_image_{counter:05}.pt"
            torch.save(image, full_output_folder / image_file)
            # np.save(full_output_folder/ image_file, image.cpu().numpy())

        if mask is not None:
            mask_file = f"{filename}_mask_{counter:05}.pt"
            torch.save(mask, full_output_folder / mask_file)
            # np.save(full_output_folder/ mask_file, mask.cpu().numpy())

        if latent is not None:
            # for latent we must use pickle
            latent_file = f"{filename}_latent_{counter:05}.pt"
            torch.save(latent, full_output_folder / latent_file)
            # pickle.dump(latent, open(full_output_folder/ latent_file, "wb"))

            # np.save(full_output_folder / latent_file,
            # latent[""].cpu().numpy())

        return f"{filename_prefix}_{counter:05}"


processors = {
    torch.Tensor: process_tensor,
    list: process_list,
    dict: process_dict,
    bool: process_bool,
}

__nodes__ = [MTB_Debug, MTB_SaveTensors]
