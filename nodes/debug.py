import base64
import io
import json
from pathlib import Path
from typing import Optional

import folder_paths
import torch

from ..log import log
from ..utils import tensor2pil


# region processors
def process_tensor(tensor):
    log.debug(f"Tensor: {tensor.shape}")

    image = tensor2pil(tensor)
    b64_imgs = []
    for im in image:
        buffered = io.BytesIO()
        im.save(buffered, format="PNG")
        b64_imgs.append(
            "data:image/png;base64,"
            + base64.b64encode(buffered.getvalue()).decode("utf-8")
        )

    return {"b64_images": b64_imgs}


def process_list(anything):
    text = []
    if not anything:
        return {"text": []}

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


def process_dict(anything):
    text = []
    if "samples" in anything:
        is_empty = (
            "(empty)" if torch.count_nonzero(anything["samples"]) == 0 else ""
        )
        text.append(f"Latent Samples: {anything['samples'].shape} {is_empty}")

    else:
        text.append(json.dumps(anything, indent=2))

    return {"text": text}


def process_bool(anything):
    return {"text": ["True" if anything else "False"]}


def process_text(anything):
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
        }

    RETURN_TYPES = ()
    FUNCTION = "do_debug"
    CATEGORY = "mtb/debug"
    OUTPUT_NODE = True

    def do_debug(self, output_to_console: bool, **kwargs):
        output = {
            "ui": {"b64_images": [], "text": []},
            # "result": ("A"),
        }

        processors = {
            torch.Tensor: process_tensor,
            list: process_list,
            dict: process_dict,
            bool: process_bool,
        }
        if output_to_console:
            for k, v in kwargs.items():
                log.info(f"{k}: {v}")

        for anything in kwargs.values():
            processor = processors.get(type(anything), process_text)

            processed_data = processor(anything)

            for ui_key, ui_value in processed_data.items():
                output["ui"][ui_key].extend(ui_value)

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
        image: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        latent: Optional[torch.Tensor] = None,
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


__nodes__ = [MTB_Debug, MTB_SaveTensors]
