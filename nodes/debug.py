import base64
import io
from pathlib import Path
from typing import Optional

import folder_paths
import open3d as o3d
import torch

from ..log import log
from ..utils import tensor2pil
from .geo_tools import mesh_to_json


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


def process_list(anything: list[object]) -> dict[str, list[str]]:
    text: list[str] = []
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
            + f"{first_element[0].shape} (x{len(anything)})"
        )

    elif isinstance(first_element, torch.Tensor):
        text.append(
            f"List of Tensors: {first_element.shape} (x{len(anything)})"
        )

    return {"text": text}


def process_dict(anything: dict[str, dict[str, any]]) -> dict[str, str]:
    if "mesh" in anything:
        m = {"geometry": {}}
        m["geometry"]["mesh"] = mesh_to_json(anything["mesh"])
        if "material" in anything:
            m["geometry"]["material"] = anything["material"]
        return m

    res = []
    if "samples" in anything:
        is_empty = (
            "(empty)" if torch.count_nonzero(anything["samples"]) == 0 else ""
        )
        res.append(f"Latent Samples: {anything['samples'].shape} {is_empty}")

    return {"text": res}


def process_bool(anything: bool) -> dict[str, str]:
    return {"text": ["True" if anything else "False"]}


def process_text(anything):
    return {"text": [str(anything)]}


# NOT USED ANYMORE
def process_geometry(anything):
    return {"geometry": [mesh_to_json(anything)]}


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

    def do_debug(self, output_to_console, **kwargs):
        output = {
            "ui": {"b64_images": [], "text": [], "geometry": []},
            # "result": ("A"),
        }

        processors = {
            torch.Tensor: process_tensor,
            list: process_list,
            dict: process_dict,
            bool: process_bool,
            o3d.geometry.Geometry: process_geometry,
        }

        for anything in kwargs.values():
            processor = processors.get(type(anything))
            if processor is None:
                if isinstance(anything, o3d.geometry.Geometry):
                    processor = process_geometry
                else:
                    processor = process_text
            log.debug(
                f"Processing: {anything} with processor: {processor.__name__} for type {type(anything)}"
            )
            processed_data = processor(anything)

            for ui_key, ui_value in processed_data.items():
                if isinstance(ui_value, list):
                    output["ui"][ui_key].extend(ui_value)
                else:
                    output["ui"][ui_key].append(ui_value)
            # log.debug(
            #     f"Processed input {k}, found {len(processed_data.get('b64_images', []))} images and {len(processed_data.get('text', []))} text items."
            # )

        if output_to_console:
            from rich.console import Console

            cons = Console()
            cons.print("OUTPUT:")
            cons.print(output)

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
