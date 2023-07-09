from pytoshop.user import nested_layers

# from pytoshop.image_data import ImageData
from .. import utils
from ..log import log
from uuid import uuid4
from pathlib import Path
import folder_paths
from importlib import reload


class PsdSave:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_1": ("PSDLAYER",),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "psd_save"
    CATEGORY = "psd"
    OUTPUT_NODE = True

    def psd_save(self, **kwargs):
        groups = {
            "main": [],
        }
        out_layers = []
        for input, item in kwargs.items():
            for group, layer in item.items():
                if group not in groups:
                    groups[group] = []
                groups[group].append(layer)

        for group, layers in groups.items():
            current_group = nested_layers.Group(
                group, visible=True, opacity=255, layers=layers
            )
            out_layers.append(current_group)

        out_layers = nested_layers.nested_layers_to_psd(out_layers, color_mode=3)
        output_name = f"{uuid4()}.psd"
        output_path = Path(folder_paths.output_directory) / output_name

        log.info(f"Saving PSD to {output_name}")

        with open(output_path, "wb") as f:
            out_layers.write(f)

        return ()


class PsdLayer:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "layer_name": ("STRING", {"default": "layer"}),
                "image": ("IMAGE",),
            },
            "optional": {"mask": ("MASK",)},
        }

    RETURN_TYPES = ("PSDLAYER",)
    FUNCTION = "psd_layer"
    CATEGORY = "psd"

    def psd_layer(self, layer_name, image, mask=None):
        reload(utils)
        group = "main"
        if "/" in layer_name:
            sepname = layer_name.split("/")
            # layer_name = sepname.pop() # todo: support nesting?
            group = sepname[0]
            layer_name = sepname[1]

        log.warning("Mask is currently ignored for PSD Layers...")
        return ({group: utils.tensor2pytolayer(image, layer_name)},)


__nodes__ = [PsdLayer, PsdSave]
