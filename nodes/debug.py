from ..utils import tensor2pil
from ..log import log
import io, base64
import torch
import folder_paths
from typing import Optional
from pathlib import Path


class Debug:
    """Experimental node to debug any Comfy values, support for more types and widgets is planned"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"anything_1": ("*")},
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "do_debug"
    CATEGORY = "mtb/debug"
    OUTPUT_NODE = True

    def do_debug(self, **kwargs):
        output = {
            "ui": {"b64_images": [], "text": []},
            "result": ("A"),
        }
        for k, v in kwargs.items():
            anything = v
            text = ""
            if isinstance(anything, torch.Tensor):
                log.debug(f"Tensor: {anything.shape}")

                # write the images to temp

                image = tensor2pil(anything)
                b64_imgs = []
                for im in image:
                    buffered = io.BytesIO()
                    im.save(buffered, format="PNG")
                    b64_imgs.append(
                        "data:image/png;base64,"
                        + base64.b64encode(buffered.getvalue()).decode("utf-8")
                    )

                output["ui"]["b64_images"] += b64_imgs
                log.debug(f"Input {k} contains {len(b64_imgs)} images")
            elif isinstance(anything, bool):
                log.debug(f"Input {k} contains boolean: {anything}")
                output["ui"]["text"] += ["True" if anything else "False"]
            else:
                text = str(anything)
                log.debug(f"Input {k} contains text: {text}")
                output["ui"]["text"] += [text]

        return output


class SaveTensors:
    """Save torch tensors (image, mask or latent) to disk, useful to debug things outside comfy"""

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

            # np.save(full_output_folder/ latent_file, latent[""].cpu().numpy())

        return f"{filename_prefix}_{counter:05}"


__nodes__ = [Debug, SaveTensors]
