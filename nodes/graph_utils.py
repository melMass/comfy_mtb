import torch
import folder_paths
import os


class SaveTensors:
    """Debug node that will probably be removed in the future"""

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

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
    CATEGORY = "utils"

    def save(
        self,
        filename_prefix,
        image: torch.Tensor = None,
        mask: torch.Tensor = None,
        latent: torch.Tensor = None,
    ):
        (
            full_output_folder,
            filename,
            counter,
            subfolder,
            filename_prefix,
        ) = folder_paths.get_save_image_path(filename_prefix, self.output_dir)

        if image is not None:
            image_file = f"{filename}_image_{counter:05}.pt"
            torch.save(image, os.path.join(full_output_folder, image_file))
            # np.save(os.path.join(full_output_folder, image_file), image.cpu().numpy())

        if mask is not None:
            mask_file = f"{filename}_mask_{counter:05}.pt"
            torch.save(mask, os.path.join(full_output_folder, mask_file))
            # np.save(os.path.join(full_output_folder, mask_file), mask.cpu().numpy())

        if latent is not None:
            # for latent we must use pickle
            latent_file = f"{filename}_latent_{counter:05}.pt"
            torch.save(latent, os.path.join(full_output_folder, latent_file))
            # pickle.dump(latent, open(os.path.join(full_output_folder, latent_file), "wb"))

            # np.save(os.path.join(full_output_folder, latent_file), latent[""].cpu().numpy())

        return f"{filename_prefix}_{counter:05}"


__nodes__ = [
    SaveTensors,
]
