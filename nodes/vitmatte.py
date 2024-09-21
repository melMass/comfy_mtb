import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download

from ..utils import models_dir, np2tensor

# TODO: check if I can make a torch script device independant
# for now I forced it to use cuda.


class MTB_LoadVitMatteModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "kind": (("Composition-1K", "Distinctions-646"),),
                "autodownload": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("VITMATTE_MODEL",)
    RETURN_NAMES = ("torch_script",)
    CATEGORY = "mtb/vitmatte"
    FUNCTION = "execute"

    def execute(self, *, kind: str, autodownload: bool):
        dest = models_dir / "vitmatte"
        dest.mkdir(exist_ok=True)
        name = "dist" if kind == "Distinctions-646" else "com"

        file = hf_hub_download(
            repo_id="melmass/pytorch-scripts",
            filename=f"vitmatte_b_{name}.pt",
            local_dir=dest.as_posix(),
            local_files_only=not autodownload,
        )
        model = torch.jit.load(file).to("cuda")

        return (model,)


class MTB_GenerateTrimap:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # "image": ("IMAGE",),
                "mask": ("MASK",),
                "erode": ("INT", {"default": 10}),
                "dilate": ("INT", {"default": 10}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("trimap",)

    CATEGORY = "mtb/vitmatte"
    FUNCTION = "execute"

    def execute(
        self,
        # image:torch.Tensor,
        mask: torch.Tensor,
        erode: int = 10,
        dilate: int = 10,
    ):
        # TODO: not sure what's the most practical between IMAGE or MASK

        # image = image.to("cuda").half()
        mask = mask.to("cuda").half()

        trimaps = []
        for m in mask:
            mask_arr = m.squeeze(0).to(torch.uint8).cpu().numpy() * 255
            erode_kernel = np.ones((erode, erode), np.uint8)
            dilate_kernel = np.ones((dilate, dilate), np.uint8)
            eroded = cv2.erode(mask_arr, erode_kernel, iterations=5)
            dilated = cv2.dilate(mask_arr, dilate_kernel, iterations=5)
            trimap = np.zeros_like(mask_arr)
            trimap[dilated == 255] = 128
            trimap[eroded == 255] = 255
            trimaps.append(trimap)

        return (np2tensor(trimaps),)


class MTB_ApplyVitMatte:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("VITMATTE_MODEL",),
                "image": ("IMAGE",),
                "trimap": ("IMAGE",),
                "returns": (("RGB", "RGBA"),),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image (rgba)", "mask")
    CATEGORY = "mtb/utils"
    FUNCTION = "execute"

    def execute(
        self, model, image: torch.Tensor, trimap: torch.Tensor, returns: str
    ):
        im_count = image.shape[0]
        tm_count = trimap.shape[0]

        if im_count != tm_count:
            raise ValueError("image and trimap must have the same batch size")

        outputs_m: list[torch.Tensor] = []
        outputs_i: list[torch.Tensor] = []
        for i, im in enumerate(image):
            tm = trimap[i].half().unsqueeze(2).permute(2, 0, 1).to("cuda")
            im = im.half().permute(2, 0, 1).to("cuda")

            inputs = {"image": im.unsqueeze(0), "trimap": tm.unsqueeze(0)}

            fine_mask = model(inputs)
            foreground = im * fine_mask + (1 - fine_mask)

            if returns == "RGBA":
                rgba_image = torch.cat(
                    (foreground, fine_mask.unsqueeze(0)), dim=0
                )
                outputs_i.append(rgba_image.unsqueeze(0))
            else:
                outputs_i.append(foreground.unsqueeze(0))

            outputs_m.append(fine_mask.unsqueeze(0))

        result_m = torch.cat(outputs_m, dim=0)
        result_i = torch.cat(outputs_i, dim=0)

        return (result_i.permute(0, 2, 3, 1), result_m)


__nodes__ = [MTB_LoadVitMatteModel, MTB_GenerateTrimap, MTB_ApplyVitMatte]
