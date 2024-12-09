import copy

import torch
from torch.nn import functional as F
from torch.nn.modules.utils import _pair

from ..log import log


class MTB_VaeDecode:
    """Wrapper for the 2 core decoders but also adding the sd seamless hack, taken from: FlyingFireCo/tiled_ksampler"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
                "seamless_model": ("BOOLEAN", {"default": False}),
                "use_tiling_decoder": ("BOOLEAN", {"default": True}),
                "tile_size": (
                    "INT",
                    {"default": 512, "min": 320, "max": 4096, "step": 64},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"

    CATEGORY = "mtb/decode"

    def decode(
        self,
        vae,
        samples,
        seamless_model,
        use_tiling_decoder=True,
        tile_size=512,
    ):
        if seamless_model:
            if use_tiling_decoder:
                log.error(
                    "You cannot use seamless mode with tiling decoder together, skipping tiling."
                )
                use_tiling_decoder = False
            for layer in [
                layer
                for layer in vae.first_stage_model.modules()
                if isinstance(layer, torch.nn.Conv2d)
            ]:
                layer.padding_mode = "circular"
        if use_tiling_decoder:
            return (
                vae.decode_tiled(
                    samples["samples"],
                    tile_x=tile_size // 8,
                    tile_y=tile_size // 8,
                ),
            )
        else:
            return (vae.decode(samples["samples"]),)


def conv_forward(lyr, tensor, weight, bias):
    step = lyr.timestep
    if (lyr.paddingStartStep < 0 or step >= lyr.paddingStartStep) and (
        lyr.paddingStopStep < 0 or step <= lyr.paddingStopStep
    ):
        working = F.pad(tensor, lyr.paddingX, mode=lyr.padding_modeX)
        working = F.pad(working, lyr.paddingY, mode=lyr.padding_modeY)
    else:
        working = F.pad(tensor, lyr.paddingX, mode="constant")
        working = F.pad(working, lyr.paddingY, mode="constant")

    lyr.timestep += 1

    return F.conv2d(
        working, weight, bias, lyr.stride, _pair(0), lyr.dilation, lyr.groups
    )


class MTB_ModelPatchSeamless:
    """Uses the stable diffusion 'hack' to infer seamless images by setting the model layers padding mode to circular (experimental)"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "startStep": ("INT", {"default": 0}),
                "stopStep": ("INT", {"default": 999}),
                "tilingX": (
                    "BOOLEAN",
                    {"default": True},
                ),
                "tilingY": (
                    "BOOLEAN",
                    {"default": True},
                ),
            }
        }

    RETURN_TYPES = ("MODEL", "MODEL")
    RETURN_NAMES = (
        "Original Model (passthrough)",
        "Patched Model",
    )
    FUNCTION = "hack"

    CATEGORY = "mtb/textures"

    def apply_circular(self, model, startStep, stopStep, x, y):
        for layer in [
            layer
            for layer in model.modules()
            if isinstance(layer, torch.nn.Conv2d)
        ]:
            layer.padding_modeX = "circular" if x else "constant"
            layer.padding_modeY = "circular" if y else "constant"
            layer.paddingX = (
                layer._reversed_padding_repeated_twice[0],
                layer._reversed_padding_repeated_twice[1],
                0,
                0,
            )
            layer.paddingY = (
                0,
                0,
                layer._reversed_padding_repeated_twice[2],
                layer._reversed_padding_repeated_twice[3],
            )
            layer.paddingStartStep = startStep
            layer.paddingStopStep = stopStep
            layer.timestep = 0
            layer._conv_forward = conv_forward.__get__(layer, torch.nn.Conv2d)

        return model

    def hack(
        self,
        model,
        startStep,
        stopStep,
        tilingX,
        tilingY,
    ):
        hacked_model = copy.deepcopy(model)
        self.apply_circular(
            hacked_model.model, startStep, stopStep, tilingX, tilingY
        )
        return (model, hacked_model)


__nodes__ = [MTB_ModelPatchSeamless, MTB_VaeDecode]
