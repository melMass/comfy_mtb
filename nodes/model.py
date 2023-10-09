import copy

import torch

from ..log import log


class VaeDecode_:
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
        self, vae, samples, seamless_model, use_tiling_decoder=True, tile_size=512
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


class ModelPatchSeamless:
    """Uses the stable diffusion 'hack' to infer seamless images by setting the model layers padding mode to circular (experimental)"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "tiling": (
                    "BOOLEAN",
                    {"default": True},
                ),  # kept for testing not sure why it should be false
            }
        }

    RETURN_TYPES = ("MODEL", "MODEL")
    RETURN_NAMES = (
        "Original Model (passthrough)",
        "Patched Model",
    )
    FUNCTION = "hack"

    CATEGORY = "mtb/textures"

    def apply_circular(self, model, enable):
        for layer in [
            layer for layer in model.modules() if isinstance(layer, torch.nn.Conv2d)
        ]:
            layer.padding_mode = "circular" if enable else "zeros"
        return model

    def hack(
        self,
        model,
        tiling,
    ):
        hacked_model = copy.deepcopy(model)
        self.apply_circular(hacked_model.model, tiling)
        return (model, hacked_model)


__nodes__ = [ModelPatchSeamless, VaeDecode_]
