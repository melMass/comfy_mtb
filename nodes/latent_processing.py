import node_helpers
import torch

from ..log import log


class MTB_LatentLerp:
    """Linear interpolation (blend) between two latent vectors"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "A": ("LATENT",),
                "B": ("LATENT",),
                "t": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "lerp_latent"

    # should fix or remove
    DEPRECATED = True

    CATEGORY = "mtb/latent"

    def lerp_latent(self, A, B, t):
        a = A.copy()
        b = B.copy()

        torch.lerp(a["samples"], b["samples"], t, out=a["samples"])

        return (a,)


class MTB_ReferenceLatents:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "execute"
    CATEGORY = "mtb/latent"

    def execute(self, positive, negative, vae, **kwargs):
        if not kwargs:
            raise ValueError("At least one image must be provided.")

        image_refs = list(kwargs.values())
        # device = image_refs[0].device

        for im in image_refs:
            # encode
            log.debug("Encoding reference image to latents")
            log.debug(f"Image shape: {im.shape}")
            latent = vae.encode(im)

            if positive is not None:
                positive = node_helpers.conditioning_set_values(
                    positive,
                    {"reference_latents": [latent]},
                    append=True,
                )
            if negative is not None:
                negative = node_helpers.conditioning_set_values(
                    negative,
                    {"reference_latents": [latent]},
                    append=True,
                )

        return (positive, negative)


__nodes__ = [
    MTB_LatentLerp,
    MTB_ReferenceLatents,
]
