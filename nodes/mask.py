import comfy.utils
from PIL import Image

from ..utils import pil2tensor, tensor2pil


class MTB_ImageRemoveBackgroundRembg:
    """Removes the background from the input using Rembg."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "alpha_matting": (
                    "BOOLEAN",
                    {"default": False},
                ),
                "alpha_matting_foreground_threshold": (
                    "INT",
                    {"default": 240, "min": 0, "max": 255},
                ),
                "alpha_matting_background_threshold": (
                    "INT",
                    {"default": 10, "min": 0, "max": 255},
                ),
                "alpha_matting_erode_size": (
                    "INT",
                    {"default": 10, "min": 0, "max": 255},
                ),
                "post_process_mask": (
                    "BOOLEAN",
                    {"default": False},
                ),
                "bgcolor": (
                    "COLOR",
                    {"default": "#000000"},
                ),
            },
        }

    RETURN_TYPES = (
        "IMAGE",
        "MASK",
        "IMAGE",
    )
    RETURN_NAMES = (
        "Image (rgba)",
        "Mask",
        "Image",
    )
    FUNCTION = "remove_background"
    CATEGORY = "mtb/image"

    # bgcolor: Optional[Tuple[int, int, int, int]]
    def remove_background(
        self,
        image,
        alpha_matting,
        alpha_matting_foreground_threshold,
        alpha_matting_background_threshold,
        alpha_matting_erode_size,
        post_process_mask,
        bgcolor,
    ):
        from rembg import remove

        pbar = comfy.utils.ProgressBar(image.size(0))
        images = tensor2pil(image)

        out_img = []
        out_mask = []
        out_img_on_bg = []

        for img in images:
            img_rm = remove(
                data=img,
                alpha_matting=alpha_matting,
                alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
                alpha_matting_background_threshold=alpha_matting_background_threshold,
                alpha_matting_erode_size=alpha_matting_erode_size,
                session=None,
                only_mask=False,
                post_process_mask=post_process_mask,
                bgcolor=None,
            )

            # extract the alpha to a new image
            mask = img_rm.getchannel(3)

            # add our bgcolor behind the image
            image_on_bg = Image.new("RGBA", img_rm.size, bgcolor)

            image_on_bg.paste(img_rm, mask=mask)

            image_on_bg = image_on_bg.convert("RGB")

            out_img.append(img_rm)
            out_mask.append(mask)
            out_img_on_bg.append(image_on_bg)

            pbar.update(1)

        return (
            pil2tensor(out_img),
            pil2tensor(out_mask),
            pil2tensor(out_img_on_bg),
        )


__nodes__ = [
    MTB_ImageRemoveBackgroundRembg,
]
