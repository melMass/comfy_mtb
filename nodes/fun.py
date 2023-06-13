import qrcode
from ..utils import pil2tensor, tensor2pil
from PIL import Image
class QRNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"default": "https://www.github.com"}),
                "width": (
                    "INT",
                    {"default": 256, "max": 8096, "min": 0, "step": 1},
                ),
                "height": (
                    "INT",
                    {"default": 256, "max": 8096, "min": 0, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "do_qr"
    CATEGORY = "fun"

    def do_qr(self, url, width, height):
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(url)
        qr.make(fit=True)
        # make the pil image
        code = img = qr.make_image(back_color=(0, 0, 0), fill_color=(255, 255, 255))

        # that we now resize without filtering
        code = code.resize((width, height), Image.NEAREST)

        return (pil2tensor(code),)