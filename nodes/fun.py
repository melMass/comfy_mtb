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
                "error_correct": (("L", "M", "Q", "H"), {"default": "L"}),
                "box_size": ("INT", {"default": 10, "max": 8096, "min": 0, "step": 1}),
                "border": ("INT", {"default": 4, "max": 8096, "min": 0, "step": 1}),
                "invert": (("True", "False"), {"default": "False"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "do_qr"
    CATEGORY = "fun"

    def do_qr(self, url, width, height,error_correct, box_size, border,invert):
        
        if error_correct == "L" or error_correct not in ["M", "Q", "H"]:
            error_correct = qrcode.constants.ERROR_CORRECT_L
        elif error_correct == "M":
            error_correct = qrcode.constants.ERROR_CORRECT_M
        elif error_correct == "Q":
            error_correct = qrcode.constants.ERROR_CORRECT_Q
        else:
            error_correct = qrcode.constants.ERROR_CORRECT_H
            
        qr = qrcode.QRCode(
            version=1,
            error_correction=error_correct,
            box_size=box_size,
            border=border,
        )
        qr.add_data(url)
        qr.make(fit=True)

        back_color = (255, 255, 255) if invert == "True" else (0, 0, 0)
        fill_color = (0, 0, 0) if invert == "True" else (255, 255, 255)
        
        code = img = qr.make_image(back_color=back_color, fill_color=fill_color)

        # that we now resize without filtering
        code = code.resize((width, height), Image.NEAREST)

        return (pil2tensor(code),)
    
    
__nodes__ = [QRNode]