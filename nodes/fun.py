import qrcode
from ..utils import pil2tensor
from PIL import Image
from ..log import log

# class MtbExamples:
#     """MTB Example Images"""

#     def __init__(self):
#         pass

#     @classmethod
#     @lru_cache(maxsize=1)
#     def get_root(cls):
#         return here / "examples" / "samples"

#     @classmethod
#     def INPUT_TYPES(cls):
#         input_dir = cls.get_root()
#         files = [f.name for f in input_dir.iterdir() if f.is_file()]
#         return {
#             "required": {"image": (sorted(files),)},
#         }

#     RETURN_TYPES = ("IMAGE", "MASK")
#     FUNCTION = "do_mtb_examples"
#     CATEGORY = "fun"

#     def do_mtb_examples(self, image, index):
#         image_path = (self.get_root() / image).as_posix()

#         i = Image.open(image_path)
#         i = ImageOps.exif_transpose(i)
#         image = i.convert("RGB")
#         image = np.array(image).astype(np.float32) / 255.0
#         image = torch.from_numpy(image)[None,]
#         if "A" in i.getbands():
#             mask = np.array(i.getchannel("A")).astype(np.float32) / 255.0
#             mask = 1.0 - torch.from_numpy(mask)
#         else:
#             mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
#         return (image, mask)

#     @classmethod
#     def IS_CHANGED(cls, image):
#         image_path = (cls.get_root() / image).as_posix()

#         m = hashlib.sha256()
#         with open(image_path, "rb") as f:
#             m.update(f.read())
#         return m.digest().hex()


class UnsplashImage:
    """Unsplash Image given a keyword and a size"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "max": 8096, "min": 0, "step": 1}),
                "height": ("INT", {"default": 512, "max": 8096, "min": 0, "step": 1}),
                "random_seed": ("INT", {"default": 0, "max": 1e5, "min": 0, "step": 1}),
            },
            "optional": {
                "keyword": ("STRING", {"default": "nature"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "do_unsplash_image"
    CATEGORY = "mtb/generate"

    def do_unsplash_image(self, width, height, random_seed, keyword=None):
        import requests
        import io

        base_url = "https://source.unsplash.com/random/"

        if width and height:
            base_url += f"/{width}x{height}"

        if keyword:
            keyword = keyword.replace(" ", "%20")
            base_url += f"?{keyword}&{random_seed}"
        else:
            base_url += f"?&{random_seed}"
        try:
            log.debug(f"Getting unsplash image from {base_url}")
            response = requests.get(base_url)
            response.raise_for_status()

            image = Image.open(io.BytesIO(response.content))
            return (
                pil2tensor(
                    image,
                ),
            )

        except requests.exceptions.RequestException as e:
            print("Error retrieving image:", e)
            return (None,)


class QrCode:
    """Basic QR Code generator"""

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
                "invert": (("BOOLEAN",), {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "do_qr"
    CATEGORY = "mtb/generate"

    def do_qr(self, url, width, height, error_correct, box_size, border, invert):
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

        back_color = (255, 255, 255) if invert else (0, 0, 0)
        fill_color = (0, 0, 0) if invert else (255, 255, 255)

        code = img = qr.make_image(back_color=back_color, fill_color=fill_color)

        # that we now resize without filtering
        code = code.resize((width, height), Image.NEAREST)

        return (pil2tensor(code),)


__nodes__ = [
    QrCode,
    UnsplashImage
    #  MtbExamples,
]
