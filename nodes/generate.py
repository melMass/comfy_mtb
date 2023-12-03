import threading
from typing import cast

import qrcode
from PIL import Image

from ..log import log
from ..utils import comfy_dir, pil2tensor

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
                "width": (
                    "INT",
                    {"default": 512, "max": 8096, "min": 0, "step": 1},
                ),
                "height": (
                    "INT",
                    {"default": 512, "max": 8096, "min": 0, "step": 1},
                ),
                "random_seed": (
                    "INT",
                    {"default": 0, "max": 1e5, "min": 0, "step": 1},
                ),
            },
            "optional": {
                "keyword": ("STRING", {"default": "nature"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "do_unsplash_image"
    CATEGORY = "mtb/generate"

    def do_unsplash_image(self, width, height, random_seed, keyword=None):
        import io

        import requests

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
                "box_size": (
                    "INT",
                    {"default": 10, "max": 8096, "min": 0, "step": 1},
                ),
                "border": (
                    "INT",
                    {"default": 4, "max": 8096, "min": 0, "step": 1},
                ),
                "invert": (("BOOLEAN",), {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "do_qr"
    CATEGORY = "mtb/generate"

    def do_qr(
        self, url, width, height, error_correct, box_size, border, invert
    ):
        log.warning(
            "This node will soon be deprecated, there are much better alternatives like https://github.com/coreyryanhanson/comfy-qr"
        )
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

        code = img = qr.make_image(
            back_color=back_color, fill_color=fill_color
        )

        # that we now resize without filtering
        code = code.resize((width, height), Image.NEAREST)

        return (pil2tensor(code),)


def bbox_dim(bbox):
    left, upper, right, lower = bbox
    width = right - left
    height = lower - upper
    return width, height


class TextToImage:
    """Utils to convert text to image using a font.

    The tool looks for any .ttf file in the Comfy folder hierarchy.
    """

    fonts = {}

    def __init__(self):
        # - This is executed when the graph is executed, we could conditionaly reload fonts there
        pass

    @classmethod
    def CACHE_FONTS(cls):
        font_extensions = ["*.ttf", "*.otf", "*.woff", "*.woff2", "*.eot"]
        fonts = []

        for extension in font_extensions:
            try:
                if comfy_dir.exists():
                    fonts.extend(comfy_dir.glob(f"**/{extension}"))
                else:
                    log.warn(f"Directory {comfy_dir} does not exist.")
            except Exception as e:
                log.error(f"Error during font caching: {e}")

        if not fonts:
            log.warn(
                "> No fonts found in the comfy folder, place at least one font file somewhere in ComfyUI's hierarchy"
            )
        else:
            log.debug(f"> Found {len(fonts)} fonts")

        for font in fonts:
            log.debug(f"Adding font {font}")
            TextToImage.fonts[font.stem] = font.as_posix()

    @classmethod
    def INPUT_TYPES(cls):
        if not cls.fonts:
            thread = threading.Thread(target=cls.CACHE_FONTS)
            thread.start()
        else:
            log.debug(f"Using cached fonts (count: {len(cls.fonts)})")
        return {
            "required": {
                "text": (
                    "STRING",
                    {"default": "Hello world!"},
                ),
                "font": ((sorted(cls.fonts.keys())),),
                "wrap": (
                    "INT",
                    {"default": 120, "min": 0, "max": 8096, "step": 1},
                ),
                "font_size": (
                    "INT",
                    {"default": 12, "min": 1, "max": 2500, "step": 1},
                ),
                "width": (
                    "INT",
                    {"default": 512, "min": 1, "max": 8096, "step": 1},
                ),
                "height": (
                    "INT",
                    {"default": 512, "min": 1, "max": 8096, "step": 1},
                ),
                "color": (
                    "COLOR",
                    {"default": "black"},
                ),
                "background": (
                    "COLOR",
                    {"default": "white"},
                ),
                "h_align": (("left", "center", "right"), {"default": "left"}),
                "v_align": (("top", "center", "bottom"), {"default": "top"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "text_to_image"
    CATEGORY = "mtb/generate"

    def text_to_image(
        self,
        text,
        font,
        wrap,
        font_size,
        width,
        height,
        color,
        background,
        h_align="left",
        v_align="top",
    ):
        import textwrap

        from PIL import Image, ImageDraw, ImageFont

        font_path = self.fonts[font]

        # Handle word wrapping
        if wrap:
            lines = textwrap.wrap(text, width=wrap)
        else:
            lines = [text]
        font = ImageFont.truetype(font_path, font_size)
        # font = ImageFont.truetype(font_path, font_size)
        # if wrap == 0:
        #     wrap = width / font_size

        log.debug(f"Lines: {lines}")
        img = Image.new("RGBA", (width, height), background)
        draw = ImageDraw.Draw(img)

        text_height = sum(font.getsize(line)[1] for line in lines)

        # Vertical alignment
        if v_align == "top":
            y_text = 0
        elif v_align == "center":
            y_text = (height - text_height) // 2
        else:  # bottom
            y_text = height - text_height

        # Draw each line of text
        for line in lines:
            line_width, line_height = font.getsize(line)

            # Horizontal alignment
            if h_align == "left":
                x_text = 0
            elif h_align == "center":
                x_text = (width - line_width) // 2
            else:  # right
                x_text = width - line_width

            draw.text((x_text, y_text), line, color, font=font)
            y_text += line_height

        # img.save(os.path.join(folder_paths.base_path, f'{str(uuid.uuid4())}.png'))
        return (pil2tensor(img),)


__nodes__ = [
    QrCode,
    UnsplashImage,
    TextToImage,
    #  MtbExamples,
]
