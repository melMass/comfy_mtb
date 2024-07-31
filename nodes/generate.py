from PIL import Image

from ..log import log
from ..utils import comfy_dir, font_path, pil2tensor

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


class MTB_UnsplashImage:
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


def bbox_dim(bbox):
    left, upper, right, lower = bbox
    width = right - left
    height = lower - upper
    return width, height


# TODO: Auto install the base font to ComfyUI/fonts


class MTB_TextToImage:
    """Utils to convert text to image using a font.

    The tool looks for any .ttf file in the Comfy folder hierarchy.
    """

    fonts = {}
    DESCRIPTION = """# Text to Image

This node look for any font files in comfy_dir/fonts.
by default it fallsback to a default font.

![img](https://i.imgur.com/3GT92hy.gif)
"""

    def __init__(self):
        # - This is executed when the graph is executed,
        # - we could conditionaly reload fonts there
        pass

    @classmethod
    def CACHE_FONTS(cls):
        font_extensions = ["*.ttf", "*.otf", "*.woff", "*.woff2", "*.eot"]
        fonts = [font_path]

        for extension in font_extensions:
            try:
                if comfy_dir.exists():
                    fonts.extend(comfy_dir.glob(f"fonts/**/{extension}"))
                else:
                    log.warn(f"Directory {comfy_dir} does not exist.")
            except Exception as e:
                log.error(f"Error during font caching: {e}")

        for font in fonts:
            log.debug(f"Adding font {font}")
            MTB_TextToImage.fonts[font.stem] = font.as_posix()

    @classmethod
    def INPUT_TYPES(cls):
        if not cls.fonts:
            cls.CACHE_FONTS()
        else:
            log.debug(f"Using cached fonts (count: {len(cls.fonts)})")
        return {
            "required": {
                "text": (
                    "STRING",
                    {"default": "Hello world!"},
                ),
                "font": ((sorted(cls.fonts.keys())),),
                "wrap": ("BOOLEAN", {"default": True}),
                "trim": ("BOOLEAN", {"default": True}),
                "line_height": (
                    "FLOAT",
                    {"default": 1.0, "min": 0, "step": 0.1},
                ),
                "font_size": (
                    "INT",
                    {"default": 32, "min": 1, "max": 2500, "step": 1},
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
                "h_offset": (
                    "INT",
                    {"default": 0, "min": 0, "max": 8096, "step": 1},
                ),
                "v_offset": (
                    "INT",
                    {"default": 0, "min": 0, "max": 8096, "step": 1},
                ),
                "h_coverage": (
                    "INT",
                    {"default": 100, "min": 1, "max": 100, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "text_to_image"
    CATEGORY = "mtb/generate"

    def text_to_image(
        self,
        text: str,
        font,
        wrap,
        trim,
        line_height,
        font_size,
        width,
        height,
        color,
        background,
        h_align="left",
        v_align="top",
        h_offset=0,
        v_offset=0,
        h_coverage=100,
    ):
        import textwrap

        from PIL import Image, ImageDraw, ImageFont

        font_path = self.fonts[font]

        text = (
            text.encode("ascii", "ignore").decode().strip() if trim else text
        )
        # Handle word wrapping
        if wrap:
            wrap_width = (((width / 100) * h_coverage) / font_size) * 2
            lines = textwrap.wrap(text, width=wrap_width)
        else:
            lines = [text]
        font = ImageFont.truetype(font_path, size=font_size)
        log.debug(f"Lines: {lines}")
        img = Image.new("RGBA", (width, height), background)
        draw = ImageDraw.Draw(img)

        line_height_px = line_height * font_size

        # Vertical alignment
        if v_align == "top":
            y_text = v_offset
        elif v_align == "center":
            y_text = ((height - (line_height_px * len(lines))) // 2) + v_offset
        else:  # bottom
            y_text = (height - (line_height_px * len(lines))) - v_offset

        def get_width(line):
            if hasattr(font, "getsize"):
                return font.getsize(line)[0]
            else:
                return font.getlength(line)

        # Draw each line of text
        for line in lines:
            line_width = get_width(line)
            # Horizontal alignment
            if h_align == "left":
                x_text = h_offset
            elif h_align == "center":
                x_text = ((width - line_width) // 2) + h_offset
            else:  # right
                x_text = (width - line_width) - h_offset

            draw.text((x_text, y_text), line, fill=color, font=font)
            y_text += line_height_px

        return (pil2tensor(img),)


__nodes__ = [
    MTB_UnsplashImage,
    MTB_TextToImage,
    #  MtbExamples,
]
