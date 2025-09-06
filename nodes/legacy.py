# from ..utils import hex_to_rgb
class MTB_ColorInput:
    RETURN_TYPES = ("COLOR",)
    FUNCTION = "color"
    CATEGORY = "mtb/color"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"color": ("MTB_COLOR", {"default": "#ffffff"})},
        }

    def color(self, color):
        return (color,)


__nodes__ = [MTB_ColorInput]
