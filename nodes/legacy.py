from ..utils import hex_to_rgb


class MTB_ColorInput:
    RETURN_TYPES = ("COLOR","STRING","STRING")
    RETURN_NAMES = ("color","hex","r,g,b")
    OUTPUT_TOOLTIPS = (
        "Color in mtb format (internaly just a hex string)",
        "Hex color string",
        "RGB values as comma-separated string",
    )
    FUNCTION = "color"
    CATEGORY = "mtb/color"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"color": ("MTB_COLOR", {"default": "#ffffff"})},
        }

    def color(self, color:str):
        # convert hex to rgb
        r, g, b = hex_to_rgb(color)

        # TODO: official COLOR will be without the #
        return (color,color,f"{r},{g},{b}")


__nodes__ = [MTB_ColorInput]
