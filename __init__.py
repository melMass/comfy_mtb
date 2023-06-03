from .nodes.image_processing import (
    ImageCompare,
    Denoise,
    Blur,
    HSVtoRGB,
    RGBtoHSV,
    ColorCorrect,
)
from .nodes.crop import Crop, Uncrop, BoundingBox
from .nodes.graph_utils import IntToNumber, Modulo

  

# NODE MAPPING
NODE_CLASS_MAPPINGS = {
    "Int to Number (mtb)": IntToNumber,
    "Bounding Box (mtb)": BoundingBox,
    "Crop (mtb)": Crop,
    "Uncrop (mtb)": Uncrop,
    "ImageBlur (mtb)": Blur,
    "Denoise (mtb)": Denoise,
    "ImageCompare (mtb)": ImageCompare,
    "RGB to HSV (mtb)": RGBtoHSV,
    "HSV to RGB (mtb)": HSVtoRGB,
    "Color Correct (mtb)": ColorCorrect,
    "Modulo (mtb)": Modulo,
}
