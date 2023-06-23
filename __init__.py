from .utils import log

NODE_CLASS_MAPPINGS = {}
try:
    from .nodes.deep_bump import DeepBump
    NODE_CLASS_MAPPINGS["Deep Bump (mtb)"] = DeepBump
except Exception:
    log.error("DeepBump nodes failed to load.")
from .nodes.latent_processing import LatentLerp
from .nodes.roop import Roop
# from .nodes.geometries import LoadGeometry, GeometryInfo
try:
    from .nodes.fun import QRNode
    NODE_CLASS_MAPPINGS["QR Code (mtb)"] = QRNode
except Exception:
    log.error("QRNode failed to load.")

from .nodes.image_processing import (
    ImageCompare,
    Denoise,
    Blur,
    HSVtoRGB,
    RGBtoHSV,
    ColorCorrect,
    MaskToImage,
    ColoredImage,
)
try:
    from .nodes.image_processing import DeglazeImage
except Exception:
    log.error("DeglazeImage failed to load. This is probably an opencv mismatch. This node requires opencv-python-contrib.")

from .nodes.crop import Crop, Uncrop, BoundingBox
from .nodes.graph_utils import IntToNumber, Modulo
from .nodes.conditions import SmartStep
# NODE MAPPING
NODE_CLASS_MAPPINGS = {
    "Latent Lerp (mtb) [DEPRECATED]": LatentLerp,
    "Deep Bump (mtb)": DeepBump,
    
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
    "Deglaze Image (mtb)": DeglazeImage,
    "Smart Step (mtb)": SmartStep,
    "Mask to Image (mtb)": MaskToImage,
    "Colored Image (mtb)": ColoredImage,
    # "Load Geometry (mtb)": LoadGeometry,
    # "Geometry Info (mtb)": GeometryInfo,
}
