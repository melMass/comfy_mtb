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
    ImagePremultiply
)
try:
    from .nodes.image_processing import DeglazeImage
except Exception:
    log.error("DeglazeImage failed to load. This is probably an opencv mismatch. This node requires opencv-python-contrib.")

from .nodes.crop import Crop, Uncrop, BoundingBox, BBoxFromMask
from .nodes.conditions import (
    SmartStep, 
    StylesLoader, 
    TextToImage
)
from .nodes.video import LoadImageSequence, SaveImageSequence
from .nodes.mask import ImageRemoveBackgroundRembg
# from .nodes.videopose import MMPoseEstimation
# NODE MAPPING
NODE_CLASS_MAPPINGS = {
    "Latent Lerp (mtb) [DEPRECATED]": LatentLerp,
    "Deep Bump (mtb)": DeepBump,
    
    "Int to Number (mtb)": IntToNumber,
    "Bounding Box (mtb)": BoundingBox,
    "Bounding Box From Mask (mtb)": BBoxFromMask,
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
    "Styles Loader (mtb)": StylesLoader,
    "Load Image Sequence (mtb)": LoadImageSequence,
    "Save Image Sequence (mtb)": SaveImageSequence,
    "Mask to Image (mtb)": MaskToImage,
    "Image Remove Background RemBG (mtb)": ImageRemoveBackgroundRembg,
    "Colored Image (mtb)": ColoredImage,
    "Image Premultiply (mtb)": ImagePremultiply,
    # "Load Geometry (mtb)": LoadGeometry,
    # "Geometry Info (mtb)": GeometryInfo,
}
