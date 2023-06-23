# region imports
from ifnude import detect
from logging import getLogger
from pathlib import Path
from PIL import Image
from typing import List, Set, Tuple
import cv2
import folder_paths
import glob
import insightface
import numpy as np
import onnxruntime
import os
import tempfile
import torch

from ..utils import pil2tensor, tensor2pil

# endregion

logger = getLogger(__name__)
providers = onnxruntime.get_available_providers()

# region roop node
class Roop:
    model = None
    model_path = None

    def __init__(self) -> None:
        pass

    @staticmethod
    def get_models() -> List[Path]:
        models_path = os.path.join(folder_paths.models_dir, "roop/*")
        models = glob.glob(models_path)
        models = [Path(x) for x in models if x.endswith(".onnx") or x.endswith(".pth")]
        return models

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "reference": ("IMAGE",),
                "faces_index": ("STRING", {"default": "0"}),
                "roop_model": ([x.name for x in cls.get_models()], {"default": "None"}),
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "swap"
    CATEGORY = "image"

    def swap(
        self,
        image: torch.Tensor,
        reference: torch.Tensor,
        faces_index: str,
        roop_model: str,
    ):
        image = tensor2pil(image)
        reference = tensor2pil(reference)
        faces_index = {
            int(x) for x in faces_index.strip(",").split(",") if x.isnumeric()
        }

        roop_model = self.getFaceSwapModel(roop_model)
        swapped = swap_face(reference, image, roop_model, faces_index)
        return (pil2tensor(swapped),)

    def getFaceSwapModel(self, model_path: str):
        model_path = os.path.join(folder_paths.models_dir, "roop", model_path)
        if self.model_path is None or self.model_path != model_path:
            self.model_path = model_path
            self.model = insightface.model_zoo.get_model(
                model_path, providers=providers
            )

        return self.model


# endregion

# region face swap utils
def get_face_single(img_data: np.ndarray, face_index=0, det_size=(640, 640)):
    face_analyser = insightface.app.FaceAnalysis(name="buffalo_l", providers=providers)
    face_analyser.prepare(ctx_id=0, det_size=det_size)
    face = face_analyser.get(img_data)

    if len(face) == 0 and det_size[0] > 320 and det_size[1] > 320:
        det_size_half = (det_size[0] // 2, det_size[1] // 2)
        return get_face_single(img_data, face_index=face_index, det_size=det_size_half)

    try:
        return sorted(face, key=lambda x: x.bbox[0])[face_index]
    except IndexError:
        return None


def convert_to_sd(img) -> Tuple[bool, str]:
    chunks = detect(img)
    shapes = [chunk["score"] > 0.7 for chunk in chunks]
    return [any(shapes), tempfile.NamedTemporaryFile(delete=False, suffix=".png")]


def swap_face(
    source_img: Image.Image,
    target_img: Image.Image,
    face_swapper_model=None,
    faces_index: Set[int] = None,
) -> Image.Image:
    if faces_index is None:
        faces_index = {0}
    result_image = target_img
    converted = convert_to_sd(target_img)
    scale, fn = converted[0], converted[1]
    if face_swapper_model is not None and not scale:
        if isinstance(source_img, str):  # source_img is a base64 string
            import base64, io

            if (
                "base64," in source_img
            ):  # check if the base64 string has a data URL scheme
                base64_data = source_img.split("base64,")[-1]
                img_bytes = base64.b64decode(base64_data)
            else:
                # if no data URL scheme, just decode
                img_bytes = base64.b64decode(source_img)
            source_img = Image.open(io.BytesIO(img_bytes))
        source_img = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)
        target_img = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
        source_face = get_face_single(source_img, face_index=0)
        if source_face is not None:
            result = target_img

            for face_num in faces_index:
                target_face = get_face_single(target_img, face_index=face_num)
                if target_face is not None:
                    result = face_swapper_model.get(result, target_face, source_face)
                else:
                    logger.info(f"No target face found for {face_num}")

            result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        else:
            logger.info("No source face found")
    return result_image


# endregion face swap utils
