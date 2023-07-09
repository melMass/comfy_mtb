# region imports
import onnxruntime
from pathlib import Path
from PIL import Image
from typing import List, Set, Tuple, Union, Optional
import cv2
import folder_paths
import glob
import insightface
import numpy as np
import os
import tempfile
import torch
from insightface.model_zoo.inswapper import INSwapper
from ..utils import pil2tensor, tensor2pil
from ..log import mklog, NullWriter
import sys
import comfy.model_management as model_management


# endregion

log = mklog(__name__)


class LoadFaceSwapModel:
    """Loads a faceswap model"""

    @staticmethod
    def get_models() -> List[Path]:
        models_path = os.path.join(folder_paths.models_dir, "insightface/*")
        models = glob.glob(models_path)
        models = [Path(x) for x in models if x.endswith(".onnx") or x.endswith(".pth")]
        return models

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "faceswap_model": (
                    [x.name for x in cls.get_models()],
                    {"default": "None"},
                ),
            },
        }

    RETURN_TYPES = ("FACESWAP_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "face"

    def load_model(self, faceswap_model: str):
        model_path = os.path.join(
            folder_paths.models_dir, "insightface", faceswap_model
        )
        log.info(f"Loading model {model_path}")
        return (
            INSwapper(
                model_path,
                onnxruntime.InferenceSession(
                    path_or_bytes=model_path,
                    providers=onnxruntime.get_available_providers(),
                ),
            ),
        )


# region roop node
class FaceSwap:
    """Face swap using deepinsight/insightface models"""

    model = None
    model_path = None

    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "reference": ("IMAGE",),
                "faces_index": ("STRING", {"default": "0"}),
                "faceswap_model": ("FACESWAP_MODEL", {"default": "None"}),
            },
            "optional": {"debug": (["true", "false"], {"default": "false"})},
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "swap"
    CATEGORY = "face"

    def swap(
        self,
        image: torch.Tensor,
        reference: torch.Tensor,
        faces_index: str,
        faceswap_model,
        debug="false",
    ):
        def do_swap(img):
            model_management.throw_exception_if_processing_interrupted()
            img = tensor2pil(img)
            ref = tensor2pil(reference)
            face_ids = {
                int(x) for x in faces_index.strip(",").split(",") if x.isnumeric()
            }
            sys.stdout = NullWriter()
            swapped = swap_face(ref, img, faceswap_model, face_ids)
            sys.stdout = sys.__stdout__
            return pil2tensor(swapped)

        batch_count = image.size(0)

        log.info(f"Running insightface swap (batch size: {batch_count})")

        if reference.size(0) != 1:
            raise ValueError("Reference image must have batch size 1")
        if batch_count == 1:
            image = do_swap(image)

        else:
            image = [do_swap(image[i]) for i in range(batch_count)]
            image = torch.cat(image, dim=0)

        return (image,)


# endregion


# region face swap utils
def get_face_single(img_data: np.ndarray, face_index=0, det_size=(640, 640)):
    face_analyser = insightface.app.FaceAnalysis(
        name="buffalo_l", root=os.path.join(folder_paths.models_dir, "insightface")
    )
    face_analyser.prepare(ctx_id=0, det_size=det_size)
    face = face_analyser.get(img_data)

    if len(face) == 0 and det_size[0] > 320 and det_size[1] > 320:
        log.debug("No face ed, trying again with smaller image")
        det_size_half = (det_size[0] // 2, det_size[1] // 2)
        return get_face_single(img_data, face_index=face_index, det_size=det_size_half)

    try:
        return sorted(face, key=lambda x: x.bbox[0])[face_index]
    except IndexError:
        return None


def swap_face(
    source_img: Union[Image.Image, List[Image.Image]],
    target_img: Union[Image.Image, List[Image.Image]],
    face_swapper_model,
    faces_index: Optional[Set[int]] = None,
) -> Image.Image:
    if faces_index is None:
        faces_index = {0}
    log.debug(f"Swapping faces: {faces_index}")
    result_image = target_img

    if face_swapper_model is not None:
        source_img = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)
        target_img = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
        source_face = get_face_single(source_img, face_index=0)
        if source_face is not None:
            result = target_img

            for face_num in faces_index:
                target_face = get_face_single(target_img, face_index=face_num)
                if target_face is not None:
                    sys.stdout = NullWriter()
                    result = face_swapper_model.get(result, target_face, source_face)
                    sys.stdout = sys.__stdout__
                else:
                    log.warning(f"No target face found for {face_num}")

            result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        else:
            log.warning("No source face found")
    else:
        log.error("No face swap model provided")
    return result_image


# endregion face swap utils


__nodes__ = [FaceSwap, LoadFaceSwapModel]
