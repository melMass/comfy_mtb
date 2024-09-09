# Optional face enhance nodes
# region imports
import sys
from pathlib import Path

import comfy.model_management as model_management
import cv2
import insightface
import numpy as np
import onnxruntime
import torch
from insightface.model_zoo.inswapper import INSwapper
from PIL import Image

from ..errors import ModelNotFound
from ..log import NullWriter, mklog
from ..utils import download_antelopev2, get_model_path, pil2tensor, tensor2pil

# endregion

log = mklog(__name__)


class MTB_LoadFaceAnalysisModel:
    """Loads a face analysis model"""

    models = []

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "faceswap_model": (
                    ["antelopev2", "buffalo_l", "buffalo_m", "buffalo_sc"],
                    {"default": "buffalo_l"},
                ),
            },
        }

    RETURN_TYPES = ("FACE_ANALYSIS_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "mtb/facetools"
    DEPRECATED = True

    def load_model(self, faceswap_model: str):
        if faceswap_model == "antelopev2":
            download_antelopev2()

        face_analyser = insightface.app.FaceAnalysis(
            name=faceswap_model,
            root=get_model_path("insightface").as_posix(),
        )
        return (face_analyser,)


class MTB_LoadFaceSwapModel:
    """Loads a faceswap model"""

    @staticmethod
    def get_models() -> list[Path]:
        models_path = get_model_path("insightface")
        if models_path.exists():
            models = models_path.iterdir()
            return [x for x in models if x.suffix in [".onnx", ".pth"]]
        return []

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
    CATEGORY = "mtb/facetools"
    DEPRECATED = True

    def load_model(self, faceswap_model: str):
        model_path = get_model_path("insightface", faceswap_model)
        if not model_path or not model_path.exists():
            raise ModelNotFound(f"{faceswap_model} ({model_path})")

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
class MTB_FaceSwap:
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
                "faceanalysis_model": (
                    "FACE_ANALYSIS_MODEL",
                    {"default": "None"},
                ),
                "faceswap_model": ("FACESWAP_MODEL", {"default": "None"}),
            },
            "optional": {
                "preserve_alpha": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "swap"
    CATEGORY = "mtb/facetools"
    DEPRECATED = True

    def swap(
        self,
        image: torch.Tensor,
        reference: torch.Tensor,
        faces_index: str,
        faceanalysis_model,
        faceswap_model,
        preserve_alpha=False,
    ):
        def do_swap(img):
            model_management.throw_exception_if_processing_interrupted()
            img = tensor2pil(img)[0]
            ref = tensor2pil(reference)[0]

            alpha_channel = None
            if preserve_alpha and img.mode == "RGBA":
                alpha_channel = img.getchannel("A")
                img = img.convert("RGB")

            face_ids = {
                int(x)
                for x in faces_index.strip(",").split(",")
                if x.isnumeric()
            }
            sys.stdout = NullWriter()
            swapped = swap_face(
                faceanalysis_model, ref, img, faceswap_model, face_ids
            )
            sys.stdout = sys.__stdout__
            if alpha_channel:
                swapped.putalpha(alpha_channel)
            return pil2tensor(swapped)

        batch_count = image.size(0)

        log.info(f"Running insightface swap (batch size: {batch_count})")

        if reference.size(0) != 1:
            raise ValueError("Reference image must have batch size 1")
        if batch_count == 1:
            image = do_swap(image)

        else:
            image_batch = [do_swap(image[i]) for i in range(batch_count)]
            image = torch.cat(image_batch, dim=0)

        return (image,)


# endregion


# region face swap utils
def get_face_single(
    face_analyser, img_data: np.ndarray, face_index=0, det_size=(640, 640)
):
    face_analyser.prepare(ctx_id=0, det_size=det_size)
    face = face_analyser.get(img_data)

    if len(face) == 0 and det_size[0] > 320 and det_size[1] > 320:
        log.debug("No face ed, trying again with smaller image")
        det_size_half = (det_size[0] // 2, det_size[1] // 2)
        return get_face_single(
            face_analyser,
            img_data,
            face_index=face_index,
            det_size=det_size_half,
        )

    try:
        return sorted(face, key=lambda x: x.bbox[0])[face_index]
    except IndexError:
        return None


def swap_face(
    face_analyser,
    source_img: Image.Image | list[Image.Image],
    target_img: Image.Image | list[Image.Image],
    face_swapper_model,
    faces_index: set[int] | None = None,
) -> Image.Image:
    if faces_index is None:
        faces_index = {0}
    log.debug(f"Swapping faces: {faces_index}")
    result_image = target_img

    if face_swapper_model is not None:
        cv_source_img = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)
        cv_target_img = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
        source_face = get_face_single(
            face_analyser, cv_source_img, face_index=0
        )
        if source_face is not None:
            result = cv_target_img

            for face_num in faces_index:
                target_face = get_face_single(
                    face_analyser, cv_target_img, face_index=face_num
                )
                if target_face is not None:
                    sys.stdout = NullWriter()
                    result = face_swapper_model.get(
                        result, target_face, source_face
                    )
                    sys.stdout = sys.__stdout__
                else:
                    log.warning(f"No target face found for {face_num}")

            result_image = Image.fromarray(
                cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            )
        else:
            log.warning("No source face found")
    else:
        log.error("No face swap model provided")
    return result_image


# endregion face swap utils


__nodes__ = [MTB_FaceSwap, MTB_LoadFaceSwapModel, MTB_LoadFaceAnalysisModel]
