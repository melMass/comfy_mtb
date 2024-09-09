from pathlib import Path

import comfy
import comfy.model_management as model_management
import comfy.utils
import numpy as np
import tensorflow as tf
import torch
from frame_interpolation.eval import interpolator, util

from ..errors import ModelNotFound
from ..log import log
from ..utils import get_model_path


class MTB_LoadFilmModel:
    """Loads a FILM model

    [DEPRECATED] Use ComfyUI-FrameInterpolation instead
    """

    @staticmethod
    def get_models() -> list[Path]:
        models_paths = get_model_path("FILM").iterdir()

        return [x for x in models_paths if x.suffix in [".onnx", ".pth"]]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "film_model": (
                    ["L1", "Style", "VGG"],
                    {"default": "Style"},
                ),
            },
        }

    RETURN_TYPES = ("FILM_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "mtb/frame iterpolation"
    DEPRECATED = True

    def load_model(self, film_model: str):
        model_path = get_model_path("FILM", film_model)
        if not model_path or not model_path.exists():
            raise ModelNotFound(f"FILM ({model_path})")

        if not (model_path / "saved_model.pb").exists():
            model_path = model_path / "saved_model"

        if not model_path.exists():
            log.error(f"Model {model_path} does not exist")
            raise ValueError(f"Model {model_path} does not exist")

        log.info(f"Loading model {model_path}")

        return (interpolator.Interpolator(model_path.as_posix(), None),)


class MTB_FilmInterpolation:
    """Google Research FILM frame interpolation for large motion

    [DEPRECATED] Use ComfyUI-FrameInterpolation instead
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "interpolate": ("INT", {"default": 2, "min": 1, "max": 50}),
                "film_model": ("FILM_MODEL",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "do_interpolation"
    CATEGORY = "mtb/frame iterpolation"
    DEPRECATED = True

    def do_interpolation(
        self,
        images: torch.Tensor,
        interpolate: int,
        film_model: interpolator.Interpolator,
    ):
        n = images.size(0)
        # check if images is an empty tensor and return it...
        if n == 0:
            return (images,)

        # check if tensorflow GPU is available
        available_gpus = tf.config.list_physical_devices("GPU")
        if not len(available_gpus):
            log.warning(
                "Tensorflow GPU not available, falling back to CPU this will be very slow"
            )
        else:
            log.debug(f"Tensorflow GPU available, using {available_gpus}")

        num_frames = (n - 1) * (2 ** (interpolate) - 1)
        log.debug(f"Will interpolate into {num_frames} frames")

        in_frames = [images[i] for i in range(n)]
        out_tensors = []

        pbar = comfy.utils.ProgressBar(num_frames)

        for frame in util.interpolate_recursively_from_memory(
            in_frames, interpolate, film_model
        ):
            out_tensors.append(
                torch.from_numpy(frame)
                if isinstance(frame, np.ndarray)
                else frame
            )
            model_management.throw_exception_if_processing_interrupted()
            pbar.update(1)

        out_tensors = torch.cat(
            [tens.unsqueeze(0) for tens in out_tensors], dim=0
        )

        log.debug(f"Returning {len(out_tensors)} tensors")
        log.debug(f"Output shape {out_tensors.shape}")
        log.debug(f"Output type {out_tensors.dtype}")
        return (out_tensors,)


__nodes__ = [MTB_LoadFilmModel, MTB_FilmInterpolation]
