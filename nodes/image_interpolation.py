from typing import List
from pathlib import Path
import os
import glob
import folder_paths
from ..log import log
import torch
from frame_interpolation.eval import util, interpolator
from ..utils import tensor2np
import uuid
import numpy as np
import subprocess
import comfy


class LoadFilmModel:
    """Loads a FILM model"""

    @staticmethod
    def get_models() -> List[Path]:
        models_path = os.path.join(folder_paths.models_dir, "FILM/*")
        models = glob.glob(models_path)
        models = [Path(x) for x in models if x.endswith(".onnx") or x.endswith(".pth")]
        return models

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
    CATEGORY = "face"

    def load_model(self, film_model: str):
        model_path = Path(folder_paths.models_dir) / "FILM" / film_model
        if not (model_path / "saved_model.pb").exists():
            model_path = model_path / "saved_model"

        if not model_path.exists():
            log.error(f"Model {model_path} does not exist")
            raise ValueError(f"Model {model_path} does not exist")

        log.info(f"Loading model {model_path}")

        return (interpolator.Interpolator(model_path.as_posix(), None),)


class FilmInterpolation:
    """Google Research FILM frame interpolation for large motion"""

    def __init__(self):
        pass

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
    CATEGORY = "animation"

    def do_interpolation(
        self,
        images: torch.Tensor,
        interpolate: int,
        film_model: interpolator.Interpolator,
    ):
        n = images.size(0)

        num_frames = (n - 1) * (2 ** (interpolate) - 1)
        log.debug(f"Will interpolate into {num_frames} frames")

        in_frames = [images[i] for i in range(n)]
        out_tensors = []

        pbar = comfy.utils.ProgressBar(num_frames)

        for frame in util.interpolate_recursively_from_memory(
            in_frames, interpolate, film_model
        ):
            out_tensors.append(
                torch.from_numpy(frame) if isinstance(frame, np.ndarray) else frame
            )
            pbar.update(1)

        out_tensors = torch.cat([tens.unsqueeze(0) for tens in out_tensors], dim=0)

        log.debug(f"Returning {len(out_tensors)} tensors")
        log.debug(f"Output shape {out_tensors.shape}")
        log.debug(f"Output type {out_tensors.dtype}")
        return (out_tensors,)


class ConcatImages:
    """Add images to batch"""

    def __init__(self):
        pass

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "concat_images"
    CATEGORY = "animation"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "imageA": ("IMAGE",),
                "imageB": ("IMAGE",),
            },
        }

    @classmethod
    def concatenate_tensors(cls, A: torch.Tensor, B: torch.Tensor):
        # Get the batch sizes of A and B
        batch_size_A = A.size(0)
        batch_size_B = B.size(0)

        # Concatenate the tensors along the batch dimension
        concatenated = torch.cat((A, B), dim=0)

        # Update the batch size in the concatenated tensor
        concatenated_size = list(concatenated.size())
        concatenated_size[0] = batch_size_A + batch_size_B
        concatenated = concatenated.view(*concatenated_size)

        return concatenated

    def concat_images(self, imageA: torch.Tensor, imageB: torch.Tensor):
        log.debug(f"Concatenating A ({imageA.shape}) and B ({imageB.shape})")
        return (self.concatenate_tensors(imageA, imageB),)


class ExportToProRes:
    """Export to ProRes 4444 (Experimental)"""

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                # "frames": ("FRAMES",),
                "fps": ("FLOAT", {"default": 24, "min": 1}),
                "prefix": ("STRING", {"default": "export"}),
            }
        }

    RETURN_TYPES = ("VIDEO",)
    OUTPUT_NODE = True
    FUNCTION = "export_prores"
    CATEGORY = "animation"

    def export_prores(
        self,
        images: torch.Tensor,
        fps: float,
        prefix: str,
    ):
        output_dir = Path(folder_paths.get_output_directory())
        id = f"{prefix}_{uuid.uuid4()}.mov"

        log.debug(f"Exporting to {output_dir / id}")

        frames = tensor2np(images)
        log.debug(f"Frames type {type(frames)}")
        log.debug(f"Exporting {len(frames)} frames")

        frames = [frame.astype(np.uint16) for frame in frames]

        height, width, _ = frames[0].shape

        out_path = (output_dir / id).as_posix()

        # Prepare the FFmpeg command
        command = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{width}x{height}",
            "-pix_fmt",
            "rgb48le",
            "-r",
            str(fps),
            "-i",
            "-",
            "-c:v",
            "prores_ks",
            "-profile:v",
            "4",
            "-pix_fmt",
            "yuva444p10le",
            "-r",
            str(fps),
            "-y",
            out_path,
        ]

        process = subprocess.Popen(command, stdin=subprocess.PIPE)

        for frame in frames:
            process.stdin.write(frame.tobytes())

        process.stdin.close()
        process.wait()

        return (out_path,)


__nodes__ = [LoadFilmModel, FilmInterpolation, ExportToProRes, ConcatImages]
