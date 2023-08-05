from ..utils import tensor2np
import uuid
import folder_paths
from ..log import log
import comfy.model_management as model_management
import subprocess
import torch
from pathlib import Path
import numpy as np


class ExportToProres:
    """Export to ProRes 4444 (Experimental)"""

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
    CATEGORY = "mtb/IO"

    def export_prores(
        self,
        images: torch.Tensor,
        fps: float,
        prefix: str,
    ):
        if images.size(0) == 0:
            return ("",)
        output_dir = Path(folder_paths.get_output_directory())
        id = f"{prefix}_{uuid.uuid4()}.mov"

        log.debug(f"Exporting to {output_dir / id}")

        frames = tensor2np(images)
        log.debug(f"Frames type {type(frames[0])}")
        log.debug(f"Exporting {len(frames)} frames")

        frames = [frame.astype(np.uint16) * 257 for frame in frames]

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
            model_management.throw_exception_if_processing_interrupted()
            process.stdin.write(frame.tobytes())

        process.stdin.close()
        process.wait()

        return (out_path,)


class SaveGif:
    """Save the images from the batch as a GIF"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "fps": ("INT", {"default": 12, "min": 1, "max": 120}),
                "resize_by": ("FLOAT", {"default": 1.0, "min": 0.1}),
                "optimize": ("BOOLEAN", {"default": False}),
                "pingpong": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    CATEGORY = "mtb/IO"
    FUNCTION = "save_gif"

    def save_gif(self, image, fps=12, resize_by=1.0, pingpong=False):
        if image.size(0) == 0:
            return ("",)

        images = tensor2np(image)
        images = [frame.astype(np.uint8) for frame in images]
        if pingpong:
            reversed_frames = images[::-1]
            images.extend(reversed_frames)

        height, width, _ = image[0].shape

        ruuid = uuid.uuid4()

        ruuid = ruuid.hex[:10]

        out_path = f"{folder_paths.output_directory}/{ruuid}.gif"

        log.debug(f"Saving a gif file {width}x{height} as {ruuid}.gif")

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
            "rgb24",  # GIF only supports rgb24
            "-r",
            str(fps),
            "-i",
            "-",
            "-vf",
            f"fps={fps},scale={width * resize_by}:-1",  # Set frame rate and resize if necessary
            "-y",
            out_path,
        ]

        process = subprocess.Popen(command, stdin=subprocess.PIPE)

        for frame in images:
            model_management.throw_exception_if_processing_interrupted()
            process.stdin.write(frame.tobytes())

        process.stdin.close()
        process.wait()
        results = []
        results.append({"filename": f"{ruuid}.gif", "subfolder": "", "type": "output"})
        return {"ui": {"gif": results}}


__nodes__ = [SaveGif, ExportToProres]
