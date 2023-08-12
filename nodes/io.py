from ..utils import tensor2np, PIL_FILTER_MAP
import uuid
import folder_paths
from ..log import log
import comfy.model_management as model_management
import subprocess
import torch
from pathlib import Path
import numpy as np
from PIL import Image
from typing import Optional, List


class ExportWithFfmpeg:
    """Export with FFmpeg (Experimental)"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                # "frames": ("FRAMES",),
                "fps": ("FLOAT", {"default": 24, "min": 1}),
                "prefix": ("STRING", {"default": "export"}),
                "format": (["mov", "mp4", "mkv", "avi"], {"default": "mov"}),
                "codec": (
                    ["prores_ks", "libx264", "libx265"],
                    {"default": "prores_ks"},
                ),
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
        format: str,
        codec: str,
    ):
        if images.size(0) == 0:
            return ("",)
        output_dir = Path(folder_paths.get_output_directory())
        pix_fmt = "rgb48le" if codec == "prores_ks" else "yuv420p"
        file_ext = format
        file_id = f"{prefix}_{uuid.uuid4()}.{file_ext}"

        log.debug(f"Exporting to {output_dir / file_id}")

        frames = tensor2np(images)
        log.debug(f"Frames type {type(frames[0])}")
        log.debug(f"Exporting {len(frames)} frames")

        frames = [frame.astype(np.uint16) * 257 for frame in frames]

        height, width, _ = frames[0].shape

        out_path = (output_dir / file_id).as_posix()

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
            pix_fmt,
            "-r",
            str(fps),
            "-i",
            "-",
            "-c:v",
            codec,
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


def prepare_animated_batch(
    batch: torch.Tensor,
    pingpong=False,
    resize_by=1.0,
    resample_filter: Optional[Image.Resampling] = None,
    image_type=np.uint8,
) -> List[Image.Image]:
    images = tensor2np(batch)
    images = [frame.astype(image_type) for frame in images]

    height, width, _ = batch[0].shape

    if pingpong:
        reversed_frames = images[::-1]
        images.extend(reversed_frames)
    pil_images = [Image.fromarray(frame) for frame in images]

    # Resize frames if necessary
    if abs(resize_by - 1.0) > 1e-6:
        new_width = int(width * resize_by)
        new_height = int(height * resize_by)
        pil_images_resized = [
            frame.resize((new_width, new_height), resample=resample_filter)
            for frame in pil_images
        ]
        pil_images = pil_images_resized

    return pil_images


# todo: deprecate for apng
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
            },
            "optional": {
                "resample_filter": (list(PIL_FILTER_MAP.keys()),),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    CATEGORY = "mtb/IO"
    FUNCTION = "save_gif"

    def save_gif(
        self,
        image,
        fps=12,
        resize_by=1.0,
        optimize=False,
        pingpong=False,
        resample_filter=None,
    ):
        if image.size(0) == 0:
            return ("",)

        if resample_filter is not None:
            resample_filter = PIL_FILTER_MAP.get(resample_filter)

        pil_images = prepare_animated_batch(
            image,
            pingpong,
            resize_by,
            resample_filter,
        )

        ruuid = uuid.uuid4()
        ruuid = ruuid.hex[:10]
        out_path = f"{folder_paths.output_directory}/{ruuid}.gif"

        # Create the GIF from PIL images
        pil_images[0].save(
            out_path,
            save_all=True,
            append_images=pil_images[1:],
            optimize=optimize,
            duration=int(1000 / fps),
            loop=0,
        )

        results = [{"filename": f"{ruuid}.gif", "subfolder": "", "type": "output"}]
        return {"ui": {"gif": results}}


__nodes__ = [SaveGif, ExportWithFfmpeg]
