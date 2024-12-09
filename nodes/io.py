import json
import subprocess
import uuid
from pathlib import Path

import comfy.model_management as model_management
import comfy.utils
import folder_paths
import numpy as np
import torch
from PIL import Image

from ..log import log
from ..utils import PIL_FILTER_MAP, output_dir, session_id, tensor2np


def get_playlist_path(playlist_name: str, persistant_playlist=False):
    if persistant_playlist:
        return output_dir / "playlists" / f"{playlist_name}.json"

    return output_dir / "playlists" / session_id / f"{playlist_name}.json"


class MTB_ReadPlaylist:
    """Read a playlist"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enable": ("BOOLEAN", {"default": True}),
                "persistant_playlist": ("BOOLEAN", {"default": False}),
                "playlist_name": (
                    "STRING",
                    {"default": "playlist_{index:04d}"},
                ),
                "index": ("INT", {"default": 0, "min": 0}),
            }
        }

    RETURN_TYPES = ("PLAYLIST",)
    FUNCTION = "read_playlist"
    CATEGORY = "mtb/IO"
    EXPERIMENTAL = True

    def read_playlist(
        self,
        enable: bool,
        persistant_playlist: bool,
        playlist_name: str,
        index: int,
    ):
        playlist_name = playlist_name.format(index=index)
        playlist_path = get_playlist_path(playlist_name, persistant_playlist)
        if not enable:
            return (None,)

        if not playlist_path.exists():
            log.warning(f"Playlist {playlist_path} does not exist, skipping")
            return (None,)

        log.debug(f"Reading playlist {playlist_path}")
        return (json.loads(playlist_path.read_text(encoding="utf-8")),)


class MTB_AddToPlaylist:
    """Add a video to the playlist"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "relative_paths": ("BOOLEAN", {"default": False}),
                "persistant_playlist": ("BOOLEAN", {"default": False}),
                "playlist_name": (
                    "STRING",
                    {"default": "playlist_{index:04d}"},
                ),
                "index": ("INT", {"default": 0, "min": 0}),
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "add_to_playlist"
    CATEGORY = "mtb/IO"
    EXPERIMENTAL = True

    def add_to_playlist(
        self,
        relative_paths: bool,
        persistant_playlist: bool,
        playlist_name: str,
        index: int,
        **kwargs,
    ):
        playlist_name = playlist_name.format(index=index)
        playlist_path = get_playlist_path(playlist_name, persistant_playlist)

        if not playlist_path.parent.exists():
            playlist_path.parent.mkdir(parents=True, exist_ok=True)

        playlist = []
        if not playlist_path.exists():
            playlist_path.write_text("[]")
        else:
            playlist = json.loads(playlist_path.read_text())
        log.debug(f"Playlist {playlist_path} has {len(playlist)} items")
        for video in kwargs.values():
            if relative_paths:
                video = Path(video).relative_to(output_dir).as_posix()

            log.debug(f"Adding {video} to playlist")
            playlist.append(video)

        log.debug(f"Writing playlist {playlist_path}")
        playlist_path.write_text(json.dumps(playlist), encoding="utf-8")
        return ()


class MTB_ExportWithFfmpeg:
    """Export with FFmpeg (Experimental).

    [DEPRACATED] Use VHS nodes instead
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "images": ("IMAGE",),
                "playlist": ("PLAYLIST",),
            },
            "required": {
                "fps": ("FLOAT", {"default": 24, "min": 1}),
                "prefix": ("STRING", {"default": "export"}),
                "format": (
                    ["mov", "mp4", "mkv", "gif", "avi"],
                    {"default": "mov"},
                ),
                "codec": (
                    ["prores_ks", "libx264", "libx265", "gif"],
                    {"default": "prores_ks"},
                ),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    OUTPUT_NODE = True
    FUNCTION = "export_prores"
    DEPRECATED = True
    CATEGORY = "mtb/IO"

    def export_prores(
        self,
        fps: float,
        prefix: str,
        format: str,
        codec: str,
        images: torch.Tensor | None = None,
        playlist: list[str] | None = None,
    ):
        file_ext = format
        file_id = f"{prefix}_{uuid.uuid4()}.{file_ext}"

        if playlist is not None and images is not None:
            log.info(f"Exporting to {output_dir / file_id}")

        if playlist is not None:
            if len(playlist) == 0:
                log.debug("Playlist is empty, skipping")
                return ("",)

            temp_playlist_path = (
                output_dir / f"temp_playlist_{uuid.uuid4()}.txt"
            )
            log.debug(
                f"Create a temporary file to list the videos for concatenation to {temp_playlist_path}"
            )

            with open(temp_playlist_path, "w") as f:
                for video_path in playlist:
                    f.write(f"file '{video_path}'\n")

            out_path = (output_dir / file_id).as_posix()

            # Prepare the FFmpeg command for concatenating videos from the playlist
            command = [
                "ffmpeg",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                temp_playlist_path.as_posix(),
                "-c",
                "copy",
                "-y",
                out_path,
            ]
            log.debug(f"Executing {command}")
            subprocess.run(command)

            temp_playlist_path.unlink()

            return (out_path,)

        if (
            images is None or images.size(0) == 0
        ):  # the is None check is just for the type checker
            return ("",)

        frames = tensor2np(images)
        log.debug(f"Frames type {type(frames[0])}")
        log.debug(f"Exporting {len(frames)} frames")
        height, width, channels = frames[0].shape
        has_alpha = channels == 4
        out_path = (output_dir / file_id).as_posix()

        if codec == "gif":
            command = [
                "ffmpeg",
                "-f",
                "image2pipe",
                "-vcodec",
                "png",
                "-r",
                str(fps),
                "-i",
                "-",
                "-vcodec",
                "gif",
                "-y",
                out_path,
            ]
            process = subprocess.Popen(command, stdin=subprocess.PIPE)
            for frame in frames:
                model_management.throw_exception_if_processing_interrupted()
                Image.fromarray(frame).save(process.stdin, "PNG")

            process.stdin.close()
            process.wait()
            return (out_path,)
        else:
            if has_alpha:
                if codec in ["prores_ks", "libx264", "libx265"]:
                    pix_fmt = (
                        "yuva444p" if codec == "prores_ks" else "yuva420p"
                    )
                    frames = [
                        frame.astype(np.uint16) * 257 for frame in frames
                    ]
                else:
                    log.warning(
                        f"Alpha channel not supported for codec {codec}. Alpha will be ignored."
                    )
                    frames = [
                        frame[:, :, :3].astype(np.uint16) * 257
                        for frame in frames
                    ]
                    pix_fmt = "rgb48le" if codec == "prores_ks" else "yuv420p"
            else:
                pix_fmt = "rgb48le" if codec == "prores_ks" else "yuv420p"
                frames = [frame.astype(np.uint16) * 257 for frame in frames]

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
        ]
        if codec == "prores_ks":
            command.extend(["-profile:v", "4444"])

        command.extend(
            [
                "-r",
                str(fps),
                "-y",
                out_path,
            ]
        )

        process = subprocess.Popen(command, stdin=subprocess.PIPE)

        pbar = comfy.utils.ProgressBar(len(frames))

        for frame in frames:
            process.stdin.write(frame.tobytes())
            pbar.update(1)

        process.stdin.close()
        process.wait()

        return (out_path,)


def prepare_animated_batch(
    batch: torch.Tensor,
    pingpong=False,
    resize_by=1.0,
    resample_filter: Image.Resampling | None = None,
    image_type=np.uint8,
) -> list[Image.Image]:
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
class MTB_SaveGif:
    """Save the images from the batch as a GIF.

    [DEPRACATED] Use VHS nodes instead
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "fps": ("INT", {"default": 12, "min": 1, "max": 120}),
                "resize_by": ("FLOAT", {"default": 1.0, "min": 0.1}),
                "optimize": ("BOOLEAN", {"default": False}),
                "pingpong": ("BOOLEAN", {"default": False}),
                "resample_filter": (list(PIL_FILTER_MAP.keys()),),
                "use_ffmpeg": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    CATEGORY = "mtb/IO"
    FUNCTION = "save_gif"
    DEPRECATED = True

    def save_gif(
        self,
        image,
        fps=12,
        resize_by=1.0,
        optimize=False,
        pingpong=False,
        resample_filter=None,
        use_ffmpeg=False,
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

        if use_ffmpeg:
            # Use FFmpeg to create the GIF from PIL images
            command = [
                "ffmpeg",
                "-f",
                "image2pipe",
                "-vcodec",
                "png",
                "-r",
                str(fps),
                "-i",
                "-",
                "-vcodec",
                "gif",
                "-y",
                out_path,
            ]
            process = subprocess.Popen(command, stdin=subprocess.PIPE)
            for image in pil_images:
                model_management.throw_exception_if_processing_interrupted()
                image.save(process.stdin, "PNG")
            process.stdin.close()
            process.wait()

        else:
            pil_images[0].save(
                out_path,
                save_all=True,
                append_images=pil_images[1:],
                optimize=optimize,
                duration=int(1000 / fps),
                loop=0,
            )
        results = [
            {"filename": f"{ruuid}.gif", "subfolder": "", "type": "output"}
        ]
        return {"ui": {"gif": results}}


__nodes__ = [
    MTB_SaveGif,
    MTB_ExportWithFfmpeg,
    MTB_AddToPlaylist,
    MTB_ReadPlaylist,
]
