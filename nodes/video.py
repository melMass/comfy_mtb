import os
import re
import torch
import numpy as np
import hashlib
from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo
import folder_paths
from pathlib import Path
import json

from ..log import log


class LoadImageSequence:
    """Load an image sequence from a folder. The current frame is used to determine which image to load.

    Usually used in conjunction with the `Primitive` node set to increment to load a sequence of images from a folder.
    Use -1 to load all matching frames as a batch.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": "videos/####.png"}),
                "current_frame": (
                    "INT",
                    {"default": 0, "min": -1, "max": 9999999},
                ),
            }
        }

    CATEGORY = "mtb/IO"
    FUNCTION = "load_image"
    RETURN_TYPES = (
        "IMAGE",
        "MASK",
        "INT",
    )
    RETURN_NAMES = (
        "image",
        "mask",
        "current_frame",
    )

    def load_image(self, path=None, current_frame=0):
        load_all = current_frame == -1

        if load_all:
            log.debug(f"Loading all frames from {path}")
            frames = resolve_all_frames(path)
            log.debug(f"Found {len(frames)} frames")

            imgs = []
            masks = []

            for frame in frames:
                img, mask = img_from_path(frame)
                imgs.append(img)
                masks.append(mask)

            out_img = torch.cat(imgs, dim=0)
            out_mask = torch.cat(masks, dim=0)

            return (
                out_img,
                out_mask,
            )

        log.debug(f"Loading image: {path}, {current_frame}")
        print(f"Loading image: {path}, {current_frame}")
        resolved_path = resolve_path(path, current_frame)
        image_path = folder_paths.get_annotated_filepath(resolved_path)
        image, mask = img_from_path(image_path)
        return (
            image,
            mask,
            current_frame,
        )

    @staticmethod
    def IS_CHANGED(path="", current_frame=0):
        print(f"Checking if changed: {path}, {current_frame}")
        resolved_path = resolve_path(path, current_frame)
        image_path = folder_paths.get_annotated_filepath(resolved_path)
        if os.path.exists(image_path):
            m = hashlib.sha256()
            with open(image_path, "rb") as f:
                m.update(f.read())
            return m.digest().hex()
        return "NONE"

    # @staticmethod
    # def VALIDATE_INPUTS(path="", current_frame=0):

    #     print(f"Validating inputs: {path}, {current_frame}")
    #     resolved_path = resolve_path(path, current_frame)
    #     if not folder_paths.exists_annotated_filepath(resolved_path):
    #         return f"Invalid image file: {resolved_path}"
    #     return True


import glob


def img_from_path(path):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    image = img.convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image)[None,]
    if "A" in img.getbands():
        mask = np.array(img.getchannel("A")).astype(np.float32) / 255.0
        mask = 1.0 - torch.from_numpy(mask)
    else:
        mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
    return (
        image,
        mask,
    )


def resolve_all_frames(pattern):
    folder_path, file_pattern = os.path.split(pattern)

    log.debug(f"Resolving all frames in {folder_path}")
    frames = []
    hash_count = file_pattern.count("#")
    frame_pattern = re.sub(r"#+", "*", file_pattern)

    log.debug(f"Found pattern: {frame_pattern}")

    matching_files = glob.glob(os.path.join(folder_path, frame_pattern))

    log.debug(f"Found {len(matching_files)} matching files")

    frame_regex = re.escape(file_pattern).replace(r"\#", r"(\d+)")

    frame_number_regex = re.compile(frame_regex)

    for file in matching_files:
        match = frame_number_regex.search(file)
        if match:
            frame_number = match.group(1)
            log.debug(f"Found frame number: {frame_number}")
            # resolved_file = pattern.replace("*" * frame_number.count("#"), frame_number)
            frames.append(file)

    frames.sort()  # Sort frames alphabetically
    return frames


def resolve_path(path, frame):
    hashes = path.count("#")
    padded_number = str(frame).zfill(hashes)
    return re.sub("#+", padded_number, path)


class SaveImageSequence:
    """Save an image sequence to a folder. The current frame is used to determine which image to save.

    This is merely a wrapper around the `save_images` function with formatting for the output folder and filename.
    """

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "Sequence"}),
                "current_frame": ("INT", {"default": 0, "min": 0, "max": 9999999}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "mtb/IO"

    def save_images(
        self,
        images,
        filename_prefix="Sequence",
        current_frame=0,
        prompt=None,
        extra_pnginfo=None,
    ):
        # full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        # results = list()
        # for image in images:
        #     i = 255. * image.cpu().numpy()
        #     img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        #     metadata = PngInfo()
        #     if prompt is not None:
        #         metadata.add_text("prompt", json.dumps(prompt))
        #     if extra_pnginfo is not None:
        #         for x in extra_pnginfo:
        #             metadata.add_text(x, json.dumps(extra_pnginfo[x]))

        #     file = f"{filename}_{counter:05}_.png"
        #     img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=4)
        #     results.append({
        #         "filename": file,
        #         "subfolder": subfolder,
        #         "type": self.type
        #     })
        #     counter += 1

        if len(images) > 1:
            raise ValueError("Can only save one image at a time")

        resolved_path = Path(self.output_dir) / filename_prefix
        resolved_path.mkdir(parents=True, exist_ok=True)

        resolved_img = resolved_path / f"{filename_prefix}_{current_frame:05}.png"

        output_image = images[0].cpu().numpy()
        img = Image.fromarray(np.clip(output_image * 255.0, 0, 255).astype(np.uint8))
        metadata = PngInfo()
        if prompt is not None:
            metadata.add_text("prompt", json.dumps(prompt))
        if extra_pnginfo is not None:
            for x in extra_pnginfo:
                metadata.add_text(x, json.dumps(extra_pnginfo[x]))

        img.save(resolved_img, pnginfo=metadata, compress_level=4)
        return {
            "ui": {
                "images": [
                    {
                        "filename": resolved_img.name,
                        "subfolder": resolved_path.name,
                        "type": self.type,
                    }
                ]
            }
        }


__nodes__ = [
    LoadImageSequence,
    SaveImageSequence,
]
