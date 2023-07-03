import logging
from gfpgan import GFPGANer
import cv2
import numpy as np
import os
from pathlib import Path
import folder_paths
from basicsr.utils import imwrite
from PIL import Image
from ..utils import pil2tensor, tensor2pil, np2tensor, tensor2np
import torch
from munch import Munch
from ..log import NullWriter, log
from comfy import model_management
import comfy


class LoadFaceEnhanceModel:
    def __init__(self) -> None:
        pass

    @classmethod
    def get_models_root(cls):
        return Path(folder_paths.models_dir) / "upscale_models"

    @classmethod
    def get_models(cls):
        models_path = cls.get_models_root()

        return [
            x
            for x in models_path.iterdir()
            if x.name.endswith(".pth")
            and ("GFPGAN" in x.name or "RestoreFormer" in x.name)
        ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (
                    [x.name for x in cls.get_models()],
                    {"default": "None"},
                ),
                "upscale": ("INT", {"default": 2}),
            },
            "optional": {"bg_upsampler": ("UPSCALE_MODEL", {"default": None})},
        }

    RETURN_TYPES = ("FACEENHANCE_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "face"

    def load_model(self, model_name, upscale=2, bg_upsampler=None):
        basic = "RestoreFormer" not in model_name

        root = self.get_models_root()

        if bg_upsampler is not None:
            log.warning(
                f"Upscale value overridden to {bg_upsampler.scale} from bg_upsampler"
            )
            upscale = bg_upsampler.scale
            bg_upsampler = BGUpscaleWrapper(bg_upsampler)

        sys.stdout = NullWriter()
        model = GFPGANer(
            model_path=(root / model_name).as_posix(),
            upscale=upscale,
            arch="clean" if basic else "RestoreFormer",  # or original for v1.0 only
            channel_multiplier=2,  # 1 for v1.0 only
            bg_upsampler=bg_upsampler,
        )

        sys.stdout = sys.__stdout__
        return (model,)


class BGUpscaleWrapper:
    def __init__(self, upscale_model) -> None:
        self.upscale_model = upscale_model

    def enhance(self, img: Image, outscale=2):
        device = model_management.get_torch_device()
        self.upscale_model.to(device)

        tile = 128 + 64
        overlap = 8

        imgt = np2tensor(img)
        imgt = imgt.movedim(-1, -3).to(device)

        steps = imgt.shape[0] * comfy.utils.get_tiled_scale_steps(
            imgt.shape[3], imgt.shape[2], tile_x=tile, tile_y=tile, overlap=overlap
        )

        log.debug(f"Steps: {steps}")

        pbar = comfy.utils.ProgressBar(steps)

        s = comfy.utils.tiled_scale(
            imgt,
            lambda a: self.upscale_model(a),
            tile_x=tile,
            tile_y=tile,
            overlap=overlap,
            upscale_amount=self.upscale_model.scale,
            pbar=pbar,
        )

        self.upscale_model.cpu()
        s = torch.clamp(s.movedim(-3, -1), min=0, max=1.0)
        return (tensor2np(s),)


import sys


class RestoreFace:
    def __init__(self) -> None:
        pass

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "restore"
    CATEGORY = "face"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": ("FACEENHANCE_MODEL",),
                # Input are aligned faces
                "aligned": (["true", "false"], {"default": "false"}),
                # Only restore the center face
                "only_center_face": (["true", "false"], {"default": "false"}),
                # Adjustable weights
                "weight": ("FLOAT", {"default": 0.5}),
                "save_tmp_steps": (["true", "false"], {"default": "true"}),
            }
        }

    def restore(
        self,
        image: torch.Tensor,
        model: GFPGANer,
        aligned="false",
        only_center_face="false",
        weight=0.5,
        save_tmp_steps="true",
    ):
        save_tmp_steps = save_tmp_steps == "true"
        aligned = aligned == "true"
        only_center_face = only_center_face == "true"

        image = tensor2pil(image)
        width, height = image.size

        source_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        sys.stdout = NullWriter()
        cropped_faces, restored_faces, restored_img = model.enhance(
            source_img,
            has_aligned=aligned,
            only_center_face=only_center_face,
            paste_back=True,
            # TODO: weight has no effect in 1.3 and 1.4 (only tested these for now...)
            weight=weight,
        )
        sys.stdout = sys.__stdout__
        log.warning(f"Weight value has no effect for now. (value: {weight})")

        if save_tmp_steps:
            self.save_intermediate_images(cropped_faces, restored_faces, height, width)
        output = None
        if restored_img is not None:
            output = Image.fromarray(cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB))
            # imwrite(restored_img, save_restore_path)

        return (pil2tensor(output),)

    def get_step_image_path(self, step, idx):
        (
            full_output_folder,
            filename,
            counter,
            _subfolder,
            _filename_prefix,
        ) = folder_paths.get_save_image_path(
            f"{step}_{idx:03}",
            folder_paths.temp_directory,
        )
        file = f"{filename}_{counter:05}_.png"

        return os.path.join(full_output_folder, file)

    def save_intermediate_images(self, cropped_faces, restored_faces, height, width):
        for idx, (cropped_face, restored_face) in enumerate(
            zip(cropped_faces, restored_faces)
        ):
            face_id = idx + 1
            file = self.get_step_image_path("cropped_faces", face_id)
            imwrite(cropped_face, file)

            file = self.get_step_image_path("cropped_faces_restored", face_id)
            imwrite(restored_face, file)

            file = self.get_step_image_path("cropped_faces_compare", face_id)

            # save comparison image
            cmp_img = np.concatenate((cropped_face, restored_face), axis=1)
            imwrite(cmp_img, file)


__nodes__ = [RestoreFace, LoadFaceEnhanceModel]
