import os
import subprocess
import tempfile

import comfy.utils
import torch

from ..log import log
from ..utils import nextAvailable, tensor2pil

RELATIVE_NOTICE = """
Absolute paths are kept as is, relatives are from the output directory.
"""


class MTB_PostshotTrain:
    CATEGORY = "mtb/postshot"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": (
                    "IMAGE",
                    {"tooltip": "These image will get save to disk first"},
                ),
                "profile": (
                    [
                        "NeRF L",
                        "NeRF M",
                        "NeRF S",
                        "NeRF XL",
                        "NeRF XXL",
                        "Splat ADC",
                        "Splat MCMC",
                    ],
                    {
                        "default": "Splat MCMC",
                        "tooltip": "The radiance field model profile to train",
                    },
                ),
                "image_select": (
                    ["all", "best"],
                    {
                        "default": "best",
                        "tooltip": "How to select training images from the source image sets",
                    },
                ),
                "train_steps_limit": (
                    "INT",
                    {
                        "default": 30,
                        "min": 1,
                        "max": 1000,
                        "tooltip": "Number of kSteps to train the model for",
                    },
                ),
                "output_path": (
                    "STRING",
                    {
                        "default": "output",
                        "tooltip": (
                            "path to save the project to" f"{RELATIVE_NOTICE}"
                        ),
                    },
                ),
                "postshot_cli": (
                    "STRING",
                    {
                        "default": "C:/Program Files/Jawset Postshot/bin/postshot-cli.exe"
                    },
                ),
            },
            "optional": {
                "gpu": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 255,
                        "tooltip": "Specify the index of the GPU to use",
                    },
                ),
                "num_train_images": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "tooltip": "If image-select best is used, specifies the number of training images to select",
                    },
                ),
                "max_image_size": (
                    "INT",
                    {
                        "default": 1600,
                        "min": 0,
                        "tooltip": "Downscale training images such that their longer edge is at most this value in pixels. Disabled if zero.",
                    },
                ),
                "max_num_features": (
                    "INT",
                    {
                        "default": 8,
                        "min": 1,
                        "tooltip": "Maximum number of 2D kFeatures extracted from each image.",
                    },
                ),
                "splat_density": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.125,
                        "max": 8.0,
                        "tooltip": (
                            "Controls how much additional splats "
                            "are generated during training."
                            "Applies only in 'Splat ADC' profile."
                        ),
                    },
                ),
                "max_num_splats": (
                    "INT",
                    {
                        "default": 3000,
                        "min": 1,
                        "tooltip": (
                            "Sets the maximum number of splats (in kSplats)"
                            " created during training. "
                            "Applies only in 'Splat MCMC' profile."
                        ),
                    },
                ),
                "export_splat_ply": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "If not empty will also save a ply file."
                            f"{RELATIVE_NOTICE}"
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    OUTPUT_NODE = True
    RETURN_NAMES = ("project_file_path",)
    FUNCTION = "train_model"

    def train_model(
        self,
        images: torch.Tensor,
        profile: str,
        image_select: str,
        train_steps_limit: int,
        output_path: str,
        gpu=0,
        num_train_images=0,
        max_image_size=1600,
        max_num_features=8,
        splat_density=1.0,
        max_num_splats=3000,
        export_splat_ply="",
        postshot_cli="",
    ):
        if not output_path.endswith(".psht"):
            output_path += ".psht"

        output_path = nextAvailable(output_path)
        output_path.parent.mkdir(exist_ok=True)

        pbar = comfy.utils.ProgressBar(200 + images.size(0))

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                image_paths = []
                images_pil = tensor2pil(images)
                for i, img in enumerate(images_pil):
                    try:
                        img_path = os.path.join(temp_dir, f"image_{i:04d}.png")
                        img.save(img_path)
                        image_paths.append(img_path)
                    except Exception as e:
                        raise RuntimeError(
                            f"Failed to save image {i}: {str(e)}"
                        ) from e
                    pbar.update(1)

                if not image_paths:
                    raise ValueError("No valid images to process")

                cmd = [postshot_cli, "train"]

                for img_path in image_paths:
                    cmd.extend(["-i", img_path])

                cmd.extend(
                    [
                        "-p",
                        profile,
                        "--image-select",
                        image_select,
                        "-s",
                        str(train_steps_limit),
                        "-o",
                        output_path.as_posix(),
                    ]
                )

                if gpu is not None:
                    cmd.extend(["--gpu", str(gpu)])
                if num_train_images > 0 and image_select == "best":
                    cmd.extend(["--num-train-images", str(num_train_images)])
                if max_image_size > 0:
                    cmd.extend(["--max-image-size", str(max_image_size)])
                if max_num_features != 8:
                    cmd.extend(["--max-num-features", str(max_num_features)])
                if profile == "Splat ADC" and splat_density != 1.0:
                    cmd.extend(["--splat-density", str(splat_density)])
                if profile == "Splat MCMC" and max_num_splats != 3000:
                    cmd.extend(["--max-num-splats", str(max_num_splats)])
                if export_splat_ply:
                    export_splat_ply = nextAvailable(export_splat_ply)
                    cmd.extend(
                        ["--export-splat-ply", export_splat_ply.as_posix()]
                    )

                log.debug(f"Running {cmd}")

                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                )

                last_step_c = 0
                last_step_t = 0
                while True:
                    output = process.stdout.readline()
                    if output == "" and process.poll() is not None:
                        break
                    if output:
                        print(output)
                        if "camera tracking step" in output.lower():
                            try:
                                current_step = int(
                                    output.split("%")[0].split(":")[1].strip()
                                )
                                if current_step > last_step_c:
                                    pbar.update(1)
                                    last_step_c = current_step

                            except (ValueError, IndexError):
                                continue

                        if "training radiance field:" in output.lower():
                            try:
                                current_step = int(
                                    output.split("%")[0].split(":")[1].strip()
                                )
                                if current_step > last_step_t:
                                    pbar.update(1)
                                    last_step_t = current_step

                            except (ValueError, IndexError):
                                continue

                if process.returncode != 0:
                    _, stderr = process.communicate()
                    raise RuntimeError(f"Postshot training failed: {stderr}")

                if not os.path.exists(output_path):
                    raise RuntimeError("Output file was not created")

                return (output_path.as_posix(),)

        except Exception as e:
            raise RuntimeError(f"Training failed: {str(e)}")
        finally:
            pbar.update(train_steps_limit)


class MTB_PostshotExport:
    CATEGORY = "mtb/postshot"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "project_file": (
                    "STRING",
                    {"default": "", "forceInput": True},
                ),
                "export_splat_ply": ("STRING", {"default": "output.ply"}),
                "postshot_cli": (
                    "STRING",
                    {
                        "default": "C:/Program Files/Jawset Postshot/bin/postshot-cli.exe"
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("exported_ply_path",)
    FUNCTION = "export_model"

    def export_model(
        self, project_file: str, export_splat_ply: str, postshot_cli: str
    ):
        if not project_file.endswith(".psht"):
            raise ValueError("Project file must have .psht extension")

        if not os.path.exists(project_file):
            raise FileNotFoundError(f"Project file not found: {project_file}")

        if not export_splat_ply.endswith(".ply"):
            export_splat_ply += ".ply"

        _export_splat_ply = nextAvailable(export_splat_ply)
        _export_splat_ply.parent.mkdir(exist_ok=True)

        cmd = [
            postshot_cli,
            "export",
            "-f",
            project_file,
            "--export-splat-ply",
            _export_splat_ply.as_posix(),
        ]

        try:
            _result = subprocess.run(
                cmd, check=True, capture_output=True, text=True
            )

            if not _export_splat_ply.exists():
                log.error("Export file was not created")

            return (_export_splat_ply.as_posix(),)

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Export failed: {e.stderr}")
        except Exception as e:
            raise RuntimeError(f"Export failed: {str(e)}")


__nodes__ = [MTB_PostshotExport, MTB_PostshotTrain]
