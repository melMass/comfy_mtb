from PIL import Image
import numpy as np
import torch
from pathlib import Path
import sys
from typing import List
from .log import log
import signal
from contextlib import suppress
from queue import Queue, Empty
import subprocess
import threading
import os

# - detect mode
comfy_mode = None
if os.environ.get("COLAB_GPU"):
    comfy_mode = "colab"
elif "python_embeded" in sys.executable:
    comfy_mode = "embeded"
elif ".venv" in sys.executable:
    comfy_mode = "venv"


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


# region MISC Utilities
def add_path(path, prepend=False):
    if isinstance(path, list):
        for p in path:
            add_path(p, prepend)
        return

    if isinstance(path, Path):
        path = path.resolve().as_posix()

    if path not in sys.path:
        if prepend:
            sys.path.insert(0, path)
        else:
            sys.path.append(path)


def enqueue_output(out, queue):
    for line in iter(out.readline, b""):
        queue.put(line)
    out.close()


def run_command(cmd):
    if isinstance(cmd, str):
        shell_cmd = cmd
    elif isinstance(cmd, list):
        shell_cmd = ""
        for arg in cmd:
            if isinstance(arg, Path):
                arg = arg.as_posix()
            shell_cmd += f"{arg} "
    else:
        raise ValueError(
            "Invalid 'cmd' argument. It must be a string or a list of arguments."
        )

    process = subprocess.Popen(
        shell_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        shell=True,
    )

    # Create separate threads to read standard output and standard error streams
    stdout_queue = Queue()
    stderr_queue = Queue()
    stdout_thread = threading.Thread(
        target=enqueue_output, args=(process.stdout, stdout_queue)
    )
    stderr_thread = threading.Thread(
        target=enqueue_output, args=(process.stderr, stderr_queue)
    )
    stdout_thread.daemon = True
    stderr_thread.daemon = True
    stdout_thread.start()
    stderr_thread.start()

    interrupted = False

    def signal_handler(signum, frame):
        nonlocal interrupted
        interrupted = True
        print("Command execution interrupted.")

    # Register the signal handler for keyboard interrupts (SIGINT)
    signal.signal(signal.SIGINT, signal_handler)

    # Process output from both streams until the process completes or interrupted
    while not interrupted and (
        process.poll() is None or not stdout_queue.empty() or not stderr_queue.empty()
    ):
        with suppress(Empty):
            stdout_line = stdout_queue.get_nowait()
            if stdout_line.strip() != "":
                print(stdout_line.strip())
        with suppress(Empty):
            stderr_line = stderr_queue.get_nowait()
            if stderr_line.strip() != "":
                print(stderr_line.strip())
    return_code = process.returncode

    if return_code == 0 and not interrupted:
        print("Command executed successfully!")
    else:
        if not interrupted:
            print(f"Command failed with return code: {return_code}")


# todo use the requirements library
reqs_map = {
    "onnxruntime": "onnxruntime-gpu==1.15.1",
    "basicsr": "basicsr==1.4.2",
    "rembg": "rembg==2.0.50",
    "qrcode": "qrcode[pil]",
}


def import_install(package_name):
    from pip._internal import main as pip_main

    try:
        __import__(package_name)
    except ImportError:
        package_spec = reqs_map.get(package_name)
        if package_spec is None:
            print(f"Installing {package_name}")
            package_spec = package_name

        pip_main(["install", package_spec])
        __import__(package_name)


# endregion

# region GLOBAL VARIABLES
# - Get the absolute path of the parent directory of the current script
here = Path(__file__).parent.resolve()

# - Construct the absolute path to the ComfyUI directory
comfy_dir = here.parent.parent

# - Construct the path to the font file
font_path = here / "font.ttf"

# - Add extern folder to path
extern_root = here / "extern"
add_path(extern_root)
for pth in extern_root.iterdir():
    if pth.is_dir():
        add_path(pth)

# - Add the ComfyUI directory and custom nodes path to the sys.path list
add_path(comfy_dir)
add_path((comfy_dir / "custom_nodes"))

PIL_FILTER_MAP = {
    "nearest": Image.Resampling.NEAREST,
    "box": Image.Resampling.BOX,
    "bilinear": Image.Resampling.BILINEAR,
    "hamming": Image.Resampling.HAMMING,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}


# endregion


# region TENSOR UTILITIES
def tensor2pil(image: torch.Tensor) -> List[Image.Image]:
    batch_count = image.size(0) if len(image.shape) > 3 else 1
    if batch_count > 1:
        out = []
        for i in range(batch_count):
            out.extend(tensor2pil(image[i]))
        return out

    return [
        Image.fromarray(
            np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
        )
    ]


def pil2tensor(image: Image.Image | List[Image.Image]) -> torch.Tensor:
    if isinstance(image, list):
        return torch.cat([pil2tensor(img) for img in image], dim=0)

    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def np2tensor(img_np: np.ndarray | List[np.ndarray]) -> torch.Tensor:
    if isinstance(img_np, list):
        return torch.cat([np2tensor(img) for img in img_np], dim=0)

    return torch.from_numpy(img_np.astype(np.float32) / 255.0).unsqueeze(0)


def tensor2np(tensor: torch.Tensor) -> List[np.ndarray]:
    batch_count = tensor.size(0) if len(tensor.shape) > 3 else 1
    if batch_count > 1:
        out = []
        for i in range(batch_count):
            out.extend(tensor2np(tensor[i]))
        return out

    return [np.clip(255.0 * tensor.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)]


# endregion


# region MODEL Utilities
def download_antelopev2():
    antelopev2_url = "https://drive.google.com/uc?id=18wEUfMNohBJ4K3Ly5wpTejPfDzp-8fI8"

    try:
        import gdown

        import folder_paths

        log.debug("Loading antelopev2 model")

        dest = Path(folder_paths.models_dir) / "insightface"
        archive = dest / "antelopev2.zip"
        final_path = dest / "models" / "antelopev2"
        if not final_path.exists():
            log.info(f"antelopev2 not found, downloading to {dest}")
            gdown.download(
                antelopev2_url,
                archive.as_posix(),
                resume=True,
            )

            log.info(f"Unzipping antelopev2 to {final_path}")

            if archive.exists():
                # we unzip it
                import zipfile

                with zipfile.ZipFile(archive.as_posix(), "r") as zip_ref:
                    zip_ref.extractall(final_path.parent.as_posix())

    except Exception as e:
        log.error(
            f"Could not load or download antelopev2 model, download it manually from {antelopev2_url}"
        )
        raise e


# endregion
