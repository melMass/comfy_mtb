from PIL import Image
import numpy as np
import torch
from pathlib import Path
import sys
from typing import List
import signal
from contextlib import suppress
from queue import Queue, Empty
import subprocess
import threading
import os
import math
import functools
import socket
import requests

try:
    from .log import log
except ImportError:
    try:
        from log import log

        log.warn("Imported log without relative path")
    except ImportError:
        import logging

        log = logging.getLogger("comfy mtb utils")
        log.warn("[comfy mtb] You probably called the file outside a module.")


class IPChecker:
    def __init__(self):
        self.ips = list(self.get_local_ips())
        log.debug(f"Found {len(self.ips)} local ips")
        self.checked_ips = set()

    def get_working_ip(self, test_url_template):
        for ip in self.ips:
            if ip not in self.checked_ips:
                self.checked_ips.add(ip)
                test_url = test_url_template.format(ip)
                if self._test_url(test_url):
                    return ip
        return None

    @staticmethod
    def get_local_ips(prefix="192.168."):
        hostname = socket.gethostname()
        log.debug(f"Getting local ips for {hostname}")
        for info in socket.getaddrinfo(hostname, None):
            # Filter out IPv6 addresses if you only want IPv4
            log.debug(info)
            # if info[1] == socket.SOCK_STREAM and
            if info[0] == socket.AF_INET and info[4][0].startswith(prefix):
                yield info[4][0]

    def _test_url(self, url):
        try:
            response = requests.get(url)
            return response.status_code == 200
        except Exception:
            return False


# region MISC Utilities
@functools.lru_cache(maxsize=1)
def get_server_info():
    from comfy.cli_args import args

    ip_checker = IPChecker()
    base_url = args.listen
    if base_url == "0.0.0.0":
        log.debug("Server set to 0.0.0.0, we will try to resolve the host IP")
        base_url = ip_checker.get_working_ip(f"http://{{}}:{args.port}/history")
        log.debug(f"Setting ip to {base_url}")
    return (base_url, args.port)


def hex_to_rgb(hex_color):
    try:
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        log.error(f"Invalid hex color: {hex_color}")
        return (0, 0, 0)


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
# - detect mode
comfy_mode = None
if os.environ.get("COLAB_GPU"):
    comfy_mode = "colab"
elif "python_embeded" in sys.executable:
    comfy_mode = "embeded"
elif ".venv" in sys.executable:
    comfy_mode = "venv"

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


# region UV Utilities


def create_uv_map_tensor(width=512, height=512):
    u = torch.linspace(0.0, 1.0, steps=width)
    v = torch.linspace(0.0, 1.0, steps=height)

    U, V = torch.meshgrid(u, v)

    uv_map = torch.zeros(height, width, 3, dtype=torch.float32)
    uv_map[:, :, 0] = U.t()
    uv_map[:, :, 1] = V.t()

    return uv_map.unsqueeze(0)


# endregion


# region ANIMATION Utilities
def apply_easing(value, easing_type):
    if value < 0 or value > 1:
        raise ValueError("The value should be between 0 and 1.")

    if easing_type == "Linear":
        return value

    # Back easing functions
    def easeInBack(t):
        s = 1.70158
        return t * t * ((s + 1) * t - s)

    def easeOutBack(t):
        s = 1.70158
        return ((t - 1) * t * ((s + 1) * t + s)) + 1

    def easeInOutBack(t):
        s = 1.70158 * 1.525
        if t < 0.5:
            return (t * t * (t * (s + 1) - s)) * 2
        return ((t - 2) * t * ((s + 1) * t + s) + 2) * 2

    # Elastic easing functions
    def easeInElastic(t):
        if t == 0:
            return 0
        if t == 1:
            return 1
        p = 0.3
        s = p / 4
        return -(math.pow(2, 10 * (t - 1)) * math.sin((t - 1 - s) * (2 * math.pi) / p))

    def easeOutElastic(t):
        if t == 0:
            return 0
        if t == 1:
            return 1
        p = 0.3
        s = p / 4
        return math.pow(2, -10 * t) * math.sin((t - s) * (2 * math.pi) / p) + 1

    def easeInOutElastic(t):
        if t == 0:
            return 0
        if t == 1:
            return 1
        p = 0.3 * 1.5
        s = p / 4
        t = t * 2
        if t < 1:
            return -0.5 * (
                math.pow(2, 10 * (t - 1)) * math.sin((t - 1 - s) * (2 * math.pi) / p)
            )
        return (
            0.5 * math.pow(2, -10 * (t - 1)) * math.sin((t - 1 - s) * (2 * math.pi) / p)
            + 1
        )

    # Bounce easing functions
    def easeInBounce(t):
        return 1 - easeOutBounce(1 - t)

    def easeOutBounce(t):
        if t < (1 / 2.75):
            return 7.5625 * t * t
        elif t < (2 / 2.75):
            t -= 1.5 / 2.75
            return 7.5625 * t * t + 0.75
        elif t < (2.5 / 2.75):
            t -= 2.25 / 2.75
            return 7.5625 * t * t + 0.9375
        else:
            t -= 2.625 / 2.75
            return 7.5625 * t * t + 0.984375

    def easeInOutBounce(t):
        if t < 0.5:
            return easeInBounce(t * 2) * 0.5
        return easeOutBounce(t * 2 - 1) * 0.5 + 0.5

    # Quart easing functions
    def easeInQuart(t):
        return t * t * t * t

    def easeOutQuart(t):
        t -= 1
        return -(t**2 * t * t - 1)

    def easeInOutQuart(t):
        t *= 2
        if t < 1:
            return 0.5 * t * t * t * t
        t -= 2
        return -0.5 * (t**2 * t * t - 2)

    # Cubic easing functions
    def easeInCubic(t):
        return t * t * t

    def easeOutCubic(t):
        t -= 1
        return t**2 * t + 1

    def easeInOutCubic(t):
        t *= 2
        if t < 1:
            return 0.5 * t * t * t
        t -= 2
        return 0.5 * (t**2 * t + 2)

    # Circ easing functions
    def easeInCirc(t):
        return -(math.sqrt(1 - t * t) - 1)

    def easeOutCirc(t):
        t -= 1
        return math.sqrt(1 - t**2)

    def easeInOutCirc(t):
        t *= 2
        if t < 1:
            return -0.5 * (math.sqrt(1 - t**2) - 1)
        t -= 2
        return 0.5 * (math.sqrt(1 - t**2) + 1)

    # Sine easing functions
    def easeInSine(t):
        return -math.cos(t * (math.pi / 2)) + 1

    def easeOutSine(t):
        return math.sin(t * (math.pi / 2))

    def easeInOutSine(t):
        return -0.5 * (math.cos(math.pi * t) - 1)

    easing_functions = {
        "Sine In": easeInSine,
        "Sine Out": easeOutSine,
        "Sine In/Out": easeInOutSine,
        "Quart In": easeInQuart,
        "Quart Out": easeOutQuart,
        "Quart In/Out": easeInOutQuart,
        "Cubic In": easeInCubic,
        "Cubic Out": easeOutCubic,
        "Cubic In/Out": easeInOutCubic,
        "Circ In": easeInCirc,
        "Circ Out": easeOutCirc,
        "Circ In/Out": easeInOutCirc,
        "Back In": easeInBack,
        "Back Out": easeOutBack,
        "Back In/Out": easeInOutBack,
        "Elastic In": easeInElastic,
        "Elastic Out": easeOutElastic,
        "Elastic In/Out": easeInOutElastic,
        "Bounce In": easeInBounce,
        "Bounce Out": easeOutBounce,
        "Bounce In/Out": easeInOutBounce,
    }

    function_ease = easing_functions.get(easing_type)
    if function_ease:
        return function_ease(value)

    log.error(f"Unknown easing type: {easing_type}")
    log.error(f"Available easing types: {list(easing_functions.keys())}")
    raise ValueError(f"Unknown easing type: {easing_type}")


# endregion
