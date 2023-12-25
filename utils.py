import contextlib, functools, math, os, shlex, shutil, socket, subprocess, sys, uuid
from pathlib import Path
from typing import List, Optional, Union

import folder_paths
import numpy as np
import requests
import torch
from PIL import Image

from .install import pip_map

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


# region SANITY_CHECK Utilities


def make_report():
    pass


# endregion


# region SERVER Utilities
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


# endregion


# region MISC Utilities
def backup_file(
    fp: Path,
    target: Optional[Path] = None,
    backup_dir: str = ".bak",
    suffix: Optional[str] = None,
    prefix: Optional[str] = None,
):
    if not fp.exists():
        raise FileNotFoundError(f"No file found at {fp}")

    backup_directory = target or fp.parent / backup_dir
    backup_directory.mkdir(parents=True, exist_ok=True)

    stem = fp.stem

    if suffix or prefix:
        new_stem = f"{prefix or ''}{stem}{suffix or ''}"
    else:
        new_stem = f"{stem}_{uuid.uuid4()}"

    backup_file_path = backup_directory / f"{new_stem}{fp.suffix}"

    # Perform the backup
    shutil.copy(fp, backup_file_path)
    log.debug(f"File backed up to {backup_file_path}")


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


def run_command(cmd, ignored_lines_start=None):
    if ignored_lines_start is None:
        ignored_lines_start = []

    if isinstance(cmd, str):
        shell_cmd = cmd
    elif isinstance(cmd, list):
        shell_cmd = " ".join(
            arg.as_posix() if isinstance(arg, Path) else shlex.quote(str(arg))
            for arg in cmd
        )
    else:
        raise ValueError(
            "Invalid 'cmd' argument. It must be a string or a list of arguments."
        )

    try:
        _run_command(shell_cmd, ignored_lines_start)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code: {e.returncode}", file=sys.stderr)
        print(e.stderr.strip(), file=sys.stderr)

    except KeyboardInterrupt:
        print("Command execution interrupted.")


def _run_command(shell_cmd, ignored_lines_start):
    log.debug(f"Running {shell_cmd}")

    result = subprocess.run(
        shell_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=True,
        check=True,
    )

    stdout_lines = result.stdout.strip().split("\n")
    stderr_lines = result.stderr.strip().split("\n")

    # Print stdout, skipping ignored lines
    for line in stdout_lines:
        if not any(line.startswith(ign) for ign in ignored_lines_start):
            print(line)

    # Print stderr
    for line in stderr_lines:
        print(line, file=sys.stderr)

    print("Command executed successfully!")


# todo use the requirements library
reqs_map = {value: key for key, value in pip_map.items()}

import importlib


def import_install(package_name):
    package_spec = reqs_map.get(package_name, package_name)

    try:
        importlib.import_module(package_name)

    except Exception:  # (ImportError, ModuleNotFoundError):
        run_command(
            [Path(sys.executable).as_posix(), "-m", "pip", "install", package_spec]
        )
        importlib.import_module(package_name)


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
here = Path(__file__).parent.absolute()

# - Construct the absolute path to the ComfyUI directory
comfy_dir = Path(folder_paths.base_path)
models_dir = Path(folder_paths.models_dir)
output_dir = Path(folder_paths.output_directory)
styles_dir = comfy_dir / "styles"
session_id = str(uuid.uuid4())
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


# region TENSOR Utilities
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


def pil2tensor(image: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
    if isinstance(image, list):
        return torch.cat([pil2tensor(img) for img in image], dim=0)

    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def np2tensor(img_np: Union[np.ndarray, List[np.ndarray]]) -> torch.Tensor:
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


def pad(img, left, right, top, bottom):
    pad_width = np.array(((0, 0), (top, bottom), (left, right)))
    print(f"pad_width: {pad_width}, shape: {pad_width.shape}")  # Debugging line
    return np.pad(img, pad_width, mode="wrap")


def tiles_infer(tiles, ort_session, progress_callback=None):
    """Infer each tile with the given model. progress_callback will be called with
    arguments : current tile idx and total tiles amount (used to show progress on
    cursor in Blender)."""

    out_channels = 3  # normal map RGB channels
    tiles_nb = tiles.shape[0]
    pred_tiles = np.empty((tiles_nb, out_channels, tiles.shape[2], tiles.shape[3]))

    for i in range(tiles_nb):
        if progress_callback != None:
            progress_callback(i + 1, tiles_nb)
        pred_tiles[i] = ort_session.run(
            None, {"input": tiles[i : i + 1].astype(np.float32)}
        )[0]

    return pred_tiles


def generate_mask(tile_size, stride_size):
    """Generates a pyramidal-like mask. Used for mixing overlapping predicted tiles."""

    tile_h, tile_w = tile_size
    stride_h, stride_w = stride_size
    ramp_h = tile_h - stride_h
    ramp_w = tile_w - stride_w

    mask = np.ones((tile_h, tile_w))

    # ramps in width direction
    mask[ramp_h:-ramp_h, :ramp_w] = np.linspace(0, 1, num=ramp_w)
    mask[ramp_h:-ramp_h, -ramp_w:] = np.linspace(1, 0, num=ramp_w)
    # ramps in height direction
    mask[:ramp_h, ramp_w:-ramp_w] = np.transpose(
        np.linspace(0, 1, num=ramp_h)[None], (1, 0)
    )
    mask[-ramp_h:, ramp_w:-ramp_w] = np.transpose(
        np.linspace(1, 0, num=ramp_h)[None], (1, 0)
    )

    # Assume tiles are squared
    assert ramp_h == ramp_w
    # top left corner
    corner = np.rot90(corner_mask(ramp_h), 2)
    mask[:ramp_h, :ramp_w] = corner
    # top right corner
    corner = np.flip(corner, 1)
    mask[:ramp_h, -ramp_w:] = corner
    # bottom right corner
    corner = np.flip(corner, 0)
    mask[-ramp_h:, -ramp_w:] = corner
    # bottom right corner
    corner = np.flip(corner, 1)
    mask[-ramp_h:, :ramp_w] = corner

    return mask


def corner_mask(side_length):
    """Generates the corner part of the pyramidal-like mask.
    Currently, only for square shapes."""

    corner = np.zeros([side_length, side_length])

    for h in range(0, side_length):
        for w in range(0, side_length):
            if h >= w:
                sh = h / (side_length - 1)
                corner[h, w] = 1 - sh
            if h <= w:
                sw = w / (side_length - 1)
                corner[h, w] = 1 - sw

    return corner - 0.25 * scaling_mask(side_length)


def scaling_mask(side_length):
    scaling = np.zeros([side_length, side_length])

    for h in range(0, side_length):
        for w in range(0, side_length):
            sh = h / (side_length - 1)
            sw = w / (side_length - 1)
            if h >= w and h <= side_length - w:
                scaling[h, w] = sw
            if h <= w and h <= side_length - w:
                scaling[h, w] = sh
            if h >= w and h >= side_length - w:
                scaling[h, w] = 1 - sh
            if h <= w and h >= side_length - w:
                scaling[h, w] = 1 - sw

    return 2 * scaling


def tiles_merge(tiles, stride_size, img_size, paddings):
    """Merges the list of tiles into one image. img_size is the original size, before
    padding."""

    _, tile_h, tile_w = tiles[0].shape
    pad_left, pad_right, pad_top, pad_bottom = paddings
    height = img_size[1] + pad_top + pad_bottom
    width = img_size[2] + pad_left + pad_right
    stride_h, stride_w = stride_size

    # stride must be even
    assert (stride_h % 2 == 0) and (stride_w % 2 == 0)
    # stride must be greater or equal than half tile
    assert (stride_h >= tile_h / 2) and (stride_w >= tile_w / 2)
    # stride must be smaller or equal tile size
    assert (stride_h <= tile_h) and (stride_w <= tile_w)

    merged = np.zeros((img_size[0], height, width))
    mask = generate_mask((tile_h, tile_w), stride_size)

    h_range = ((height - tile_h) // stride_h) + 1
    w_range = ((width - tile_w) // stride_w) + 1

    idx = 0
    for h in range(0, h_range):
        for w in range(0, w_range):
            h_from, h_to = h * stride_h, h * stride_h + tile_h
            w_from, w_to = w * stride_w, w * stride_w + tile_w
            merged[:, h_from:h_to, w_from:w_to] += tiles[idx] * mask
            idx += 1

    return merged[:, pad_top:-pad_bottom, pad_left:-pad_right]


def tiles_split(img, tile_size, stride_size):
    """Returns list of tiles from the given image and the padding used to fit the tiles
    in it. Input image must have dimension C,H,W."""
    log.debug(f"Splitting img: tile {tile_size}, stride {stride_size} ")
    tile_h, tile_w = tile_size
    stride_h, stride_w = stride_size
    img_h, img_w = img.shape[0], img.shape[1]

    # stride must be even
    assert (stride_h % 2 == 0) and (stride_w % 2 == 0)
    # stride must be greater or equal than half tile
    assert (stride_h >= tile_h / 2) and (stride_w >= tile_w / 2)
    # stride must be smaller or equal tile size
    assert (stride_h <= tile_h) and (stride_w <= tile_w)

    # find total height & width padding sizes
    pad_h, pad_w = 0, 0
    remainer_h = (img_h - tile_h) % stride_h
    remainer_w = (img_w - tile_w) % stride_w
    if remainer_h != 0:
        pad_h = stride_h - remainer_h
    if remainer_w != 0:
        pad_w = stride_w - remainer_w

    # if tile bigger than image, pad image to tile size
    if tile_h > img_h:
        pad_h = tile_h - img_h
    if tile_w > img_w:
        pad_w = tile_w - img_w

    # pad image, add extra stride to padding to avoid pyramid
    # weighting leaking onto the valid part of the picture
    pad_left = pad_w // 2 + stride_w
    pad_right = pad_left if pad_w % 2 == 0 else pad_left + 1
    pad_top = pad_h // 2 + stride_h
    pad_bottom = pad_top if pad_h % 2 == 0 else pad_top + 1
    img = pad(img, pad_left, pad_right, pad_top, pad_bottom)
    img_h, img_w = img.shape[1], img.shape[2]

    # extract tiles
    h_range = ((img_h - tile_h) // stride_h) + 1
    w_range = ((img_w - tile_w) // stride_w) + 1
    tiles = np.empty([h_range * w_range, img.shape[0], tile_h, tile_w])
    idx = 0
    for h in range(0, h_range):
        for w in range(0, w_range):
            h_from, h_to = h * stride_h, h * stride_h + tile_h
            w_from, w_to = w * stride_w, w * stride_w + tile_w
            tiles[idx] = img[:, h_from:h_to, w_from:w_to]
            idx += 1

    return tiles, (pad_left, pad_right, pad_top, pad_bottom)


# endregion


# region MODEL Utilities
def download_antelopev2():
    antelopev2_url = "https://drive.google.com/uc?id=18wEUfMNohBJ4K3Ly5wpTejPfDzp-8fI8"

    try:
        import gdown

        log.debug("Loading antelopev2 model")

        dest = get_model_path("insightface")
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


def get_model_path(fam, model=None):
    log.debug(f"Requesting {fam} with model {model}")
    res = None
    if model:
        res = folder_paths.get_full_path(fam, model)
    else:
        # this one can raise errors...
        with contextlib.suppress(KeyError):
            res = folder_paths.get_folder_paths(fam)

    if res:
        if isinstance(res, list):
            if len(res) > 1:
                log.warning(
                    f"Found multiple match, we will pick the first {res[0]}\n{res}"
                )
            res = res[0]
        res = Path(res)
        log.debug(f"Resolved model path from folder_paths: {res}")
    else:
        res = models_dir / fam
        if model:
            res /= model

    return res


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
