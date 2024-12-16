"""Support utilities for the nodes package."""

import base64
import contextlib
import copy
import functools
import importlib
import io
import itertools
import json
import math
import operator
import os
import shlex
import shutil
import socket
import subprocess
import sys
import uuid
from collections.abc import Callable, Sequence
from enum import Enum
from functools import reduce
from pathlib import Path
from typing import TypeVar

import folder_paths
import numpy as np
import numpy.typing as npt
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

try:
    import open3d as o3d
except:
    log.warn("You do not have open3D installed, 3d utils won't work")
    o3d = {}


# region SANITY_CHECK Utilities


def make_report():
    """Generate user report about the mtb installation."""
    pass


# endregion


# region NFOV
class numpy_NFOV:
    def __init__(self, fov=None, height: int = 400, width: int = 800):
        self.field_of_view = fov or [0.45, 0.45]
        self.PI = np.pi
        self.PI_2 = np.pi * 0.5
        self.PI2 = np.pi * 2.0
        self.height = height
        self.width = width
        self.screen_points = self._get_screen_img()

    def _get_coord_rad(self, is_center_point, center_point=None):
        if is_center_point:
            center_point = np.array(center_point)
            return (center_point * 2 - 1) * np.array([self.PI, self.PI_2])
        else:
            return (
                (self.screen_points * 2 - 1)
                * np.array([self.PI, self.PI_2])
                * (np.ones(self.screen_points.shape) * self.field_of_view)
            )

    def _get_screen_img(self):
        xx, yy = np.meshgrid(
            np.linspace(0, 1, self.width), np.linspace(0, 1, self.height)
        )
        return np.array([xx.ravel(), yy.ravel()]).T

    def _calc_spherical_to_gnomonic(self, converted_screen_coord):
        x = converted_screen_coord.T[0]
        y = converted_screen_coord.T[1]

        rou = np.sqrt(x**2 + y**2)
        c = np.arctan(rou)
        sin_c = np.sin(c)
        cos_c = np.cos(c)

        lat = np.arcsin(
            cos_c * np.sin(self.cp[1]) + (y * sin_c * np.cos(self.cp[1])) / rou
        )
        lon = self.cp[0] + np.arctan2(
            x * sin_c,
            rou * np.cos(self.cp[1]) * cos_c - y * np.sin(self.cp[1]) * sin_c,
        )

        lat = (lat / self.PI_2 + 1.0) * 0.5
        lon = (lon / self.PI + 1.0) * 0.5

        return np.array([lon, lat]).T

    def _bilinear_interpolation(self, screen_coord):
        uf = np.mod(screen_coord.T[0], 1) * self.frame_width  # long - width
        vf = np.mod(screen_coord.T[1], 1) * self.frame_height  # lat - height

        x0 = np.floor(uf).astype(int)  # coord of pixel to bottom left
        y0 = np.floor(vf).astype(int)
        x2 = np.add(
            x0, np.ones(uf.shape).astype(int)
        )  # coords of pixel to top right
        y2 = np.add(y0, np.ones(vf.shape).astype(int))

        base_y0 = np.multiply(y0, self.frame_width)
        base_y2 = np.multiply(y2, self.frame_width)

        A_idx = np.add(base_y0, x0)
        B_idx = np.add(base_y2, x0)
        C_idx = np.add(base_y0, x2)
        D_idx = np.add(base_y2, x2)

        flat_img = np.reshape(self.frame, [-1, self.frame_channel])

        A = np.take(flat_img, A_idx, axis=0)
        B = np.take(flat_img, B_idx, axis=0)
        C = np.take(flat_img, C_idx, axis=0)
        D = np.take(flat_img, D_idx, axis=0)

        wa = np.multiply(x2 - uf, y2 - vf)
        wb = np.multiply(x2 - uf, vf - y0)
        wc = np.multiply(uf - x0, y2 - vf)
        wd = np.multiply(uf - x0, vf - y0)

        # interpolate
        AA = np.multiply(A, np.array([wa, wa, wa]).T)
        BB = np.multiply(B, np.array([wb, wb, wb]).T)
        CC = np.multiply(C, np.array([wc, wc, wc]).T)
        DD = np.multiply(D, np.array([wd, wd, wd]).T)
        nfov = np.reshape(
            np.round(AA + BB + CC + DD).astype(np.uint8),
            [self.height, self.width, 3],
        )

        return nfov

    def to_nfov(self, frame, center_point):
        self.frame = frame
        self.frame_height = frame.shape[0]
        self.frame_width = frame.shape[1]
        self.frame_channel = frame.shape[2]

        self.cp = self._get_coord_rad(
            center_point=center_point, is_center_point=True
        )
        converted_screen_coord = self._get_coord_rad(is_center_point=False)
        return self._bilinear_interpolation(
            self._calc_spherical_to_gnomonic(converted_screen_coord)
        )


# endregion


# region SERVER Utilities
class IPChecker:
    """Checks for a working ip address used by the server."""

    def __init__(self):
        self.ips = list(self.get_local_ips())
        log.debug(f"Found {len(self.ips)} local ips")
        self.checked_ips: set[str] = set()

    def get_working_ip(self, test_url_template: str):
        for ip in self.ips:
            if ip not in self.checked_ips:
                self.checked_ips.add(ip)
                test_url = test_url_template.format(ip)
                if self._test_url(test_url):
                    return ip
        return None

    @staticmethod
    def get_local_ips(prefix: str = "192.168."):
        hostname = socket.gethostname()
        log.debug(f"Getting local ips for {hostname}")
        for info in socket.getaddrinfo(hostname, None):
            # Filter out IPv6 addresses if you only want IPv4
            log.debug(info)
            # if info[1] == socket.SOCK_STREAM and
            if info[0] == socket.AF_INET and info[4][0].startswith(prefix):
                yield info[4][0]

    def _test_url(self, url: str):
        try:
            response = requests.get(url, timeout=10)
            return response.status_code == 200
        except Exception:
            return False


@functools.lru_cache(maxsize=1)
def get_server_info():
    """Get server info."""
    from comfy.cli_args import args

    ip_checker = IPChecker()
    base_url: str = args.listen
    if base_url == "0.0.0.0":  # noqa: S104
        log.debug("Server set to 0.0.0.0, we will try to resolve the host IP")
        base_url = ip_checker.get_working_ip(
            f"http://{{}}:{args.port}/history"
        )
        log.debug(f"Setting ip to {base_url}")
    return (base_url, args.port)


# endregion


# region MISC Utilities
def glob_multiple(
    path: Path, patterns: list[str], recursive: bool = False
) -> list[Path]:
    """Combine multiple glob patterns into a single iterator."""
    return list(reduce(operator.or_, (set(path.glob(p)) for p in patterns)))


def build_glob_patterns(
    extensions: list[str], recursive: bool = False
) -> list[str]:
    """Build glob patterns for given extensions."""
    prefix = "**/" if recursive else ""
    return [f"{prefix}*.{ext}" for ext in extensions]


class SortMode(Enum):
    NONE = "none"
    MODIFIED = "modified"
    MODIFIED_REVERSE = "modified-reverse"
    NAME = "name"
    NAME_REVERSE = "name-reverse"

    @classmethod
    def from_str(cls, value: str | None) -> "SortMode|None":
        if not value:
            return None
        try:
            return cls(value.lower())
        except ValueError:
            log.warning(f"Sort mode {value} not supported")
            return None


# TODO: use mtb.core directly instead of copying parts here
T = TypeVar("T", bound="StringConvertibleEnum")


class StringConvertibleEnum(Enum):
    """Base class for enums with utility methods for string conversion and member listing."""

    @classmethod
    def from_str(cls: type[T], label: str | T) -> T:
        """
        Convert a string to the corresponding enum value (case sensitive).

        Args:
            label (Union[str, T]): The string or enum value to convert.

        Returns
        -------
            T: The corresponding enum value.

        Raises
        ------
            ValueError: If the label does not correspond to any enum member.
        """
        if isinstance(label, cls):
            return label
        if isinstance(label, str):
            # from key
            if label in cls.__members__:
                return cls[label]

            for member in cls:
                if member.value == label:
                    return member

        raise ValueError(
            f"Unknown label: '{label}'. Valid members: {list(cls.__members__.keys())}, "
            f"valid values: {cls.list_members()}"
        )

    @classmethod
    def to_str(cls: type[T], enum_value: T) -> str:
        """
        Convert an enum value to its string representation.

        Args:
            enum_value (T): The enum value to convert.

        Returns
        -------
            str: The string representation of the enum value.

        Raises
        ------
            ValueError: If the enum value is invalid.
        """
        if isinstance(enum_value, cls):
            return enum_value.value
        raise ValueError(f"Invalid Enum: {enum_value}")

    @classmethod
    def list_members(cls: type[T]) -> list[str]:
        """
        Return a list of string representations of all enum members.

        Returns
        -------
            list[str]: List of all enum member values.
        """
        return [enum.value for enum in cls]

    def __str__(self) -> str:
        """
        Returns the string representation of the enum value.

        Returns
        -------
            str: The string representation of the enum value.
        """
        return self.value


class Precision(StringConvertibleEnum):
    FULL = "full"
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8 = "fp8"

    def to_dtype(self):
        match self:
            case Precision.FP32 | Precision.FULL:
                return torch.float32
            case Precision.FP16:
                return torch.float16
            case Precision.BF16:
                return torch.bfloat16
            case Precision.FP8:
                return torch.float8_e4m3fn


class Operation(StringConvertibleEnum):
    COPY = "copy"
    CONVERT = "convert"
    DELETE = "delete"


def backup_file(
    fp: Path,
    target: Path | None = None,
    backup_dir: str = ".bak",
    suffix: str | None = None,
    prefix: str | None = None,
):
    """Backup a file by copying it to a backup directory."""
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


def hex_to_rgb(hex_color, *, bgr=False):
    """Convert hex to rgb tuple."""
    try:
        hex_color = hex_color.lstrip("#")
        if bgr:
            return tuple(int(hex_color[i : i + 2], 16) for i in (4, 2, 0))
        return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))

    except ValueError:
        log.error(f"Invalid hex color: {hex_color}")
        return (0, 0, 0)


def add_path(path: str | Path | list[str] | list[Path], *, prepend=False):
    """Add a path(s) to the system path.

    Supports Path, str, and list (of either).
    """
    if isinstance(path, list):
        for p in path:
            add_path(p, prepend=prepend)
        return

    if isinstance(path, Path):
        path = path.resolve().as_posix()

    if path not in sys.path:
        if prepend:
            sys.path.insert(0, path)
        else:
            sys.path.append(path)


def run_command(cmd, ignored_lines_start: None | str | list[str] = None):
    """Unsafe command runner."""
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
            f"Invalid 'cmd' argument. \
            It must be a string or a list of arguments. {type(cmd)}"
        )

    try:
        _run_command(shell_cmd, ignored_lines_start)
    except subprocess.CalledProcessError as e:
        print(
            f"Command failed with return code: {e.returncode}", file=sys.stderr
        )
        print(e.stderr.strip(), file=sys.stderr)

    except KeyboardInterrupt:
        print("Command execution interrupted.")


def _run_command(shell_cmd, ignored_lines_start):
    log.debug(f"Running {shell_cmd}")

    result = subprocess.run(
        shell_cmd,
        capture_output=True,
        text=True,
        shell=True,  # noqa: S602
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


def import_install(package_name):
    """Install and import a package if it is not already imported."""
    package_spec = reqs_map.get(package_name, package_name)

    try:
        importlib.import_module(package_name)

    except Exception:  # (ImportError, ModuleNotFoundError):
        run_command(
            [
                Path(sys.executable).as_posix(),
                "-m",
                "pip",
                "install",
                package_spec,
            ]
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


# NOTE: these aren't reliable, better call the getters each time
output_dir = Path(folder_paths.output_directory)
input_dir = Path(folder_paths.input_directory)

styles_dir = comfy_dir / "styles"
session_id = str(uuid.uuid4())
# - Construct the path to the font file
font_path = here / "data" / "font.ttf"

# - Add extern folder to path
extern_root = here / "extern"
add_path(extern_root)

for pth in extern_root.iterdir():
    if pth.is_dir():
        add_path(pth)

# - Add the ComfyUI directory and custom nodes path to the sys.path list
add_path(comfy_dir)
add_path(comfy_dir / "custom_nodes")

# TODO: use the requirements library
reqs_map = {value: key for key, value in pip_map.items()}

# NOTE: store already logged warnings to only alert once.
warned_messages: set[str] = set()


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
def to_numpy(image: torch.Tensor) -> npt.NDArray[np.uint8]:
    """Converts a tensor to a ndarray with proper scaling and type conversion."""
    log.debug(f"Converting tensor to numpy array with shape {image.shape}")
    np_array = np.clip(255.0 * image.cpu().numpy(), 0, 255).astype(np.uint8)
    log.debug(f"Numpy array shape after conversion: {np_array.shape}")
    return np_array


def handle_batch(
    tensor: torch.Tensor,
    func: Callable[[torch.Tensor], Image.Image | npt.NDArray[np.uint8]],
) -> list[Image.Image] | list[npt.NDArray[np.uint8]]:
    """Handles batch processing for a given tensor and conversion function."""
    return [func(tensor[i]) for i in range(tensor.shape[0])]


def tensor2b64(tensor: torch.Tensor) -> list[str]:
    images = tensor2pil(tensor)
    res: list[str] = []
    for img in images:
        frame_bytes = io.BytesIO()
        img.save(frame_bytes, format="PNG")
        res.append(
            "data:image/png;base64,"
            + base64.b64encode(frame_bytes.getvalue()).decode("utf-8")
        )
    return res


def tensor2pil(tensor: torch.Tensor) -> list[Image.Image]:
    """Converts a batch of tensors to a list of PIL Images."""

    def single_tensor2pil(t: torch.Tensor) -> Image.Image:
        np_array = to_numpy(t)
        if np_array.ndim == 2:  # (H, W) for masks
            return Image.fromarray(np_array, mode="L")
        elif np_array.ndim == 3:  # (H, W, C) for RGB/RGBA
            if np_array.shape[2] == 3:
                return Image.fromarray(np_array, mode="RGB")
            elif np_array.shape[2] == 4:
                return Image.fromarray(np_array, mode="RGBA")
        raise ValueError(f"Invalid tensor shape: {t.shape}")

    return handle_batch(tensor, single_tensor2pil)


def pil2tensor(images: Image.Image | list[Image.Image]) -> torch.Tensor:
    """Converts a PIL Image or a list of PIL Images to a tensor."""

    def single_pil2tensor(image: Image.Image) -> torch.Tensor:
        np_image = np.array(image).astype(np.float32) / 255.0
        if np_image.ndim == 2:  # Grayscale
            return torch.from_numpy(np_image).unsqueeze(0)  # (1, H, W)
        else:  # RGB or RGBA
            return torch.from_numpy(np_image).unsqueeze(0)  # (1, H, W, C)

    if isinstance(images, Image.Image):
        return single_pil2tensor(images)
    else:
        return torch.cat([single_pil2tensor(img) for img in images], dim=0)


def np2tensor(
    np_array: npt.NDArray[np.float32] | Sequence[npt.NDArray[np.float32]],
) -> torch.Tensor:
    """Converts a NumPy array or a list of NumPy arrays to a tensor."""

    def single_np2tensor(array: npt.NDArray[np.float32]) -> torch.Tensor:
        if array.ndim == 2:  # (H, W) for masks
            return torch.from_numpy(
                array.astype(np.float32) / 255.0
            ).unsqueeze(0)  # (1, H, W)
        elif array.ndim == 3:  # (H, W, C) for RGB/RGBA
            return torch.from_numpy(
                array.astype(np.float32) / 255.0
            ).unsqueeze(0)  # (1, H, W, C)
        raise ValueError(f"Invalid array shape: {array.shape}")

    if isinstance(np_array, np.ndarray):
        return single_np2tensor(np_array)
    else:
        return torch.cat([single_np2tensor(arr) for arr in np_array], dim=0)


def tensor2np(tensor: torch.Tensor) -> list[npt.NDArray[np.uint8]]:
    """Converts a batch of tensors to a list of NumPy arrays."""

    def single_tensor2np(t: torch.Tensor) -> npt.NDArray[np.uint8]:
        t = t.squeeze()  # Remove any singleton dimensions
        if t.ndim == 2:  # (H, W) for masks
            return to_numpy(t)
        elif t.ndim == 3:  # (C, H, W) for RGB/RGBA
            if t.shape[0] in [1, 3, 4]:  # Channel-first format
                t = t.permute(1, 2, 0)
            return to_numpy(t)
        else:
            raise ValueError(f"Invalid tensor shape: {t.shape}")

    return handle_batch(tensor, single_tensor2np)


def nextAvailable(path: Path | str) -> Path:
    """
    Find the next available path by adding a numbered suffix. (mimics comfy's version).

    Args:
        path (Path): The original path to check

    Returns
    -------
        Path: A path that doesn't exist yet
    """
    path = Path(path)

    if not path.is_absolute():
        path = output_dir / path

    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    parent = path.parent

    counter = 1
    while True:
        new_path = parent / f"{stem}_{counter:04d}{suffix}"
        if not new_path.exists():
            return new_path
        counter += 1


def pad(img: np.ndarray, left, right, top, bottom):
    """Pad an image with wrap padding on edges."""
    pad_width = np.array(((0, 0), (top, bottom), (left, right)))
    print(
        f"pad_width: {pad_width}, shape: {pad_width.shape}"
    )  # Debugging line
    return np.pad(img, pad_width, mode="wrap")  # type: ignore


def tiles_infer(tiles, ort_session, progress_callback=None):
    """Infer each tile with the given model.

    progress_callback will be called with
    arguments : current tile idx and total tiles amount
                (used to show progress on cursor in Blender).
    """
    out_channels = 3  # normal map RGB channels
    tiles_nb = tiles.shape[0]
    pred_tiles = np.empty(
        (tiles_nb, out_channels, tiles.shape[2], tiles.shape[3])
    )
    pred_tiles = np.empty(
        (tiles_nb, out_channels, tiles.shape[2], tiles.shape[3])
    )

    for i in range(tiles_nb):
        if progress_callback is not None:
            progress_callback(i + 1, tiles_nb)
        pred_tiles[i] = ort_session.run(
            None, {"input": tiles[i : i + 1].astype(np.float32)}
        )[0]

    return pred_tiles


def generate_mask(tile_size, stride_size):
    """Generate a pyramidal-like mask.

    Used for mixing overlapping predicted tiles.
    """
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
        np.linspace(0, 1, num=ramp_h)[None],
        (1, 0),  # type: ignore
    )
    mask[-ramp_h:, ramp_w:-ramp_w] = np.transpose(
        np.linspace(1, 0, num=ramp_h)[None],
        (1, 0),  # type: ignore
    )

    # Assume tiles are squared
    if ramp_h != ramp_w:
        raise ValueError("Ramps in both height and width directions required")

    # top left corner
    corner = np.rot90(corner_mask(ramp_h), 2)  # type: ignore
    mask[:ramp_h, :ramp_w] = corner
    # top right corner
    corner = np.flip(corner, 1)  # type: ignore
    mask[:ramp_h, -ramp_w:] = corner
    # bottom right corner
    corner = np.flip(corner, 0)  # type: ignore
    mask[-ramp_h:, -ramp_w:] = corner
    # bottom right corner
    corner = np.flip(corner, 1)  # type: ignore
    mask[-ramp_h:, :ramp_w] = corner

    return mask


def corner_mask(side_length):
    """Generate the corner part of the pyramidal-like mask.

    Currently, only for square shapes.
    """
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


def scaling_mask(side_length: int):
    """Generate a scaling mask for image blending."""
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


def _validate_stride(w, h, tile_w, tile_h):
    if (h % 2 != 0) or (w % 2 != 0):
        raise ValueError("Stride must be even")

    if (h < tile_h / 2) or (w < tile_w / 2):
        raise ValueError("Stride must be greater or equal than half tile size")

    if (h > tile_h) or (w > tile_w):
        raise ValueError("Stride must be smaller or equal than tile size")


def tiles_merge(tiles, stride_size, img_size, paddings):
    """Merge the list of tiles into one image.

    img_size is the original size, before padding.
    """
    _, tile_h, tile_w = tiles[0].shape
    pad_left, pad_right, pad_top, pad_bottom = paddings
    height = img_size[1] + pad_top + pad_bottom
    width = img_size[2] + pad_left + pad_right
    stride_h, stride_w = stride_size

    _validate_stride(stride_w, stride_h, tile_w, tile_h)

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
    """Return list of tiles from the given image.

    Also returns the padding used to fit the tiles in it.
    Input image must have dimension C,H,W.
    """
    log.debug(f"Splitting img: tile {tile_size}, stride {stride_size} ")
    tile_h, tile_w = tile_size
    stride_h, stride_w = stride_size
    img_h, img_w = img.shape[0], img.shape[1]

    _validate_stride(stride_w, stride_h, tile_w, tile_h)

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
    """Dumb hardcoded method to download antelopev2 model.

    TODO: remove and use the system from download_models.py
    """
    antelopev2_url = (
        "https://drive.google.com/uc?id=18wEUfMNohBJ4K3Ly5wpTejPfDzp-8fI8"
    )

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
            f"Could not load or download antelopev2 model, \
                download it manually from {antelopev2_url}"
        )
        raise e


def get_model_path(fam, model=None):
    """Get any model path given it's family and name.

    This supports extra_models and models folders.
    """
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
                warn_msg = f"Found multiple match, we will pick the last {res[-1]}\n{res}"
                if warn_msg not in warned_messages:
                    log.info(warn_msg)
                    warned_messages.add(warn_msg)
            res = res[-1]
        res = Path(res)
        log.debug(f"Resolved model path from folder_paths: {res}")
    else:
        res = models_dir / fam
        if model:
            res /= model

    return res


# endregion

# region geo


def euler_to_rotation_matrix(
    x_deg: float, y_deg: float, z_deg: float
) -> np.ndarray[np.float64]:
    # Convert degrees to radians
    x = np.radians(x_deg)
    y = np.radians(y_deg)
    z = np.radians(z_deg)

    # Rotation matrix around x-axis
    Rx = np.array(
        [[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]]
    )

    # Rotation matrix around y-axis
    Ry = np.array(
        [[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]]
    )

    # Rotation matrix around z-axis
    Rz = np.array(
        [[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]]
    )

    return Rz @ Ry @ Rx


def rotate_mesh(mesh, x_deg: float, y_deg: float, z_deg: float, center=None):
    R = euler_to_rotation_matrix(x_deg, y_deg, z_deg)
    return mesh.rotate(R, center) if center is not None else mesh.rotate(R)


def get_transformation_matrix(position, rotation, scale):
    # translation matrix
    T = np.eye(4)
    T[:3, 3] = position

    # rotation matrix from Euler angles
    R = euler_to_rotation_matrix(*rotation)
    R_homo = np.eye(4)
    R_homo[:3, :3] = R

    # scaling matrix
    S = np.eye(4)
    S[0, 0] = scale[0]
    S[1, 1] = scale[1]
    S[2, 2] = scale[2]

    # combine
    return T @ R_homo @ S


def spread_geo(geo, *, cp=False):
    """Spreads a GEOMETRY type into (mesh,material)."""
    mesh = geo["mesh"] if not cp else copy.copy(geo["mesh"])
    material = geo.get("material", {})
    return (mesh, material)


def json_to_mesh(json_data: str):
    """Convert JSON to an Open3D mesh."""
    data = json.loads(json_data)
    mesh = o3d.geometry.TriangleMesh()

    if "vertices" in data:
        mesh.vertices = o3d.utility.Vector3dVector(
            np.array(data["vertices"]).reshape(-1, 3)
        )

    if "triangles" in data:
        mesh.triangles = o3d.utility.Vector3iVector(
            np.array(data["triangles"]).reshape(-1, 3)
        )

    if "vertex_normals" in data:
        mesh.vertex_normals = o3d.utility.Vector3dVector(
            np.array(data["vertex_normals"]).reshape(-1, 3)
        )

    if "vertex_colors" in data:
        mesh.vertex_colors = o3d.utility.Vector3dVector(
            np.array(data["vertex_colors"]).reshape(-1, 3)
        )

    if "triangle_uvs" in data:
        mesh.triangle_uvs = o3d.utility.Vector2dVector(
            np.array(data["triangle_uvs"]).reshape(-1, 2)
        )

    return mesh


def mesh_to_json(mesh: "o3d.geometry.MeshBase"):
    """Convert an Open3D mesh to JSON."""
    mesh_dict = {
        "vertices": np.asarray(mesh.vertices).tolist(),
        "triangles": np.asarray(mesh.triangles).tolist(),
    }

    if mesh.has_vertex_normals():
        mesh_dict["vertex_normals"] = np.asarray(mesh.vertex_normals).tolist()

    if mesh.has_vertex_colors():
        mesh_dict["vertex_colors"] = np.asarray(mesh.vertex_colors).tolist()

    if mesh.has_triangle_uvs():
        mesh_dict["triangle_uvs"] = np.asarray(mesh.triangle_uvs).tolist()

    return json.dumps(mesh_dict)


def create_grid(scale=(1, 1, 1), rows=10, columns=10):
    dx, dy, dz = scale

    # Create vertices
    vertices = []
    for i in np.linspace(-dy / 2, dy / 2, rows + 1):
        vertices.extend(
            [j, 0, i] for j in np.linspace(-dx / 2, dx / 2, columns + 1)
        )
    # Generate triangles
    triangles = []
    for i, j in itertools.product(range(rows), range(columns)):
        p1 = i * (columns + 1) + j
        p2 = i * (columns + 1) + j + 1
        p3 = (i + 1) * (columns + 1) + j
        p4 = (i + 1) * (columns + 1) + j + 1

        triangles.extend(([p1, p2, p3], [p2, p4, p3]))
    vertices = o3d.utility.Vector3dVector(np.array(vertices))
    triangles = o3d.utility.Vector3iVector(np.array(triangles))
    mesh = o3d.geometry.TriangleMesh(vertices, triangles)

    mesh.compute_vertex_normals()

    return mesh


def create_box(scale=(1, 1, 1), divisions=(1, 1, 1)):
    dx, dy, dz = scale
    div_x, div_y, div_z = divisions

    vertices = []
    for i in np.linspace(-dx / 2, dx / 2, div_x + 1):
        for j in np.linspace(-dy / 2, dy / 2, div_y + 1):
            vertices.extend(
                [i, j, k] for k in np.linspace(-dz / 2, dz / 2, div_z + 1)
            )
    # Generate triangles for the box faces
    triangles = []
    for x, y in itertools.product(range(div_x), range(div_y)):
        for z in range(div_z):
            # Define base index for this cube
            base = z * (div_x + 1) * (div_y + 1) + y * (div_x + 1) + x

            # Indices for the 8 vertices of the cube
            v0 = base
            v1 = base + 1
            v2 = base + (div_x + 1) + 1
            v3 = base + (div_x + 1)
            v4 = base + (div_x + 1) * (div_y + 1)
            v5 = v4 + 1
            v6 = v4 + (div_x + 1) + 1
            v7 = v4 + (div_x + 1)

            triangles.extend(
                (
                    [v0, v1, v2],
                    [v2, v3, v0],
                    [v4, v5, v6],
                    [v6, v7, v4],
                    [v0, v3, v7],
                    [v7, v4, v0],
                    [v1, v5, v6],
                    [v6, v2, v1],
                    [v0, v4, v5],
                    [v5, v1, v0],
                    [v3, v2, v6],
                    [v6, v7, v3],
                )
            )
    vertices = o3d.utility.Vector3dVector(np.array(vertices))
    triangles = o3d.utility.Vector3iVector(np.array(triangles))
    mesh = o3d.geometry.TriangleMesh(vertices, triangles)

    mesh.compute_vertex_normals()

    return mesh


def create_sphere(radius=1, columns=10, rows=10):
    # Create vertex positions
    vertices = []
    for i in range(rows + 1):
        lat = i * np.pi / rows
        sin_lat = np.sin(lat)
        cos_lat = np.cos(lat)

        for j in range(columns + 1):
            lon = j * 2 * np.pi / columns
            sin_lon = np.sin(lon)
            cos_lon = np.cos(lon)

            x = radius * cos_lon * sin_lat
            y = radius * cos_lat
            z = radius * sin_lon * sin_lat
            vertices.append([x, y, z])

    # Create triangles
    triangles = []
    for i in range(rows):
        for j in range(columns):
            p1 = i * (columns + 1) + j
            p2 = i * (columns + 1) + j + 1
            p3 = (i + 1) * (columns + 1) + j
            p4 = (i + 1) * (columns + 1) + j + 1

            triangles.extend(([p1, p2, p3], [p2, p4, p3]))
    vertices = o3d.utility.Vector3dVector(np.array(vertices))
    triangles = o3d.utility.Vector3iVector(np.array(triangles))
    mesh = o3d.geometry.TriangleMesh(vertices, triangles)

    # Assigning random colors to vertices
    N = len(vertices)
    mesh.vertex_colors = o3d.utility.Vector3dVector(
        np.random.uniform(0, 1, size=(N, 3))
    )
    mesh.compute_vertex_normals()

    return mesh


def create_torus(torus_radius=1, ring_radius=0.5, rows=10, columns=10):
    vertices = []
    for i in range(rows + 1):
        theta = i * 2 * np.pi / rows
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        circle_center = torus_radius + ring_radius * cos_theta

        for j in range(columns + 1):
            phi = j * 2 * np.pi / columns
            cos_phi = np.cos(phi)
            sin_phi = np.sin(phi)

            x = circle_center * cos_phi
            y = ring_radius * sin_theta
            z = circle_center * sin_phi
            vertices.append([x, y, z])

    triangles = []
    for i in range(rows):
        for j in range(columns):
            p1 = i * (columns + 1) + j
            p2 = i * (columns + 1) + j + 1
            p3 = (i + 1) * (columns + 1) + j
            p4 = (i + 1) * (columns + 1) + j + 1

            triangles.extend(([p1, p2, p3], [p2, p4, p3]))
    vertices = o3d.utility.Vector3dVector(np.array(vertices))
    triangles = o3d.utility.Vector3iVector(np.array(triangles))
    mesh = o3d.geometry.TriangleMesh(vertices, triangles)

    mesh.compute_vertex_normals()

    return mesh


# endregion

# region UV Utilities


def create_uv_map_tensor(width=512, height=512):
    """Create a UV map tensor."""
    # Create u coordinates
    u_coords = torch.linspace(0.0, 1.0, steps=width)
    # Create v coordinates
    v_coords = torch.linspace(0.0, 1.0, steps=height)

    # Generate UV grid
    u_grid, v_grid = torch.meshgrid(u_coords, v_coords)

    uv_map = torch.zeros(height, width, 3, dtype=torch.float32)
    # Fill U channel with U grid transposed
    uv_map[:, :, 0] = u_grid.t()
    # Fill V channel with V grid transposed
    uv_map[:, :, 1] = v_grid.t()

    return uv_map.unsqueeze(0)


# endregion


# region ANIMATION Utilities
EASINGS = [
    "Linear",
    "Sine In",
    "Sine Out",
    "Sine In/Out",
    "Quart In",
    "Quart Out",
    "Quart In/Out",
    "Cubic In",
    "Cubic Out",
    "Cubic In/Out",
    "Circ In",
    "Circ Out",
    "Circ In/Out",
    "Back In",
    "Back Out",
    "Back In/Out",
    "Elastic In",
    "Elastic Out",
    "Elastic In/Out",
    "Bounce In",
    "Bounce Out",
    "Bounce In/Out",
]


def apply_easing(value, easing_type):
    if easing_type == "Linear":
        return value

    # Back easing functions
    def easeInBack(t):
        s = 1.70158
        return t * t * ((s + 1) * t - s)


def _out_back(t):
    s = 1.70158
    return ((t - 1) * t * ((s + 1) * t + s)) + 1


def _inout_back(t):
    s = 1.70158 * 1.525
    if t < 0.5:
        return (t * t * (t * (s + 1) - s)) * 2
    return ((t - 2) * t * ((s + 1) * t + s) + 2) * 2


# Elastic easing functions
def _in_elastic(t):
    if t == 0:
        return 0
    if t == 1:
        return 1
    p = 0.3
    s = p / 4
    return -(
        math.pow(2, 10 * (t - 1)) * math.sin((t - 1 - s) * (2 * math.pi) / p)
    )


def _out_elastic(t):
    if t == 0:
        return 0
    if t == 1:
        return 1
    p = 0.3
    s = p / 4
    return math.pow(2, -10 * t) * math.sin((t - s) * (2 * math.pi) / p) + 1


def _inout_elastic(t):
    if t == 0:
        return 0
    if t == 1:
        return 1
    p = 0.3 * 1.5
    s = p / 4
    t = t * 2
    if t < 1:
        return -0.5 * (
            math.pow(2, 10 * (t - 1))
            * math.sin((t - 1 - s) * (2 * math.pi) / p)
        )
    return (
        0.5
        * math.pow(2, -10 * (t - 1))
        * math.sin((t - 1 - s) * (2 * math.pi) / p)
        + 1
    )


# Bounce easing functions
def _in_bounce(t):
    return 1 - _out_bounce(1 - t)


def _out_bounce(t):
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


def _inout_bounce(t):
    if t < 0.5:
        return _in_bounce(t * 2) * 0.5
    return _out_bounce(t * 2 - 1) * 0.5 + 0.5


# Quart easing functions
def _in_quart(t):
    return t * t * t * t


def _out_quart(t):
    t -= 1
    return -(t**2 * t * t - 1)


def _inout_quart(t):
    t *= 2
    if t < 1:
        return 0.5 * t * t * t * t
    t -= 2
    return -0.5 * (t**2 * t * t - 2)


# Cubic easing functions
def _in_cubic(t):
    return t * t * t


def _out_cubic(t):
    t -= 1
    return t**2 * t + 1


def _inout_cubic(t):
    t *= 2
    if t < 1:
        return 0.5 * t * t * t
    t -= 2
    return 0.5 * (t**2 * t + 2)


# Circ easing functions
def _in_circ(t):
    return -(math.sqrt(1 - t * t) - 1)


def _out_circ(t):
    t -= 1
    return math.sqrt(1 - t**2)


def _inout_circ(t):
    t *= 2
    if t < 1:
        return -0.5 * (math.sqrt(1 - t**2) - 1)
    t -= 2
    return 0.5 * (math.sqrt(1 - t**2) + 1)


# Sine easing functions
def _in_sine(t):
    return -math.cos(t * (math.pi / 2)) + 1


def _out_sine(t):
    return math.sin(t * (math.pi / 2))


def _inout_sine(t):
    return -0.5 * (math.cos(math.pi * t) - 1)


def apply_easing(value: float, easing_type: str):
    """Apply easing function to value based on easing type."""
    if easing_type == "Linear":
        return value

    easing_functions = {
        "Sine In": _in_sine,
        "Sine Out": _out_sine,
        "Sine In/Out": _inout_sine,
        "Quart In": _in_quart,
        "Quart Out": _out_quart,
        "Quart In/Out": _inout_quart,
        "Cubic In": _in_cubic,
        "Cubic Out": _out_cubic,
        "Cubic In/Out": _inout_cubic,
        "Circ In": _in_circ,
        "Circ Out": _out_circ,
        "Circ In/Out": _inout_circ,
        # "Back In": _in_back,
        "Back Out": _out_back,
        "Back In/Out": _inout_back,
        "Elastic In": _in_elastic,
        "Elastic Out": _out_elastic,
        "Elastic In/Out": _inout_elastic,
        "Bounce In": _in_bounce,
        "Bounce Out": _out_bounce,
        "Bounce In/Out": _inout_bounce,
    }

    function_ = easing_functions.get(easing_type)
    if function_:
        return function_(value)

    log.error(f"Unknown easing type: {easing_type}")
    log.error(f"Available easing types: {list(easing_functions.keys())}")
    raise ValueError(f"Unknown easing type: {easing_type}")


# endregion
