import tempfile
from pathlib import Path

import numpy as np

# torch must be imported prior to onnx for the CUDAProvider.
import torch  # isort:skip
import onnxruntime as ort
from PIL import Image

from ..errors import ModelNotFound
from ..log import mklog
from ..utils import (
    download_model,
    get_model_path,
    tensor2pil,
    tiles_infer,
    tiles_merge,
    tiles_split,
)

# Disable MS telemetry
ort.disable_telemetry_events()
log = mklog(__name__)


# - COLOR to NORMALS
def color_to_normals(
    color_img,
    overlap,
    progress_callback,
    *,
    save_temp=False,
    auto_download=False,
):
    """Compute a normal map from the given color map.

    'color_img' must be a numpy array in C,H,W format (with C as RGB).
    'overlap' must be one of 'SMALL', 'MEDIUM', 'LARGE'.
    """
    temp_dir = Path(tempfile.mkdtemp()) if save_temp else None

    # Remove alpha & convert to grayscale
    img = np.mean(color_img[:3], axis=0, keepdims=True)

    if temp_dir:
        Image.fromarray((img[0] * 255).astype(np.uint8)).save(
            temp_dir / "grayscale_img.png"
        )

    log.debug(
        "Converting color image to grayscale by taking "
        f"the mean over color channels: {img.shape}"
    )

    # Split image in tiles
    log.debug("DeepBump Color → Normals : tilling")
    tile_size = 256
    overlaps = {
        "SMALL": tile_size // 6,
        "MEDIUM": tile_size // 4,
        "LARGE": tile_size // 2,
    }
    stride_size = tile_size - overlaps[overlap]
    tiles, paddings = tiles_split(
        img, (tile_size, tile_size), (stride_size, stride_size)
    )
    if temp_dir:
        for i, tile in enumerate(tiles):
            Image.fromarray((tile[0] * 255).astype(np.uint8)).save(
                temp_dir / f"tile_{i}.png"
            )

    # Load model
    log.debug("DeepBump Color → Normals : loading model")
    model = get_model_path("deepbump", "deepbump256.onnx")
    if not model or not model.exists():
        if not auto_download:
            raise ModelNotFound(f"deepbump ({model})")
        log.debug("Downloading models...")
        download_model(
            "https://github.com/HugoTini/DeepBump/raw/master/deepbump256.onnx",
            "deepbump",
        )

    providers = [
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "CoreMLProvider",
        "CPUExecutionProvider",
    ]
    available_providers = [
        provider
        for provider in providers
        if provider in ort.get_available_providers()
    ]

    if not available_providers:
        raise RuntimeError(
            "No valid ONNX Runtime providers available on this machine."
        )
    log.debug(f"Using ONNX providers: {available_providers}")
    ort_session = ort.InferenceSession(
        model.as_posix(), providers=available_providers
    )

    # Predict normal map for each tile
    log.debug("DeepBump Color → Normals : generating")
    pred_tiles = tiles_infer(
        tiles, ort_session, progress_callback=progress_callback
    )

    if temp_dir:
        for i, pred_tile in enumerate(pred_tiles):
            Image.fromarray(
                (pred_tile.transpose(1, 2, 0) * 255).astype(np.uint8)
            ).save(temp_dir / f"pred_tile_{i}.png")

    # Merge tiles
    log.debug("DeepBump Color → Normals : merging")
    pred_img = tiles_merge(
        pred_tiles,
        (stride_size, stride_size),
        (3, img.shape[1], img.shape[2]),
        paddings,
    )

    if temp_dir:
        Image.fromarray(
            (pred_img.transpose(1, 2, 0) * 255).astype(np.uint8)
        ).save(temp_dir / "merged_img.png")

    # Normalize each pixel to unit vector
    pred_img = normalize(pred_img)

    if temp_dir:
        Image.fromarray(
            (pred_img.transpose(1, 2, 0) * 255).astype(np.uint8)
        ).save(temp_dir / "final_img.png")

        log.debug(f"Debug images saved in {temp_dir}")

    return pred_img


# - NORMALS to CURVATURE
def conv_1d(array, kernel_1d):
    """Perform row by row 1D convolutions.

    of the given 2D image with the given 1D kernel.
    """
    # Input kernel length must be odd
    k_l = len(kernel_1d)

    assert k_l % 2 != 0
    # Convolution is repeat-padded
    extended = np.pad(array, k_l // 2, mode="wrap")
    # Output has same size as input (padded, valid-mode convolution)
    output = np.empty(array.shape)
    for i in range(array.shape[0]):
        output[i] = np.convolve(
            extended[i + (k_l // 2)], kernel_1d, mode="valid"
        )

    return output * -1


def gaussian_kernel(length, sigma):
    """Return a 1D gaussian kernel of size 'length'."""
    space = np.linspace(-(length - 1) / 2, (length - 1) / 2, length)
    kernel = np.exp(-0.5 * np.square(space) / np.square(sigma))
    return kernel / np.sum(kernel)


def normalize(np_array):
    """Normalize all elements of the given numpy array to [0,1]."""
    return (np_array - np.min(np_array)) / (
        np.max(np_array) - np.min(np_array)
    )


def normals_to_curvature(normals_img, blur_radius, progress_callback):
    """Compute a curvature map from the given normal map.

    'normals_img' must be a numpy array in C,H,W format (with C as RGB).
    'blur_radius' must be one of:
        'SMALLEST', 'SMALLER', 'SMALL', 'MEDIUM', 'LARGE', 'LARGER', 'LARGEST'.
    """
    # Convolutions on normal map red & green channels
    if progress_callback is not None:
        progress_callback(0, 4)
    diff_kernel = np.array([-1, 0, 1])
    h_conv = conv_1d(normals_img[0, :, :], diff_kernel)
    if progress_callback is not None:
        progress_callback(1, 4)
    v_conv = conv_1d(-1 * normals_img[1, :, :].T, diff_kernel).T
    if progress_callback is not None:
        progress_callback(2, 4)

    # Sum detected edges
    edges_conv = h_conv + v_conv

    # Blur radius size is proportional to img sizes
    blur_factors = {
        "SMALLEST": 1 / 256,
        "SMALLER": 1 / 128,
        "SMALL": 1 / 64,
        "MEDIUM": 1 / 32,
        "LARGE": 1 / 16,
        "LARGER": 1 / 8,
        "LARGEST": 1 / 4,
    }
    if blur_radius not in blur_factors:
        raise ValueError(f"{blur_radius} not found in {blur_factors}")

    blur_radius_px = int(
        np.mean(normals_img.shape[1:3]) * blur_factors[blur_radius]
    )

    # If blur radius too small, do not blur
    if blur_radius_px < 2:
        edges_conv = normalize(edges_conv)
        return np.stack([edges_conv, edges_conv, edges_conv])

    # Make sure blur kernel length is odd
    if blur_radius_px % 2 == 0:
        blur_radius_px += 1

    # Blur curvature with separated convolutions
    sigma = blur_radius_px // 8
    if sigma == 0:
        sigma = 1
    g_kernel = gaussian_kernel(blur_radius_px, sigma)
    h_blur = conv_1d(edges_conv, g_kernel)
    if progress_callback is not None:
        progress_callback(3, 4)
    v_blur = conv_1d(h_blur.T, g_kernel).T
    if progress_callback is not None:
        progress_callback(4, 4)

    # Normalize to [0,1]
    curvature = normalize(v_blur)

    # Expand single channel the three channels (RGB)
    return np.stack([curvature, curvature, curvature])


# - NORMALS to HEIGHT
def normals_to_grad(normals_img):
    return (normals_img[0] - 0.5) * 2, (normals_img[1] - 0.5) * 2


def copy_flip(grad_x, grad_y):
    """Concat 4 flipped copies of input gradients (makes them wrap).

    Output is twice bigger in both dimensions.
    """
    grad_x_top = np.hstack([grad_x, -np.flip(grad_x, axis=1)])
    grad_x_bottom = np.hstack([np.flip(grad_x, axis=0), -np.flip(grad_x)])
    new_grad_x = np.vstack([grad_x_top, grad_x_bottom])

    grad_y_top = np.hstack([grad_y, np.flip(grad_y, axis=1)])
    grad_y_bottom = np.hstack([-np.flip(grad_y, axis=0), -np.flip(grad_y)])
    new_grad_y = np.vstack([grad_y_top, grad_y_bottom])

    return new_grad_x, new_grad_y


def frankot_chellappa(grad_x, grad_y, progress_callback=None):
    """Frankot-Chellappa depth-from-gradient algorithm."""
    if progress_callback is not None:
        progress_callback(0, 3)

    rows, cols = grad_x.shape

    rows_scale = (np.arange(rows) - (rows // 2 + 1)) / (rows - rows % 2)
    cols_scale = (np.arange(cols) - (cols // 2 + 1)) / (cols - cols % 2)

    u_grid, v_grid = np.meshgrid(cols_scale, rows_scale)

    u_grid = np.fft.ifftshift(u_grid)
    v_grid = np.fft.ifftshift(v_grid)

    if progress_callback is not None:
        progress_callback(1, 3)

    grad_x_F = np.fft.fft2(grad_x)
    grad_y_F = np.fft.fft2(grad_y)

    if progress_callback is not None:
        progress_callback(2, 3)

    nominator = (-1j * u_grid * grad_x_F) + (-1j * v_grid * grad_y_F)
    denominator = (u_grid**2) + (v_grid**2) + 1e-16

    Z_F = nominator / denominator
    Z_F[0, 0] = 0.0

    Z = np.real(np.fft.ifft2(Z_F))

    if progress_callback is not None:
        progress_callback(3, 3)

    return (Z - np.min(Z)) / (np.max(Z) - np.min(Z))


def normals_to_height(normals_img, seamless, progress_callback):
    """Computes a height map from the given normal map. 'normals_img' must be a numpy array
    in C,H,W format (with C as RGB). 'seamless' is a bool that should indicates if 'normals_img'
    is seamless.
    """
    # Flip height axis
    flip_img = np.flip(normals_img, axis=1)

    # Get gradients from normal map
    grad_x, grad_y = normals_to_grad(flip_img)
    grad_x = np.flip(grad_x, axis=0)
    grad_y = np.flip(grad_y, axis=0)

    # If non-seamless chosen, expand gradients
    if not seamless:
        grad_x, grad_y = copy_flip(grad_x, grad_y)

    # Compute height
    pred_img = frankot_chellappa(
        -grad_x, grad_y, progress_callback=progress_callback
    )

    # Cut to valid part if gradients were expanded
    if not seamless:
        height, width = normals_img.shape[1], normals_img.shape[2]
        pred_img = pred_img[:height, :width]

    # Expand single channel the three channels (RGB)
    return np.stack([pred_img, pred_img, pred_img])


# - ADDON
class MTB_DeepBump:
    """Normal & height maps generation from single pictures"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (
                    [
                        "Color to Normals",
                        "Normals to Curvature",
                        "Normals to Height",
                    ],
                ),
                "color_to_normals_overlap": (["SMALL", "MEDIUM", "LARGE"],),
                "normals_to_curvature_blur_radius": (
                    [
                        "SMALLEST",
                        "SMALLER",
                        "SMALL",
                        "MEDIUM",
                        "LARGE",
                        "LARGER",
                        "LARGEST",
                    ],
                ),
                "normals_to_height_seamless": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "auto_download": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"

    CATEGORY = "mtb/textures"

    def apply(
        self,
        *,
        image,
        mode="Color to Normals",
        color_to_normals_overlap="SMALL",
        normals_to_curvature_blur_radius="SMALL",
        normals_to_height_seamless=True,
        auto_download=False,
    ):
        images = tensor2pil(image)
        out_images = []

        for image in images:
            log.debug(f"Input image shape: {image}")

            in_img = np.transpose(image, (2, 0, 1)) / 255
            log.debug(f"transposed for deep image shape: {in_img.shape}")
            out_img = None

            # Apply processing
            if mode == "Color to Normals":
                out_img = color_to_normals(
                    in_img,
                    color_to_normals_overlap,
                    None,
                    auto_download=auto_download,
                )
            if mode == "Normals to Curvature":
                out_img = normals_to_curvature(
                    in_img, normals_to_curvature_blur_radius, None
                )
            if mode == "Normals to Height":
                out_img = normals_to_height(
                    in_img, normals_to_height_seamless, None
                )

            if out_img is not None:
                log.debug(f"Output image shape: {out_img.shape}")
                out_images.append(
                    torch.from_numpy(
                        np.transpose(out_img, (1, 2, 0)).astype(np.float32)
                    ).unsqueeze(0)
                )
            else:
                log.error("No out img... This should not happen")
        for outi in out_images:
            log.debug(f"Shape fed to utils: {outi.shape}")
        return (torch.cat(out_images, dim=0),)


__nodes__ = [MTB_DeepBump]
