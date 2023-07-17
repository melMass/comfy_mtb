from PIL import Image
import numpy as np
import torch
from pathlib import Path
import sys

from typing import Union, List
from .log import log


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


# Get the absolute path of the parent directory of the current script
here = Path(__file__).parent.resolve()

# Construct the absolute path to the ComfyUI directory
comfy_dir = here.parent.parent

# Construct the path to the font file
font_path = here / "font.ttf"

# Add extern folder to path
extern_root = here / "extern"
add_path(extern_root)
for pth in extern_root.iterdir():
    if pth.is_dir():
        add_path(pth)


# Add the ComfyUI directory and custom nodes path to the sys.path list
add_path(comfy_dir)
add_path((comfy_dir / "custom_nodes"))


def tensor2pil(image: torch.Tensor) -> List[Image.Image]:
    batch_count = 1
    if len(image.shape) > 3:
        batch_count = image.size(0)

    if batch_count > 1:
        out = []
        out.extend([tensor2pil(image[i]) for i in range(batch_count)])
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
    batch_count = 1
    if len(tensor.shape) > 3:
        batch_count = tensor.size(0)
    if batch_count > 1:
        out = []
        for i in range(batch_count):
            out.extend(tensor2np(tensor[i]))
        return out

    return [np.clip(255.0 * tensor.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)]
