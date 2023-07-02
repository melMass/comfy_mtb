from PIL import Image
import numpy as np
import torch
from pathlib import Path
import sys


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


# Tensor to PIL (grabbed from WAS Suite)
def tensor2pil(image: torch.Tensor) -> Image.Image:
    return Image.fromarray(
        np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    )


# TODO: write pil2tensor counterpart (batch support)
# def tensor2pil(image: torch.Tensor) -> Union[Image.Image, List[Image.Image]]:
#     batch_count = 1
#     if len(image.shape) > 3:
#         batch_count = image.size(0)

#     if batch_count == 1:
#         return Image.fromarray(
#             np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
#         )
#     return [tensor2pil(image[i]) for i in range(batch_count)]


# Convert PIL to Tensor (grabbed from WAS Suite)
def pil2tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def img_np_to_tensor(img_np):
    return torch.from_numpy(img_np / 255.0)[None,]


def img_tensor_to_np(img_tensor):
    img_tensor = img_tensor.clone()
    img_tensor = img_tensor * 255.0
    return img_tensor.squeeze(0).numpy().astype(np.float32)


