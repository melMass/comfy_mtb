from PIL import Image
import numpy as np
import torch
from pathlib import Path
import sys
from logging import getLogger
import logging

log = getLogger(__package__)
log.setLevel(logging.DEBUG)

# Get the absolute path of the parent directory of the current script
here = Path(__file__).parent.resolve()

# Construct the absolute path to the ComfyUI directory
comfy_dir = here.parent.parent

# Construct the path to the font file
font_path = here / "font.ttf"

# Add extern folder to path
extern = (here / "extern", here / "extern" / "SadTalker")
sys.path.extend([ x.as_posix() for x in extern])

# Add the ComfyUI directory path to the sys.path list
sys.path.append(comfy_dir.resolve().as_posix())

# Tensor to PIL (grabbed from WAS Suite)
def tensor2pil(image: torch.Tensor) -> Image.Image:
    return Image.fromarray(
        np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    )


# Convert PIL to Tensor (grabbed from WAS Suite)
def pil2tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def img_np_to_tensor(img_np):
    return torch.from_numpy(img_np / 255.0)[None,]

def img_tensor_to_np(img_tensor):
    img_tensor = img_tensor.clone()
    img_tensor = img_tensor * 255.0
    return img_tensor.squeeze(0).numpy().astype(np.float32)


