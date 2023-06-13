from PIL import Image
import numpy as np
import torch
from pathlib import Path
import sys

# Get the absolute path of the parent directory of the current script
here = Path(__file__).parent.resolve()
# Tensor to PIL (grabbed from WAS Suite)
def tensor2pil(image: torch.Tensor) -> Image.Image:
    return Image.fromarray(
        np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    )


# Convert PIL to Tensor (grabbed from WAS Suite)
def pil2tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

