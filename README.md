## MTB Nodes

Feel free to do whatever you want with this codebase, I'm mainly using Comfy to build POCs to implement in [MLOPs](https://github.com/Bismuth-Consultancy-BV/MLOPs). And a lot of nodes are inspired by existing ones from the community or builtin 
Just beware of the licenses of some libraries (deepbump for instance is [GPLv3](https://github.com/HugoTini/DeepBump/blob/master/LICENSE))

## Install

From within the python environment you already use for ComfyUI install the requirements.
```bash
pip install -r comfy_mtb/requirements.txt
```

### Node List

- `Latent Lerp`: Linear Interpolate between two latents,
- `Int to Number`: Supplement for WASSuite number nodes,
- `Bounding Box`: BBox constructor (custom type),
- `Crop`: Crop image from BBox,
- `Uncrop`: Uncrop image from BBox,
- `ImageBlur`: Blur the input image,
- `Denoise`: Denoise the input image,
- `ImageCompare`: Compare image,
- `RGB to HSV`: -,
- `HSV to RGB`: -,
- `Color Correct`: Basic color correction tools,
- `Modulo`: Modulo (useful for loops),
- `Deglaze Image`: taken from [FN16](https://github.com/Fannovel16/FN16-ComfyUI-nodes/blob/main/DeglazeImage.py),
- `Smart Step`: A very basic node to get step percent to use in KSampler advanced,


### Comfy Resources

**Guides**:
- [Official Examples (eng)](https://comfyanonymous.github.io/ComfyUI_examples/)
- [ComfyUI Community Manual (eng)](https://blenderneko.github.io/ComfyUI-docs/) by @BlenderNeko
  
- [Tomoaki's personal Wiki (jap)](https://comfyui.creamlab.net/guides/) by @tjhayasaka

**Extensions and Custom Nodes**:
- [Plugins for Comfy List (eng)](https://github.com/WASasquatch/comfyui-plugins) by @WASasquatch

- [ComfyUI tag on CivitAI (eng)](https://civitai.com/tag/comfyui)