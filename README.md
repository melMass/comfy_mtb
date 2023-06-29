## MTB Nodes

Feel free to do whatever you want with this codebase, I'm mainly using Comfy to build POCs to implement in [MLOPs](https://github.com/Bismuth-Consultancy-BV/MLOPs). And a lot of nodes are inspired by existing ones from the community or builtin 
Just beware of the licenses of some libraries (deepbump for instance is [GPLv3](https://github.com/HugoTini/DeepBump/blob/master/LICENSE))

## Install

> **Warning**
> If you use the `python-embed` mode of comfyui (the standalone release) then you might not be able to pip install
> dependencies that don't have wheels, in this case check the last [release](https://github.com/melMass/comfy_mtb/releases) there is a bundle for linux and windows

**Dependencies:**  
From within the python environment you already use for ComfyUI install the requirements.
```bash
pip install -r comfy_mtb/requirements.txt
```
**Models Download:**  
Some nodes require extra models to be downloaded, you can interactively do it using the same python environment as above:
```bash
python scripts/download_models.py
```

then follow the prompt or just press enter to download every models.

On first run the script [tries to symlink](https://github.com/melMass/comfy_mtb/blob/d982b69a58c05ccead9c49370764beaa4549992a/__init__.py#L45-L61) the web extensions to your comfy `web/extensions` folder. In case it fails you can manually copy the web scripts it only provides a color widget for now:  
![image](https://github.com/melMass/comfy_mtb/assets/7041726/cff7e66a-4cc4-4866-b35b-10af0bb2d110)


You should also see the nodes that failed loading with the reason next to them.

### Node List
In the JS Side a new widget type is added: COLOR, for now both the input and the widget are exposed, the input takes precendence over the widget.


**misc utils**  
- `Text To Image`: Utils to convert text to image using a font
- `Styles Loader`: Load csv files and populate a dropdown from the rows (à la A111)
- `Smart Step`: A very basic node to get step percent to use in KSampler advanced,
- `Qr Code`: Basic QR Code generator
- `Save Tensors`: Debug node that will probably be removed in the future
- `Int to Number`: Supplement for WASSuite number nodes

**face detection / swapping**
- `Face Swap`: Face swap using deepinsight/insightface models (this node used to be called `Roop` in early versions, it does the same)
  
**latent**  
- `Latent Lerp`: Linear interpolation (blend) between two latent 


**bbox**
- `Bounding Box`: BBox constructor (custom type),
- `BBox From Mask`: From a mask extract the bounding box
- `Crop`: Crop image from BBox
- `Uncrop`: Uncrop image from BBox

**image ops**
- `Image Remove Background Rembg`: [RemBG](https://github.com/danielgatis/rembg) powered background removal.
- `Blur`: Blur an image using a Gaussian filter.
- `Denoise`: Denoise the input image,
- `Image Compare`: Compare two images and return a difference image
- `Deglaze Image`: taken from [FN16](https://github.com/Fannovel16/FN16-ComfyUI-nodes/blob/main/DeglazeImage.py),
- `Mask To Image`: Converts a mask (alpha) to an RGB image with a color and background
- `Image Premultiply`: Premultiply image with mask
- `Image Resize Factor`: Extracted mostly from [WAS Node Suite](https://github.com/WASasquatch/was-node-suite-comfyui), with a few edits (most notably multiple image support) and less features.
- `Save Image Grid`: Save all the images in the input batch as a grid of images.

**colors**
- `Colored Image`: Constant color image of given size
- `RGB to HSV`: -,
- `HSV to RGB`: -,
- `Color Correct`: Basic color correction tools,

        
## Screenshots

- **FaceSwap (formely named roop)** (using [insight-face](https://github.com/deepinsight/insightface))  
  The face index allow you to choose which face to replace as you can see here:
  ![ComfyUI_909](https://github.com/melMass/comfy_mtb/assets/7041726/2e9d6066-c466-4a01-bd6c-315f7f1e8b42)

- **Style Loader**: A111 like csv styles in Comfy  
  ![image](https://github.com/melMass/comfy_mtb/assets/7041726/02fe3211-18ee-4e54-a029-931388f5fde8)

- **Color Correction**: basic color correction node  
  ![image](https://github.com/melMass/comfy_mtb/assets/7041726/7c20ac83-31ff-40ea-a1a0-06c2acefb2ef)

- **Image Remove Background [RemBG]**: (using [rembg](https://github.com/danielgatis/rembg))
  ![image](https://github.com/melMass/comfy_mtb/assets/7041726/e69253b4-c03c-45e9-92b5-aa46fb887be8)


### Comfy Resources

**Guides**:
- [Official Examples (eng)](https://comfyanonymous.github.io/ComfyUI_examples/)
- [ComfyUI Community Manual (eng)](https://blenderneko.github.io/ComfyUI-docs/) by @BlenderNeko
  
- [Tomoaki's personal Wiki (jap)](https://comfyui.creamlab.net/guides/) by @tjhayasaka

**Extensions and Custom Nodes**:
- [Plugins for Comfy List (eng)](https://github.com/WASasquatch/comfyui-plugins) by @WASasquatch

- [ComfyUI tag on CivitAI (eng)](https://civitai.com/tag/comfyui)
