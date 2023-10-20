# MTB Nodes
[![embedded test](https://github.com/melMass/comfy_mtb/actions/workflows/test_embedded.yml/badge.svg)](https://github.com/melMass/comfy_mtb/actions/workflows/test_embedded.yml)

![home](https://repository-images.githubusercontent.com/649047066/a3eef9a7-20dd-4ef9-b839-884502d4e873)

<!-- omit in toc -->

**Translated Readme (using DeepTranslate, PRs are welcome)**:  
![image](https://github.com/melMass/comfy_mtb/assets/7041726/f8429c14-3521-4e28-82a3-863d781976c0)
[日本語による説明](./README-JP.md)  
![image](https://github.com/melMass/comfy_mtb/assets/7041726/d5cc1fdd-2820-4a5c-b2d7-482f1c222063)
[中文说明](./README-CN.md)

<a href="https://www.buymeacoffee.com/melmass" target="_blank"><img src="https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png" alt="Buy Me A Coffee" style="height: 32px !important;width: 140px !important;box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;-webkit-box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;" ></a>

[**Install Guide**](./INSTALL.md) | [**Examples**](https://github.com/melMass/comfy_mtb/wiki/Examples)

Welcome to the MTB Nodes project! This codebase is open for you to explore and utilize as you wish. Its primary purpose is to build proof-of-concepts (POCs) for implementation in [MLOPs](https://github.com/Bismuth-Consultancy-BV/MLOPs). Many nodes in this project are inspired by existing community contributions or built-in functionalities.

Before proceeding, please be aware of the licenses associated with certain libraries used in this project. For example, the `deepbump` library is licensed under [GPLv3](https://github.com/HugoTini/DeepBump/blob/master/LICENSE).

- [Web Extensions](#web-extensions)
- [Node List](#node-list)
  - [Animation](#animation)
  - [bbox](#bbox)
  - [colors](#colors)
  - [image ops](#image-ops)
  - [latent utils](#latent-utils)
  - [textures](#textures)
  - [misc utils](#misc-utils)
  - [Optional nodes](#optional-nodes)
    - [face detection / swapping](#face-detection--swapping)
    - [image interpolation (animation)](#image-interpolation-animation)
- [Comfy Resources](#comfy-resources)

# Web Extensions
mtb add a few widgets like `COLOR`

<img alt="color widget preview" src="https://github.com/melMass/comfy_mtb/assets/7041726/cff7e66a-4cc4-4866-b35b-10af0bb2d110" width=450>

A few nodes have the concept of "dynamic" inputs:  
<img alt="dynamic inputs" width=450  src="https://github.com/melMass/comfy_mtb/assets/7041726/10b3976e-b212-4968-91eb-f34c02bb80c3" />


# Node List

## Animation
- `Animation Builder`: Convenient way to manage basic animation maths at the core of many of my workflows (both worflows for the following GIFs are in the [examples](https://github.com/melMass/comfy_mtb/wiki/Examples))  

  **[Example lerping two conditions (blue car -> yellow car)](https://github.com/melMass/comfy_mtb/blob/main/examples/03-animation_builder-condition-lerp.json)**
  
  <img width=300 src="https://user-images.githubusercontent.com/7041726/260258970-d6d66d96-fb34-40d0-9038-cbabf0714c5d.gif"/>  
  

  **[Example using image transforms a feedback for a fake deforum effect](https://github.com/melMass/comfy_mtb/blob/main/examples/04-animation_builder-deforum.json)**
  
  <img width=300 src="https://user-images.githubusercontent.com/7041726/260261504-303a1037-60d3-4b31-a589-b15d549752f6.gif"/>
  
- `Batch Float`: Generates a batch of float values with interpolation.
- `Batch Shape`: Generates a batch of 2D shapes with optional shading (experimental).
- `Batch Transform`: Transform a batch of images using a batch of keyframes.  
  <img width=400 src="https://github.com/melMass/comfy_mtb/assets/7041726/3f217de1-79aa-49b0-a66a-35cf29dd8f01"/>
- `Export With Ffmpeg`: Export with FFmpeg, it used to be export to Proress and is still tailored for YUV
- `Fit Number` : Fit the input float using a source and target range, you can also control the interpolation curve from a list of presets (default to linear)
  
## bbox
- `Bounding Box`: BBox constructor (custom type),
- `BBox From Mask`: From a mask extract the bounding box
- `Crop`: Crop image from BBox
- `Uncrop`: Uncrop image from BBox

## colors
- `Colored Image`: Constant color image of given size
- `RGB to HSV`: -,
- `HSV to RGB`: -,
- `Color Correct`: Basic color correction tools  
  <img src="https://github.com/melMass/comfy_mtb/assets/7041726/7c20ac83-31ff-40ea-a1a0-06c2acefb2ef" width=400/>

## image ops
- `Blur`: Blur an image using a Gaussian filter.
- `Deglaze Image`: taken from [FN16](https://github.com/Fannovel16/FN16-ComfyUI-nodes/blob/main/DeglazeImage.py),
- `Denoise`: Denoise the input image,
- `Image Compare`: Compare two images and return a difference image
- `Image Premultiply`: Premultiply image with mask
- `Image Remove Background Rembg`: [RemBG](https://github.com/danielgatis/rembg) powered background removal.  
  <img src="https://github.com/melMass/comfy_mtb/assets/7041726/e69253b4-c03c-45e9-92b5-aa46fb887be8" width=320/>
- `Image Resize Factor`: Extracted mostly from [WAS Node Suite](https://github.com/WASasquatch/was-node-suite-comfyui), with a few edits (most notably multiple image support) and less features.
- `Mask To Image`: Converts a mask (alpha) to an RGB image with a color and background
- `Save Image Grid`: Save all the images in the input batch as a grid of images.

## latent utils
- `Latent Lerp`: Linear interpolation (blend) between two latent 

## textures
- `Model Patch Seamless`: Use the [seamless diffusion "hack"](https://gitlab.com/-/snippets/2395088) to patch any model to infere seamless images, check the [examples](https://github.com/melMass/comfy_mtb/wiki/Examples) to see how to use all those textures node together  
  <img width=500 src="https://user-images.githubusercontent.com/7041726/272970506-9db516b5-45d2-4389-b904-b3a94660f24c.png"/>
- `DeepBump`: Normal & height maps generation from single pictures  
  <img width=500 src="https://user-images.githubusercontent.com/7041726/272970715-7e4477f6-8e18-4839-9864-83d07d6690a1.png"/>
- `Image Tile Offset`: Mimics an old photoshop technique to check for seamless textures by offsetting tiles of the image.  
  <img width=600 src="https://github.com/melMass/comfy_mtb/assets/7041726/cbcc51fb-922f-433f-acf1-c6c6c2a7ffc4" />

## misc utils
- `Any To String`: Tries to take any input and convert it to a string.
- `Concat Images`: Takes two image stream and merge them as a batch of images supported by other Comfy pipelines.
- `Image Resize Factor`: **Deprecated**, I since discovered the builtin image resize.
- `Text To Image`: Utils to convert text to image using a font
- `Styles Loader`: Load csv files and populate a dropdown from the rows (à la A111)  
  <img src="https://github.com/melMass/comfy_mtb/assets/7041726/02fe3211-18ee-4e54-a029-931388f5fde8" width=320/>
- `Smart Step`: A very basic node to get step percent to use in KSampler advanced,
- `Qr Code`: Basic QR Code generator
- `Save Tensors`: Debug node that will probably be removed in the future
- `Int to Number`: Supplement for WASSuite number nodes
- `Smart Step`: A very basic tool to control the steps (start/stop) of the `KAdvancedSampler` using percentage
- `Load Image From Url`: Load an image from the given URL


## Optional nodes

These nodes are still bundled in mtb, but moving forward (>0.2.0) they won't
be setup by the install script and their dependencies won't install either.
The reason is mostly that they all have a better alternatives available and tensorflow on windows was not a fun experience and since Python 3.11 not an experience at all.

For linux and mac users though these nodes didn't cause any issue and I personally still use them, these are the extra requirements needed:

```console
.venv/python -m pip install tensorflow facexlib insightface basicsr
```

### face detection / swapping
> **Warning**
> Those nodes were among the first to be implemented they do work, but on windows the installation is still not properly handled for everyone  
> As alternatives you can use [reactor](https://github.com/Gourieff/comfyui-reactor-node) for face swap and [facerestore](https://github.com/Haidra-Org/hordelib/tree/main/hordelib/nodes/facerestore) for restoration  
> You can check [this video](https://www.youtube.com/watch?v=FShlpMxbU0E) for a tutorial by Ferniclestix using these alternatives  

- `Face Swap`: Face swap using deepinsight/insightface models (this node used to be called `Roop` in early versions, it does the same, roop is *just* an app that uses those model)  
  <img  width=320 src="https://user-images.githubusercontent.com/7041726/260261217-54e33446-183f-4dda-88b3-d38a1e6de980.gif"/>
- `Load Face Swap Model`: Load an insightface model for face swapping
- `Restore Face`: Using [GFPGan](https://github.com/TencentARC/GFPGAN) to restore faces, works great in conjunction with `Face Swap` and supports Comfy native upscalers for the `bg_upscaler`
  
### image interpolation (animation)
> **Warning**
> The FILM nodes will be deprecated at some point after 0.2.0, [Fannovel16](https://github.com/Fannovel16/ComfyUI-Frame-Interpolation)'s interpolation nodes implement it and they rely on a pytorch implementation of FILM
> which solves the issues related to the ones included in mtb. They will probably remain available if your system meet the requirements and ignored otherwise.

<details><summary>Why?</summary>
  
> **Windows only issue**: This requires tensorflow-gpu that is unfortunately not a thing anymore on Windows since 2.10.1 (unless you use a complex WSL passthrough setup but it's still not "Windows")  
> Using this old version is quite clunky and require some patching that install.py does automatically, but the main issue is that no wheels are available for python > 3.10
> Comfy-nightly is already using Python 11 so installing this old tf version won't work there.  
> You can in any case install the normal up to date tensorflow but that will run on CPU and is much MUCH slower for FILM inference.
</details>

- `Load Film Model`: Loads a [FILM](https://github.com/google-research/frame-interpolation) model
- `Film Interpolation`: Process input frames using [FILM](https://github.com/google-research/frame-interpolation)  
  <img width=400 src="https://github.com/melMass/comfy_mtb/assets/7041726/3afd1647-6634-4b92-a34b-51432e6a9834"/>  
  <img width=400 src="https://user-images.githubusercontent.com/7041726/260259079-c0f04a63-960c-43a7-ba78-a45cd5ac7514.gif"/>
- `Export to Prores (experimental)`: Exports the input frames to a ProRes 4444 mov file. This is using ffmpeg stdin to send raw numpy arrays, used with `Film Interpolation` and very simple for now but could be expanded upon.

# Comfy Resources

**Misc**

- [Slick ComfyUI by NoCrypt](https://colab.research.google.com/drive/1ZMvLWEiYITmBJngtqeIQToeNuiydwI0z#scrollTo=1fWMaexXS188): A colab notebook with batteries included!

**Guides**:
- [Official Examples (eng)](https://comfyanonymous.github.io/ComfyUI_examples/)
- [ComfyUI Community Manual (eng)](https://blenderneko.github.io/ComfyUI-docs/) by @BlenderNeko
  
- [Tomoaki's personal Wiki (jap)](https://comfyui.creamlab.net/guides/) by @tjhayasaka

**Extensions and Custom Nodes**:
- [Plugins for Comfy List (eng)](https://github.com/WASasquatch/comfyui-plugins) by @WASasquatch

- [ComfyUI tag on CivitAI (eng)](https://civitai.com/tag/comfyui)
