# MTB Nodes

[![embedded test](https://github.com/melMass/comfy_mtb/actions/workflows/test_embedded.yml/badge.svg)](https://github.com/melMass/comfy_mtb/actions/workflows/test_embedded.yml)

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

- [Node List](#node-list)
  - [bbox](#bbox)
  - [colors](#colors)
  - [face detection / swapping](#face-detection--swapping)
  - [image interpolation (animation)](#image-interpolation-animation)
  - [image ops](#image-ops)
  - [latent utils](#latent-utils)
  - [misc utils](#misc-utils)
  - [textures](#textures)
- [Comfy Resources](#comfy-resources)


# Node List

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
  <img src="https://github.com/melMass/comfy_mtb/assets/7041726/7c20ac83-31ff-40ea-a1a0-06c2acefb2ef" width=345/>

## face detection / swapping
- `Face Swap`: Face swap using deepinsight/insightface models (this node used to be called `Roop` in early versions, it does the same, roop is *just* an app that uses those model)
  > **Note**
  > The face index allow you to choose which face to replace as you can see here:  
  <img src="https://github.com/melMass/comfy_mtb/assets/7041726/2e9d6066-c466-4a01-bd6c-315f7f1e8b42" width=320/>
- `Load Face Swap Model`: Load an insightface model for face swapping
- `Restore Face`: Using [GFPGan](https://github.com/TencentARC/GFPGAN) to restore faces, works great in conjunction with `Face Swap` and supports Comfy native upscalers for the `bg_upscaler`
  
## image interpolation (animation)
- `Load Film Model`: Loads a [FILM](https://github.com/google-research/frame-interpolation) model
- `Film Interpolation`: Process input frames using [FILM](https://github.com/google-research/frame-interpolation)  
  <img src="https://github.com/melMass/comfy_mtb/assets/7041726/3afd1647-6634-4b92-a34b-51432e6a9834" width=400/>
- `Export to Prores (experimental)`: Exports the input frames to a ProRes 4444 mov file. This is using ffmpeg stdin to send raw numpy arrays, used with `Film Interpolation` and very simple for now but could be expanded upon.

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


## misc utils
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

## textures

- `DeepBump`: Normal & height maps generation from single pictures

# Comfy Resources

**Guides**:
- [Official Examples (eng)](https://comfyanonymous.github.io/ComfyUI_examples/)
- [ComfyUI Community Manual (eng)](https://blenderneko.github.io/ComfyUI-docs/) by @BlenderNeko
  
- [Tomoaki's personal Wiki (jap)](https://comfyui.creamlab.net/guides/) by @tjhayasaka

**Extensions and Custom Nodes**:
- [Plugins for Comfy List (eng)](https://github.com/WASasquatch/comfyui-plugins) by @WASasquatch

- [ComfyUI tag on CivitAI (eng)](https://civitai.com/tag/comfyui)
