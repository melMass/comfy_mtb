# Installation
- [Installation](#installation)
    - [Dependencies](#dependencies)
      - [Custom virtualenv (I use this mainly)](#custom-virtualenv-i-use-this-mainly)
      - [Comfy-portable / standalone (from ComfyUI releases)](#comfy-portable--standalone-from-comfyui-releases)
      - [Google Colab](#google-colab)
    - [Models Download](#models-download)
    - [Web Extensions](#web-extensions)


### Dependencies

#### Custom virtualenv (I use this mainly)

1. Make sure you are in the Python environment you use for ComfyUI.
2. Install the required dependencies by running the following command:
  ```bash
  pip install -r comfy_mtb/requirements.txt
  ```
#### Comfy-portable / standalone (from ComfyUI releases)

If you use the `python-embeded` from ComfyUI standalone then you are not able to pip install dependencies with binaries when they don't have wheels, in this case check the last [release](https://github.com/melMass/comfy_mtb/releases) there is a bundle for linux and windows with prebuilt wheels (only the ones that require building from source), check [this issue (#1)](https://github.com/melMass/comfy_mtb/issues/1) for more info.
![image](https://github.com/melMass/comfy_mtb/assets/7041726/2934fa14-3725-427c-8b9e-2b4f60ba1b7b)


#### Google Colab

Add a new code cell just after the **Run ComfyUI with localtunnel (Recommended Way)** header (before the code cell)
![preview of where to add it on colab](https://github.com/melMass/comfy_mtb/assets/7041726/35df2ef1-14f9-44cd-aa65-353829188cd7)


```python
# download the nodes
!git clone --recursive https://github.com/melMass/comfy_mtb.git custom_nodes/comfy_mtb

# download all models
!python custom_nodes/comfy_mtb/scripts/download_models.py -y

# install the dependencies
!pip install -r custom_nodes/comfy_mtb/requirements.txt -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
```
If after running this, colab complains about needing to restart runtime, do it, and then do not rerun earlier cells, just the one to run the localtunnel. (you might have to add a cell with `%cd ComfyUI` first...)


> **Note**:
> If you don't need all models, remove the `-y` as collab actually supportd user input: ![image](https://github.com/melMass/comfy_mtb/assets/7041726/40fc3602-f1d4-432a-98fd-ce2240f5ad06)

> **Preview**
> ![image](https://github.com/melMass/comfy_mtb/assets/7041726/b5b2b2d9-f1e8-4c43-b1db-7dfc5e07be86)



### Models Download
Some nodes require extra models to be downloaded, you can interactively do it using the same python environment as above:
```bash
python scripts/download_models.py
```

then follow the prompt or just press enter to download every models.

> **Note**
> You can use the following to download all models without prompt:
  ```bash
  python scripts/download_models.py -y
  ```

### Web Extensions

On first run the script [tries to symlink](https://github.com/melMass/comfy_mtb/blob/d982b69a58c05ccead9c49370764beaa4549992a/__init__.py#L45-L61) the [web extensions](https://github.com/melMass/comfy_mtb/tree/main/web) to your comfy `web/extensions` folder. In case it fails you can manually copy the mtb folder to `ComfyUI/web/extensions` it only provides a color widget for now shared by a few nodes:  

<img alt="color widget preview" src="https://github.com/melMass/comfy_mtb/assets/7041726/cff7e66a-4cc4-4866-b35b-10af0bb2d110" width=450>
