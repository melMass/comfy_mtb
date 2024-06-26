# Installation
- [Installation](#installation)
  - [Automatic Install (Recommended)](#automatic-install-recommended)
    - [ComfyUI Manager](#comfyui-manager)
    - [Virtual Env](#virtual-env)
  - [Models Download](#models-download)
  - [Old installation method (MANUAL)](#old-installation-method-manual)
    - [Dependencies](#dependencies)

## Automatic Install (Recommended)

### ComfyUI Manager

As of version 0.1.0, this extension is meant to be installed with the [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager), which helps a lot with handling the various install issues faced by various environments.

### Virtual Env
There is also an experimental one liner install using the following command from ComfyUI's root. It will download the code, install the dependencies and run the install script:

```bash
curl -sSL "https://raw.githubusercontent.com/username/repo/main/install.py" | python3 -
```

## Models Download
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


## Old installation method (MANUAL)
### Dependencies
<details><summary><h4>Custom Virtualenv (I use this mainly)</h4></summary>
    
1. Make sure you are in the Python environment you use for ComfyUI.
2. Install the required dependencies by running the following command:
  ```bash
  pip install -r comfy_mtb/requirements.txt
  ```

</details>

<details><summary><h4>Comfy-portable / standalone (from ComfyUI releases)</h4></summary>
    
If you use the `python-embeded` from ComfyUI standalone then you are not able to pip install dependencies with binaries when they don't have wheels, in this case check the last [release](https://github.com/melMass/comfy_mtb/releases) there is a bundle for linux and windows with prebuilt wheels (only the ones that require building from source), check [this issue (#1)](https://github.com/melMass/comfy_mtb/issues/1) for more info.
![image](https://github.com/melMass/comfy_mtb/assets/7041726/2934fa14-3725-427c-8b9e-2b4f60ba1b7b)



</details>

<details><summary><h4>Google Colab</h4></summary>

Add a new code cell just after the **Run ComfyUI with localtunnel (Recommended Way)** header (before the code cell)
![preview of where to add it on colab](https://github.com/melMass/comfy_mtb/assets/7041726/35df2ef1-14f9-44cd-aa65-353829188cd7)


```python
# download the nodes
!git clone --recursive https://github.com/melMass/comfy_mtb.git custom_nodes/comfy_mtb

# download all models
!python custom_nodes/comfy_mtb/scripts/download_models.py -y

# install the dependencies
!pip install -r custom_nodes/comfy_mtb/reqs.txt -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
```
If after running this, colab complains about needing to restart runtime, do it, and then do not rerun earlier cells, just the one to run the localtunnel. (you might have to add a cell with `%cd ComfyUI` first...)


> **Note**:
> If you don't need all models, remove the `-y` as collab actually supports user input: ![image](https://github.com/melMass/comfy_mtb/assets/7041726/40fc3602-f1d4-432a-98fd-ce2240f5ad06)

> **Preview**
> ![image](https://github.com/melMass/comfy_mtb/assets/7041726/b5b2b2d9-f1e8-4c43-b1db-7dfc5e07be86)

</details>

