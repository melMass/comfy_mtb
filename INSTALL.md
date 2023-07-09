# Installation
- [Installation](#installation)
    - [Dependencies](#dependencies)
    - [Models Download](#models-download)
    - [Web Extensions](#web-extensions)


### Dependencies

1. Make sure you are in the Python environment you use for ComfyUI.
2. Install the required dependencies by running the following command:
  ```bash
  pip install -r comfy_mtb/requirements.txt
  ```

> **Note**
> If you use the `python-embed` mode of comfyui (the standalone release) then you might not be able to pip install
> dependencies that don't have wheels, in this case check the last [release](https://github.com/melMass/comfy_mtb/releases) there is a bundle for linux and windows, check [this issue (#1)](https://github.com/melMass/comfy_mtb/issues/1) for more info.

### Models Download
Some nodes require extra models to be downloaded, you can interactively do it using the same python environment as above:
```bash
python scripts/download_models.py
```

then follow the prompt or just press enter to download every models.

If you use Comfy from withing a notebook/colab, you can use the following to download all models without prompt:

```bash
python scripts/download_models.py -y
```


### Web Extensions

On first run the script [tries to symlink](https://github.com/melMass/comfy_mtb/blob/d982b69a58c05ccead9c49370764beaa4549992a/__init__.py#L45-L61) the [web extensions](https://github.com/melMass/comfy_mtb/tree/main/web) to your comfy `web/extensions` folder. In case it fails you can manually copy the mtb folder to `ComfyUI/web/extensions` it only provides a color widget for now shared by a few nodes:  

<img alt="color widget preview" src="https://github.com/melMass/comfy_mtb/assets/7041726/cff7e66a-4cc4-4866-b35b-10af0bb2d110" width=320>