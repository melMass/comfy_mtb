# 安装

- [安装](#安装)
    - [依赖项](#依赖项)
      - [自定义虚拟环境（我主要使用此方法）](#自定义虚拟环境我主要使用此方法)
      - [Comfy-portable / 独立安装（来自 ComfyUI 发布）](#comfy-portable--独立安装来自-comfyui-发布)
      - [Google Colab](#google-colab)
    - [模型下载](#模型下载)
    - [Web 扩展](#web-扩展)


### 依赖项

#### 自定义虚拟环境（我主要使用此方法）

1. 确保您处于用于 ComfyUI 的 Python 环境中。
2. 运行以下命令安装所需的依赖项：
   ```bash
   pip install -r comfy_mtb/requirements.txt
   ```

#### Comfy-portable / 独立安装（来自 ComfyUI 发布）

如果您使用的是 ComfyUI 独立版的 `python-embeded`，则无法使用二进制文件进行依赖项的 pip 安装，特别是当它们没有预编译的 wheels 时。在这种情况下，请查看最新的[发布版](https://github.com/melMass/comfy_mtb/releases)，其中包含了为 Linux 和 Windows 构建的预编译 wheels（仅对需要从源代码构建的部分）。有关更多信息，请参阅[此问题（#1）](https://github.com/melMass/comfy_mtb/issues/1)。
![image](https://github.com/melMass/comfy_mtb/assets/7041726/2934fa14-3725-427c-8b9e-2b4f60ba1b7b)


#### Google Colab

在 **运行 ComfyUI with localtunnel（推荐方式）** 标题下方添加一个新的代码单元格（在代码单元格之前）。
![在 Colab 中添加位置的预览](https://github.com/melMass/comfy_mtb/assets/7041726/35df2ef1-14f9-44cd-aa65-353829188cd7)


```python
# 下载节点
!git clone --recursive https://github.com/melMass/comfy_mtb.git custom_nodes/comfy_mtb

# 下载所有模型
!python custom_nodes/comfy_mtb/scripts/download_models.py -y

# 安装依赖项
!pip install -r custom_nodes/comfy_mtb/requirements.txt -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
```
如果运行后，Colab 抱怨需要重新启动运行时，请重新启动，并且不要重新运行之前的单元格，只需运行一个用于运行 localtunnel 的单元格即可（您可能需要先添加一个 `%cd ComfyUI` 的单元格...）。

> **注意**：
> 如果您不需要所有模型，请将 `-y` 删除，因为 Colab 实际上支持用户输入： ![image](https://github.com/melMass/comfy_mtb/assets/7041726/40fc3602-f1d4-432a-98fd-ce2240f5ad06)

> **预览**
> ![image](https://github.com/melMass/comfy_mtb/assets/7041726/b5b2b2d9-f1e8-4c43-b1db-7dfc5e07be86)



### 模型下载
某些节点需要下载额外的模型，您可以使用与上述相同的 Python 环境进行交互式下载：
```bash
python scripts/download_models.py
```

然后按照提示操作或只需按 Enter 键下载所有模型。

> **注意**
> 您可以使用以下命令无需提示下载所有模型：
> ```bash
> python scripts/download_models.py -y
> ```

### Web 扩展

第一次运行脚本时，会[尝试创建符号链接](https://github.com/melMass/comfy_mtb/blob/d982b69a58c05ccead9c49370764beaa4549992a/__init__.py#L45-L61)，将[Web 扩展](https://github.com/melMass/comfy_mtb/tree/main/web)链接到 Comfy 的 `web/extensions` 文件夹。如果失败，您可以手动将 `mtb` 文件夹复制到 `ComfyUI/web/extensions`，目前它只提供了一个颜色小部件，由几个节点共享：

<img alt="color widget preview" src="https://github.com/melMass/comfy_mtb/assets/7041726/cff7e66a-4cc4-4866-b35b-10af0bb2d110" width=450>