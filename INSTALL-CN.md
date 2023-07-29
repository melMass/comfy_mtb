# 安装
- [安装](#安装)
    - [自动安装（推荐）](#自动安装推荐)
    - [ComfyUI 管理器](#comfyui-管理器)
    - [虚拟环境](#虚拟环境)
  - [模型下载](#模型下载)
      - [网络扩展](#网络扩展)
    - [旧的安装方法 (MANUAL)](#旧的安装方法-manual)
    - [依赖关系](#依赖关系)
### 自动安装（推荐）

### ComfyUI 管理器

从 0.1.0 版开始，该扩展将使用 [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager) 进行安装，这对处理各种环境下的各种安装问题大有帮助。

### 虚拟环境
还有一种试验性的单行安装方法，即在 ComfyUI 根目录下使用以下命令进行安装。它将下载代码、安装依赖项并运行安装脚本：


```bash
curl -sSL "https://raw.githubusercontent.com/username/repo/main/install.py" | python3 -
```

## 模型下载
某些节点需要下载额外的模型，您可以使用与上述相同的 python 环境以交互方式完成下载：

```bash
python scripts/download_models.py
```

然后根据提示或直接按回车键下载每个模型。

> **Note**
> 您可以使用以下方法下载所有型号，无需提示：
  ```bash
  python scripts/download_models.py -y
  ```

#### 网络扩展

首次运行时，脚本会尝试将 [网络扩展](https://github.com/melMass/comfy_mtb/tree/main/web)链接到你的 "web/extensions "文件夹，[请参阅](https://github.com/melMass/comfy_mtb/blob/d982b69a58c05ccead9c49370764beaa4549992a/__init__.py#L45-L61)。

<img alt="color widget preview" src="https://github.com/melMass/comfy_mtb/assets/7041726/cff7e66a-4cc4-4866-b35b-10af0bb2d110" width=450>

### 旧的安装方法 (MANUAL)
### 依赖关系
<details><summary><h4>Custom Virtualenv（我主要用这个）</h4></summary
    
1. 确保您处于用于 ComfyUI 的 Python 环境中。
2. 运行以下命令安装所需的依赖项：
  ```bash
  pip install -r comfy_mtb/reqs.txt
  ```

</details>

<details><summary><h4>Comfy 便携式/单机版（来自 ComfyUI 版本）</h4></summary>

如果您使用 ComfyUI 单机版中的 `python-embeded `，那么当二进制文件没有轮子时，您就无法使用 pip 安装二进制文件的依赖项，在这种情况下，请查看最近的 [发布](https://github.com/melMass/comfy_mtb/releases)，那里有一个预编译轮子的 linux 和 windows 捆绑包（只有那些需要从源代码编译的轮子），请查看 [此问题 (#1)](https://github.com/melMass/comfy_mtb/issues/1) 以获取更多信息。
![image](https://github.com/melMass/comfy_mtb/assets/7041726/2934fa14-3725-427c-8b9e-2b4f60ba1b7b)


</details>

<details><summary><h4>Google Colab</h4></summary>

在 **Run ComfyUI with localtunnel (Recommended Way)** 标题之后（代码单元格之前）添加一个新的代码单元格

![preview of where to add it on colab](https://github.com/melMass/comfy_mtb/assets/7041726/35df2ef1-14f9-44cd-aa65-353829188cd7)


```python
# download the nodes
!git clone --recursive https://github.com/melMass/comfy_mtb.git custom_nodes/comfy_mtb

# download all models
!python custom_nodes/comfy_mtb/scripts/download_models.py -y

# install the dependencies
!pip install -r custom_nodes/comfy_mtb/reqs.txt -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
```

如果运行后 colab 抱怨需要重新启动运行时，请重新启动，然后不要重新运行之前的单元格，只运行运行本地隧道的单元格。(可能需要先添加一个包含 `%cd ComfyUI` 的单元格）


> **Note**:
> If you don't need all models, remove the `-y` as collab actually supports user input: ![image](https://github.com/melMass/comfy_mtb/assets/7041726/40fc3602-f1d4-432a-98fd-ce2240f5ad06)

> **Preview**
> ![image](https://github.com/melMass/comfy_mtb/assets/7041726/b5b2b2d9-f1e8-4c43-b1db-7dfc5e07be86)

</details>

