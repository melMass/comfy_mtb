# MTB Nodes

<a href="https://www.buymeacoffee.com/melmass" target="_blank"><img src="https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png" alt="Buy Me A Coffee" style="height: 32px !important;width: 140px !important;box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;-webkit-box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;" ></a>

[** 安装指南**](./INSTALL-CN.md) | [** 示例**](https://github.com/melMass/comfy_mtb/wiki/Examples)

欢迎使用 MTB Nodes 项目！这个代码库是开放的，您可以自由地探索和利用。它的主要目的是构建用于 [MLOPs](https://github.com/Bismuth-Consultancy-BV/MLOPs) 中的概念验证（POCs）。该项目中的许多节点都是受到现有社区贡献或内置功能的启发而创建的。

在继续之前，请注意与此项目中使用的某些库相关的许可证。例如，`deepbump` 库采用 [GPLv3](https://github.com/HugoTini/DeepBump/blob/master/LICENSE) 许可证。

- [节点列表](#节点列表)
  - [bbox](#bbox)
  - [colors](#colors)
  - [人脸检测/交换](#人脸检测交换)
  - [图像插值（动画）](#图像插值动画)
  - [图像操作](#图像操作)
  - [潜在变量工具](#潜在变量工具)
  - [其他工具](#其他工具)
  - [纹理](#纹理)
- [Comfy 资源](#comfy-资源)




# 节点列表

## bbox
- `Bounding Box`: BBox 构造函数（自定义类型）
- `BBox From Mask`: 从遮罩中提取边界框
- `Crop`: 根据边界框裁剪图像
- `Uncrop`: 根据边界框还原图像

## colors
- `Colored Image`: 给定尺寸的纯色图像
- `RGB to HSV`: -
- `HSV to RGB`: -
- `Color Correct`: 基本颜色校正工具  
  <img src="https://github.com/melMass/comfy_mtb/assets/7041726/7c20ac83-31ff-40ea-a1a0-06c2acefb2ef" width=345/>

## 人脸检测/交换
- `Face Swap`: 使用 deepinsight/insightface 模型进行人脸交换（该节点在早期版本中称为 `Roop`，功能相同，`Roop` 只是使用这些模型的应用程序）
  > **注意**
  > 人脸索引允许您选择要替换的人脸，如下所示：
  <img src="https://github.com/melMass/comfy_mtb/assets/7041726/2e9d6066-c466-4a01-bd6c-315f7f1e8b42" width=320/>
- `Load Face Swap Model`: 加载 insightface 模型用于人脸交换
- `Restore Face`: 使用 [GFPGan](https://github.com/TencentARC/GFPGAN) 还原人脸，与 `Face Swap` 配合使用效果很好，并支持 `bg_upscaler` 的 Comfy 原生放大器
  
## 图像插值（动画）
- `Load Film Model`: 加载 [FILM](https://github.com/google-research/frame-interpolation) 模型
- `Film Interpolation`: 使用 [FILM](https://github.com/google-research/frame-interpolation) 处理输入帧  
  <img src="https://github.com/melMass/comfy_mtb/assets/7041726/3afd1647-6634-4b92-a34b-51432e6a9834" width=400/>
- `Export to Prores (experimental)`: 将输入帧导出为 ProRes 4444 mov 文件。这使用 ffmpeg stdin 发送原始的 NumPy 数组，与 `Film Interpolation` 一起使用，目前很简单，但可以进一步扩展。

## 图像操作
- `Blur`: 使用高斯滤波器对图像进行模糊处理。
- `Deglaze Image`: 从 [FN16](https://github.com/Fannovel16/FN16-ComfyUI-nodes/blob/main/DeglazeImage.py) 中提取
- `Denoise`: 对输入图像进行降噪处理
- `Image Compare`: 比较两个图像并返回差异图像
- `Image Premultiply`: 使用掩码对图像进行预乘处理
- `Image Remove Background Rembg`: 使用 [RemBG](https://github.com/danielgatis/rembg) 进行背景去除  
  <img src="https://github.com/melMass/comfy_mtb/assets/7041726/e69253b4-c03c-45e9-92b5-aa46fb887be8" width=320/>
- `Image Resize Factor`: 大部分提取自 [WAS Node Suite](https://github.com/WASasquatch/was-node-suite-comfyui)，经过一些编辑（特别是支持多个图像）和较少的功能。
- `Mask To Image`: 将遮罩（Alpha）转换为带有颜色和背景的 RGB 图像
- `Save Image Grid`: 将输入批次中的所有图像保存为图像网格。

## 潜在变量工具
- `Latent Lerp`: 两个潜在变量之间的线性插值（混合）


## 其他工具
- `Concat Images`: 接受两个图像流，并将它们合并为其他 Comfy 管道支持的图像批次。
- `Image Resize Factor`: **已弃用**，因为我后来发现了内

置的图像调整大小功能。
- `Text To Image`: 使用字体将文本转换为图像的工具
- `Styles Loader`: 加载 csv 文件并从行中填充下拉列表（类似于 A111）  
  <img src="https://github.com/melMass/comfy_mtb/assets/7041726/02fe3211-18ee-4e54-a029-931388f5fde8" width=320/>
- `Smart Step`: 一个非常基本的节点，用于获取在 KSampler 高级中使用的步骤百分比
- `Qr Code`: 基本的 QR Code 生成器
- `Save Tensors`: 调试节点，将来可能会被删除
- `Int to Number`: 用于 WASSuite 数字节点的补充
- `Smart Step`: 使用百分比来控制 `KAdvancedSampler` 的步骤（开始/停止）

## 纹理

- `DeepBump`: 从单张图片生成法线图和高度图

# Comfy 资源

**指南**:
- [官方示例（英文）](https://comfyanonymous.github.io/ComfyUI_examples/)
- @BlenderNeko 的[ComfyUI 社区手册（英文）](https://blenderneko.github.io/ComfyUI-docs/)

- @tjhayasaka 的[Tomoaki 个人 Wiki（日文）](https://comfyui.creamlab.net/guides/)

**扩展和自定义节点**:
- @WASasquatch 的[Comfy 列表插件（英文）](https://github.com/WASasquatch/comfyui-plugins)

- [CivitAI 上的 ComfyUI 标签（英文）](https://civitai.com/tag/comfyui)
