# MTB Nodes

<a href="https://www.buymeacoffee.com/melmass" target="_blank"><img src="https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png" alt="Buy Me A Coffee" style="height: 32px !important;width: 140px !important;box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;-webkit-box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;" ></a>

[**インストールガイド**](./INSTALL-JP.md) | [**サンプル**](https://github.com/melMass/comfy_mtb/wiki/Examples)

MTB Nodesプロジェクトへようこそ！このコードベースは、自由に探索し、利用することができます。主な目的は、[MLOPs](https://github.com/Bismuth-Consultancy-BV/MLOPs)の実装のための概念実証（POC）を構築することです。このプロジェクトの多くのノードは、既存のコミュニティの貢献や組み込みの機能に触発されています。

続行する前に、このプロジェクトで使用されている特定のライブラリに関連するライセンスに注意してください。たとえば、「deepbump」ライブラリは、[GPLv3](https://github.com/HugoTini/DeepBump/blob/master/LICENSE)の下でライセンスされています。

- [ノードリスト](#ノードリスト)
  - [bbox](#bbox)
  - [colors](#colors)
  - [顔検出 / スワッピング](#顔検出--スワッピング)
  - [画像補間（アニメーション）](#画像補間アニメーション)
  - [画像操作](#画像操作)
  - [潜在的なユーティリティ](#潜在的なユーティリティ)
  - [その他のユーティリティ](#その他のユーティリティ)
  - [テクスチャ](#テクスチャ)
- [Comfyリソース](#comfyリソース)


# ノードリスト

## bbox
- `Bounding Box`: BBoxコンストラクタ（カスタムタイプ）
- `BBox From Mask`: マスクからバウンディングボックスを抽出
- `Crop`: BBoxから画像を切り抜く
- `Uncrop`: BBoxから画像を元に戻す

## colors
- `Colored Image`: 指定されたサイズの一定の色の画像
- `RGB to HSV`: -
- `HSV to RGB`: -
- `Color Correct`: 基本的なカラーコレクションツール  
  <img src="https://github.com/melMass/comfy_mtb/assets/7041726/7c20ac83-31ff-40ea-a1a0-06c2acefb2ef" width=345/>

## 顔検出 / スワッピング
- `Face Swap`: deepinsight/insightfaceモデルを使用した顔の入れ替え（このノードは初期バージョンでは「Roop」と呼ばれていましたが、同じ機能を提供します。Roopは単にこれらのモデルを使用するアプリです）
  > **注意**
  > 顔のインデックスを使用して置き換える顔を選択できます。以下を参照してください：
  <img src="https://github.com/melMass/comfy_mtb/assets/7041726/2e9d6066-c466-4a01-bd6c-315f7f1e8b42" width=320/>
- `Load Face Swap Model`: 顔の交換のためのinsightfaceモデルを読み込む
- `Restore Face`: [GFPGan](https://github.com/TencentARC/GFPGAN)を使用して顔を復元し、`Face Swap`と組み合わせて使用すると非常に効果的であり、`bg_upscaler`のComfyネイティブアップスケーラーもサポートしています。

## 画像補間（アニメーション）
- `Load Film Model`: [FILM](https://github.com/google-research/frame-interpolation)モデルを読み込む
- `Film Interpolation`: [FILM](https://github.com/google-research/frame-interpolation)を使用して入力フレームを処理する  
  <img src="https://github.com/melMass/comfy_mtb/assets/7041726/3afd1647-6634-4b92-a34b-51432e6a9834" width=400/>
- `Export to Prores (experimental)`: 入力フレームをProRes 4444 movファイルにエクスポートします。これは現在は単純なものですが、`Film Interpolation`と組み合わせて使用するためのffmpegのstdinを使用して生のNumPy配列を送信するもので、拡張することもできます。

## 画像操作
- `Blur`: ガウスフィルタを使用して画像をぼかす
- `Deglaze Image`: [FN16](https://github.com/Fannovel16/FN16-ComfyUI-nodes/blob/main/DeglazeImage.py)から取得
- `Denoise`: 入力画像のノイズを除去する
- `Image Compare`: 2つの画像を比較し、差分画像を返す
- `Image Premultiply`: 画像をマスクで乗算
- `Image Remove Background Rembg`: [RemBG](https://github.com/danielgatis/rembg)を使用した背景除去  
  <img src="https://github.com/melMass/comfy_mtb/assets/704172

6/e69253b4-c03c-45e9-92b5-aa46fb887be8" width=320/>
- `Image Resize Factor`: [WAS Node Suite](https://github.com/WASasquatch/was-node-suite-comfyui)から抽出され、いくつかの編集（特に複数の画像のサポート）と機能の削減が行われました。
- `Mask To Image`: マスク（アルファ）をカラーと背景を持つRGBイメージに変換します。
- `Save Image Grid`: 入力バッチのすべての画像を画像グリッドとして保存します。

## 潜在的なユーティリティ
- `Latent Lerp`: 2つの潜在的なベクトルの間の線形補間（ブレンド）

## その他のユーティリティ
- `Concat Images`: 2つの画像ストリームを取り、他のComfyパイプラインでサポートされている画像のバッチとしてマージします。
- `Image Resize Factor`: **非推奨**。組み込みの画像リサイズ機能を発見したため、削除される予定です。
- `Text To Image`: フォントを使用してテキストを画像に変換するためのユーティリティ
- `Styles Loader`: csvファイルをロードし、行からドロップダウンを作成します（A111のようなもの）  
  <img src="https://github.com/melMass/comfy_mtb/assets/7041726/02fe3211-18ee-4e54-a029-931388f5fde8" width=320/>
- `Smart Step`: KSamplerの高度な使用に使用するステップパーセントを取得する非常に基本的なノード
- `Qr Code`: 基本的なQRコード生成器
- `Save Tensors`: 将来的に削除される可能性のあるデバッグノード
- `Int to Number`: WASSuiteの数値ノードの補完
- `Smart Step`: `KAdvancedSampler`のステップ（開始/停止）を制御するための非常に基本的なツールで、パーセンテージを使用します。

## テクスチャ

- `DeepBump`: 1枚の画像から法線マップと高さマップを生成します。

# Comfyリソース

**ガイド**:
- [公式の例（英語）](https://comfyanonymous.github.io/ComfyUI_examples/)
- @BlenderNekoによる[ComfyUIコミュニティマニュアル（英語）](https://blenderneko.github.io/ComfyUI-docs/)

- @tjhayasakaによる[Tomoakiの個人Wiki（日本語）](https://comfyui.creamlab.net/guides/)

**拡張機能とカスタムノード**:
- @WASasquatchによる[Comfyリスト用のプラグイン（英語）](https://github.com/WASasquatch/comfyui-plugins)

- [CivitAIのComfyUIタグ（英語）](https://civitai.com/tag/comfyui)
