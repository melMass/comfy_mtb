# インストール

- [インストール](#インストール)
  - [自動インストール (推奨)](#自動インストール-推奨)
    - [ComfyUI マネージャ](#comfyui-マネージャ)
    - [仮想環境](#仮想環境)
  - [モデルのダウンロード](#モデルのダウンロード)
    - [ウェブ拡張機能](#ウェブ拡張機能)
  - [旧インストール方法 (MANUAL)](#旧インストール方法-manual)
    - [依存関係](#依存関係)


## 自動インストール (推奨)

### ComfyUI マネージャ

バージョン0.1.0では、この拡張機能は[ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager)と一緒にインストールすることを想定しています。これは、様々な環境で直面する様々なインストール問題を処理するのに非常に役立ちます。

### 仮想環境
また、ComfyUIのルートから以下のコマンドを使用する実験的なワンライナー・インストールもあります。これはコードをダウンロードし、依存関係をインストールし、インストールスクリプトを実行します：

```bash
curl -sSL "https://raw.githubusercontent.com/username/repo/main/install.py" | python3 -
```

## モデルのダウンロード
ノードによっては、追加モデルのダウンロードが必要な場合があるので、上記と同じ python 環境を使って対話的に行うことができる：
```bash
python scripts/download_models.py
```

プロンプトに従うか、Enterを押すだけで全てのモデルをダウンロードできます。


> **Note**
> プロンプトを出さずに全てのモデルをダウンロードするには、以下のようにします:
  ```bash
  python scripts/download_models.py -y
  ```

### ウェブ拡張機能

初回実行時にスクリプトは[web extensions](https://github.com/melMass/comfy_mtb/tree/main/web)をあなたの快適な `web/extensions` フォルダに[シンボリックリンク](https://github.com/melMass/comfy_mtb/blob/d982b69a58c05ccead9c49370764beaa4549992a/__init__.py#L45-L61)しようとします。万が一失敗した場合は、mtbフォルダを手動で`ComfyUI/web/extensions`にコピーしてください：

<img alt="color widget preview" src="https://github.com/melMass/comfy_mtb/assets/7041726/cff7e66a-4cc4-4866-b35b-10af0bb2d110" width=450>

## 旧インストール方法 (MANUAL)
### 依存関係

<details><summary><h4>カスタム Virtualenv (私は主にこれを使っています)</h4></summary>
    
1. ComfyUIで使用しているPython環境であることを確認してください。
2. 以下のコマンドを実行して、必要な依存関係をインストールします：
  ```bash
  pip install -r comfy_mtb/reqs.txt
  ```

</details> 

<details><summary><h4>Comfy-portable / standalone (ComfyUI リリースより)</h4></summary>。
    
もしあなたがComfyUIスタンドアロンから`python-embeded`を使用している場合、バイナリがホイールを持っていない場合、依存関係をpipでインストールすることができません。この場合、最後の[リリース](https://github.com/melMass/comfy_mtb/releases)をチェックしてください。(ソースからのビルドが必要なもののみ)あらかじめビルドされたホイールがあるlinuxとwindows用のバンドルがあります。詳細は[この問題(#1)](https://github.com/melMass/comfy_mtb/issues/1)をチェックしてください。

![image](https://github.com/melMass/comfy_mtb/assets/7041726/2934fa14-3725-427c-8b9e-2b4f60ba1b7b)

</details>

<details><summary><h4>Google Colab</h4></summary>

ComfyUI with localtunnel (Recommended Way)**ヘッダーのすぐ後（コードセルの前）に、新しいコードセルを追加してください。
![colabに追加する場所のプレビュー](https://github.com/melMass/comfy_mtb/assets/7041726/35df2ef1-14f9-44cd-aa65-353829188cd7)

```python
# download the nodes
!git clone --recursive https://github.com/melMass/comfy_mtb.git custom_nodes/comfy_mtb

# download all models
!python custom_nodes/comfy_mtb/scripts/download_models.py -y

# install the dependencies
!pip install -r custom_nodes/comfy_mtb/reqs.txt -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
```
これを実行した後、colabがランタイムを再起動する必要があると文句を言ったら、それを実行し、それ以前のセルは再実行せず、localtunnelを実行するセルだけを再実行してください。(最初に`%cd ComfyUI`のセルを追加する必要があるかもしれません...)


> **Note**:
> すべてのモデルが必要でない場合は、`-y`を削除してください : ![image](https://github.com/melMass/comfy_mtb/assets/7041726/40fc3602-f1d4-432a-98fd-ce2240f5ad06)

> **プレビュー**
> ![image](https://github.com/melMass/comfy_mtb/assets/7041726/b5b2b2d9-f1e8-4c43-b1db-7dfc5e07be86)

</details>

