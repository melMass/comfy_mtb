# インストール
- [インストール](#インストール)
    - [依存関係](#依存関係)
      - [カスタム仮想環境（主に使用）](#カスタム仮想環境主に使用)
      - [Comfyポータブル/スタンドアロン（ComfyUIリリースから）](#comfyポータブルスタンドアロンcomfyuiリリースから)
      - [Google Colab](#google-colab)
    - [モデルのダウンロード](#モデルのダウンロード)
    - [Web拡張機能](#web拡張機能)


### 依存関係

#### カスタム仮想環境（主に使用）

1. ComfyUIで使用するPython環境にいることを確認してください。
2. 次のコマンドを実行して、必要な依存関係をインストールします。
  ```bash
  pip install -r comfy_mtb/requirements.txt
  ```
#### Comfyポータブル/スタンドアロン（ComfyUIリリースから）

ComfyUIスタンドアロンの`python-embeded`を使用している場合、ホイールがない場合にはバイナリの依存関係をpipでインストールすることはできません。この場合は、最新の[リリース](https://github.com/melMass/comfy_mtb/releases)を確認してください。ビルド元のソースが必要な場合は、linuxとwindows用のプリビルドホイールがバンドルされています。詳細については、[この問題（#1）](https://github.com/melMass/comfy_mtb/issues/1)を参照してください。
![image](https://github.com/melMass/comfy_mtb/assets/7041726/2934fa14-3725-427c-8b9e-2b4f60ba1b7b)


#### Google Colab

**Run ComfyUI with localtunnel (Recommended Way)**ヘッダーのすぐ後ろに新しいコードセルを追加します（コードセルの前）
![colabでの追加方法のプレビュー](https://github.com/melMass/comfy_mtb/assets/7041726/35df2ef1-14f9-44cd-aa65-353829188cd7)


```python
# ノードをダウンロード
!git clone --recursive https://github.com/melMass/comfy_mtb.git custom_nodes/comfy_mtb

# すべてのモデルをダウンロード
!python custom_nodes/comfy_mtb/scripts/download_models.py -y

# 依存関係をインストール
!pip install -r custom_nodes/comfy_mtb/requirements.txt -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
```
これを実行した後、colabがランタイムを再起動するようにクレームをつける場合は、再起動してから以前のセルを再実行せずに、ローカルトンネルを実行するセルのみを実行します（最初に`%cd ComfyUI`というセルを追加する必要があるかもしれません...）


> **注意**：
> すべてのモデルが必要ない場合は、collabは実際にユーザー入力をサポートしているため、`-y`を削除します： ![image](https://github.com/melMass/comfy_mtb/assets/7041726/40fc3602-f1d4-432a-98fd-ce2240f5ad06)

> **プレビュー**
> ![image](https://github.com/melMass/comfy_mtb/assets/7041726/b5b2b2d9-f1e8-4c43-b1db-7dfc5e07be86)



### モデルのダウンロード
一部のノードには追加のモデルのダウンロードが必要です。上記と同じPython環境を使用して対話的に行うことができます。
```bash
python scripts/download_models.py
```

その後、プロンプトに従ってモデルをダウンロードするか、単にEnterキーを押してすべてのモデルをダウンロードします。

> **注意**
> プロンプトなしですべてのモデルをダウンロードするには、次のコマンドを使用します：
  ```bash
  python scripts/download_models.py -y
  ```

### Web拡張機能

最初の実行時に、スクリプトは[web拡張機能](https://github.com/melMass/comfy_mtb/tree/main/web)をComfyの`web/extensions`フォルダにシンボリックリンクしようとします。これが失敗する場合は、`mtb`フォルダを手動で`ComfyUI/web/extensions`にコピーしてください。現時点では、いくつかのノードで共有されるカラーウィジェットのみが提供されています。

<img alt="color widget preview" src="https://github.com/melMass/comfy_mtb/assets/7041726/cff7e66a-4cc4-4866-b35b-10af0bb2d110" width=450>