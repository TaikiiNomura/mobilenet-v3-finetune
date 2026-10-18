# MobileNet V3 Smallの追加学習

`data/normal` を0、`data/Strand Coreout` を1として画像全体を分類します。
`extra` と `ritec_data` は学習対象に含めません。JSONのポリゴンは使用しません。
画像データとImageNet初期重みを残し、以前の結果は削除しています。

## 必要なもの

- Python 3.11、PyTorch、torchvision、Pillow、NumPy、Matplotlib
- `data/normal/*.png` と `data/Strand Coreout/*.png`
- `weights/mobilenet_v3_small-047dcff4.pth`（学習前のImageNet重み）

既存の画像データは学習プロジェクト側の `data/` にあります。
コードを別の場所へ移した場合は `--data-root` でデータの場所を指定します。

## 実行

必要なライブラリのある仮想環境を有効化して実行します。

```bash
bash run_exp.bash
bash run_exp.bash --lr 0.0005 --bs 16 --epochs 10
```

`main.py` のargparseで条件を受け取る構成です。`--config config.toml` も使えます。
`--data-root`、`--weights`、`--results-root`、`--output` は相対パスで指定できます。
CLIの相対パスは `main.py`、設定内部の相対パスは設定ファイルを基準に解決します。
`run_exp.bash` は有効化した環境の `python` を使います。別の実行環境は `RITEC_PYTHON` で指定できます。

## コードと学習条件

- `main.py`：set_device、set_seed、make_dataloader、train_model、test_model、保存処理
- `models.py`：network関数によるモデル構築と初期重みの読込
- `dataset.py`：画像読込、分割、固定した特徴抽出部での特徴計算
- `config.toml`：任意で使用する設定
- `run_exp.bash`：標準の実行引数

特徴抽出部とそのBatchNormを固定し、分類部全体をAdamWで学習します。
既定値は20エポック、batch size32、lr0.001、weight decay0.0001、seed42。
デバイスはCUDA、MPS、CPUの順に選択します。
画像はRGB化し、縦横比を維持して224×224へ余白付きで変換・正規化します。
特徴は一度だけ計算してRAMに保持します。データ拡張は行いません。

各クラスを約70%・15%・15%で学習・検証・最終テストへ分割します。
完全一致画像は同じ分割にまとめます。似た画像や同じロープ単位の分割は未対応です。
検証損失が最小の重みを選択し、その後で最終テストを1回評価します。
コード内の `test_loader` は各エポックの検証用、`final_loader` は最終テスト用です。

## 出力

`results/lr=学習率_bs=バッチサイズ/seed=seed/run_日時/` に保存します。
`--output` で明示した場合はその場所を使い、既存の出力先は上書きしません。

- `mobilenet_v3_small_rope_finetuned_YYYYMMDD.pt`：学習済み重み、ラベル、前処理、学習日
- `config.json`：実行条件・デバイス・初期重みの情報
- `split.json`：画像のハッシュ、ラベル、分割と各クラスの件数
- `results.csv`：エポックごとの学習・検証指標と選択エポックの最終テスト指標
- `metrics.json`、`verification.json`：評価結果・保存モデルの再読込確認
- `img/`：損失、正解率、混同行列のグラフ

重み名の日付は学習開始日です。同日の実験は結果フォルダの日時で区別します。
