# MotionBERT（ベンダリング）

単眼 2D → 3D リフティングに使う **MotionBERT** のモデル定義を、変更を最小限にして取り込んだもの。

| 項目 | 内容 |
|---|---|
| 出典 | [Walter0807/MotionBERT](https://github.com/Walter0807/MotionBERT)（ICCV 2023） |
| ライセンス | **Apache-2.0**（[LICENSE](LICENSE) は上流のものをそのまま同梱） |
| 取り込んだファイル | `lib/model/DSTformer.py` → `DSTformer.py`、`lib/model/drop.py` → `drop.py` |
| 加えた変更 | `DSTformer.py` の import を 1 行だけ書き換え（`from lib.model.drop import DropPath` → `from pose_viz.vendor.motionbert.drop import DropPath`）。**それ以外は一切変更していない。** |
| 重み | 取り込まない。実行時に HuggingFace [`walterzhu/MotionBERT`](https://huggingface.co/walterzhu/MotionBERT) から取得する |

## なぜベンダリングするか

MotionBERT は pip パッケージとして配布されていない。実行時に GitHub や HuggingFace から
`.py` を落として import する形にすると、**外部から取得したコードをその場で実行する**ことになり、
再現性の面でも安全性の面でも受け入れられない。Apache-2.0 は改変・再配布を許諾しているので、
必要な 2 ファイルだけをここに固定し、差分を上表に明示する形をとった。

上流を追随する場合は 2 ファイルを取り直し、上記の import 1 行を当て直せばよい。
