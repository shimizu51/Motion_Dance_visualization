# CLAUDE.md

このファイルは、このリポジトリで作業する際の Claude Code 向けガイドです。

## プロジェクト概要

**Pose_visualization** は映像から人間の骨格を取り出し、それを**アート作品として可視化**するリポジトリ。
単なる骨格描画ではなく、**残差を残す**（軌跡が一定時間で消える）ことと、**人物本体を薄く重ねる**
ことで、曲調や見た目からはわからない「激しさ」を別の側面として見せることを目的とする。

出力は**黒背景の中に骨格が最も強く表示され、その周りに本人が薄く映る**映像。

### モデルの構成要素

| 要素 | 実体 | 役割 | 備考 |
|------|-----|-----|-----|
| 人物検出・セグメンテーション | **RF-DETR**（`rfdetr` の `RFDETRSegSmall`） | 人物 box + mask を1パスで取得 | box/mask を同時取得できるため追加のセグメンテーションモデルは使わない |
| トラッキング | 自前 IoU トラッカ（`tracking.py`） | フレーム間で人物 ID を維持 | 残差の連続性は ID の安定性に依存するため丁寧に実装 |
| 骨格推定 | **ViTPose**（`transformers.VitPoseForPoseEstimation` / `usyd-community/vitpose-base-simple`） | 人物 box ごとに17点キーポイント推定 | top-down モデルなので box が必須。box は **COCO 形式 `(x,y,w,h)`** で渡す（xyxy ではない） |
| 平滑化 | One-Euro フィルタ（`pose.py`） | 関節のジッタ除去 | トラック ID・関節ごとに独立して適用。**描画用**であり、平滑化前の生値も `keypoints_raw` として別途保存する |
| 残差抽出 | **AKAZE**（`akaze.py`） | 人物マスク内の特徴点をフレーム間対応付け | `estimateAffinePartial2D` で大域運動を推定し、そこからのズレ（＝残差）を「見た目からわからない激しさ」として使う |
| カメラ運動推定 | **AKAZE**（`akaze.py` の `CameraMotionEstimator`） | 人物を除いた背景からフレーム間の相似変換を推定 | 特徴量側でカメラのパン・ズームを差し引くための**計測専用**。描画には使わない |
| 残差の寿命管理 | `residual.py` | 粒子・関節軌跡を一定時間でフェードアウト | 残差が大きいほど寿命を延ばす |
| 合成 | `render.py` | ゴースト層・残差層・骨格残像層・骨格層を加算合成 | 骨格層が必ず最高輝度になるよう最後に描く |

### 処理フロー

**extract（重い・1回だけ）→ render（軽い・何度も回す）の2ステージ**に分離する。

```
extract: 動画 → 検出/追跡/骨格/AKAZE軌跡 → data/cache/<name>.pkl.gz
render:  キャッシュ + 動画（薄い人物レイヤー用） + config → 出力mp4
```

アート作品として見た目の調整を何十回も繰り返す前提のため、**推論結果はキャッシュし、render は
モデル推論なしで完結させる**（`render.py` はキャッシュと生フレームだけを読む）。

## リポジトリの現状

**実装初期段階**。`src/pose_viz/` に extract/render の各モジュールを構築中。

- `src/pose_viz/` … 本体パッケージ（`detect.py`／`tracking.py`／`pose.py`／`akaze.py`／`cache.py`／
  `residual.py`／`render.py`／`palette.py`／`video_io.py`／`config.py`／`cli.py`）
- `configs/` … `default.yaml`（全パラメータ既定値）と動画ごとの上書き設定
- `data/input/` … 入力動画（`Magnetic.mp4`：3840x2160・AV1・23.976fps・166秒・Opus音声）。**git 管理外**
- `data/cache/` … extract の出力（`.pkl.gz`：gzip 圧縮した pickle）。**git 管理外**（重いので再生成する前提）
- `data/output/` … render の出力動画。**git 管理外**

### 未対応（今後）

1. **軌跡の滑らかさの検証** — 純粋な AKAZE 記述子マッチングはフレーム毎に検出点が入れ替わりやすい。
   `config` の `akaze.trail_mode: akaze | akaze_lk`（LK 光学フローで伝播）を実映像で比較して既定を決める。
2. **マスク品質の検証** — `RFDETRSegSmall` のマスクは 512x512 ベースで輪郭がやや粗い。ゴースト層は
   ぼかして低 alpha で敷くだけなので実用上問題ない見込みだが、気になる場合は `PersonDetector`
   プロトコルの差し替え（SAM2 等）で対応できる形にしてある。
3. **rfdetr の MPS 対応** — 公式に MPS 対応の明記がないため、動かない場合は検出のみ CPU にフォールバックする。
4. **全尺（166秒）の処理時間・メモリ計測** — 現状は 10〜20秒の試作クリップで検証する段階。
5. **テスト・lint の自動化** — pytest／ruff の設定はまだ無い。

## ディレクトリ構成

| パス | 役割 |
|------|-----|
| `src/pose_viz/cli.py` | サブコマンド（`extract`／`render`／`run`）のエントリポイント |
| `src/pose_viz/config.py` | dataclass 定義と YAML の読み込み・マージ |
| `src/pose_viz/video_io.py` | ffmpeg サブプロセスによる rawvideo パイプ I/O（`FrameReader`／`FrameWriter`／`mux_audio`） |
| `src/pose_viz/detect.py` | RF-DETR Seg Small ラッパ。person クラスでフィルタし box・score・mask を返す |
| `src/pose_viz/tracking.py` | IoU ベースの簡易トラッカ。ID の生成・維持・失効を管理 |
| `src/pose_viz/pose.py` | ViTPose ラッパ＋ One-Euro フィルタによる平滑化 |
| `src/pose_viz/akaze.py` | AKAZE 抽出・BFMatcher 対応付け・アフィン推定による残差ベクトル算出、および背景からのカメラ大域運動推定 |
| `src/pose_viz/cache.py` | extract 結果（pose／akaze／mask／カメラ運動）の `.pkl.gz` 保存・復元・スキーマ版と設定ハッシュの検証 |
| `src/pose_viz/residual.py` | 粒子系・関節残像系の寿命とフェードカーブ |
| `src/pose_viz/render.py` | ゴースト層・残差層・骨格残像層・骨格層のレイヤ合成コンポジタ |
| `src/pose_viz/palette.py` | トラック ID ごとの配色・グロー・ブレンド関数 |
| `configs/default.yaml` | 全パラメータの既定値 |
| `configs/magnetic.yaml` | `data/input/Magnetic.mp4` 用の上書き設定 |

## 実装（src/pose_viz）

Python 3.12（**uv** venv）。パッケージ化して `pose-viz` コマンドとしても呼べるようにする。

- **2ステージの境界を越えない**: `render.py` はモデル（RF-DETR／ViTPose）を一切 import しない。
  推論が必要な処理はすべて `extract` 側（`detect.py`／`pose.py`／`akaze.py`）に閉じる。
- **box 形式は COCO `(x,y,w,h)`**: ViTPose の `AutoProcessor` に渡す box はこの形式のみ受け付ける。
  RF-DETR や内部トラッカは xyxy を使うため、`pose.py` の境界で必ず変換する。
- **マスクは bbox クロップ＋PNG エンコードで保存**（`cache.py`）。フル解像度の二値マスクを
  そのまま持つと動画全体でサイズが膨れるため（解像度の縮小はしていない）。
- **トラック ID の安定性が残差の質を決める**: `tracking.py` の `max_age`／`min_hits` を緩めると
  ID 入れ替わりで軌跡が破綻するので、パラメータを変えたら必ず `--debug-overlay` で確認する。

### キャッシュのスキーマ（`CACHE_VERSION = 3`）

`ExtractCache` は `frames: dict[frame_idx -> list[TrackFrame]]` を持つ。**`track_id` はキーではなくフィールド**
なので、1トラック分の時系列が欲しいときは `ExtractCache.by_track()` を使う（全フレーム走査を1回で済ませる）。

読み解くときに間違えやすい点:

- **欠損は「レコードが存在しない」ことで表現される。** NaN もゼロ埋めも補間もない。トラックがそのフレームで
  マッチしなければ、`frames[i]` にそのトラックの要素が入らないだけ。
- **`min_hits` の分だけ先頭フレームは必ず空になる**（既定 3 なら先頭2フレーム）。
- **`keypoint_scores` はヒートマップのピーク値であり確率ではない。** 実測で 1.0 を超えるため
  `[0,1]` を前提にしたコードを書かない。
- **`depth_rank`／`depth_score` は深度ではない。** レイヤ合成順のための序数であり、metric な意味も
  動画間で一貫したスケールも持たない。重なりが無いフレームでは前回値を引き継ぐ。
- **`camera_affine` は `(frame_count, 2, 3)`。** i 行目が「フレーム i-1 → i」の相似変換で、先頭フレームと
  推定失敗フレームは NaN（単位行列で埋めていないので、失敗を判別できる）。

### 未対応の設定・既知の落とし穴

- `akaze.trail_mode` は**どのコードからも読まれていない**（`akaze_lk` は未実装）。
- `pose.keypoint_score_threshold` も**読まれていない**。`render.py` が `0.3` をハードコードしている。
- `video.fps` を既定の `null` 以外にすると壊れる。`FrameReader` の ffmpeg コマンドに `-r` が無いため
  **実際にはフレームが間引かれない**のに、タイムスタンプ・粒子寿命・出力 fps だけがずれる。

### 厳守する不変条件（崩すと描画が壊れる／キャッシュが壊れる）

1. **ViTPose への box は COCO 形式 `(x, y, w, h)`**。xyxy のまま渡すと骨格がズレる。
2. **AKAZE は人物マスクでクロップした ROI にのみ適用する**。マスク無しで動画全体にかけると背景の
   特徴点まで残差として拾ってしまい、「その人の激しさ」という意味が失われる。
3. **extract と render は完全分離**。render 側でモデル推論を呼び出さない（重い処理を毎回の見た目調整で
   繰り返さないため）。
4. **キャッシュには設定ハッシュを埋め込む**。抽出時のパラメータ（検出閾値・トラッキング閾値等）が
   変わったら再抽出が必要になるため、`render` 側で不整合を検出して警告する。
5. **骨格レイヤーは必ず最後に最高輝度で描く**（`render.py` の合成順）。ゴースト層・残差層より後に、
   ブルームをかけた上で鋭い線を再度重ねる。
6. **One-Euro フィルタの状態はトラック ID・関節ごとに独立させる**。共有すると ID 交代時に前の人物の
   平滑化状態が新しい人物に漏れる。
7. **平滑化は「描画用」、計測は「生値」から行う**。One-Euro は低遅延・非対称なオンラインフィルタなので、
   その出力を微分すると速度・加速度が減衰する。`keypoints`（平滑化後）は描画に、`keypoints_raw`
   （平滑化前）は特徴量計算に使い、計測側では Savitzky-Golay のようなゼロ位相フィルタを別途かける。

## 実行環境

単一環境（Apple M5・32GB ユニファイドメモリ・macOS）を対象にする。学習は行わず、すべて推論のみ。

- **デバイス**: PyTorch は `mps` を優先し、未対応 op は `PYTORCH_ENABLE_MPS_FALLBACK=1` で CPU に
  フォールバックさせる。`torch.backends.mps.is_available()` が `False` の場合は CPU 実行に切り替える。
- **rfdetr の MPS 対応は未確認**。動かない場合は検出フェーズのみ CPU 実行にする分岐を `detect.py` に
  持たせる（検出は毎フレーム軽量に済ませられるため、全体の律速にはなりにくい想定）。
- **AV1 デコードが重い場合**: `ffmpeg -i in.mp4 -vf scale=1920:-2 -c:v libx264 -crf 14 proxy.mp4` で
  H.264 プロキシを作り、以降はそちらを入力にする。
- 依存は **uv**（`pyproject.toml`＋`uv.lock`＋`.python-version`=3.12。**conda は使わない**）。

## 開発・起動・検証

- **セットアップ**: `uv sync` → `uv run python -c "import torch, transformers, rfdetr, cv2; print(torch.backends.mps.is_available())"` で `True` になることを確認。
- **試作（10〜20秒クリップ）**:
  ```bash
  uv run pose-viz extract data/input/Magnetic.mp4 --start 30 --duration 15 --config configs/magnetic.yaml
  uv run pose-viz render --cache data/cache/Magnetic.pkl.gz --out data/output/magnetic_v1.mp4
  ```
- **検証はまず `--debug-overlay` から**（見た目の調整より先に検出・追跡・姿勢の正しさを確認する）:
  ```bash
  uv run pose-viz render --cache data/cache/Magnetic.pkl.gz --debug-overlay --out data/output/debug.mp4
  ```
  box が人物に追従しているか・ID が入れ替わっていないか・骨格が破綻していないか・AKAZE 点が
  人物の上にだけ乗っているかを目視確認する。
- **キャッシュの再利用確認**: 同じ設定で `extract` を2回走らせて2回目がスキップされること、
  設定を変えると警告が出ることを確認する。
- **自動テストは未整備**。配線チェックは短尺クリップ＋`--debug-overlay` を最短の検証手段とする。

## 作業時の指針

- **判断が必要なときは質問する**: 見た目のパラメータ（配色・寿命・ブルーム量など）以外の設計判断
  （モデル選定・キャッシュ形式・トラッキング方式など）で複数の妥当解があるときは、推測で進めず
  `AskUserQuestion` で確認する（推奨案を第一候補に提示）。
- **見た目の調整と配線の正しさを分けて考える**: パラメータが期待通りに見えない場合、まず
  `--debug-overlay` で検出・追跡・姿勢が正しいかを切り分けてから配色・寿命などのアート的パラメータを
  疑う。
- **モデル・アルゴリズムの差し替えは想定しておく**: マスク生成（RF-DETR → SAM2 等）や軌跡生成
  （AKAZE → AKAZE+LK 等）は `config` で切り替えられる形にし、既定以外を選ぶ場合は理由を記録する。
- **外部データ・成果物はコミットしない**: `data/input`／`data/cache`／`data/output` は `.gitignore`
  済み（入力動画・中間キャッシュ・出力映像はサイズが大きいため）。

## コミット規約（毎回の作業完了時に必ず実施）

**各作業の完了時には毎回コミットする。** メッセージは [この Qiita 記事](https://qiita.com/konatsu_p/items/dfe199ebe3a7d2010b3e) の規約（Angular ベース）に従う。

> **⚠ push は必ず事前に確認する（commit は自動でよい）**: `git push` はリモートの状態を変更するため、
> **push の前は毎回ユーザーに確認を取る**。commit はローカル操作なので確認不要で進めてよい。

### フォーマット

```
[prefix]: [理由]、[変更内容]
```

例:
- `feat: 〇〇のため、△△を追加`
- `chore: ネットワーク通信のため、hoge を追加`

### prefix 一覧

| prefix | 意味 |
|--------|------|
| `feat` | 新機能の追加 |
| `fix` | バグ修正 |
| `docs` | ドキュメントのみの変更 |
| `style` | 意味に影響しない変更（空白・フォーマット等） |
| `refactor` | バグ修正でも機能追加でもないコード改善 |
| `perf` | パフォーマンス改善 |
| `test` | テストの追加・修正 |
| `chore` | ビルド・補助ツール・設定など保守作業 |

### ルール

1. **理由を含める** — 変更の意図・目的を書き、レビュー品質を上げる。
2. **適切な粒度** — prefix 単位でコミットを分割し、1コミットが大きくなりすぎないようにする。
3. **言語** — このプロジェクトは日本語チーム想定なので、コミットメッセージは**日本語**で書く。
