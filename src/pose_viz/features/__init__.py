"""解釈可能な動作特徴量の算出。

extract 済みキャッシュ（`ExtractCache`）だけを入力とし、モデル推論は一切行わない。
`render` からは import されない片方向依存にしてあるため、ここを触っても描画は壊れない。

設計の要点:

- **計測には `keypoints_raw`（One-Euro 平滑化「前」）を使う。** 平滑化は描画用の低遅延・非対称
  フィルタで、その出力を微分すると速度・加速度が減衰する。計測側は Savitzky-Golay という
  ゼロ位相フィルタを改めて掛ける（CLAUDE.md 不変条件⑦）。
- **全ての量は「毎秒」単位・体幹長で正規化する。** fps とカメラ距離と体格差に依存しない値にする。
- **欠損は NaN として最後まで伝播させる。** 埋められない穴を埋めない。指標ごとの欠損率を併記する。
"""

from pose_viz.features.schema import TrackSeries

__all__ = ["TrackSeries"]
