"""特徴量を「関節ごとの描画重み」に写像する。

`render.py` を特徴量パッケージから独立させたままにするための境界。render 側は
**キーポイント 17 点それぞれの 0..1 のスカラー** しか受け取らず、その値が負荷なのか速度なのかを
知らない。どの指標をどう関節へ割り当てるかという領域知識はすべてここに閉じる。
"""

from __future__ import annotations

import numpy as np

from pose_viz.features.export import TrackFeatures
from pose_viz.features.filters import nan_percentile
from pose_viz.features.schema import (
    L_ANKLE,
    L_ELBOW,
    L_HIP,
    L_KNEE,
    L_SHOULDER,
    L_WRIST,
    N_KEYPOINTS,
    R_ANKLE,
    R_ELBOW,
    R_HIP,
    R_KNEE,
    R_SHOULDER,
    R_WRIST,
)

MODES = ("off", "load", "speed")

#: 負荷を定義できる関節（角度が測れる 8 点）と、その値を継承させる遠位のキーポイント。
#: 前腕は肘の負荷で、下腿は膝の負荷で塗る、という対応にあたる。手首・足首そのものには
#: 挟角が定義できない（COCO-17 に踵・爪先が無い）ため、近位から引き継ぐ。
_ANGLE_TO_KEYPOINT: dict[str, int] = {
    "l_elbow": L_ELBOW,
    "r_elbow": R_ELBOW,
    "l_shoulder": L_SHOULDER,
    "r_shoulder": R_SHOULDER,
    "l_hip": L_HIP,
    "r_hip": R_HIP,
    "l_knee": L_KNEE,
    "r_knee": R_KNEE,
}
_INHERIT_FROM: dict[int, int] = {
    L_WRIST: L_ELBOW,
    R_WRIST: R_ELBOW,
    L_ANKLE: L_KNEE,
    R_ANKLE: R_KNEE,
}


def _load_weights(tf: TrackFeatures) -> np.ndarray:
    """関節負荷の代理値を (T, 17) の重みに展開する。負荷が定義できない点は 0。"""
    w = np.zeros((tf.series.n_frames, N_KEYPOINTS), dtype=np.float32)
    for k, name in enumerate(tf.load.names):
        kp = _ANGLE_TO_KEYPOINT.get(name)
        if kp is not None:
            w[:, kp] = np.nan_to_num(tf.load.load[:, k], nan=0.0)
    for distal, proximal in _INHERIT_FROM.items():
        w[:, distal] = w[:, proximal]
    return w


def _speed_weights(tf: TrackFeatures) -> np.ndarray:
    """関節速度を、そのトラック内の p95 で正規化した 0..1 の重みにする。"""
    speed = tf.kinematics.speed_rel
    ref = nan_percentile(speed, 95.0)
    if not np.isfinite(ref) or ref <= 0:
        return np.zeros_like(speed, dtype=np.float32)
    return np.clip(np.nan_to_num(speed, nan=0.0) / ref, 0.0, 1.0).astype(np.float32)


def build_joint_weights(
    features: dict[int, TrackFeatures], mode: str
) -> dict[int, dict[int, np.ndarray]]:
    """`frame_idx -> track_id -> (17,) の 0..1 重み` を作る。

    `mode` が "off" なら空の辞書を返す（render は従来どおりの描画になる）。
    """
    if mode not in MODES:
        raise ValueError(f"未知の feature_modulation: {mode!r}（{'|'.join(MODES)} のいずれか）")
    if mode == "off":
        return {}

    out: dict[int, dict[int, np.ndarray]] = {}
    for track_id, tf in features.items():
        w = _load_weights(tf) if mode == "load" else _speed_weights(tf)
        for i, frame_idx in enumerate(tf.series.frame_idx):
            out.setdefault(int(frame_idx), {})[track_id] = w[i]
    return out
