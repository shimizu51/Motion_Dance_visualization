"""COCO-17 の関節定義と、特徴量計算の入力になる時系列コンテナ。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# --- COCO-17 キーポイント ---------------------------------------------------
# ラベルは被写体自身の解剖学的な左右（画像の左右ではない）。ただし被写体が背を向けたとき
# top-down 2D 推定器が左右を取り違えるのは既知の failure mode なので、左右対称性を扱う
# 特徴量では posture.detect_lr_flip() で反転区間を検出して除外する。
NOSE = 0
L_EYE, R_EYE = 1, 2
L_EAR, R_EAR = 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16
N_KEYPOINTS = 17

KEYPOINT_NAMES = [
    "nose", "l_eye", "r_eye", "l_ear", "r_ear",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
    "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle",
]

#: 3点で挟角を測る関節。(近位, 頂点, 遠位)。
#: 足首は COCO-17 に踵・爪先が無いため定義できない（HALPE-26 等への差し替えが必要）。
JOINT_ANGLES: dict[str, tuple[int, int, int]] = {
    "l_elbow": (L_SHOULDER, L_ELBOW, L_WRIST),
    "r_elbow": (R_SHOULDER, R_ELBOW, R_WRIST),
    "l_shoulder": (L_HIP, L_SHOULDER, L_ELBOW),
    "r_shoulder": (R_HIP, R_SHOULDER, R_ELBOW),
    "l_hip": (L_SHOULDER, L_HIP, L_KNEE),
    "r_hip": (R_SHOULDER, R_HIP, R_KNEE),
    "l_knee": (L_HIP, L_KNEE, L_ANKLE),
    "r_knee": (R_HIP, R_KNEE, R_ANKLE),
}

#: 左右対をなす関節角度（対称性の算出に使う）
ANGLE_PAIRS = [("l_elbow", "r_elbow"), ("l_shoulder", "r_shoulder"), ("l_hip", "r_hip"), ("l_knee", "r_knee")]

#: 左右対をなす四肢のキーポイント（速度の対称性に使う）
LIMB_PAIRS = [(L_WRIST, R_WRIST), (L_ELBOW, R_ELBOW), (L_ANKLE, R_ANKLE), (L_KNEE, R_KNEE)]

#: 体節の質量比と近位端からの重心位置（Winter, *Biomechanics and Motor Control of Human Movement*）。
#: (近位キーポイント, 遠位キーポイント, 全体質量比, 近位端からの重心比)。合計 1.000。
#: 体幹・頭部は COCO-17 に対応点が無いため、肩中点・腰中点の仮想点で代用する（`_VIRTUAL_*`）。
_VIRTUAL_MID_SHOULDER = -1
_VIRTUAL_MID_HIP = -2
BODY_SEGMENTS: list[tuple[int, int, float, float]] = [
    (NOSE, _VIRTUAL_MID_SHOULDER, 0.081, 0.5),          # 頭部+頸部
    (_VIRTUAL_MID_SHOULDER, _VIRTUAL_MID_HIP, 0.497, 0.5),  # 体幹
    (L_SHOULDER, L_ELBOW, 0.028, 0.436),                # 上腕
    (R_SHOULDER, R_ELBOW, 0.028, 0.436),
    (L_ELBOW, L_WRIST, 0.022, 0.682),                   # 前腕+手
    (R_ELBOW, R_WRIST, 0.022, 0.682),
    (L_HIP, L_KNEE, 0.100, 0.433),                      # 大腿
    (R_HIP, R_KNEE, 0.100, 0.433),
    (L_KNEE, L_ANKLE, 0.061, 0.500),                    # 下腿+足部
    (R_KNEE, R_ANKLE, 0.061, 0.500),
]


@dataclass
class TrackSeries:
    """1 トラック分の、密な時間軸に並べ直したキーポイント時系列。

    `frame_idx` は観測された最初と最後のフレームの間を **1 刻みで埋めた** 軸で、キャッシュ側の
    「レコードが存在しない＝欠損」という表現を NaN に変換したもの。短い欠損だけ線形補間し
    （`interpolated`）、長い欠損は `segments` でセグメントを分割して NaN のまま残す。
    """

    track_id: int
    fps: float
    frame_idx: np.ndarray  # (T,) int32、first..last を 1 刻みで埋めた軸
    t: np.ndarray  # (T,) float32、秒（frame_idx / fps）
    xy: np.ndarray  # (T, 17, 2) float32、作業解像度のピクセル。欠損は NaN
    score: np.ndarray  # (T, 17) float32
    box: np.ndarray  # (T, 4) float32
    observed: np.ndarray  # (T,) bool、キャッシュに実レコードがあったフレーム
    interpolated: np.ndarray  # (T, 17) bool、線形補間で埋めた要素
    scale: np.ndarray  # (T,) float32、体幹長の移動中央値（px）。正規化の基準
    root: np.ndarray  # (T, 2) float32、腰中点
    xy_rel: np.ndarray  # (T, 17, 2) float32、root 相対のピクセル座標（カメラ運動に不変）
    #: 長い欠損で切った連続区間 [start, end)。特徴量はこの単位で計算する
    segments: list[tuple[int, int]] = field(default_factory=list)

    @property
    def n_frames(self) -> int:
        return len(self.frame_idx)

    def coverage(self) -> float:
        """観測率（補間・欠損を除いた実観測フレームの割合）。"""
        return float(self.observed.mean()) if self.n_frames else 0.0
