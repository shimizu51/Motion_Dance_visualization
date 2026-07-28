"""関節ごとの速度・加速度と、動きの滑らかさ（SPARC）。

単位はすべて **body-length/秒**（体幹長で割った無次元長 ÷ 秒）。これで
カメラ距離・被写体の体格差・フレームレートのいずれにも依存しない値になる。

速度は 2 系統を出す。

- **root 相対**（骨盤中点基準）… カメラのパン・ズームに原理的に不変。主指標。
- **カメラ補正した絶対** … 体幹の並進を含む。`camera_affine` で毎フレーム補正する。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pose_viz.config import FeatureConfig
from pose_viz.features.filters import savgol_nan
from pose_viz.features.schema import TrackSeries


@dataclass
class JointKinematics:
    speed_rel: np.ndarray  # (T, 17) body-length/s、root 相対
    accel_rel: np.ndarray  # (T, 17) body-length/s^2、root 相対
    speed_abs: np.ndarray  # (T, 17) body-length/s、カメラ補正済みの絶対
    root_speed: np.ndarray  # (T,) body-length/s、骨盤中点の並進（カメラ補正済み）


def _camera_velocity(xy: np.ndarray, affine: np.ndarray, fps: float) -> np.ndarray:
    """各点におけるカメラ運動由来の見かけの速度 (T,17,2) px/s を返す。

    `affine[i]` は「フレーム i-1 → i」の相似変換なので、静止点の見かけの変位は
    `A_i(p) - p`。累積変換を作らず毎フレーム局所的に評価するため、途中に推定失敗
    （NaN）があっても、そのフレームだけが NaN になり後続に伝播しない。
    """
    a = affine[:, :, :2]  # (T,2,2)
    b = affine[:, :, 2]  # (T,2)
    moved = np.einsum("tij,tkj->tki", a, xy) + b[:, None, :]
    return (moved - xy) * fps


def joint_kinematics(
    series: TrackSeries, cfg: FeatureConfig, camera_affine: np.ndarray | None = None
) -> JointKinematics:
    fps = series.fps
    scale = series.scale[:, None]  # (T,1)

    # root 相対のピクセル座標を微分してから体幹長で割る。
    # （先に割ってから微分すると、体幹長の時間変化が速度に混ざり込んでしまう）
    v_rel = savgol_nan(series.xy_rel, fps, cfg.deriv_window_sec, cfg.deriv_polyorder, deriv=1)
    a_rel = savgol_nan(series.xy_rel, fps, cfg.deriv_window_sec, cfg.deriv_polyorder, deriv=2)
    speed_rel = np.linalg.norm(v_rel, axis=2) / scale
    accel_rel = np.linalg.norm(a_rel, axis=2) / scale

    v_abs = savgol_nan(series.xy, fps, cfg.deriv_window_sec, cfg.deriv_polyorder, deriv=1)
    if camera_affine is not None:
        aff = camera_affine[series.frame_idx]
        v_cam = _camera_velocity(series.xy, aff, fps)
        # 微分側は窓で平滑化されているので、カメラ項も同じ窓で均してから引く
        v_cam = savgol_nan(v_cam, fps, cfg.deriv_window_sec, cfg.deriv_polyorder, deriv=0)
        v_abs = v_abs - np.nan_to_num(v_cam, nan=0.0)
    speed_abs = np.linalg.norm(v_abs, axis=2) / scale

    v_root = savgol_nan(series.root, fps, cfg.deriv_window_sec, cfg.deriv_polyorder, deriv=1)
    root_speed = np.linalg.norm(v_root, axis=1) / series.scale

    return JointKinematics(
        speed_rel=speed_rel.astype(np.float32),
        accel_rel=accel_rel.astype(np.float32),
        speed_abs=speed_abs.astype(np.float32),
        root_speed=root_speed.astype(np.float32),
    )


def sparc(speed: np.ndarray, fps: float, padlevel: int = 4, fc: float = 10.0, amp_th: float = 0.05) -> float:
    """Spectral Arc Length（Balasubramanian et al.）による動きの滑らかさ。

    速度プロファイルの正規化フーリエ振幅スペクトルの弧長の負値。0 に近いほど滑らかで、
    動きが分節的・ぎこちないほど負に大きくなる。**ジャーク（3階微分）の代わりに使う**：
    2D 姿勢推定の座標を 3 回微分するとノイズが支配的になるが、SPARC は速度（1階）だけを
    使い、しかも周波数領域で振幅閾値により実質的な帯域制限を掛けるため、ノイズに強い。
    """
    v = np.asarray(speed, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size < 4 or not np.any(v):
        return float("nan")

    nfft = int(2 ** (np.ceil(np.log2(v.size)) + padlevel))
    freq = np.arange(0, fps, fps / nfft)
    mag = np.abs(np.fft.fft(v, nfft))
    peak = mag.max()
    if peak <= 0:
        return float("nan")
    mag = mag / peak

    band = freq <= fc
    freq, mag = freq[band], mag[band]
    if freq.size < 2:
        return float("nan")

    # 振幅が閾値以上の範囲まで適応的にカットオフを詰める
    above = np.nonzero(mag >= amp_th)[0]
    if above.size < 2:
        return float("nan")
    freq, mag = freq[above[0] : above[-1] + 1], mag[above[0] : above[-1] + 1]
    span = freq[-1] - freq[0]
    if span <= 0:
        return float("nan")

    return float(-np.sum(np.sqrt((np.diff(freq) / span) ** 2 + np.diff(mag) ** 2)))


def windowed_sparc(speed: np.ndarray, fps: float, window_sec: float) -> np.ndarray:
    """スライド窓ごとの SPARC を毎フレーム値として返す（(T,) float32）。

    SPARC は本来ひとつの動作区間に対する要約値なので、時系列として見せたい場合は
    窓を切って算出する。窓内に有効値が半分未満のフレームは NaN。
    """
    n = len(speed)
    win = max(4, int(round(window_sec * fps)))
    half = win // 2
    out = np.full(n, np.nan, dtype=np.float32)
    for i in range(n):
        s, e = max(0, i - half), min(n, i + half + 1)
        seg = speed[s:e]
        if np.isfinite(seg).sum() < win // 2:
            continue
        out[i] = sparc(seg, fps)
    return out
