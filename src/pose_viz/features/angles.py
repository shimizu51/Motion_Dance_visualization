"""2D 関節角度と、その角速度・角加速度・信頼度。

**2D 角度は面外回転（カメラ面から外れた方向への屈曲）で系統的に歪む。** カメラ面から外れた肘は
投影上は実際より浅い角度に見える。ここではセグメントの見かけの長さがその区間の中央値に対して
どれだけ短縮しているか（短縮率）を信頼度として併記し、低信頼区間を後段でマスクできるようにする。
根本的な解決は 3D リフティング（ロードマップ Phase 4）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pose_viz.config import FeatureConfig
from pose_viz.features.filters import nan_percentile, savgol_nan
from pose_viz.features.schema import (
    JOINT_ANGLES,
    JOINT_ANGLES_3D,
    L_HIP,
    L_SHOULDER,
    R_HIP,
    R_SHOULDER,
    TrackSeries,
)

#: 投影は長さを縮める方向にしか働かないので、正規化長の高位パーセンタイルを真の長さとみなす
_TRUE_LENGTH_PERCENTILE = 90.0


@dataclass
class AngleFeatures:
    names: list[str]
    angle: np.ndarray  # (T, J) rad, 0..pi。関節の挟角（小さいほど深く曲がっている）
    ang_vel: np.ndarray  # (T, J) rad/s
    ang_accel: np.ndarray  # (T, J) rad/s^2
    #: (T, J) 0..1。**意味は `source` で変わる。**
    #: "2d" … 見かけのセグメント長の短縮率（低いほど面外回転が大きく角度が信用できない）
    #: "3d" … 入力 2D が有効だったかの 0/1（面外回転はリフティングで解消済みのため）
    confidence: np.ndarray
    trunk_lean: np.ndarray  # (T,) rad、鉛直からの符号付き傾き（正 = 画面右へ倒れる）
    source: str = "2d"  # "2d" | "3d"


def _angle_at(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """点 b における a-b-c の挟角（rad, 0..pi）。いずれかが NaN なら NaN。"""
    u, v = a - b, c - b
    nu = np.linalg.norm(u, axis=-1)
    nv = np.linalg.norm(v, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.sum(u * v, axis=-1) / (nu * nv)
    return np.arccos(np.clip(cos, -1.0, 1.0))


def _midpoint(xy: np.ndarray, i: int, j: int) -> np.ndarray:
    return 0.5 * (xy[:, i, :] + xy[:, j, :])


def resolve_angle_source(cfg: FeatureConfig, has_3d: bool) -> str:
    """設定とキャッシュの状態から、実際に使う座標系を決める。"""
    if cfg.angle_source not in ("auto", "2d", "3d"):
        raise ValueError(f"未知の angle_source: {cfg.angle_source!r}（auto|2d|3d のいずれか）")
    if cfg.angle_source == "3d" and not has_3d:
        raise ValueError("angle_source='3d' だが、キャッシュに 3D が無い。先に `pose-viz lift3d` を実行する")
    if cfg.angle_source == "2d":
        return "2d"
    return "3d" if has_3d else "2d"


def joint_angles(series: TrackSeries, cfg: FeatureConfig) -> AngleFeatures:
    """関節角度を算出する。3D が使えるならそちらを使う（面外回転の系統誤差が消える）。"""
    if resolve_angle_source(cfg, series.xyz is not None) == "3d":
        return _joint_angles_3d(series, cfg)
    return _joint_angles_2d(series, cfg)


def _joint_angles_3d(series: TrackSeries, cfg: FeatureConfig) -> AngleFeatures:
    """3D 座標から角度を出す。投影による短縮が無いので面外回転の系統誤差が生じない。"""
    fps = series.fps
    xyz = series.xyz
    n = series.n_frames
    names = list(JOINT_ANGLES_3D)

    angle = np.full((n, len(names)), np.nan, dtype=np.float32)
    confidence = np.zeros((n, len(names)), dtype=np.float32)
    valid_2d = series.score >= cfg.min_score  # (T,17) COCO 順。入力が有効だったか

    for k, name in enumerate(names):
        ia, ib, ic = JOINT_ANGLES_3D[name]
        angle[:, k] = _angle_at(xyz[:, ia, :], xyz[:, ib, :], xyz[:, ic, :])
        # 3D では短縮率という概念が無いので、代わりに「入力 2D が有効だったか」を信頼度にする。
        # 元の 2D が破綻していれば 3D も破綻するため、そこだけは引き継いで判定する必要がある。
        ja, jb, jc = JOINT_ANGLES[name]
        confidence[:, k] = (valid_2d[:, ja] & valid_2d[:, jb] & valid_2d[:, jc]).astype(np.float32)
    angle[confidence <= 0] = np.nan

    ang_vel = savgol_nan(angle, fps, cfg.deriv_window_sec, cfg.deriv_polyorder, deriv=1)
    ang_accel = savgol_nan(angle, fps, cfg.deriv_window_sec, cfg.deriv_polyorder, deriv=2)

    # 体幹の傾きは 2D と同じ「画面内の符号付き傾き」のままにする（列の意味を揃えるため）。
    # 3D の x,y は入力 2D とほぼ同じなので、ここは 3D 化の恩恵を受けない点に注意。
    trunk = _midpoint(series.xy, L_SHOULDER, R_SHOULDER) - _midpoint(series.xy, L_HIP, R_HIP)
    trunk_lean = np.arctan2(trunk[:, 0], -trunk[:, 1]).astype(np.float32)

    return AngleFeatures(
        names=names,
        angle=angle,
        ang_vel=ang_vel.astype(np.float32),
        ang_accel=ang_accel.astype(np.float32),
        confidence=confidence,
        trunk_lean=trunk_lean,
        source="3d",
    )


def _joint_angles_2d(series: TrackSeries, cfg: FeatureConfig) -> AngleFeatures:
    fps = series.fps
    xy = series.xy
    n = series.n_frames
    names = list(JOINT_ANGLES)

    angle = np.full((n, len(names)), np.nan, dtype=np.float32)
    confidence = np.full((n, len(names)), np.nan, dtype=np.float32)

    for k, name in enumerate(names):
        ia, ib, ic = JOINT_ANGLES[name]
        a, b, c = xy[:, ia, :], xy[:, ib, :], xy[:, ic, :]
        angle[:, k] = _angle_at(a, b, c)

        # 短縮率 = 見かけのセグメント長 ÷ そのセグメントの真の長さ。1 に近いほどカメラ面内。
        #
        # 真の長さの基準に移動中央値を使うと、面外回転が窓より長く続いたときに基準自体が
        # 追従してしまい検出できない。四肢の長さは体幹長に対して解剖学的にほぼ一定なので、
        # 「体幹長で割った長さ」のトラック全体での高位パーセンタイルを真の長さとみなす
        # （投影は必ず長さを縮める方向にしか働かないため、最大側が真値に近い）。
        ratios = []
        for p, q in ((ia, ib), (ib, ic)):
            seg_len = np.linalg.norm(xy[:, p, :] - xy[:, q, :], axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                norm_len = seg_len / series.scale
            ref = nan_percentile(norm_len, _TRUE_LENGTH_PERCENTILE)
            if not np.isfinite(ref) or ref <= 1e-6:
                ratios.append(np.full(n, np.nan, dtype=np.float32))
                continue
            ratios.append(norm_len / ref)
        confidence[:, k] = np.clip(np.minimum(ratios[0], ratios[1]), 0.0, 1.0)

    ang_vel = savgol_nan(angle, fps, cfg.deriv_window_sec, cfg.deriv_polyorder, deriv=1)
    ang_accel = savgol_nan(angle, fps, cfg.deriv_window_sec, cfg.deriv_polyorder, deriv=2)

    # 体幹の傾き: 腰中点→肩中点ベクトルと鉛直（画像座標では -y が上）のなす符号付き角
    trunk = _midpoint(xy, L_SHOULDER, R_SHOULDER) - _midpoint(xy, L_HIP, R_HIP)
    trunk_lean = np.arctan2(trunk[:, 0], -trunk[:, 1]).astype(np.float32)

    return AngleFeatures(
        names=names,
        angle=angle,
        ang_vel=ang_vel.astype(np.float32),
        ang_accel=ang_accel.astype(np.float32),
        confidence=confidence,
        trunk_lean=trunk_lean,
    )


def mask_low_confidence(features: AngleFeatures, threshold: float) -> np.ndarray:
    """短縮率が閾値を下回る（面外回転が大きい）要素の真偽マスク (T, J) を返す。"""
    return ~(features.confidence >= threshold)
