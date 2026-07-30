"""関節負荷の代理指標。

> **⚠ これは運動学的な代理指標であり、力学的な関節荷重ではない。**
> 真の関節負荷は逆動力学（体節質量・慣性モーメント・床反力）を要し、単眼2D映像からは求まらない。
> ここで出るのは「その関節が動画内でどれだけ激しく・偏って使われているか」の相対値であって、
> 医学的な閾値判定には使えない。値は **同じ動画の中でのみ** 比較できる。

4 つの項を等重みで合成する（重みは `feature.load_weights` で変えられる）。

1. 角速度 … どれだけ速く曲げ伸ばししているか
2. 角加速度 … どれだけ急激に速度を変えているか
3. ROM 逸脱 … その動画でのその関節の常用域からどれだけ外れているか
4. 制動 … 曲げている勢いに逆らう向きの角加速度（＝急ブレーキ）
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pose_viz.config import FeatureConfig
from pose_viz.features.angles import AngleFeatures
from pose_viz.features.filters import nan_percentile, normalize_by_percentile


@dataclass
class LoadFeatures:
    names: list[str]
    load: np.ndarray  # (T, J) 0..1、動画内相対の負荷代理値
    term_ang_vel: np.ndarray  # (T, J)
    term_ang_accel: np.ndarray  # (T, J)
    term_rom: np.ndarray  # (T, J)
    term_brake: np.ndarray  # (T, J)


def _rom_excursion(angle: np.ndarray) -> np.ndarray:
    """常用域（中央値 ± IQR）からの逸脱量を 0..1 で返す。"""
    out = np.full_like(angle, np.nan, dtype=np.float32)
    for k in range(angle.shape[1]):
        col = angle[:, k]
        q25, q50, q75 = (nan_percentile(col, q) for q in (25, 50, 75))
        band = q75 - q25
        if not np.isfinite(band) or band <= 1e-6:
            continue
        out[:, k] = np.clip((np.abs(col - q50) - band) / band, 0.0, 1.0)
    return out


def joint_load(
    angles: AngleFeatures, cfg: FeatureConfig, confidence_threshold: float | None = None
) -> LoadFeatures:
    p = cfg.load_percentile
    w = np.asarray(cfg.load_weights, dtype=np.float64)
    if w.sum() <= 0:
        w = np.ones(4)

    term_w = normalize_by_percentile(angles.ang_vel, p)
    term_a = normalize_by_percentile(angles.ang_accel, p)
    term_rom = _rom_excursion(angles.angle)
    # 制動: 角速度に逆らう向きの角加速度だけを取り出す
    brake_raw = np.where(
        np.isfinite(angles.ang_vel) & np.isfinite(angles.ang_accel),
        np.maximum(0.0, -np.sign(angles.ang_vel) * angles.ang_accel),
        np.nan,
    )
    term_brake = normalize_by_percentile(brake_raw, p)

    stack = np.stack([term_w, term_a, term_rom, term_brake], axis=0)  # (4,T,J)
    weights = w[:, None, None]
    valid = np.isfinite(stack)
    wsum = (weights * valid).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        load = (np.where(valid, stack, 0.0) * weights).sum(axis=0) / wsum
    load[wsum <= 0] = np.nan

    # 面外回転が大きく 2D 角度が信用できない区間は負荷も出さない
    thr = cfg.foreshorten_threshold if confidence_threshold is None else confidence_threshold
    load[~(angles.confidence >= thr)] = np.nan

    return LoadFeatures(
        names=angles.names,
        load=load.astype(np.float32),
        term_ang_vel=term_w,
        term_ang_accel=term_a,
        term_rom=term_rom,
        term_brake=term_brake,
    )
