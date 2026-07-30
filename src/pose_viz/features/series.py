"""キャッシュを、特徴量計算に使える密な時系列（`TrackSeries`）へ変換する。

キャッシュ側は「レコードが存在しない＝欠損」という表現なので、ここで欠損を NaN に開き直す。
短い欠損だけ線形補間し、長い欠損はセグメント境界にして NaN のまま残す（埋められない穴は埋めない）。
"""

from __future__ import annotations

import warnings

import numpy as np

from pose_viz.cache import ExtractCache
from pose_viz.config import FeatureConfig
from pose_viz.features.schema import (
    L_HIP,
    L_SHOULDER,
    N_KEYPOINTS,
    R_HIP,
    R_SHOULDER,
    TrackSeries,
)

#: 立位のとき肩–腰間距離は身長のおよそ 0.3 倍。体幹が取れないフレームの scale 代替に使う。
_TORSO_TO_BOX_HEIGHT = 0.3


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """True が連続する区間 [start, end) を列挙する。"""
    if len(mask) == 0:
        return []
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def _interp_short_gaps(x: np.ndarray, max_gap: int) -> tuple[np.ndarray, np.ndarray]:
    """NaN の連続長が `max_gap` 以下の欠損だけを線形補間する（両端の外挿はしない）。

    戻り値は (補間後の配列, 補間で埋めた位置のマスク)。
    """
    out = x.copy()
    filled = np.zeros(len(x), dtype=bool)
    valid = np.isfinite(x)
    if valid.sum() < 2:
        return out, filled

    idx = np.arange(len(x))
    first, last = idx[valid][0], idx[valid][-1]
    target = np.zeros(len(x), dtype=bool)
    target[first : last + 1] = True
    target &= ~valid
    if target.any():
        out[target] = np.interp(idx[target], idx[valid], x[valid])
        filled[target] = True

    # 長すぎる欠損は補間せず NaN に戻す
    for s, e in _runs(~valid):
        if e - s > max_gap:
            out[s:e] = np.nan
            filled[s:e] = False
    return out, filled


def rolling_nanmedian(x: np.ndarray, window: int) -> np.ndarray:
    """NaN を無視する移動中央値（中央揃え。端は窓が短くなる）。"""
    window = max(1, window | 1)  # 奇数化
    half = window // 2
    padded = np.pad(x.astype(np.float64), (half, half), constant_values=np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(padded, window)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # 全部 NaN の窓
        return np.nanmedian(windows, axis=-1).astype(np.float32)


def _midpoint(xy: np.ndarray, i: int, j: int) -> np.ndarray:
    """2 点の中点。片方が欠損ならもう片方をそのまま使う（両方欠損なら NaN）。"""
    a, b = xy[:, i, :], xy[:, j, :]
    va, vb = np.isfinite(a).all(axis=1), np.isfinite(b).all(axis=1)
    out = np.full((len(xy), 2), np.nan, dtype=np.float32)
    both = va & vb
    out[both] = 0.5 * (a[both] + b[both])
    out[va & ~vb] = a[va & ~vb]
    out[vb & ~va] = b[vb & ~va]
    return out


def build_track_series(cache: ExtractCache, cfg: FeatureConfig) -> dict[int, TrackSeries]:
    """キャッシュ全体を track_id ごとの `TrackSeries` に変換する。"""
    fps = float(cache.fps)
    max_gap = max(1, int(round(cfg.max_gap_sec * fps)))
    scale_win = max(1, int(round(cfg.scale_window_sec * fps)))
    use_raw = cfg.source == "raw"

    out: dict[int, TrackSeries] = {}
    for track_id, seq in cache.by_track().items():
        first, last = seq[0][0], seq[-1][0]
        n = last - first + 1
        frame_idx = np.arange(first, last + 1, dtype=np.int32)

        xy = np.full((n, N_KEYPOINTS, 2), np.nan, dtype=np.float32)
        score = np.full((n, N_KEYPOINTS), np.nan, dtype=np.float32)
        box = np.full((n, 4), np.nan, dtype=np.float32)
        observed = np.zeros(n, dtype=bool)
        xyz = None

        for fi, tf in seq:
            k = fi - first
            kp = tf.keypoints_raw if use_raw else tf.keypoints
            xy[k] = kp
            score[k] = tf.keypoint_scores
            box[k] = tf.box_xyxy
            observed[k] = True
            if tf.keypoints_3d is not None:
                if xyz is None:
                    xyz = np.full((n, N_KEYPOINTS, 3), np.nan, dtype=np.float32)
                xyz[k] = tf.keypoints_3d

        # 低スコアのキーポイントは欠損として扱う（スコアは確率ではないので閾値は経験則）
        low = ~(score >= cfg.min_score)
        xy[low] = np.nan

        # 関節ごと・軸ごとに短い欠損だけ補間する
        interpolated = np.zeros((n, N_KEYPOINTS), dtype=bool)
        for j in range(N_KEYPOINTS):
            fx, fill_x = _interp_short_gaps(xy[:, j, 0], max_gap)
            fy, fill_y = _interp_short_gaps(xy[:, j, 1], max_gap)
            xy[:, j, 0], xy[:, j, 1] = fx, fy
            interpolated[:, j] = fill_x & fill_y

        mid_shoulder = _midpoint(xy, L_SHOULDER, R_SHOULDER)
        root = _midpoint(xy, L_HIP, R_HIP)

        torso = np.linalg.norm(mid_shoulder - root, axis=1)
        scale = rolling_nanmedian(torso, scale_win)
        # 体幹が全く取れない区間は bbox 高さから代替する（無いよりは比較可能な基準を置く）
        box_h = box[:, 3] - box[:, 1]
        fallback = box_h * _TORSO_TO_BOX_HEIGHT
        need = ~np.isfinite(scale) | (scale <= 1e-3)
        scale[need] = fallback[need]
        scale[~np.isfinite(scale) | (scale <= 1e-3)] = np.nan

        xy_rel = xy - root[:, None, :]

        # 長い欠損でセグメントを分割する（またいで微分すると存在しない動きを作ってしまう）
        missing = ~observed
        usable = np.ones(n, dtype=bool)
        for s, e in _runs(missing):
            if e - s > max_gap:
                usable[s:e] = False
        segments = _runs(usable)

        out[track_id] = TrackSeries(
            track_id=track_id,
            fps=fps,
            frame_idx=frame_idx,
            t=(frame_idx / fps).astype(np.float32),
            xy=xy,
            score=score,
            box=box,
            observed=observed,
            interpolated=interpolated,
            scale=scale.astype(np.float32),
            root=root,
            xy_rel=xy_rel.astype(np.float32),
            xyz=xyz,
            segments=segments,
        )
    return out
