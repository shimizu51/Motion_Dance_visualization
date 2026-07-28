"""姿勢統計量 — 重心・全身速さ・収縮指数・Quantity of Motion・左右対称性。

収縮指数と QoM は **キャッシュ済みの人物マスクから計算する**（Camurri らの表情豊かなジェスチャ
解析で使われる古典的な指標）。追加のモデル推論は要らず、既にディスクにあるデータだけで求まる。
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from pose_viz.cache import TrackFrame, decode_mask_crop_only, mask_crop_origin
from pose_viz.config import FeatureConfig
from pose_viz.features.filters import savgol_nan
from pose_viz.features.schema import (
    BODY_SEGMENTS,
    LIMB_PAIRS,
    L_HIP,
    L_SHOULDER,
    R_HIP,
    R_SHOULDER,
    TrackSeries,
    _VIRTUAL_MID_HIP,
    _VIRTUAL_MID_SHOULDER,
)


@dataclass
class PostureFeatures:
    com: np.ndarray  # (T, 2) 体節質量比で加重した重心（px）
    com_speed: np.ndarray  # (T,) body-length/s
    body_speed: np.ndarray  # (T,) 体節質量比で加重した全身の動きの量 body-length/s
    mass_coverage: np.ndarray  # (T,) 重心の算出に使えた体節質量の割合（信頼度）
    contraction_index: np.ndarray  # (T,) シルエット面積 ÷ bbox 面積。縮こまるほど大きい
    qom: np.ndarray  # (T,) Quantity of Motion（シルエット差分面積 ÷ シルエット面積）
    facing_sign: np.ndarray  # (T,) 体の向き（+1/-1）。符号反転が向き変え
    lr_suspect: np.ndarray  # (T,) bool。左右取り違えが疑われるフレーム
    symmetry_index: np.ndarray  # (T,) 0..2。左右の動きの量の非対称度（0 = 完全対称）


def _resolve(xy: np.ndarray, idx: int) -> np.ndarray:
    """体節端点のインデックスを解決する（負の値は肩中点・腰中点の仮想点）。"""
    if idx == _VIRTUAL_MID_SHOULDER:
        return 0.5 * (xy[:, L_SHOULDER, :] + xy[:, R_SHOULDER, :])
    if idx == _VIRTUAL_MID_HIP:
        return 0.5 * (xy[:, L_HIP, :] + xy[:, R_HIP, :])
    return xy[:, idx, :]


def segment_com(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """各体節の重心位置 (T, S, 2) と質量比 (S,) を返す。"""
    coms = []
    masses = []
    for prox_i, dist_i, mass, ratio in BODY_SEGMENTS:
        p, d = _resolve(xy, prox_i), _resolve(xy, dist_i)
        coms.append((1.0 - ratio) * p + ratio * d)
        masses.append(mass)
    return np.stack(coms, axis=1), np.array(masses, dtype=np.float32)


def _rolling_nanrms(x: np.ndarray, window: int) -> np.ndarray:
    """NaN を無視する移動 RMS（中央揃え）。"""
    window = max(1, window | 1)
    half = window // 2
    padded = np.pad(x.astype(np.float64) ** 2, (half, half), constant_values=np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(padded, window)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.sqrt(np.nanmean(windows, axis=-1)).astype(np.float32)


def _runs_of_sign(sign: np.ndarray) -> list[tuple[int, int, float]]:
    """符号が一定な連続区間 [start, end, sign) を列挙する（0 と NaN は区切り扱い）。"""
    out = []
    start, cur = None, 0.0
    for i, s in enumerate(sign):
        if not np.isfinite(s) or s == 0:
            if start is not None:
                out.append((start, i, cur))
                start = None
            continue
        if start is None:
            start, cur = i, s
        elif s != cur:
            out.append((start, i, cur))
            start, cur = i, s
    if start is not None:
        out.append((start, len(sign), cur))
    return out


def detect_lr_flip(series: TrackSeries, cfg: FeatureConfig) -> tuple[np.ndarray, np.ndarray]:
    """左右取り違えが疑われるフレームを検出する。

    体軸ベクトル（腰中点→肩中点）と肩ベクトル（左肩→右肩）の外積の符号は、被写体が
    カメラを向いているか背を向けているかを表す。実際に向きを変えたなら符号は数フレーム
    かけて変わり、その後は持続する。**ごく短時間で元に戻る符号反転は、向き変えではなく
    推定器のラベル取り違え**とみなす。
    """
    xy = series.xy
    up = 0.5 * (xy[:, L_SHOULDER, :] + xy[:, R_SHOULDER, :]) - 0.5 * (xy[:, L_HIP, :] + xy[:, R_HIP, :])
    lr = xy[:, R_SHOULDER, :] - xy[:, L_SHOULDER, :]
    cross = up[:, 0] * lr[:, 1] - up[:, 1] * lr[:, 0]
    sign = np.sign(cross).astype(np.float32)
    sign[~np.isfinite(cross)] = np.nan

    min_frames = max(1, int(round(cfg.flip_min_sec * series.fps)))
    suspect = np.zeros(series.n_frames, dtype=bool)
    for s, e, _ in _runs_of_sign(sign):
        if e - s < min_frames:
            suspect[s:e] = True
    return sign, suspect


def _mask_features(
    series: TrackSeries, seq: list[tuple[int, TrackFrame]]
) -> tuple[np.ndarray, np.ndarray]:
    """収縮指数と QoM をキャッシュ済みマスクから計算する。

    QoM は連続フレームのシルエット差分面積。2 フレームの bbox が動くため、両者を包含する
    最小の矩形に載せ替えてから XOR を取る（フル解像度キャンバスを毎回作らない）。
    """
    n = series.n_frames
    first = int(series.frame_idx[0])
    ci = np.full(n, np.nan, dtype=np.float32)
    qom = np.full(n, np.nan, dtype=np.float32)

    prev = None  # (frame_idx, crop, x0, y0)
    for fi, tf in seq:
        k = fi - first
        if tf.mask_png is None:
            prev = None
            continue
        crop = decode_mask_crop_only(tf.mask_png)
        if crop is None or crop.size == 0:
            prev = None
            continue
        x0, y0 = mask_crop_origin(tf.box_xyxy)
        area = int(crop.sum())
        ci[k] = area / float(crop.size)

        if prev is not None and fi - prev[0] == 1 and area > 0:
            pfi, pcrop, px0, py0 = prev
            ux0, uy0 = min(x0, px0), min(y0, py0)
            ux1 = max(x0 + crop.shape[1], px0 + pcrop.shape[1])
            uy1 = max(y0 + crop.shape[0], py0 + pcrop.shape[0])
            a = np.zeros((uy1 - uy0, ux1 - ux0), dtype=bool)
            b = np.zeros_like(a)
            a[y0 - uy0 : y0 - uy0 + crop.shape[0], x0 - ux0 : x0 - ux0 + crop.shape[1]] = crop
            b[py0 - uy0 : py0 - uy0 + pcrop.shape[0], px0 - ux0 : px0 - ux0 + pcrop.shape[1]] = pcrop
            qom[k] = float((a ^ b).sum()) / float(area)

        prev = (fi, crop, x0, y0)
    return ci, qom


def posture_features(
    series: TrackSeries,
    seq: list[tuple[int, TrackFrame]],
    speed_rel: np.ndarray,
    cfg: FeatureConfig,
) -> PostureFeatures:
    fps = series.fps
    scale = series.scale

    coms, masses = segment_com(series.xy)  # (T,S,2), (S,)
    valid = np.isfinite(coms).all(axis=2)  # (T,S)
    w = np.where(valid, masses[None, :], 0.0)
    total = w.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        com = np.einsum("ts,tsi->ti", w, np.nan_to_num(coms)) / total[:, None]
    com[total <= 0] = np.nan
    mass_coverage = (total / masses.sum()).astype(np.float32)

    v_com = savgol_nan(com, fps, cfg.deriv_window_sec, cfg.deriv_polyorder, deriv=1)
    com_speed = (np.linalg.norm(v_com, axis=1) / scale).astype(np.float32)

    # 全身の動きの量: 体節重心速度の質量加重平均
    v_seg = savgol_nan(coms, fps, cfg.deriv_window_sec, cfg.deriv_polyorder, deriv=1)
    seg_speed = np.linalg.norm(v_seg, axis=2)  # (T,S)
    with np.errstate(invalid="ignore", divide="ignore"):
        body_speed = np.einsum("ts,ts->t", w, np.nan_to_num(seg_speed)) / total
    body_speed = (body_speed / scale).astype(np.float32)
    body_speed[total <= 0] = np.nan

    ci, qom = _mask_features(series, seq)
    facing_sign, lr_suspect = detect_lr_flip(series, cfg)

    # 左右対称性: 対になる四肢の速さの移動 RMS を比べる
    sym_win = max(1, int(round(cfg.symmetry_window_sec * fps)))
    sis = []
    for li, ri in LIMB_PAIRS:
        rl = _rolling_nanrms(speed_rel[:, li], sym_win)
        rr = _rolling_nanrms(speed_rel[:, ri], sym_win)
        denom = 0.5 * (rl + rr)
        with np.errstate(invalid="ignore", divide="ignore"):
            sis.append(np.where(denom > 1e-6, np.abs(rl - rr) / denom, np.nan))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        symmetry_index = np.nanmean(np.stack(sis, axis=1), axis=1).astype(np.float32)
    # 左右取り違えが疑われる区間の対称性は信用できない
    symmetry_index[lr_suspect] = np.nan

    return PostureFeatures(
        com=com.astype(np.float32),
        com_speed=com_speed,
        body_speed=body_speed,
        mass_coverage=mass_coverage,
        contraction_index=ci,
        qom=qom,
        facing_sign=facing_sign,
        lr_suspect=lr_suspect,
        symmetry_index=symmetry_index,
    )
