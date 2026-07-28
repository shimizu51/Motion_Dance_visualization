"""動きの周期性（音声を使わないリズム推定）。

全身の動きの量の時系列から自己相関で支配周期を求める。音楽のビートとの同期は
ロードマップ Phase 3（librosa を導入して拍時刻と照合する）で扱う。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RhythmFeatures:
    period_sec: float  # 支配周期（秒）。求まらなければ NaN
    bpm: float  # 60 / period_sec
    confidence: float  # 0..1、正規化自己相関のピーク高（低いほど周期性が弱い）


def _prepare(signal: np.ndarray) -> np.ndarray | None:
    """NaN を内挿して平均を除去する。有効値が少なすぎれば None。"""
    x = np.asarray(signal, dtype=np.float64)
    valid = np.isfinite(x)
    if valid.sum() < 8:
        return None
    idx = np.arange(len(x))
    x = np.interp(idx, idx[valid], x[valid])
    x = x - x.mean()
    return x if np.any(x) else None


def dominant_period(
    signal: np.ndarray, fps: float, min_period_sec: float = 0.25, max_period_sec: float = 4.0
) -> RhythmFeatures:
    """自己相関のピークから支配周期を推定する。"""
    nan = RhythmFeatures(float("nan"), float("nan"), float("nan"))
    x = _prepare(signal)
    if x is None:
        return nan

    n = len(x)
    nfft = int(2 ** np.ceil(np.log2(2 * n)))
    spec = np.fft.rfft(x, nfft)
    acf = np.fft.irfft(spec * np.conj(spec), nfft)[:n]
    if acf[0] <= 0:
        return nan
    acf = acf / acf[0]

    lo = max(1, int(round(min_period_sec * fps)))
    hi = min(n - 1, int(round(max_period_sec * fps)))
    if hi <= lo:
        return nan

    band = acf[lo : hi + 1]
    k = int(np.argmax(band)) + lo
    peak = float(acf[k])
    if peak <= 0:
        return nan

    # 放物線内挿でサブサンプル精度にする
    if 0 < k < n - 1:
        y0, y1, y2 = acf[k - 1], acf[k], acf[k + 1]
        denom = y0 - 2 * y1 + y2
        if abs(denom) > 1e-12:
            k = k + 0.5 * (y0 - y2) / denom

    period = float(k / fps)
    return RhythmFeatures(period_sec=period, bpm=60.0 / period, confidence=float(np.clip(peak, 0.0, 1.0)))
