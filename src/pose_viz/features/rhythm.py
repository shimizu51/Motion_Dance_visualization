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


@dataclass
class BeatSync:
    """音楽の拍に対して動きがどれだけ揃っているか（位相固定値による）。

    「最も近い拍までの平均距離」で測ると、**拍から一定量ずれたまま正確に刻んでいる**踊りを
    非同期と誤判定する。実際には一定のずれを保っていること自体が同期なので、円統計の
    **位相固定値（resultant vector length）** を使う。位相の一様なばらつきだけを見て、
    平均的なずれ（`mean_phase`）は別に報告する。
    """

    phase_lock: float  # 0..1。1 = 位相が完全に一点、0 = 一様にばらつく
    #: 一様分布でも有限個の標本では R が 0 にならない。その期待値 √π/(2√n)。
    #: **`phase_lock` はこの値と比べて初めて意味を持つ。**
    chance_level: float
    lock_ratio: float  # phase_lock / chance_level（1 倍ならランダムと区別できない）
    mean_phase: float  # 0..1。どの位相に寄っているか（0 = 拍の上、0.5 = 拍の裏）
    subdivision: int  # 1 = 拍そのもの、2 = 8分、4 = 16分
    n_onsets: int


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

    # **探索範囲の最大値ではなく、局所ピークを取る。**
    # 自己相関は原点から単調に下がる中央ローブを持つため、単純な argmax では
    # 「範囲の左端」＝中央ローブの裾を掴んでしまい、平滑化フィルタの窓幅を周期として
    # 報告してしまう（実測で全トラックが探索下限に張り付いた）。真の周期は局所ピーク。
    from scipy.signal import find_peaks

    peaks, _ = find_peaks(acf[: hi + 2])
    peaks = peaks[(peaks >= lo) & (peaks <= hi)]
    if peaks.size == 0:
        return nan
    k = int(peaks[np.argmax(acf[peaks])])
    peak = float(acf[k])
    if peak <= 0:
        return nan

    # 放物線内挿でサブサンプル精度にする。
    # 補正量は必ず ±0.5 サンプル以内に収める。ピーク近傍が平坦だと分母が 0 に近づき、
    # 補正が発散して負の周期などあり得ない値になるため（実測で -1.18 秒が出た）。
    if 0 < k < n - 1:
        y0, y1, y2 = acf[k - 1], acf[k], acf[k + 1]
        denom = y0 - 2 * y1 + y2
        if abs(denom) > 1e-12:
            k = k + float(np.clip(0.5 * (y0 - y2) / denom, -0.5, 0.5))

    period = float(k / fps)
    return RhythmFeatures(period_sec=period, bpm=60.0 / period, confidence=float(np.clip(peak, 0.0, 1.0)))


def motion_onsets(
    signal: np.ndarray,
    fps: float,
    t: np.ndarray,
    min_interval_sec: float = 0.25,
    prominence_k: float = 0.5,
) -> np.ndarray:
    """動きの「節目」の時刻を返す（全身速さの極大点）。

    高さの閾値だけで拾うと **平滑化フィルタのリップルを節目として数えてしまう**
    （実測で、検出間隔の中央値が Savitzky-Golay の窓幅とぴたり一致してしまった）。
    周囲からどれだけ突き出ているかを見る prominence で選ぶことで、その混入を防ぐ。
    """
    from scipy.signal import find_peaks

    x = np.asarray(signal, dtype=np.float64)
    finite = np.isfinite(x)
    if finite.sum() < 8:
        return np.zeros(0, dtype=np.float32)

    filled = x.copy()
    idx = np.arange(len(x))
    filled[~finite] = np.interp(idx[~finite], idx[finite], x[finite])
    prominence = prominence_k * float(np.std(filled))
    peaks, _ = find_peaks(
        filled,
        prominence=prominence if prominence > 0 else None,
        distance=max(1, int(round(min_interval_sec * fps))),
    )
    return np.asarray(t, dtype=np.float32)[peaks]


def subdivide_beats(beat_times: np.ndarray, subdivision: int) -> np.ndarray:
    """拍と拍の間を `subdivision` 等分する（8分・16分での同期を見るため）。

    **等間隔のグリッドを合成してはいけない。** 実際の拍は演奏の揺れに追従して等間隔ではなく
    （実測で標準偏差 0.02 秒、最小 0.65 〜 最大 0.86 秒）、合成グリッドで剰余を取ると
    数分の間に位相がずれ切ってしまう。必ず実測の拍の間を分割する。
    """
    b = np.asarray(beat_times, dtype=np.float64)
    if subdivision <= 1 or b.size < 2:
        return b
    parts = [b[:-1] + (b[1:] - b[:-1]) * k / subdivision for k in range(subdivision)]
    return np.sort(np.concatenate(parts + [b[-1:]]))


def beat_sync(onset_times: np.ndarray, beat_times: np.ndarray, subdivision: int = 1) -> BeatSync:
    """動きの節目が拍のどの位相に集まるかを、円統計の位相固定値で測る。"""
    from pose_viz.audio import beat_phase

    onsets = np.asarray(onset_times, dtype=np.float64)
    grid = subdivide_beats(beat_times, subdivision)
    nan = BeatSync(float("nan"), float("nan"), float("nan"), float("nan"), subdivision, len(onsets))
    if onsets.size == 0 or grid.size < 2:
        return nan

    phase = beat_phase(onsets, grid)
    phase = phase[np.isfinite(phase)]
    if phase.size < 5:
        return nan

    vec = np.mean(np.exp(2j * np.pi * phase))
    r = float(abs(vec))
    chance = float(np.sqrt(np.pi) / (2 * np.sqrt(phase.size)))
    return BeatSync(
        phase_lock=r,
        chance_level=chance,
        lock_ratio=float(r / chance) if chance > 0 else float("nan"),
        mean_phase=float((np.angle(vec) / (2 * np.pi)) % 1.0),
        subdivision=subdivision,
        n_onsets=int(phase.size),
    )
