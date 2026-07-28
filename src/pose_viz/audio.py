"""音楽のビート・テンポ解析。

Phase 1 で **動きの周期は音楽のテンポと同一視できない** ことが実測で分かった
（最も長い 2 トラックが 119.7 / 103.7 BPM と一致せず、自己相関の信頼度も 0.2〜0.54 と低い）。
ダンサーごとに「何拍で 1 動作を作るか」が違うため、動きだけでは拍の倍数が定まらない。
音楽側から拍の絶対時刻を持ってくることで初めて「拍に対して動きが揃っているか」を測れる。

音声のデコードは既存の ffmpeg パイプ（`video_io.read_audio`）を使い、librosa には
波形だけを渡す（音声デコード用の依存を増やさないため）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_SAMPLE_RATE = 22050


@dataclass
class BeatInfo:
    """解析区間の先頭を 0 秒とした拍情報。"""

    tempo_bpm: float
    beat_times: np.ndarray  # (B,) float32、秒
    sample_rate: int

    def __len__(self) -> int:
        return len(self.beat_times)


def analyze_beats(samples: np.ndarray, sample_rate: int = DEFAULT_SAMPLE_RATE) -> BeatInfo | None:
    """波形からテンポと拍時刻を推定する。推定できなければ None。"""
    import librosa

    y = np.asarray(samples, dtype=np.float32)
    if y.size < sample_rate:  # 1 秒未満は諦める
        return None
    if not np.any(y):
        return None

    tempo, beats = librosa.beat.beat_track(y=y, sr=sample_rate, units="time")
    beats = np.asarray(beats, dtype=np.float32)
    if beats.size < 2:
        return None
    # librosa は tempo を配列で返すことがある
    tempo_val = float(np.atleast_1d(tempo)[0])
    return BeatInfo(tempo_bpm=tempo_val, beat_times=beats, sample_rate=sample_rate)


def analyze_video_beats(
    video_path: Path | str,
    start: float = 0.0,
    duration: float | None = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> BeatInfo | None:
    """動画の音声から拍を推定する。音声が無い・解析できない場合は None。"""
    from pose_viz.video_io import read_audio

    samples = read_audio(video_path, start=start, duration=duration, sample_rate=sample_rate)
    if samples is None:
        return None
    return analyze_beats(samples, sample_rate)


def beat_phase(times: np.ndarray, beat_times: np.ndarray) -> np.ndarray:
    """各時刻が拍の周期のどこにいるかを 0..1 で返す（0 = 拍のちょうど上）。

    拍の外側（最初の拍より前・最後の拍より後）は NaN。
    """
    times = np.asarray(times, dtype=np.float64)
    beats = np.asarray(beat_times, dtype=np.float64)
    out = np.full(len(times), np.nan)
    if len(beats) < 2:
        return out

    idx = np.searchsorted(beats, times, side="right") - 1
    inside = (idx >= 0) & (idx < len(beats) - 1)
    i = idx[inside]
    interval = beats[i + 1] - beats[i]
    good = interval > 0
    phase = np.full(inside.sum(), np.nan)
    phase[good] = (times[inside][good] - beats[i][good]) / interval[good]
    out[inside] = phase
    return out


def beat_pulse(times: np.ndarray, beat_times: np.ndarray, decay_sec: float) -> np.ndarray:
    """直近の拍からの経過時間で指数減衰する 0..1 の包絡（拍の瞬間が 1）。"""
    times = np.asarray(times, dtype=np.float64)
    beats = np.asarray(beat_times, dtype=np.float64)
    if len(beats) == 0 or decay_sec <= 0:
        return np.zeros(len(times))

    idx = np.clip(np.searchsorted(beats, times, side="right") - 1, 0, len(beats) - 1)
    elapsed = times - beats[idx]
    elapsed[elapsed < 0] = np.inf  # 最初の拍より前は光らせない
    return np.exp(-elapsed / decay_sec)
