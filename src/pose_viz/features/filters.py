"""計測用のゼロ位相微分フィルタ。

One-Euro（`pose.py`）は描画用の低遅延・非対称なオンラインフィルタで、位相遅れを持つため
その出力を微分すると速度・加速度が減衰する。計測側では Savitzky-Golay を使う。

- **ゼロ位相**（対称窓）なので時刻がずれない
- **多項式当てはめの解析微分**なので、差分の後に平滑化するより高周波ノイズに強い
- 窓幅を「秒」で指定するため、fps が変わっても同じ帯域を落とす
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def window_length(window_sec: float, fps: float, polyorder: int) -> int:
    """秒指定の窓を、多項式次数より大きい奇数のサンプル数に変換する。"""
    win = int(round(window_sec * fps))
    win = max(win, polyorder + 1)
    if win % 2 == 0:
        win += 1
    return win


def savgol_nan(
    x: np.ndarray,
    fps: float,
    window_sec: float,
    polyorder: int = 3,
    deriv: int = 0,
    trim_edges: bool | None = None,
) -> np.ndarray:
    """NaN を含む時系列に Savitzky-Golay を適用する（`deriv` 階微分、単位は 1/秒^deriv）。

    scipy の `savgol_filter` は NaN を伝播させてしまうため、NaN で区切られた連続区間ごとに
    個別に適用する。窓に満たない短い区間は **埋めずに NaN のまま返す**（無理に値を作らない）。

    区間の端では窓を中央に取れず、`savgol_filter` は端の多項式を外挿して値を出す。この外挿は
    ノイズに弱く、**微分では区間境界に非現実的なスパイクを生む**（実測でトラック分割の直後に
    体幹長の 17 倍/秒という速度が出た）。そのため `deriv >= 1` では既定で端の半窓分を NaN に
    する。「窓が足りないところでは値を作らない」という上の方針と同じ扱いにそろえたもの。

    `x` は先頭軸を時間とする任意形状。先頭軸以外は独立した信号として扱う。
    """
    x = np.asarray(x, dtype=np.float64)
    orig_shape = x.shape
    flat = x.reshape(orig_shape[0], -1)
    out = np.full_like(flat, np.nan)
    win = window_length(window_sec, fps, polyorder)
    half = win // 2
    delta = 1.0 / fps
    if trim_edges is None:
        trim_edges = deriv >= 1

    for c in range(flat.shape[1]):
        col = flat[:, c]
        valid = np.isfinite(col)
        if not valid.any():
            continue
        d = np.diff(np.concatenate(([0], valid.astype(np.int8), [0])))
        for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
            if e - s < win:
                continue  # 窓に満たない区間は微分を定義しない
            filtered = savgol_filter(
                col[s:e], window_length=win, polyorder=polyorder, deriv=deriv, delta=delta
            )
            if trim_edges:
                out[s + half : e - half, c] = filtered[half : len(filtered) - half]
            else:
                out[s:e, c] = filtered
    return out.reshape(orig_shape)


def nan_percentile(x: np.ndarray, q: float) -> float:
    """NaN を無視したパーセンタイル。有効値が無ければ NaN。"""
    v = x[np.isfinite(x)]
    return float(np.percentile(v, q)) if v.size else float("nan")


def normalize_by_percentile(x: np.ndarray, q: float) -> np.ndarray:
    """動画内の第 q パーセンタイルで割り、0..1 に切り詰める（動画内相対の指標にする）。"""
    ref = nan_percentile(np.abs(x), q)
    if not np.isfinite(ref) or ref <= 0:
        return np.full_like(x, np.nan, dtype=np.float32)
    return np.clip(np.abs(x) / ref, 0.0, 1.0).astype(np.float32)
