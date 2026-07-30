"""Savitzky-Golay 微分が解析解と一致することを確かめる。"""

from __future__ import annotations

import numpy as np

from pose_viz.features.filters import savgol_nan, window_length

FPS = 30.0


def _sine(freq: float, duration: float = 5.0, amp: float = 1.0):
    t = np.arange(0, duration, 1.0 / FPS)
    return t, amp * np.sin(2 * np.pi * freq * t)


def test_window_length_is_odd_and_above_polyorder():
    assert window_length(0.25, 30.0, 3) % 2 == 1
    assert window_length(0.01, 30.0, 3) > 3  # 窓が短すぎても多項式次数は満たす


def test_window_covers_the_requested_duration_at_any_fps():
    """窓は「秒」指定。奇数化の丸めはあるが、実効的な窓長は fps によらず指定秒に一致する。"""
    for fps in (24.0, 29.97, 30.0, 60.0):
        win = window_length(0.25, fps, 3)
        assert win % 2 == 1
        assert abs(win / fps - 0.25) <= 2.0 / fps  # 丸め誤差は 2 サンプル以内


def test_first_derivative_matches_analytic():
    freq = 1.0
    t, x = _sine(freq)
    v = savgol_nan(x, FPS, window_sec=0.25, polyorder=3, deriv=1)
    expected = 2 * np.pi * freq * np.cos(2 * np.pi * freq * t)
    inner = slice(15, -15)  # 端は窓が外挿になるので内側で比較する
    assert np.allclose(v[inner], expected[inner], rtol=0.02, atol=0.05)


def test_second_derivative_matches_analytic():
    freq = 1.0
    t, x = _sine(freq)
    a = savgol_nan(x, FPS, window_sec=0.25, polyorder=3, deriv=2)
    expected = -((2 * np.pi * freq) ** 2) * np.sin(2 * np.pi * freq * t)
    inner = slice(15, -15)
    assert np.allclose(a[inner], expected[inner], rtol=0.05, atol=0.5)


def test_derivative_is_zero_phase():
    """ゼロ位相であること: 微分結果のピーク位置が解析解とずれない。

    One-Euro のような因果フィルタだとここがずれる。計測に SavGol を使う理由そのもの。
    """
    t, x = _sine(1.0)
    v = savgol_nan(x, FPS, window_sec=0.25, polyorder=3, deriv=1)
    expected = 2 * np.pi * np.cos(2 * np.pi * t)
    inner = slice(15, -15)
    lag = np.argmax(np.correlate(v[inner], expected[inner], mode="same")) - len(v[inner]) // 2
    assert lag == 0


def test_nan_segments_are_handled_independently():
    t, x = _sine(1.0)
    x = x.copy()
    x[60:80] = np.nan
    v = savgol_nan(x, FPS, window_sec=0.25, polyorder=3, deriv=1)
    assert np.all(np.isnan(v[60:80]))  # 欠損はそのまま欠損
    expected = 2 * np.pi * np.cos(2 * np.pi * t)
    assert np.allclose(v[15:45], expected[15:45], rtol=0.02, atol=0.05)  # 前の区間は無傷
    assert np.isfinite(v[100:130]).all()  # 後ろの区間も独立に計算される


def test_short_segment_yields_nan_not_a_guess():
    """窓に満たない区間では値を作らない（無理に埋めない）。"""
    x = np.full(200, np.nan)
    x[100:104] = [0.0, 1.0, 2.0, 3.0]  # 窓(9)より短い
    v = savgol_nan(x, FPS, window_sec=0.3, polyorder=3, deriv=1)
    assert np.all(np.isnan(v))


def test_derivative_trims_segment_edges():
    """微分では区間の端を NaN にする（外挿由来のスパイクを出さない）。"""
    t, x = _sine(1.0)
    win = window_length(0.25, FPS, 3)
    half = win // 2
    v = savgol_nan(x, FPS, 0.25, 3, deriv=1)
    assert np.all(np.isnan(v[:half])) and np.all(np.isnan(v[-half:]))
    assert np.isfinite(v[half:-half]).all()


def test_smoothing_does_not_trim_edges():
    """deriv=0（平滑化）は外挿の危険が無いので端を残す。"""
    _, x = _sine(1.0)
    y = savgol_nan(x, FPS, 0.25, 3, deriv=0)
    assert np.isfinite(y).all()


def test_edge_trim_removes_spikes_at_a_segment_boundary():
    """欠損で分割された区間の直後に、非現実的な速度が出ないこと。

    実データではトラック分割の直後に体幹長の 17 倍/秒という速度が出ていた。
    """
    _, x = _sine(1.0, duration=10.0)
    x = x.copy()
    x[100:140] = np.nan  # 長い欠損で区間を分割する
    v = savgol_nan(x, FPS, 0.25, 3, deriv=1)
    analytic_max = 2 * np.pi  # 振幅 1・1Hz の正弦波の最大速度
    assert np.nanmax(np.abs(v)) < analytic_max * 1.1


def test_all_nan_input_is_safe():
    v = savgol_nan(np.full(50, np.nan), FPS, 0.25, 3, deriv=1)
    assert v.shape == (50,) and np.all(np.isnan(v))


def test_multidimensional_input_is_per_signal():
    """(T, J, 2) のような多次元でも、先頭軸以外は独立した信号として扱う。"""
    t, x = _sine(1.0)
    stacked = np.stack([x, 2 * x], axis=1)
    v = savgol_nan(stacked, FPS, 0.25, 3, deriv=1)
    inner = slice(15, -15)
    assert np.allclose(v[inner, 1], 2 * v[inner, 0], rtol=1e-6)
