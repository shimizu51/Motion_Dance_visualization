"""音楽の拍解析と、拍に対する動きの同期度を確かめる。

既知の BPM のクリック音を合成して流す（正解が分かっている唯一の検証手段）。
"""

from __future__ import annotations

import numpy as np
import pytest

from pose_viz.audio import DEFAULT_SAMPLE_RATE, analyze_beats, beat_phase, beat_pulse
from pose_viz.config import Config
from pose_viz.features.rhythm import beat_sync, motion_onsets

SR = DEFAULT_SAMPLE_RATE
FPS = 30.0


def _click_track(bpm: float, duration: float = 12.0, sr: int = SR) -> np.ndarray:
    """一定 BPM のクリック音（短い減衰つき正弦波）を並べた波形を作る。"""
    n = int(duration * sr)
    y = np.zeros(n, dtype=np.float32)
    period = 60.0 / bpm
    click_len = int(0.05 * sr)
    env = np.exp(-np.linspace(0, 8, click_len)).astype(np.float32)
    tone = np.sin(2 * np.pi * 1000 * np.arange(click_len) / sr).astype(np.float32) * env
    t = 0.0
    while int(t * sr) + click_len < n:
        s = int(t * sr)
        y[s : s + click_len] += tone
        t += period
    return y


# --- テンポ推定 -------------------------------------------------------------


@pytest.mark.parametrize("bpm", [90.0, 120.0, 140.0])
def test_analyze_beats_recovers_a_known_tempo(bpm):
    info = analyze_beats(_click_track(bpm), SR)
    assert info is not None
    # 半分・倍のテンポに取られることがあるので、倍数関係も許容して判定する
    ratio = info.tempo_bpm / bpm
    assert min(abs(ratio - r) for r in (0.5, 1.0, 2.0)) < 0.05, f"推定 {info.tempo_bpm:.1f} BPM"


def test_detected_beats_are_evenly_spaced():
    info = analyze_beats(_click_track(120.0), SR)
    intervals = np.diff(info.beat_times)
    assert np.std(intervals) / np.mean(intervals) < 0.05  # 一定間隔で刻まれている


def test_analyze_beats_returns_none_for_silence_and_short_input():
    assert analyze_beats(np.zeros(SR * 5, dtype=np.float32), SR) is None
    assert analyze_beats(np.ones(100, dtype=np.float32), SR) is None


# --- 拍の位相・包絡 ---------------------------------------------------------


def test_beat_phase_is_zero_on_the_beat_and_half_way_between():
    beats = np.arange(0, 10, 0.5)
    assert np.allclose(beat_phase(np.array([1.0, 2.0]), beats), 0.0, atol=1e-6)
    assert np.allclose(beat_phase(np.array([1.25]), beats), 0.5, atol=1e-6)


def test_beat_phase_is_nan_outside_the_beat_range():
    beats = np.array([1.0, 2.0, 3.0])
    assert np.isnan(beat_phase(np.array([0.5]), beats)).all()
    assert np.isnan(beat_phase(np.array([9.0]), beats)).all()


def test_beat_pulse_peaks_on_the_beat_and_decays():
    beats = np.array([1.0, 2.0])
    pulse = beat_pulse(np.array([1.0, 1.05, 1.5]), beats, decay_sec=0.12)
    assert pulse[0] == pytest.approx(1.0)
    assert pulse[0] > pulse[1] > pulse[2]


def test_beat_pulse_is_zero_before_the_first_beat():
    assert beat_pulse(np.array([0.0, 0.5]), np.array([1.0, 2.0]), 0.12).max() == 0.0


def test_beat_pulse_is_zero_without_beats():
    assert not np.any(beat_pulse(np.arange(10.0), np.zeros(0), 0.12))


# --- 動きと拍の同期度 -------------------------------------------------------


def _onsets_at(times, jitter=0.0, seed=0):
    rng = np.random.default_rng(seed)
    return np.asarray(times) + rng.normal(0, jitter, len(times))


def test_perfectly_synced_motion_locks_completely():
    beats = np.arange(0.0, 20.0, 0.5)
    r = beat_sync(_onsets_at(beats[::2]), beats)
    assert r.phase_lock == pytest.approx(1.0, abs=1e-6)
    assert r.lock_ratio > 5


def test_random_motion_is_indistinguishable_from_chance():
    """拍と無関係に打つと、位相固定値は一様分布の期待値と同程度になる。"""
    beats = np.arange(0.0, 60.0, 0.5)
    rng = np.random.default_rng(3)
    r = beat_sync(np.sort(rng.uniform(0, 60, 300)), beats)
    assert r.lock_ratio < 2.0


def test_phase_lock_decreases_as_timing_gets_sloppier():
    beats = np.arange(0.0, 60.0, 0.5)
    tight = beat_sync(_onsets_at(beats[::2], jitter=0.02, seed=1), beats).phase_lock
    loose = beat_sync(_onsets_at(beats[::2], jitter=0.12, seed=1), beats).phase_lock
    assert tight > loose


def test_a_constant_offset_still_counts_as_synced():
    """拍から一定量ずれたまま正確に刻む踊りは同期とみなす（これが平均距離では測れない）。"""
    beats = np.arange(0.0, 60.0, 0.5)
    shifted = beat_sync(beats[::2] + 0.15, beats)  # 常に 0.15 秒だけ後ろ
    assert shifted.phase_lock > 0.95
    assert shifted.mean_phase == pytest.approx(0.3, abs=0.02)  # 0.15 / 0.5 拍


def test_phase_lock_is_comparable_across_tempos():
    """無次元量なので、テンポが違っても「揃い具合」として比べられる。"""
    scores = [
        beat_sync(
            _onsets_at(np.arange(0.0, 60.0, p), jitter=0.05 * p, seed=2), np.arange(0.0, 60.0, p)
        ).phase_lock
        for p in (0.4, 0.5, 0.8)
    ]
    assert max(scores) - min(scores) < 0.1


def test_subdivision_detects_offbeat_motion():
    """裏拍でも動いている踊りは、8分に分割して初めて同期として見える。"""
    beats = np.arange(0.0, 60.0, 0.5)
    onsets = np.sort(np.concatenate([beats, beats[:-1] + 0.25]))  # 表と裏の両方
    assert beat_sync(onsets, beats, subdivision=1).phase_lock < 0.2
    assert beat_sync(onsets, beats, subdivision=2).phase_lock > 0.9


def test_subdivide_beats_follows_the_actual_beats():
    """等間隔グリッドを合成せず、揺れる実測の拍の間を分割する。"""
    from pose_viz.features.rhythm import subdivide_beats

    beats = np.array([0.0, 1.0, 3.0])  # わざと不等間隔
    assert np.allclose(subdivide_beats(beats, 2), [0.0, 0.5, 1.0, 2.0, 3.0])
    assert np.allclose(subdivide_beats(beats, 1), beats)


def test_beat_sync_is_nan_without_enough_data():
    assert np.isnan(beat_sync(np.zeros(0), np.arange(10.0)).phase_lock)
    assert np.isnan(beat_sync(np.arange(10.0), np.array([1.0])).phase_lock)


def test_chance_level_shrinks_with_more_onsets():
    """位相固定値は有限標本では 0 にならない。比較の基準を必ず併記する。"""
    beats = np.arange(0.0, 200.0, 0.5)
    rng = np.random.default_rng(5)
    few = beat_sync(np.sort(rng.uniform(0, 200, 20)), beats)
    many = beat_sync(np.sort(rng.uniform(0, 200, 400)), beats)
    assert few.chance_level > many.chance_level
    assert few.phase_lock > many.phase_lock  # ランダムでも標本が少ないほど R は大きく出る


def test_motion_onsets_find_the_peaks_of_a_periodic_signal():
    t = np.arange(0, 10.0, 1.0 / FPS)
    signal = np.sin(2 * np.pi * t / 1.0) ** 2  # 1 秒ごとに 2 山
    onsets = motion_onsets(signal, FPS, t)
    assert len(onsets) >= 15
    assert np.std(np.diff(onsets)) < 0.1  # 等間隔に並ぶ


def test_motion_onsets_tolerate_missing_values():
    t = np.arange(0, 10.0, 1.0 / FPS)
    signal = np.sin(2 * np.pi * t) ** 2
    signal[100:110] = np.nan
    assert len(motion_onsets(signal, FPS, t)) > 10


# --- 後方互換 ---------------------------------------------------------------


def test_beat_effects_are_off_by_default():
    """既定では拍の演出が無効で、従来と同一の描画になる。"""
    from pose_viz.cli import DEFAULT_CONFIG_PATH

    for cfg in (Config(), Config.load(DEFAULT_CONFIG_PATH)):
        assert cfg.render.beat_bloom == 0.0


def test_beat_settings_do_not_invalidate_the_cache():
    from pose_viz.cli import DEFAULT_CONFIG_PATH

    cfg = Config.load(DEFAULT_CONFIG_PATH)
    before = cfg.extract_hash()
    cfg.render.beat_bloom = 1.5
    cfg.render.beat_decay_sec = 0.3
    assert cfg.extract_hash() == before
