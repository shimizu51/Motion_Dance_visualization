"""周期推定・左右反転検出・負荷代理指標の性質を確かめる。"""

from __future__ import annotations

import numpy as np
import pytest

from pose_viz.config import FeatureConfig
from pose_viz.features import schema as S
from pose_viz.features.angles import joint_angles
from pose_viz.features.load import joint_load
from pose_viz.features.posture import detect_lr_flip, posture_features
from pose_viz.features.rhythm import dominant_period
from pose_viz.features.series import build_track_series
from tests.synthetic import bend_elbow, make_cache, standing_pose

FPS = 30.0
STATURE = 600.0


# --- 周期推定 --------------------------------------------------------------


@pytest.mark.parametrize("period", [0.5, 0.75, 1.0, 2.0])
def test_dominant_period_recovers_a_known_period(period):
    t = np.arange(0, 12.0, 1.0 / FPS)
    signal = np.sin(2 * np.pi * t / period)
    result = dominant_period(signal, FPS)
    assert abs(result.period_sec - period) < 0.05
    assert abs(result.bpm - 60.0 / period) < 5.0
    assert result.confidence > 0.8


def test_dominant_period_reports_low_confidence_for_noise():
    rng = np.random.default_rng(1)
    result = dominant_period(rng.normal(size=int(12 * FPS)), FPS)
    assert not np.isfinite(result.period_sec) or result.confidence < 0.5


def test_dominant_period_survives_missing_values():
    t = np.arange(0, 12.0, 1.0 / FPS)
    signal = np.sin(2 * np.pi * t / 1.0)
    signal[100:110] = np.nan
    assert abs(dominant_period(signal, FPS).period_sec - 1.0) < 0.05


def test_dominant_period_ignores_the_central_lobe():
    """自己相関の中央ローブの裾を周期と誤認しないこと。

    平滑化された信号の自己相関は原点から単調に下がるため、探索範囲の単純な最大値を取ると
    「範囲の左端」を返してしまう。実データで全トラックが探索下限（＝平滑化フィルタの窓幅）に
    張り付き、音楽の拍と無関係な値を報告していた。
    """
    period = 1.0
    t = np.arange(0, 30.0, 1.0 / FPS)
    # 緩やかな正弦波（自己相関の中央ローブが広く、下限側が高い値を持つ信号）
    signal = np.sin(2 * np.pi * t / period)
    r = dominant_period(signal, FPS, min_period_sec=0.25, max_period_sec=4.0)
    assert abs(r.period_sec - period) < 0.05, f"下限に張り付いた: {r.period_sec}"


def test_dominant_period_never_returns_an_impossible_value():
    """周期は必ず正で、指定した探索範囲に収まること。

    放物線内挿の補正を制限していないと、ピーク近傍が平坦な信号で補正が発散し、
    負の周期が出てしまう（実データで -1.18 秒を観測した）。
    """
    rng = np.random.default_rng(0)
    for seed in range(30):
        rng = np.random.default_rng(seed)
        # 周期性が弱く自己相関が平坦になりやすい信号を色々流す
        t = np.arange(0, 20.0, 1.0 / FPS)
        signal = rng.normal(size=len(t)) + 0.3 * np.sin(2 * np.pi * t / rng.uniform(0.3, 3.0))
        r = dominant_period(signal, FPS, 0.25, 4.0)
        if np.isfinite(r.period_sec):
            assert 0.25 - 1 / FPS <= r.period_sec <= 4.0 + 1 / FPS, f"seed={seed}: {r.period_sec}"
            assert r.bpm > 0


def test_dominant_period_is_nan_for_degenerate_input():
    assert np.isnan(dominant_period(np.zeros(300), FPS).period_sec)
    assert np.isnan(dominant_period(np.array([1.0, 2.0]), FPS).period_sec)


# --- 左右取り違えの検出 ------------------------------------------------------


def _series(poses, **kwargs):
    cache = make_cache(poses, fps=FPS)
    cfg = FeatureConfig(**kwargs)
    return build_track_series(cache, cfg)[0], cfg


def test_no_flip_suspected_when_facing_is_stable():
    series, cfg = _series([standing_pose(STATURE) for _ in range(90)])
    _, suspect = detect_lr_flip(series, cfg)
    assert not suspect.any()


def test_momentary_label_swap_is_flagged():
    """1 フレームだけ左右の肩が入れ替わったら取り違えとして検出する。"""
    poses = [standing_pose(STATURE) for _ in range(90)]
    swapped = poses[45].copy()
    swapped[[S.L_SHOULDER, S.R_SHOULDER]] = swapped[[S.R_SHOULDER, S.L_SHOULDER]]
    poses[45] = swapped
    series, cfg = _series(poses)
    _, suspect = detect_lr_flip(series, cfg)
    assert suspect[45]
    assert suspect.sum() < 5  # 前後には広がらない


def test_a_sustained_turn_is_not_flagged_as_a_swap():
    """本当に体の向きを変えた場合は取り違え扱いにしない。"""
    poses = []
    for i in range(120):
        kp = standing_pose(STATURE)
        if i >= 60:  # 後半はずっと背を向けている（肩の左右が反転する）
            kp[[S.L_SHOULDER, S.R_SHOULDER]] = kp[[S.R_SHOULDER, S.L_SHOULDER]]
        poses.append(kp)
    series, cfg = _series(poses)
    _, suspect = detect_lr_flip(series, cfg)
    assert not suspect.any()


# --- 負荷代理指標 ------------------------------------------------------------


def _bending_track(freq: float, amp_deg: float, duration: float = 8.0):
    t = np.arange(0, duration, 1.0 / FPS)
    theta = np.deg2rad(100.0) + np.deg2rad(amp_deg) * np.sin(2 * np.pi * freq * t)
    return [bend_elbow(standing_pose(STATURE), th, "l", STATURE) for th in theta]


def test_load_is_bounded_and_higher_for_more_violent_motion():
    """負荷代理値は 0..1 に収まり、同じ動画内で激しい関節ほど高く出る。"""
    # 左肘は大きく速く、右肘はほぼ静止
    t = np.arange(0, 8.0, 1.0 / FPS)
    poses = []
    for ti in t:
        kp = standing_pose(STATURE)
        kp = bend_elbow(kp, np.deg2rad(100.0) + np.deg2rad(60.0) * np.sin(2 * np.pi * 1.5 * ti), "l", STATURE)
        kp = bend_elbow(kp, np.deg2rad(100.0) + np.deg2rad(1.0) * np.sin(2 * np.pi * 0.1 * ti), "r", STATURE)
        poses.append(kp)

    series, cfg = _series(poses)
    ang = joint_angles(series, cfg)
    load = joint_load(ang, cfg)

    finite = load.load[np.isfinite(load.load)]
    assert finite.size > 0
    assert finite.min() >= 0.0 and finite.max() <= 1.0

    li, ri = load.names.index("l_elbow"), load.names.index("r_elbow")
    assert np.nanmean(load.load[:, li]) > np.nanmean(load.load[:, ri])


def test_load_is_masked_where_the_2d_angle_is_untrustworthy():
    """面外回転で信頼度が落ちた区間は負荷を出さない（誤った数値を見せない）。"""
    poses = _bending_track(1.0, 40.0)
    for i in range(len(poses) // 2, len(poses)):
        poses[i][S.L_WRIST] = poses[i][S.L_ELBOW] + 0.2 * (poses[i][S.L_WRIST] - poses[i][S.L_ELBOW])
    series, cfg = _series(poses)
    ang = joint_angles(series, cfg)
    load = joint_load(ang, cfg)
    k = load.names.index("l_elbow")
    assert np.all(np.isnan(load.load[len(poses) // 2 + 5 :, k]))


def test_body_speed_and_symmetry_are_produced():
    """姿勢統計量が一通り算出できること（マスクが無いキャッシュでも落ちない）。"""
    from pose_viz.features.kinematics import joint_kinematics

    poses = _bending_track(1.0, 40.0)
    cache = make_cache(poses, fps=FPS)
    cfg = FeatureConfig()
    series = build_track_series(cache, cfg)[0]
    kin = joint_kinematics(series, cfg, cache.camera_affine)
    post = posture_features(series, cache.by_track()[0], kin.speed_rel, cfg)

    assert np.isfinite(post.body_speed).any()
    assert np.isfinite(post.com).any()
    assert np.allclose(post.mass_coverage[np.isfinite(post.mass_coverage)], 1.0, atol=1e-3)
    # マスクが無いキャッシュでは収縮指数・QoM は算出できず NaN のまま
    assert np.all(np.isnan(post.contraction_index))
    assert np.all(np.isnan(post.qom))
