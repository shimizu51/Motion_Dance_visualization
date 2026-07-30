"""関節角度・角速度が解析解と一致することを確かめる。"""

from __future__ import annotations

import numpy as np
import pytest

from pose_viz.config import FeatureConfig
from pose_viz.features.angles import joint_angles
from pose_viz.features.series import build_track_series
from tests.synthetic import bend_elbow, make_cache, standing_pose

FPS = 30.0
STATURE = 600.0


def _series_from(poses):
    cache = make_cache(poses, fps=FPS)
    cfg = FeatureConfig()
    return build_track_series(cache, cfg)[0], cfg


@pytest.mark.parametrize("target_deg", [30.0, 60.0, 90.0, 120.0, 170.0])
def test_static_elbow_angle_matches_construction(target_deg):
    """与えた挟角がそのまま復元されること。"""
    target = np.deg2rad(target_deg)
    poses = [bend_elbow(standing_pose(STATURE), target, "l", STATURE) for _ in range(60)]
    series, cfg = _series_from(poses)
    ang = joint_angles(series, cfg)
    k = ang.names.index("l_elbow")
    measured = np.nanmedian(ang.angle[:, k])
    assert np.rad2deg(abs(measured - target)) < 0.5


def test_sinusoidal_elbow_angle_and_angular_velocity():
    """正弦波で肘を屈伸させ、角度と角速度が解析解に一致するか。"""
    freq, amp, center = 0.8, np.deg2rad(40.0), np.deg2rad(100.0)
    t = np.arange(0, 6.0, 1.0 / FPS)
    theta = center + amp * np.sin(2 * np.pi * freq * t)
    poses = [bend_elbow(standing_pose(STATURE), th, "l", STATURE) for th in theta]

    series, cfg = _series_from(poses)
    ang = joint_angles(series, cfg)
    k = ang.names.index("l_elbow")
    inner = slice(15, -15)

    assert np.allclose(ang.angle[inner, k], theta[inner], atol=np.deg2rad(0.5))

    expected_vel = amp * 2 * np.pi * freq * np.cos(2 * np.pi * freq * t)
    assert np.allclose(ang.ang_vel[inner, k], expected_vel[inner], rtol=0.05, atol=np.deg2rad(3.0))


def test_angle_is_scale_invariant():
    """カメラ距離（見かけの大きさ）が変わっても角度は変わらない。"""
    target = np.deg2rad(75.0)
    small = [bend_elbow(standing_pose(300.0), target, "l", 300.0) for _ in range(60)]
    large = [bend_elbow(standing_pose(900.0), target, "l", 900.0) for _ in range(60)]
    ks = joint_angles(*_series_from(small))
    kl = joint_angles(*_series_from(large))
    k = ks.names.index("l_elbow")
    assert abs(np.nanmedian(ks.angle[:, k]) - np.nanmedian(kl.angle[:, k])) < np.deg2rad(0.5)


def test_foreshortening_confidence_drops_when_segment_shortens():
    """面外回転を模して前腕を短く見せると、信頼度が下がること。"""
    target = np.deg2rad(90.0)
    poses = []
    for i in range(120):
        kp = bend_elbow(standing_pose(STATURE), target, "l", STATURE)
        if i >= 90:  # 後半だけ前腕を 30% の長さに縮める（カメラ方向へ向いた状態）
            from pose_viz.features import schema as S

            kp[S.L_WRIST] = kp[S.L_ELBOW] + 0.3 * (kp[S.L_WRIST] - kp[S.L_ELBOW])
        poses.append(kp)

    series, cfg = _series_from(poses)
    ang = joint_angles(series, cfg)
    k = ang.names.index("l_elbow")
    assert np.nanmedian(ang.confidence[20:80, k]) > 0.9
    assert np.nanmedian(ang.confidence[110:, k]) < cfg.foreshorten_threshold


def test_trunk_lean_sign_and_magnitude():
    """直立で 0、右に倒せば正の符号になること。"""
    from pose_viz.features import schema as S

    upright = [standing_pose(STATURE) for _ in range(60)]
    series, cfg = _series_from(upright)
    assert abs(np.nanmedian(joint_angles(series, cfg).trunk_lean)) < np.deg2rad(0.5)

    lean_deg = 20.0
    tilted = []
    for _ in range(60):
        kp = standing_pose(STATURE)
        hip = 0.5 * (kp[S.L_HIP] + kp[S.R_HIP])
        a = np.deg2rad(lean_deg)
        rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], dtype=np.float32)
        # 画像座標は y 下向きなので、この回転は上半身を画面右へ倒す
        kp = (kp - hip) @ rot.T + hip
        tilted.append(kp.astype(np.float32))
    series, cfg = _series_from(tilted)
    measured = np.nanmedian(joint_angles(series, cfg).trunk_lean)
    assert measured > 0
    assert abs(np.rad2deg(measured) - lean_deg) < 1.0
