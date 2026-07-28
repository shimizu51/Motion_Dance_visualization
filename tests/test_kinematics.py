"""関節速度の正規化・カメラ不変性と、SPARC の性質を確かめる。"""

from __future__ import annotations

import numpy as np

from pose_viz.config import FeatureConfig
from pose_viz.features import schema as S
from pose_viz.features.kinematics import joint_kinematics, sparc
from pose_viz.features.series import build_track_series
from tests.synthetic import TORSO_RATIO, make_cache, standing_pose

FPS = 30.0
STATURE = 600.0
INNER = slice(20, -20)


def _wrist_circle(radius_px: float, freq: float, duration: float = 5.0, stature: float = STATURE, drift=None):
    """左手首だけを半径 `radius_px`・周波数 `freq` の円運動させた姿勢列を作る。"""
    t = np.arange(0, duration, 1.0 / FPS)
    poses = []
    for i, ti in enumerate(t):
        kp = standing_pose(stature)
        phase = 2 * np.pi * freq * ti
        kp[S.L_WRIST] = kp[S.L_WRIST] + radius_px * np.array([np.cos(phase), np.sin(phase)], dtype=np.float32)
        if drift is not None:
            kp = kp + np.asarray(drift, dtype=np.float32) * i
        poses.append(kp.astype(np.float32))
    return poses


def _kin(poses, camera_affine=None):
    cache = make_cache(poses, fps=FPS, camera_affine=camera_affine)
    cfg = FeatureConfig()
    series = build_track_series(cache, cfg)[0]
    return joint_kinematics(series, cfg, cache.camera_affine), series


def test_speed_matches_analytic_and_is_in_body_lengths():
    radius, freq = 60.0, 1.0
    kin, _ = _kin(_wrist_circle(radius, freq))
    expected = (2 * np.pi * freq * radius) / (TORSO_RATIO * STATURE)  # body-length/s
    measured = np.nanmedian(kin.speed_rel[INNER, S.L_WRIST])
    assert abs(measured - expected) / expected < 0.03


def test_speed_is_invariant_to_camera_distance():
    """同じ動きを 2 倍の大きさで撮っても、body-length/s の値は変わらない。"""
    a, _ = _kin(_wrist_circle(60.0, 1.0, stature=STATURE))
    b, _ = _kin(_wrist_circle(120.0, 1.0, stature=2 * STATURE))
    va = np.nanmedian(a.speed_rel[INNER, S.L_WRIST])
    vb = np.nanmedian(b.speed_rel[INNER, S.L_WRIST])
    assert abs(va - vb) / va < 0.02


def test_root_relative_speed_ignores_whole_body_translation():
    """人物全体が平行移動しても、root 相対の関節速度は変わらない。"""
    still, _ = _kin(_wrist_circle(60.0, 1.0))
    drifting, _ = _kin(_wrist_circle(60.0, 1.0, drift=(4.0, -2.0)))
    v0 = np.nanmedian(still.speed_rel[INNER, S.L_WRIST])
    v1 = np.nanmedian(drifting.speed_rel[INNER, S.L_WRIST])
    assert abs(v0 - v1) / v0 < 0.02


def test_camera_compensation_cancels_a_pure_pan():
    """静止した人物をカメラがパンしただけなら、補正後の絶対速度はほぼ 0 になる。"""
    n = 150
    shift = np.array([3.0, -1.5], dtype=np.float32)  # 毎フレームの見かけの移動量
    poses = [(standing_pose(STATURE) + shift * i).astype(np.float32) for i in range(n)]

    affine = np.tile(np.eye(2, 3, dtype=np.float32), (n, 1, 1))
    affine[:, 0, 2] = shift[0]
    affine[:, 1, 2] = shift[1]
    affine[0] = np.nan  # 先頭フレームは前フレームが無い

    with_cam, _ = _kin(poses, camera_affine=affine)
    without_cam, _ = _kin(poses, camera_affine=None)

    uncorrected = np.nanmedian(without_cam.speed_abs[INNER, S.L_WRIST])
    corrected = np.nanmedian(with_cam.speed_abs[INNER, S.L_WRIST])
    assert uncorrected > 0.5  # 補正しなければカメラの動きが速度として乗る
    assert corrected < uncorrected * 0.02


def _min_jerk_speed(duration: float = 1.0, fps: float = FPS) -> np.ndarray:
    """最小ジャーク軌道の速度プロファイル（もっとも滑らかな到達運動）。"""
    t = np.linspace(0.0, 1.0, int(duration * fps))
    return 30.0 * t**2 * (1 - t) ** 2


def test_sparc_of_smooth_movement_is_near_minus_one_point_four():
    """最小ジャーク運動の SPARC は文献値どおり -1.4 前後になる。"""
    value = sparc(_min_jerk_speed(), FPS)
    assert -1.8 < value < -1.2


def test_sparc_penalises_segmented_movement():
    """同じ移動を 4 分割してぎこちなくすると、SPARC はより負に大きくなる。"""
    smooth = _min_jerk_speed(duration=2.0)
    segmented = np.tile(_min_jerk_speed(duration=0.5), 4)
    assert sparc(segmented, FPS) < sparc(smooth, FPS)


def test_sparc_is_robust_to_small_noise():
    """SPARC を採用した理由: 微小ノイズで値が暴れない（ジャークはここで破綻する）。"""
    rng = np.random.default_rng(0)
    clean = _min_jerk_speed(duration=2.0)
    noisy = clean + rng.normal(0, 0.01 * clean.max(), clean.shape)
    assert abs(sparc(noisy, FPS) - sparc(clean, FPS)) < 0.3


def test_sparc_returns_nan_for_degenerate_input():
    assert np.isnan(sparc(np.zeros(50), FPS))
    assert np.isnan(sparc(np.array([1.0, 2.0]), FPS))
