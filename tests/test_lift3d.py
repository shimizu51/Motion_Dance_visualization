"""3D リフティングの前処理・後処理と、3D 由来の関節角度を確かめる。

モデル本体（170MB）は読み込まない。純粋な変換関数と、合成 3D を流し込んだ角度算出だけを
対象にする。モデルの重みの正しさは上流の責任で、ここで確かめるのは **こちら側の配線** である。
"""

from __future__ import annotations

import numpy as np
import pytest

from pose_viz.config import FeatureConfig
from pose_viz.features import schema as S
from pose_viz.features.angles import joint_angles, resolve_angle_source
from pose_viz.features.series import build_track_series
from pose_viz.lift3d import (
    H_HIP,
    H_LELBOW,
    H_LSHOULDER,
    H_LWRIST,
    H_THORAX,
    MAX_CLIP_LEN,
    coco2h36m,
    flip_motion,
    normalize_sequence,
    plan_windows,
)
from tests.synthetic import bend_elbow, h36m_from_coco, make_cache, standing_pose

FPS = 30.0
STATURE = 600.0


# --- COCO -> H36M ----------------------------------------------------------


def test_coco2h36m_maps_joints_to_the_right_slots():
    kp = np.zeros((1, 17, 3), dtype=np.float32)
    kp[0, :, :2] = standing_pose(STATURE)
    h = coco2h36m(kp)[0]
    src = kp[0]
    assert np.allclose(h[H_LSHOULDER, :2], src[S.L_SHOULDER, :2])
    assert np.allclose(h[H_LELBOW, :2], src[S.L_ELBOW, :2])
    assert np.allclose(h[H_LWRIST, :2], src[S.L_WRIST, :2])


def test_coco2h36m_synthesises_missing_joints():
    """COCO に無い骨盤・胸郭・脊椎は、左右の股関節・肩から合成する。"""
    kp = np.zeros((1, 17, 3), dtype=np.float32)
    kp[0, :, :2] = standing_pose(STATURE)
    h = coco2h36m(kp)[0]
    src = kp[0]
    assert np.allclose(h[H_HIP, :2], (src[S.L_HIP, :2] + src[S.R_HIP, :2]) / 2)
    assert np.allclose(h[H_THORAX, :2], (src[S.L_SHOULDER, :2] + src[S.R_SHOULDER, :2]) / 2)
    assert np.allclose(h[7, :2], (h[H_HIP, :2] + h[H_THORAX, :2]) / 2)  # spine


# --- 正規化 ----------------------------------------------------------------


def test_normalize_sequence_maps_into_unit_range():
    motion = np.zeros((50, 17, 3), dtype=np.float32)
    motion[:, :, :2] = standing_pose(STATURE)[None]
    motion[:, :, 2] = 0.9
    out = normalize_sequence(motion)
    assert out[..., :2].min() >= -1.0 and out[..., :2].max() <= 1.0
    assert np.isclose(out[..., :2].min(), -1.0, atol=1e-5)  # 外接矩形が端に張り付く


def test_normalize_sequence_is_scale_invariant():
    """撮影距離が変わっても正規化後は同じになる（これが 3D 側の入力条件）。"""
    def norm(stature):
        m = np.zeros((10, 17, 3), dtype=np.float32)
        m[:, :, :2] = standing_pose(stature, center=(960.0, 540.0))[None]
        m[:, :, 2] = 0.9
        return normalize_sequence(m)

    assert np.allclose(norm(300.0), norm(900.0), atol=1e-4)


def test_normalize_sequence_ignores_zero_confidence_points():
    motion = np.zeros((10, 17, 3), dtype=np.float32)
    motion[:, :, :2] = standing_pose(STATURE)[None]
    motion[:, :, 2] = 0.9
    outlier = motion.copy()
    outlier[:, 0, :2] = 99999.0  # 遠くに飛んだ点
    outlier[:, 0, 2] = 0.0  # ただし信頼度 0
    assert np.allclose(normalize_sequence(motion)[:, 1:], normalize_sequence(outlier)[:, 1:], atol=1e-4)


def test_normalize_sequence_handles_degenerate_input():
    assert not np.any(normalize_sequence(np.zeros((10, 17, 3), dtype=np.float32)))


# --- 反転と窓分割 ----------------------------------------------------------


def test_flip_motion_is_an_involution():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5, 17, 3)).astype(np.float32)
    assert np.allclose(flip_motion(flip_motion(x)), x)


def test_flip_motion_swaps_left_and_right():
    x = np.zeros((1, 17, 3), dtype=np.float32)
    x[0, 4] = [1.0, 2.0, 3.0]  # LHip
    y = flip_motion(x)
    assert np.allclose(y[0, 1], [-1.0, 2.0, 3.0])  # RHip に移り x が反転する


@pytest.mark.parametrize("n", [1, 50, 243, 244, 500, 5000])
def test_plan_windows_cover_every_frame(n):
    windows = plan_windows(n, MAX_CLIP_LEN, MAX_CLIP_LEN // 2)
    covered = np.zeros(n, dtype=bool)
    for s, e in windows:
        assert e - s <= MAX_CLIP_LEN
        covered[s:e] = True
    assert covered.all()


def test_plan_windows_overlap_when_the_sequence_is_long():
    """窓が重ならないと、境界で 3D が不連続になり微分がスパイクする。"""
    windows = plan_windows(1000, MAX_CLIP_LEN, MAX_CLIP_LEN // 2)
    assert len(windows) > 1
    for (s0, e0), (s1, _) in zip(windows, windows[1:]):
        assert s1 < e0  # 隣り合う窓が必ず重なる


# --- 角度の切り替え --------------------------------------------------------


def test_resolve_angle_source():
    assert resolve_angle_source(FeatureConfig(angle_source="auto"), has_3d=True) == "3d"
    assert resolve_angle_source(FeatureConfig(angle_source="auto"), has_3d=False) == "2d"
    assert resolve_angle_source(FeatureConfig(angle_source="2d"), has_3d=True) == "2d"
    with pytest.raises(ValueError, match="lift3d"):
        resolve_angle_source(FeatureConfig(angle_source="3d"), has_3d=False)
    with pytest.raises(ValueError, match="angle_source"):
        resolve_angle_source(FeatureConfig(angle_source="bogus"), has_3d=True)


def _series_with_3d(poses, depths=None):
    poses_3d = [h36m_from_coco(p, None if depths is None else depths[i]) for i, p in enumerate(poses)]
    cache = make_cache(poses, fps=FPS, poses_3d=poses_3d)
    return build_track_series(cache, FeatureConfig())[0]


def test_3d_angles_match_2d_when_the_pose_is_in_the_camera_plane():
    """z=0（カメラ面内）なら 3D 角度は 2D 角度と一致するはず。"""
    target = np.deg2rad(75.0)
    poses = [bend_elbow(standing_pose(STATURE), target, "l", STATURE) for _ in range(60)]
    series = _series_with_3d(poses)
    k = joint_angles(series, FeatureConfig(angle_source="2d")).names.index("l_elbow")
    a2 = joint_angles(series, FeatureConfig(angle_source="2d")).angle[:, k]
    a3 = joint_angles(series, FeatureConfig(angle_source="3d")).angle[:, k]
    assert np.nanmax(np.abs(np.rad2deg(a2 - a3))) < 0.5


def test_3d_recovers_an_angle_that_2d_foreshortens():
    """前腕をカメラ方向へ向けると 2D は角度を誤るが、3D は正しく出せる。

    奥行きを与えた前腕の真の挟角を解析的に置き、2D 投影の値と比べる。
    """
    forearm = 0.15 * STATURE
    poses, depths, truth = [], [], None
    for _ in range(60):
        kp = bend_elbow(standing_pose(STATURE), np.deg2rad(90.0), "l", STATURE)
        # 前腕を画面内で完全に潰し、その分を奥行きに逃がす（真横から見れば 90 度のまま）
        kp[S.L_WRIST] = kp[S.L_ELBOW].copy()
        d = np.zeros(17, dtype=np.float32)
        poses.append(kp)
        depths.append(d)
    # 合成 3D 側では手首を奥行き方向に伸ばす（上腕は面内のままなので真の挟角は 90 度）
    poses_3d = []
    for kp in poses:
        h = h36m_from_coco(kp)
        h[H_LWRIST] = h[H_LELBOW] + np.array([0.0, 0.0, forearm], dtype=np.float32)
        poses_3d.append(h)
    cache = make_cache(poses, fps=FPS, poses_3d=poses_3d)
    series = build_track_series(cache, FeatureConfig())[0]

    names = joint_angles(series, FeatureConfig(angle_source="3d")).names
    k = names.index("l_elbow")
    a3 = np.nanmedian(joint_angles(series, FeatureConfig(angle_source="3d")).angle[:, k])
    assert abs(np.rad2deg(a3) - 90.0) < 1.0  # 3D は真値 90 度を復元する


def test_3d_confidence_reflects_input_validity():
    poses = [standing_pose(STATURE) for _ in range(60)]
    poses_3d = [h36m_from_coco(p) for p in poses]
    good = build_track_series(make_cache(poses, fps=FPS, poses_3d=poses_3d), FeatureConfig())[0]
    bad = build_track_series(
        make_cache(poses, fps=FPS, poses_3d=poses_3d, scores=0.1), FeatureConfig()
    )[0]
    cfg = FeatureConfig(angle_source="3d")
    assert np.nanmin(joint_angles(good, cfg).confidence) == 1.0
    assert np.nanmax(joint_angles(bad, cfg).confidence) == 0.0


def test_features_fall_back_to_2d_without_lift3d():
    poses = [standing_pose(STATURE) for _ in range(60)]
    series = build_track_series(make_cache(poses, fps=FPS), FeatureConfig())[0]
    assert series.xyz is None
    assert joint_angles(series, FeatureConfig()).source == "2d"
