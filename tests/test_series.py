"""キャッシュ→時系列化での欠損処理・正規化を確かめる。

キャッシュ側の「レコードが無い＝欠損」という表現を取り違えると、存在しない動きを
作ってしまう（穴の両端をつないだ巨大な速度など）。ここはその防波堤。
"""

from __future__ import annotations

import numpy as np

from pose_viz.config import FeatureConfig
from pose_viz.features import schema as S
from pose_viz.features.series import build_track_series
from tests.synthetic import TORSO_RATIO, make_cache, standing_pose

FPS = 30.0
STATURE = 600.0


def _build(poses, present=None, scores=0.9, **cfg_kwargs):
    cache = make_cache(poses, fps=FPS, present=present, scores=scores)
    cfg = FeatureConfig(**cfg_kwargs)
    return build_track_series(cache, cfg)[0], cfg


def test_scale_is_the_torso_length():
    series, _ = _build([standing_pose(STATURE) for _ in range(60)])
    assert np.allclose(series.scale, TORSO_RATIO * STATURE, rtol=1e-3)


def test_root_is_the_hip_midpoint():
    kp = standing_pose(STATURE)
    series, _ = _build([kp for _ in range(30)])
    expected = 0.5 * (kp[S.L_HIP] + kp[S.R_HIP])
    assert np.allclose(series.root[0], expected, atol=1e-3)


def test_axis_is_dense_between_first_and_last_observation():
    """観測が飛んでいても、時間軸は 1 刻みで埋まる（欠損は NaN で表現される）。"""
    poses = [standing_pose(STATURE) for _ in range(50)]
    present = [True] * 50
    present[20:23] = [False] * 3
    series, _ = _build(poses, present=present)
    assert np.array_equal(series.frame_idx, np.arange(50))
    assert not series.observed[20:23].any()
    assert series.observed[:20].all() and series.observed[23:].all()


def test_short_gap_is_interpolated():
    poses = [standing_pose(STATURE) for _ in range(60)]
    present = [True] * 60
    present[30:33] = [False] * 3  # max_gap_sec=0.2s → 6 フレーム以内なので補間される
    series, _ = _build(poses, present=present)
    assert np.isfinite(series.xy[30:33]).all()
    assert series.interpolated[30:33].all()
    assert not series.interpolated[:30].any()


def test_long_gap_stays_nan_and_splits_segments():
    """長い欠損は埋めない。またいで微分すると存在しない動きを作ってしまうため。"""
    poses = [standing_pose(STATURE) for _ in range(80)]
    present = [True] * 80
    present[30:50] = [False] * 20  # 6 フレームを大きく超える
    series, _ = _build(poses, present=present)
    assert np.all(np.isnan(series.xy[30:50]))
    assert not series.interpolated[30:50].any()
    assert len(series.segments) == 2
    assert series.segments[0] == (0, 30) and series.segments[1] == (50, 80)


def test_low_score_keypoints_become_missing():
    series, _ = _build([standing_pose(STATURE) for _ in range(40)], scores=0.1)
    assert np.all(np.isnan(series.xy))


def test_score_threshold_boundary_is_inclusive():
    """min_score ちょうどの値は有効として扱う。"""
    series, _ = _build([standing_pose(STATURE) for _ in range(40)], scores=0.3, min_score=0.3)
    assert np.isfinite(series.xy).all()


def test_interpolation_does_not_extrapolate_past_the_ends():
    poses = [standing_pose(STATURE) for _ in range(40)]
    present = [True] * 40
    present[0:2] = [False, False]
    series, _ = _build(poses, present=present)
    # 先頭の欠損は前方に値が無いので、補間ではなく欠損のまま
    assert not series.interpolated[0:2].any()


def test_coverage_reports_observed_fraction():
    poses = [standing_pose(STATURE) for _ in range(100)]
    present = [True] * 100
    present[40:60] = [False] * 20
    series, _ = _build(poses, present=present)
    assert abs(series.coverage() - 0.8) < 1e-6


def test_xy_rel_is_root_relative():
    poses = [(standing_pose(STATURE) + np.float32([5.0, 3.0]) * i) for i in range(40)]
    series, _ = _build([p.astype(np.float32) for p in poses])
    # 全身が平行移動しても root 相対の座標は動かない
    assert np.allclose(series.xy_rel[0], series.xy_rel[-1], atol=1e-3)
