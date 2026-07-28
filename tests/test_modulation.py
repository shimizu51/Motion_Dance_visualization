"""特徴量→描画重みの写像と、既定設定での後方互換を確かめる。"""

from __future__ import annotations

import numpy as np
import pytest

from pose_viz.config import Config, FeatureConfig
from pose_viz.features import schema as S
from pose_viz.features.export import compute_features
from pose_viz.features.modulation import build_joint_weights
from tests.synthetic import bend_elbow, make_cache, standing_pose

FPS = 30.0
STATURE = 600.0


def _features():
    t = np.arange(0, 8.0, 1.0 / FPS)
    poses = []
    for ti in t:
        kp = standing_pose(STATURE)
        kp = bend_elbow(kp, np.deg2rad(100.0) + np.deg2rad(60.0) * np.sin(2 * np.pi * 1.5 * ti), "l", STATURE)
        poses.append(kp)
    cache = make_cache(poses, fps=FPS)
    return compute_features(cache, FeatureConfig())


def test_off_computes_nothing():
    assert build_joint_weights(_features(), "off") == {}


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="feature_modulation"):
        build_joint_weights(_features(), "bogus")


@pytest.mark.parametrize("mode", ["load", "speed"])
def test_weights_are_bounded_and_cover_every_keypoint(mode):
    weights = build_joint_weights(_features(), mode)
    assert weights
    for per_track in weights.values():
        for w in per_track.values():
            assert w.shape == (S.N_KEYPOINTS,)
            assert np.isfinite(w).all()
            assert w.min() >= 0.0 and w.max() <= 1.0


def test_load_weights_are_inherited_by_distal_keypoints():
    """手首・足首には挟角が定義できないので、肘・膝の値を引き継ぐ。"""
    weights = build_joint_weights(_features(), "load")
    per_track = next(iter(weights.values()))
    w = next(iter(per_track.values()))
    assert w[S.L_WRIST] == w[S.L_ELBOW]
    assert w[S.R_WRIST] == w[S.R_ELBOW]
    assert w[S.L_ANKLE] == w[S.L_KNEE]
    assert w[S.R_ANKLE] == w[S.R_KNEE]


def test_load_weights_are_zero_where_no_angle_is_defined():
    weights = build_joint_weights(_features(), "load")
    per_track = next(iter(weights.values()))
    w = next(iter(per_track.values()))
    for kp in (S.NOSE, S.L_EYE, S.R_EYE, S.L_EAR, S.R_EAR):
        assert w[kp] == 0.0


def test_the_moving_joint_gets_more_load_than_the_still_one():
    weights = build_joint_weights(_features(), "load")
    left = np.mean([w[S.L_ELBOW] for per in weights.values() for w in per.values()])
    right = np.mean([w[S.R_ELBOW] for per in weights.values() for w in per.values()])
    assert left > right


def test_default_config_keeps_modulation_off():
    """既定では特徴量を一切使わない＝従来の描画と完全に同一になる。"""
    cfg = Config()
    assert cfg.render.feature_modulation == "off"
    assert cfg.render.residual_streak_gain == 0.0


def test_shipped_default_yaml_keeps_modulation_off():
    from pose_viz.cli import DEFAULT_CONFIG_PATH

    cfg = Config.load(DEFAULT_CONFIG_PATH)
    assert cfg.render.feature_modulation == "off"
    assert cfg.render.residual_streak_gain == 0.0


def test_feature_settings_do_not_invalidate_the_cache():
    """特徴量・描画の設定は extract_hash に入らない（変えても再抽出が要らない）。"""
    base = Config.load(__import__("pose_viz.cli", fromlist=["x"]).DEFAULT_CONFIG_PATH)
    before = base.extract_hash()
    base.render.feature_modulation = "load"
    base.render.feature_thickness_gain = 9.0
    base.feature.min_score = 0.5
    base.feature.deriv_window_sec = 0.5
    assert base.extract_hash() == before
