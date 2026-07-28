"""特徴量の組み立てと CSV 出力。

出力は 2 種類。

- **時系列 CSV**（wide 形式）… 1 行 = 1 トラックの 1 フレーム。可視化・グラフ確認用。
- **サマリ CSV** … 1 行 = 1 トラック。各指標の平均・p95 に加えて **有効値の割合（欠損率の裏返し）**
  を必ず併記する。欠損率 40% の対称性指標を鵜呑みにしないための情報。

角度は内部では rad で扱うが、CSV には **度** で書き出す（人が検証するための出力なので）。
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pose_viz.cache import ExtractCache
from pose_viz.config import FeatureConfig
from pose_viz.features.angles import AngleFeatures, joint_angles
from pose_viz.features.filters import nan_percentile
from pose_viz.features.kinematics import JointKinematics, joint_kinematics, windowed_sparc
from pose_viz.features.load import LoadFeatures, joint_load
from pose_viz.features.posture import PostureFeatures, posture_features
from pose_viz.features.rhythm import RhythmFeatures, dominant_period
from pose_viz.features.schema import KEYPOINT_NAMES, TrackSeries
from pose_viz.features.series import build_track_series

_DEG = 180.0 / np.pi


@dataclass
class TrackFeatures:
    series: TrackSeries
    kinematics: JointKinematics
    angles: AngleFeatures
    posture: PostureFeatures
    load: LoadFeatures
    rhythm: RhythmFeatures
    #: 窓ごとの SPARC（滑らかさ）。**トラック間の比較にはこちらを使う。**
    #: SPARC は本来ひとつの動作区間に対する指標なので、数分の連続動作全体に一度だけ
    #: 適用した値は長さと内容に依存してしまい、トラック間で比較できない。
    sparc_window: np.ndarray


def compute_features(cache: ExtractCache, cfg: FeatureConfig) -> dict[int, TrackFeatures]:
    """キャッシュ全体から track_id ごとの特徴量を計算する。"""
    all_series = build_track_series(cache, cfg)
    by_track = cache.by_track()

    out: dict[int, TrackFeatures] = {}
    for track_id, series in all_series.items():
        kin = joint_kinematics(series, cfg, cache.camera_affine)
        ang = joint_angles(series, cfg)
        post = posture_features(series, by_track[track_id], kin.speed_rel, cfg)
        load = joint_load(ang, cfg)
        rhythm = dominant_period(
            post.body_speed, series.fps, cfg.rhythm_min_period_sec, cfg.rhythm_max_period_sec
        )
        out[track_id] = TrackFeatures(
            series=series,
            kinematics=kin,
            angles=ang,
            posture=post,
            load=load,
            rhythm=rhythm,
            sparc_window=windowed_sparc(post.body_speed, series.fps, cfg.sparc_window_sec),
        )
    return out


def timeseries_columns(tf: TrackFeatures) -> dict[str, np.ndarray]:
    """時系列 CSV に出す数値列（サマリの集計もこの定義を共有する）。"""
    s, kin, ang, post, load = tf.series, tf.kinematics, tf.angles, tf.posture, tf.load
    cols: dict[str, np.ndarray] = {
        "scale_px": s.scale,
        "body_speed": post.body_speed,
        "com_speed": post.com_speed,
        "root_speed": kin.root_speed,
        "mass_coverage": post.mass_coverage,
        "contraction_index": post.contraction_index,
        "qom": post.qom,
        "symmetry_index": post.symmetry_index,
        "sparc_window": tf.sparc_window,
        "trunk_lean_deg": ang.trunk_lean * _DEG,
    }
    for k, name in enumerate(ang.names):
        cols[f"angle_{name}_deg"] = ang.angle[:, k] * _DEG
        cols[f"angvel_{name}_degs"] = ang.ang_vel[:, k] * _DEG
        cols[f"angaccel_{name}_degs2"] = ang.ang_accel[:, k] * _DEG
        cols[f"conf_{name}"] = ang.confidence[:, k]
        cols[f"load_{name}"] = load.load[:, k]
    for j, kp in enumerate(KEYPOINT_NAMES):
        cols[f"speed_{kp}"] = kin.speed_rel[:, j]
    return cols


def _fmt(v: float) -> str:
    return "" if not np.isfinite(v) else f"{v:.6g}"


def write_timeseries_csv(path: Path | str, features: dict[int, TrackFeatures]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not features:
        path.write_text("", encoding="utf-8")
        return 0

    sample = next(iter(features.values()))
    metric_names = list(timeseries_columns(sample))
    header = ["track_id", "frame_idx", "t_sec", "observed", "interp_frac", "lr_suspect"] + metric_names

    rows = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for track_id in sorted(features):
            tf = features[track_id]
            s = tf.series
            cols = timeseries_columns(tf)
            interp_frac = s.interpolated.mean(axis=1)
            for i in range(s.n_frames):
                writer.writerow(
                    [
                        track_id,
                        int(s.frame_idx[i]),
                        f"{float(s.t[i]):.4f}",
                        int(s.observed[i]),
                        f"{float(interp_frac[i]):.3f}",
                        int(tf.posture.lr_suspect[i]),
                    ]
                    + [_fmt(float(cols[m][i])) for m in metric_names]
                )
                rows += 1
    return rows


def write_summary_csv(path: Path | str, features: dict[int, TrackFeatures], cache: ExtractCache) -> None:
    """トラックごとの要約。各指標に **有効値の割合** を必ず併記する。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not features:
        path.write_text("", encoding="utf-8")
        return

    sample = next(iter(features.values()))
    metric_names = list(timeseries_columns(sample))
    # 代表 SPARC には「窓ごとの値の中央値」を使う。SPARC は本来ひとつの動作区間に対する指標で、
    # 数分の連続したダンス全体に一度だけ適用した値は長さと内容に依存し、トラック間で比較できない。
    header = [
        "track_id", "n_frames", "duration_sec", "first_frame", "last_frame",
        "observed_frac", "n_segments", "lr_suspect_frac", "angle_source",
        "sparc_median", "period_sec", "bpm", "rhythm_confidence",
    ]
    # 中央値を必ず併記する。2D 姿勢推定は 0.1〜0.4% のフレームでキーポイントが飛び、
    # そこだけ非現実的な速度が出る。平均は汚染されるが中央値は汚染されないので、
    # 両者の乖離がそのまま「その指標がどれだけ外れ値を含むか」の目安になる。
    for m in metric_names:
        header += [f"{m}_mean", f"{m}_median", f"{m}_p95", f"{m}_valid_frac"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for track_id in sorted(features):
            tf = features[track_id]
            s = tf.series
            cols = timeseries_columns(tf)
            row = [
                track_id,
                s.n_frames,
                f"{s.n_frames / cache.fps:.3f}",
                int(s.frame_idx[0]),
                int(s.frame_idx[-1]),
                f"{s.coverage():.4f}",
                len(s.segments),
                f"{float(tf.posture.lr_suspect.mean()):.4f}",
                tf.angles.source,
                _fmt(nan_percentile(tf.sparc_window, 50)),
                _fmt(tf.rhythm.period_sec),
                _fmt(tf.rhythm.bpm),
                _fmt(tf.rhythm.confidence),
            ]
            for m in metric_names:
                v = np.asarray(cols[m], dtype=np.float64)
                finite = np.isfinite(v)
                mean = float(v[finite].mean()) if finite.any() else float("nan")
                row += [
                    _fmt(mean),
                    _fmt(nan_percentile(v, 50)),
                    _fmt(nan_percentile(v, 95)),
                    f"{finite.mean():.4f}",
                ]
            writer.writerow(row)
