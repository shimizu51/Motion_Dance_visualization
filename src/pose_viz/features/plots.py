"""検証用のグラフ出力。

数値が妥当かを人が確かめるための道具であって、作品としての可視化ではない
（作品側への反映はロードマップ Phase 2）。matplotlib は dev 依存なので遅延 import する。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pose_viz.features.export import TrackFeatures

_DEG = 180.0 / np.pi


def plot_track(tf: TrackFeatures, out_path: Path | str, title: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = tf.series
    t = s.t
    fig, axes = plt.subplots(5, 1, figsize=(14, 13), sharex=True)

    ax = axes[0]
    ax.plot(t, tf.posture.body_speed, lw=0.8, label="body speed (mass-weighted)")
    ax.plot(t, tf.posture.com_speed, lw=0.8, alpha=0.7, label="CoM speed")
    ax.plot(t, tf.kinematics.root_speed, lw=0.8, alpha=0.6, label="root (pelvis) speed")
    ax.set_ylabel("body-length / s")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(title or f"track {s.track_id}")

    ax = axes[1]
    for name in ("l_knee", "r_knee", "l_elbow", "r_elbow"):
        k = tf.angles.names.index(name)
        ax.plot(t, tf.angles.angle[:, k] * _DEG, lw=0.8, label=name)
    ax.set_ylabel("joint angle [deg]")
    ax.legend(loc="upper right", fontsize=8, ncol=4)

    ax = axes[2]
    im = ax.imshow(
        tf.load.load.T,
        aspect="auto",
        origin="lower",
        extent=(float(t[0]), float(t[-1]), -0.5, len(tf.load.names) - 0.5),
        vmin=0.0,
        vmax=1.0,
        cmap="magma",
        interpolation="nearest",
    )
    ax.set_yticks(range(len(tf.load.names)))
    ax.set_yticklabels(tf.load.names, fontsize=7)
    ax.set_ylabel("joint load proxy")
    fig.colorbar(im, ax=ax, pad=0.01, fraction=0.02)

    ax = axes[3]
    ax.plot(t, tf.posture.contraction_index, lw=0.8, label="contraction index")
    ax.plot(t, tf.posture.qom, lw=0.8, alpha=0.8, label="quantity of motion")
    ax.set_ylabel("silhouette")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[4]
    ax.plot(t, tf.posture.symmetry_index, lw=0.8, label="L/R symmetry index (0 = symmetric)")
    ax.plot(t, tf.sparc_window, lw=0.8, alpha=0.8, label="SPARC (windowed, 0 = smooth)")
    susp = tf.posture.lr_suspect
    if susp.any():
        ax.fill_between(t, 0, 1, where=susp, transform=ax.get_xaxis_transform(),
                        color="red", alpha=0.15, label="L/R flip suspected")
    ax.set_ylabel("symmetry / smoothness")
    ax.set_xlabel("time [s]")
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_tracks(features: dict[int, TrackFeatures], out_dir: Path | str, top_n: int = 3) -> list[Path]:
    """長く映っているトラックから順に `top_n` 件だけグラフ化する。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ranked = sorted(features.items(), key=lambda kv: kv[1].series.n_frames, reverse=True)[:top_n]
    paths = []
    for track_id, tf in ranked:
        p = out_dir / f"track_{track_id:03d}.png"
        plot_track(tf, p, title=f"track {track_id}  ({tf.series.n_frames} frames, coverage {tf.series.coverage():.1%})")
        paths.append(p)
    return paths
