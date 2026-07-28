from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pose_viz.cache import TrackFrame


@dataclass
class Particle:
    track_id: int
    point_id: int
    positions: list[np.ndarray] = field(default_factory=list)
    residual_mags: list[float] = field(default_factory=list)
    #: 残差ベクトル（向き付き）。大域運動を差し引いた瞬間的な動きの方向を表す
    residuals: list[np.ndarray] = field(default_factory=list)
    birth_frame: int = 0
    last_seen_frame: int = 0


class ParticleSystem:
    """AKAZE 残差点を粒子として保持し、一定時間で消えるように寿命を管理する。

    残差（大域運動を差し引いた動き）が大きい粒子ほど寿命を延ばす＝「見た目からわからない
    激しさ」が長く尾を引くようにする。
    """

    def __init__(
        self,
        fps: float,
        lifetime_sec: float = 0.8,
        gamma: float = 1.6,
        lifetime_boost_k: float = 1.0,
        norm_scale: float = 8.0,
        max_trail_len: int = 40,
    ):
        self.base_lifetime_frames = max(1, round(lifetime_sec * fps))
        self.gamma = gamma
        self.lifetime_boost_k = lifetime_boost_k
        self.norm_scale = norm_scale
        self.max_trail_len = max_trail_len
        self._particles: dict[tuple[int, int], Particle] = {}

    def _lifetime_frames(self, particle: Particle) -> float:
        recent = particle.residual_mags[-5:] if particle.residual_mags else [0.0]
        norm = float(np.clip(np.mean(recent) / self.norm_scale, 0.0, 3.0))
        return self.base_lifetime_frames * (1.0 + self.lifetime_boost_k * norm)

    def update(self, frame_idx: int, track_frames: list[TrackFrame]) -> None:
        for tf in track_frames:
            for pt, res, res_mag, pid in zip(
                tf.akaze_points, tf.akaze_residual, tf.akaze_residual_mag, tf.akaze_point_ids
            ):
                key = (tf.track_id, int(pid))
                p = self._particles.get(key)
                if p is None:
                    p = Particle(track_id=tf.track_id, point_id=int(pid), birth_frame=frame_idx)
                    self._particles[key] = p
                p.positions.append(pt.astype(np.float32))
                p.residuals.append(res.astype(np.float32))
                p.residual_mags.append(float(res_mag))
                if len(p.positions) > self.max_trail_len:
                    p.positions.pop(0)
                    p.residuals.pop(0)
                    p.residual_mags.pop(0)
                p.last_seen_frame = frame_idx

        dead = [k for k, p in self._particles.items() if frame_idx - p.last_seen_frame > self._lifetime_frames(p)]
        for k in dead:
            del self._particles[k]

    def iter_visible(self, frame_idx: int):
        """(particle, alpha) を yield する。alpha は直近で見えた粒子ほど1に近い。"""
        for p in self._particles.values():
            lifetime = self._lifetime_frames(p)
            age = frame_idx - p.last_seen_frame
            if age > lifetime:
                continue
            alpha = (1.0 - age / lifetime) ** self.gamma
            yield p, alpha
