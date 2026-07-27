from __future__ import annotations

import numpy as np
import cv2

from pose_viz.cache import ExtractCache, TrackFrame, decode_mask_crop
from pose_viz.config import RenderConfig, ResidualConfig
from pose_viz.palette import hue_for_track, hue_to_bgr
from pose_viz.residual import ParticleSystem

KEYPOINT_SCORE_THRESHOLD_DEFAULT = 0.3


def _odd(k: int) -> int:
    return k if k % 2 == 1 else k + 1


def _build_ghost_layer(frame_bgr: np.ndarray, track_frames: list[TrackFrame], rcfg: RenderConfig) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    mask_total = np.zeros((h, w), dtype=np.float32)
    for tf in track_frames:
        if tf.mask_png is None:
            continue
        m = decode_mask_crop(tf.mask_png, tf.box_xyxy, (h, w))
        mask_total = np.maximum(mask_total, m)

    if mask_total.max() <= 0:
        return np.zeros((h, w, 3), dtype=np.float32)

    if rcfg.ghost_blur > 0:
        k = _odd(rcfg.ghost_blur)
        mask_total = cv2.GaussianBlur(mask_total, (k, k), 0)

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32)
    frame_f = frame_bgr.astype(np.float32)

    desat = np.clip(rcfg.ghost_desaturate, 0.0, 1.0)
    tinted = frame_f * (1 - desat) + gray_bgr * desat
    # わずかに寒色寄りにティントする（ゴーストらしさを出す）
    tinted[..., 0] *= 1.05  # B
    tinted[..., 2] *= 0.9  # R

    alpha = mask_total[..., None] * rcfg.ghost_alpha
    return tinted * alpha


def _draw_debug_overlay(canvas: np.ndarray, track_frames: list[TrackFrame], edges: list[tuple[int, int]]) -> None:
    h, w = canvas.shape[:2]
    for tf in track_frames:
        color = hue_to_bgr(hue_for_track(tf.track_id))
        x1, y1, x2, y2 = tf.box_xyxy.astype(int)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, f"id={tf.track_id}", (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if tf.mask_png is not None:
            m = decode_mask_crop(tf.mask_png, tf.box_xyxy, (h, w))
            contours, _ = cv2.findContours((m > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, contours, -1, color, 1)
        for pt in tf.akaze_points:
            cv2.circle(canvas, tuple(pt.astype(int)), 2, (0, 255, 255), -1)
        _draw_skeleton(canvas, tf.keypoints, tf.keypoint_scores, edges, color, thickness=2)


def _draw_skeleton(
    canvas: np.ndarray,
    keypoints: np.ndarray,
    scores: np.ndarray,
    edges: list[tuple[int, int]],
    color: tuple[float, float, float],
    thickness: int = 3,
    score_threshold: float = KEYPOINT_SCORE_THRESHOLD_DEFAULT,
) -> None:
    for i, j in edges:
        if scores[i] < score_threshold or scores[j] < score_threshold:
            continue
        p1 = tuple(keypoints[i].astype(int))
        p2 = tuple(keypoints[j].astype(int))
        cv2.line(canvas, p1, p2, color, thickness, cv2.LINE_AA)
    for k in range(len(keypoints)):
        if scores[k] < score_threshold:
            continue
        cv2.circle(canvas, tuple(keypoints[k].astype(int)), thickness + 1, color, -1, cv2.LINE_AA)


def _build_skeleton_layer(
    h: int, w: int, track_frames: list[TrackFrame], edges: list[tuple[int, int]], rcfg: RenderConfig
) -> np.ndarray:
    base = np.zeros((h, w, 3), dtype=np.float32)
    for tf in track_frames:
        color = hue_to_bgr(hue_for_track(tf.track_id))
        _draw_skeleton(base, tf.keypoints, tf.keypoint_scores, edges, color, thickness=rcfg.bone_thickness)

    if base.max() <= 0:
        return base

    small_k, large_k = rcfg.skeleton_bloom
    bloom_small = cv2.GaussianBlur(base, (_odd(small_k), _odd(small_k)), 0)
    bloom_large = cv2.GaussianBlur(base, (_odd(large_k), _odd(large_k)), 0)
    # 骨格レイヤーは必ず他レイヤーより最も明るくする: 鋭い芯を増幅しつつグローを重ねる
    return base * 2.0 + bloom_small * 1.0 + bloom_large * 0.6


def _build_joint_trail_layer(
    h: int,
    w: int,
    cache: ExtractCache,
    frame_idx: int,
    current_track_ids: set[int],
    edges: list[tuple[int, int]],
    rescfg: ResidualConfig,
) -> np.ndarray:
    layer = np.zeros((h, w, 3), dtype=np.float32)
    trail_len = rescfg.joint_trail_len
    for lag in range(1, trail_len):
        past = cache.frames.get(frame_idx - lag)
        if not past:
            continue
        alpha = (1.0 - lag / trail_len) ** 1.5 * 0.5
        for tf in past:
            if tf.track_id not in current_track_ids:
                continue
            color = hue_to_bgr(hue_for_track(tf.track_id))
            faded = tuple(c * alpha for c in color)
            _draw_skeleton(layer, tf.keypoints, tf.keypoint_scores, edges, faded, thickness=1)
    return layer


def _build_particle_layer(
    h: int, w: int, particle_system: ParticleSystem, frame_idx: int, rcfg: RenderConfig, norm_scale: float
) -> np.ndarray:
    layer = np.zeros((h, w, 3), dtype=np.float32)
    for p, alpha in particle_system.iter_visible(frame_idx):
        if len(p.positions) < 2:
            continue
        pts = np.array(p.positions, dtype=np.int32)
        color = hue_to_bgr(hue_for_track(p.track_id))
        recent_mag = p.residual_mags[-1] if p.residual_mags else 0.0
        intensity = float(np.clip(recent_mag / norm_scale, 0.2, 1.5))
        col = tuple(c * alpha * intensity for c in color)
        cv2.polylines(layer, [pts], False, col, thickness=2, lineType=cv2.LINE_AA)

    if layer.max() <= 0 or rcfg.particle_glow <= 0:
        return layer
    k = _odd(rcfg.particle_glow)
    glow = cv2.GaussianBlur(layer, (k, k), 0)
    return layer + glow


def _post_process(canvas: np.ndarray, rcfg: RenderConfig) -> np.ndarray:
    h, w = canvas.shape[:2]
    if rcfg.vignette > 0:
        yy, xx = np.mgrid[0:h, 0:w]
        cx, cy = w / 2.0, h / 2.0
        dist = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
        vig = 1.0 - rcfg.vignette * np.clip(dist - 0.6, 0.0, 1.0)
        canvas = canvas * vig[..., None]
    if rcfg.grain > 0:
        noise = np.random.randn(h, w, 1).astype(np.float32) * 255.0 * rcfg.grain
        canvas = canvas + noise
    return canvas


def render_frame(
    frame_bgr: np.ndarray,
    frame_idx: int,
    track_frames: list[TrackFrame],
    cache: ExtractCache,
    edges: list[tuple[int, int]],
    render_cfg: RenderConfig,
    residual_cfg: ResidualConfig,
    particle_system: ParticleSystem,
    debug_overlay: bool = False,
) -> np.ndarray:
    h, w = frame_bgr.shape[:2]

    if debug_overlay:
        canvas = frame_bgr.astype(np.float32).copy()
        _draw_debug_overlay(canvas, track_frames, edges)
        return np.clip(canvas, 0, 255).astype(np.uint8)

    canvas = _build_ghost_layer(frame_bgr, track_frames, render_cfg)
    canvas += _build_particle_layer(h, w, particle_system, frame_idx, render_cfg, residual_cfg.norm_scale)
    current_ids = {tf.track_id for tf in track_frames}
    canvas += _build_joint_trail_layer(h, w, cache, frame_idx, current_ids, edges, residual_cfg)
    canvas += _build_skeleton_layer(h, w, track_frames, edges, render_cfg)
    canvas = _post_process(canvas, render_cfg)

    return np.clip(canvas, 0, 255).astype(np.uint8)
