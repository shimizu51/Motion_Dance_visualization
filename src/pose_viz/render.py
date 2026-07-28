from __future__ import annotations

import numpy as np
import cv2

from pose_viz.cache import ExtractCache, TrackFrame, decode_mask_crop
from pose_viz.config import RenderConfig, ResidualConfig
from pose_viz.palette import heat_core, hue_for_track, hue_to_bgr
from pose_viz.residual import ParticleSystem

KEYPOINT_SCORE_THRESHOLD_DEFAULT = 0.3

#: 関節ごとの重み（0..1）。track_id -> (17,) float。`feature_modulation` が "off" のときは None。
JointWeights = dict[int, np.ndarray] | None


def _odd(k: int) -> int:
    return k if k % 2 == 1 else k + 1


def _build_ghost_layer(frame_bgr: np.ndarray, track_frames_back_to_front: list[TrackFrame], rcfg: RenderConfig) -> np.ndarray:
    """人物ごとのゴーストを奥から手前へアルファ合成で塗り重ねる（手前の人が奥の人を隠す）。"""
    h, w = frame_bgr.shape[:2]

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32)
    frame_f = frame_bgr.astype(np.float32)

    desat = np.clip(rcfg.ghost_desaturate, 0.0, 1.0)
    tinted = frame_f * (1 - desat) + gray_bgr * desat
    # わずかに寒色寄りにティントする（ゴーストらしさを出す）
    tinted[..., 0] *= 1.05  # B
    tinted[..., 2] *= 0.9  # R

    canvas = np.zeros((h, w, 3), dtype=np.float32)
    any_mask = False
    for tf in track_frames_back_to_front:
        if tf.mask_png is None:
            continue
        m = decode_mask_crop(tf.mask_png, tf.box_xyxy, (h, w))
        if rcfg.ghost_blur > 0:
            k = _odd(rcfg.ghost_blur)
            m = cv2.GaussianBlur(m, (k, k), 0)
        any_mask = True
        alpha = (m * rcfg.ghost_alpha)[..., None]
        canvas = tinted * alpha + canvas * (1 - alpha)

    if not any_mask:
        return np.zeros((h, w, 3), dtype=np.float32)
    return canvas


def _draw_dashed_line(
    canvas: np.ndarray, pt1: tuple[int, int], pt2: tuple[int, int], color: tuple[float, float, float], thickness: int
) -> None:
    x1, y1 = pt1
    x2, y2 = pt2
    dist = float(np.hypot(x2 - x1, y2 - y1))
    if dist < 1:
        return
    dashes = max(1, int(dist // 10))
    for i in range(dashes):
        s = i / dashes
        e = min(1.0, s + 0.5 / dashes)
        p1 = (int(x1 + (x2 - x1) * s), int(y1 + (y2 - y1) * s))
        p2 = (int(x1 + (x2 - x1) * e), int(y1 + (y2 - y1) * e))
        cv2.line(canvas, p1, p2, color, thickness)


def _draw_dashed_rect(
    canvas: np.ndarray, pt1: tuple[int, int], pt2: tuple[int, int], color: tuple[float, float, float], thickness: int
) -> None:
    x1, y1 = pt1
    x2, y2 = pt2
    _draw_dashed_line(canvas, (x1, y1), (x2, y1), color, thickness)
    _draw_dashed_line(canvas, (x2, y1), (x2, y2), color, thickness)
    _draw_dashed_line(canvas, (x2, y2), (x1, y2), color, thickness)
    _draw_dashed_line(canvas, (x1, y2), (x1, y1), color, thickness)


#: デバッグ表示で数値を出すキーポイント（挟角が定義できる 8 点）
_DEBUG_VALUE_KEYPOINTS = (5, 6, 7, 8, 11, 12, 13, 14)


def _draw_debug_overlay(
    canvas: np.ndarray,
    track_frames: list[TrackFrame],
    edges: list[tuple[int, int]],
    joint_weights: JointWeights = None,
) -> None:
    h, w = canvas.shape[:2]
    for tf in track_frames:
        color = hue_to_bgr(hue_for_track(tf.track_id))
        x1, y1, x2, y2 = tf.box_xyxy.astype(int)
        if tf.recovered:
            # low_threshold 帯の検出で救済されたフレーム: 破線で区別する
            _draw_dashed_rect(canvas, (x1, y1), (x2, y2), color, 2)
        else:
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"id={tf.track_id} d={tf.depth_rank} s={tf.det_score:.2f}"
        cv2.putText(canvas, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if tf.mask_png is not None:
            m = decode_mask_crop(tf.mask_png, tf.box_xyxy, (h, w))
            contours, _ = cv2.findContours((m > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, contours, -1, color, 1)
        for pt in tf.akaze_points:
            cv2.circle(canvas, tuple(pt.astype(int)), 2, (0, 255, 255), -1)
        _draw_skeleton(canvas, tf.keypoints, tf.keypoint_scores, edges, color, thickness=2)

        # 特徴量を使う設定のときは、映像と数値を突き合わせられるよう関節ごとの値を出す
        weights = joint_weights.get(tf.track_id) if joint_weights else None
        if weights is None:
            continue
        for k in _DEBUG_VALUE_KEYPOINTS:
            if tf.keypoint_scores[k] < KEYPOINT_SCORE_THRESHOLD_DEFAULT or not np.isfinite(weights[k]):
                continue
            px, py = tf.keypoints[k].astype(int)
            cv2.putText(
                canvas, f"{float(weights[k]):.2f}", (px + 6, py - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
            )


def _draw_skeleton(
    canvas: np.ndarray,
    keypoints: np.ndarray,
    scores: np.ndarray,
    edges: list[tuple[int, int]],
    color: tuple[float, float, float],
    thickness: int = 3,
    score_threshold: float = KEYPOINT_SCORE_THRESHOLD_DEFAULT,
    weights: np.ndarray | None = None,
    thickness_gain: float = 0.0,
    highlight: float = 0.0,
) -> None:
    """骨格を描く。`weights`（関節ごとの 0..1）を渡すと太さと白熱コアで強調する。

    重みは **足すだけで、引かない**。重みが低い関節を暗くしてしまうと「骨格レイヤーは必ず
    最高輝度」（不変条件⑤）が崩れるため、基準の線は常に元の明るさで描き、その上に
    太さと白熱コアを重ねる形にしている。
    """
    def _weight(*idx: int) -> float:
        if weights is None:
            return 0.0
        vals = [float(weights[i]) for i in idx if np.isfinite(weights[i])]
        return sum(vals) / len(vals) if vals else 0.0

    for i, j in edges:
        if scores[i] < score_threshold or scores[j] < score_threshold:
            continue
        p1 = tuple(keypoints[i].astype(int))
        p2 = tuple(keypoints[j].astype(int))
        w = _weight(i, j)
        cv2.line(canvas, p1, p2, color, thickness + int(round(thickness_gain * w)), cv2.LINE_AA)
        if w > 0 and highlight > 0:
            cv2.line(canvas, p1, p2, heat_core(color, w * highlight), max(1, thickness - 1), cv2.LINE_AA)

    for k in range(len(keypoints)):
        if scores[k] < score_threshold:
            continue
        center = tuple(keypoints[k].astype(int))
        w = _weight(k)
        cv2.circle(canvas, center, thickness + 1 + int(round(thickness_gain * w)), color, -1, cv2.LINE_AA)
        if w > 0 and highlight > 0:
            cv2.circle(canvas, center, max(1, thickness), heat_core(color, w * highlight), -1, cv2.LINE_AA)


def _build_skeleton_layer(
    h: int,
    w: int,
    track_frames_back_to_front: list[TrackFrame],
    edges: list[tuple[int, int]],
    rcfg: RenderConfig,
    depth_body_occlude: bool = False,
    joint_weights: JointWeights = None,
) -> np.ndarray:
    base = np.zeros((h, w, 3), dtype=np.float32)
    for tf in track_frames_back_to_front:
        if depth_body_occlude and tf.mask_png is not None:
            # 手前の人物の胴体マスクで、奥側からここまで描いた骨格線をくり抜く
            m = decode_mask_crop(tf.mask_png, tf.box_xyxy, (h, w))
            base *= (1.0 - (m > 0.5).astype(np.float32))[..., None]
        color = hue_to_bgr(hue_for_track(tf.track_id))
        _draw_skeleton(
            base,
            tf.keypoints,
            tf.keypoint_scores,
            edges,
            color,
            thickness=rcfg.bone_thickness,
            weights=joint_weights.get(tf.track_id) if joint_weights else None,
            thickness_gain=rcfg.feature_thickness_gain,
            highlight=rcfg.feature_highlight,
        )

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
        pts = list(p.positions)
        if rcfg.residual_streak_gain > 0 and p.residuals:
            # 残差は向き付きのベクトルなので、先頭を「大域運動から外れた方向」へ伸ばして
            # 瞬間の動きの向きを見せる（キャッシュ済みで今まで使っていなかった情報）
            pts.append(pts[-1] + p.residuals[-1] * rcfg.residual_streak_gain)
        pts = np.array(pts, dtype=np.int32)
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
    joint_weights: JointWeights = None,
) -> np.ndarray:
    h, w = frame_bgr.shape[:2]

    if debug_overlay:
        canvas = frame_bgr.astype(np.float32).copy()
        _draw_debug_overlay(canvas, track_frames, edges, joint_weights)
        return np.clip(canvas, 0, 255).astype(np.uint8)

    # depth_rank 降順 = 奥(値が大きい)から手前(0)の順。骨格層・ゴースト層はこの順で描き、
    # 手前の人物が奥の人物を自然に覆うようにする。
    ordered = sorted(track_frames, key=lambda tf: tf.depth_rank, reverse=True)

    canvas = _build_ghost_layer(frame_bgr, ordered, render_cfg)
    canvas += _build_particle_layer(h, w, particle_system, frame_idx, render_cfg, residual_cfg.norm_scale)
    current_ids = {tf.track_id for tf in track_frames}
    canvas += _build_joint_trail_layer(h, w, cache, frame_idx, current_ids, edges, residual_cfg)
    canvas += _build_skeleton_layer(
        h, w, ordered, edges, render_cfg, render_cfg.depth_body_occlude, joint_weights
    )
    canvas = _post_process(canvas, render_cfg)

    return np.clip(canvas, 0, 255).astype(np.uint8)
