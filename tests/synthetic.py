"""解析解が分かっている合成データを組み立てるヘルパー。

正解データが存在しない領域なので、**既知の入力に対して既知の答えが出るか** を確かめることが
唯一の客観的な検証手段になる。ここで作った骨格は実写ではないが、角度・速度・周期は解析的に
分かっているため、実装のバグはここで落ちる。
"""

from __future__ import annotations

import numpy as np

from pose_viz.cache import ExtractCache, TrackFrame
from pose_viz.features import schema as S

#: 腰中点を原点、上向きを +y とした立位の骨格（身長 1.0 に対する比率）
_STANDING = {
    S.L_HIP: (-0.05, 0.00),
    S.R_HIP: (0.05, 0.00),
    S.L_SHOULDER: (-0.10, 0.30),
    S.R_SHOULDER: (0.10, 0.30),
    S.NOSE: (0.00, 0.45),
    S.L_EYE: (-0.02, 0.47),
    S.R_EYE: (0.02, 0.47),
    S.L_EAR: (-0.05, 0.46),
    S.R_EAR: (0.05, 0.46),
    S.L_ELBOW: (-0.12, 0.15),
    S.R_ELBOW: (0.12, 0.15),
    S.L_WRIST: (-0.13, 0.00),
    S.R_WRIST: (0.13, 0.00),
    S.L_KNEE: (-0.06, -0.25),
    S.R_KNEE: (0.06, -0.25),
    S.L_ANKLE: (-0.06, -0.50),
    S.R_ANKLE: (0.06, -0.50),
}

#: 立位骨格の体幹長（肩中点 - 腰中点）。身長 1.0 に対する比率
TORSO_RATIO = 0.30


def standing_pose(stature_px: float = 600.0, center: tuple[float, float] = (960.0, 540.0)) -> np.ndarray:
    """立位の (17,2) キーポイント（画像座標。y は下向き）を作る。"""
    cx, cy = center
    kp = np.zeros((S.N_KEYPOINTS, 2), dtype=np.float32)
    for idx, (x, y) in _STANDING.items():
        kp[idx] = (cx + x * stature_px, cy - y * stature_px)
    return kp


def bend_elbow(kp: np.ndarray, angle_rad: float, side: str = "l", stature_px: float = 600.0) -> np.ndarray:
    """肘の挟角がちょうど `angle_rad` になるように手首を置き直す。

    肘での挟角 φ は (肩-肘) と (手首-肘) のなす角。肩→肘の向きを基準に φ だけ回した先に
    手首を置けば、`_angle_at` はちょうど φ を返すはずである。
    """
    kp = kp.copy()
    sh, el, wr = (
        (S.L_SHOULDER, S.L_ELBOW, S.L_WRIST) if side == "l" else (S.R_SHOULDER, S.R_ELBOW, S.R_WRIST)
    )
    upper = kp[sh] - kp[el]
    upper = upper / np.linalg.norm(upper)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    direction = np.array([c * upper[0] - s * upper[1], s * upper[0] + c * upper[1]], dtype=np.float32)
    forearm_len = 0.15 * stature_px
    kp[wr] = kp[el] + direction * forearm_len
    return kp


def make_cache(
    poses: list[np.ndarray],
    fps: float = 30.0,
    track_id: int = 0,
    present: list[bool] | None = None,
    scores: float = 0.9,
    camera_affine: np.ndarray | None = None,
) -> ExtractCache:
    """`poses[i]` をフレーム i のキーポイントとする最小のキャッシュを作る。

    `present[i]` が False のフレームはレコードごと落とす（キャッシュの欠損表現に合わせる）。
    """
    n = len(poses)
    present = [True] * n if present is None else present
    frames: dict[int, list[TrackFrame]] = {}
    for i, kp in enumerate(poses):
        if not present[i]:
            frames[i] = []
            continue
        lo = kp.min(axis=0) - 10.0
        hi = kp.max(axis=0) + 10.0
        frames[i] = [
            TrackFrame(
                track_id=track_id,
                box_xyxy=np.array([lo[0], lo[1], hi[0], hi[1]], dtype=np.float32),
                keypoints=kp.astype(np.float32),
                keypoints_raw=kp.astype(np.float32),
                keypoint_scores=np.full(S.N_KEYPOINTS, scores, dtype=np.float32),
                mask_png=None,
                akaze_points=np.zeros((0, 2), dtype=np.float32),
                akaze_residual=np.zeros((0, 2), dtype=np.float32),
                akaze_residual_mag=np.zeros(0, dtype=np.float32),
                akaze_point_ids=np.zeros(0, dtype=np.int64),
            )
        ]
    return ExtractCache(
        video_path="synthetic",
        width=1920,
        height=1080,
        fps=fps,
        frame_count=n,
        config_hash="synthetic",
        edges=[],
        frames=frames,
        camera_affine=camera_affine,
    )
