from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


class OneEuroFilter:
    """One-Euro フィルタ（Casiez et al. 2012）。ndarray 全体に要素ごとに適用する。"""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.01, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev: np.ndarray | None = None
        self._dx_prev: np.ndarray | None = None
        self._t_prev: float | None = None

    @staticmethod
    def _alpha(dt: float, cutoff: np.ndarray) -> np.ndarray:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, t: float, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if self._x_prev is None:
            self._x_prev = x.copy()
            self._dx_prev = np.zeros_like(x)
            self._t_prev = t
            return x

        dt = max(t - self._t_prev, 1e-6)
        dx = (x - self._x_prev) / dt
        a_d = self._alpha(dt, np.full_like(x, self.d_cutoff))
        edx = a_d * dx + (1 - a_d) * self._dx_prev

        cutoff = self.min_cutoff + self.beta * np.abs(edx)
        a_x = self._alpha(dt, cutoff)
        ex = a_x * x + (1 - a_x) * self._x_prev

        self._x_prev, self._dx_prev, self._t_prev = ex, edx, t
        return ex


@dataclass
class PoseResult:
    keypoints: np.ndarray  # (17, 2) float32、平滑化済み
    scores: np.ndarray  # (17,) float32


class PoseSmoother:
    """トラック ID ごとに独立した One-Euro フィルタ状態を保持する。"""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.01):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self._filters: dict[int, OneEuroFilter] = {}

    def smooth(self, track_id: int, t: float, keypoints: np.ndarray) -> np.ndarray:
        f = self._filters.get(track_id)
        if f is None:
            f = OneEuroFilter(self.min_cutoff, self.beta)
            self._filters[track_id] = f
        return f(t, keypoints)

    def drop(self, track_id: int) -> None:
        self._filters.pop(track_id, None)


class PoseEstimator:
    """ViTPose (transformers) のラッパ。box は xyxy で受け取り、内部で COCO 形式に変換する。"""

    def __init__(self, model_name: str = "usyd-community/vitpose-base-simple", device: str | None = None):
        import torch
        from transformers import AutoProcessor, VitPoseForPoseEstimation

        self._torch = torch
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = VitPoseForPoseEstimation.from_pretrained(model_name)
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = self.model.to(self.device).eval()
        self.edges: list[tuple[int, int]] = [tuple(e) for e in self.model.config.edges]

    def estimate(self, frame_bgr: np.ndarray, boxes_xyxy: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """box ごとに (keypoints (17,2), scores (17,)) を返す。"""
        if len(boxes_xyxy) == 0:
            return []
        from PIL import Image

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)

        boxes_coco = boxes_xyxy.copy()
        boxes_coco[:, 2] = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
        boxes_coco[:, 3] = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]

        inputs = self.processor(image, boxes=[boxes_coco], return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            outputs = self.model(**inputs)
        pose_results = self.processor.post_process_pose_estimation(outputs, boxes=[boxes_coco])[0]
        return [
            (r["keypoints"].detach().cpu().numpy().astype(np.float32), r["scores"].detach().cpu().numpy().astype(np.float32))
            for r in pose_results
        ]
