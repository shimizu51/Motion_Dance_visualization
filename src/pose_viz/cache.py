from __future__ import annotations

import gzip
import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class TrackFrame:
    track_id: int
    box_xyxy: np.ndarray  # (4,) float32, 作業解像度でのピクセル座標
    keypoints: np.ndarray  # (17, 2) float32
    keypoint_scores: np.ndarray  # (17,) float32
    mask_png: bytes | None  # box_xyxy でクロップした人物マスクの PNG バイト列
    akaze_points: np.ndarray  # (K, 2) float32
    akaze_residual: np.ndarray  # (K, 2) float32
    akaze_residual_mag: np.ndarray  # (K,) float32
    akaze_point_ids: np.ndarray  # (K,) int64, トラック内で永続する特徴点 ID


@dataclass
class ExtractCache:
    video_path: str
    width: int
    height: int
    fps: float
    frame_count: int
    config_hash: str
    edges: list[tuple[int, int]]
    start: float = 0.0
    duration: float | None = None
    frames: dict[int, list[TrackFrame]] = field(default_factory=dict)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: Path | str) -> "ExtractCache":
        with gzip.open(path, "rb") as f:
            return pickle.load(f)

    def check_hash(self, expected_hash: str) -> None:
        if self.config_hash != expected_hash:
            warnings.warn(
                f"キャッシュの設定ハッシュ({self.config_hash})が現在の設定({expected_hash})と一致しません。"
                " extract をやり直すことを推奨します。",
                stacklevel=2,
            )


def encode_mask_crop(mask_full: np.ndarray, box_xyxy: np.ndarray) -> bytes | None:
    """box_xyxy でクロップしたマスクを PNG エンコードして保存する（二値なので高圧縮）。"""
    h, w = mask_full.shape[:2]
    x1 = int(max(0, np.floor(box_xyxy[0])))
    y1 = int(max(0, np.floor(box_xyxy[1])))
    x2 = int(min(w, np.ceil(box_xyxy[2])))
    y2 = int(min(h, np.ceil(box_xyxy[3])))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = (mask_full[y1:y2, x1:x2].astype(np.uint8)) * 255
    ok, buf = cv2.imencode(".png", crop)
    if not ok:
        return None
    return buf.tobytes()


def decode_mask_crop(mask_png: bytes, box_xyxy: np.ndarray, frame_shape: tuple[int, int]) -> np.ndarray:
    """PNG バイト列を復元し、フルフレームサイズのマスク(float32, 0..1)に配置する。"""
    h, w = frame_shape
    full = np.zeros((h, w), dtype=np.float32)
    crop = cv2.imdecode(np.frombuffer(mask_png, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if crop is None:
        return full
    x1 = int(max(0, np.floor(box_xyxy[0])))
    y1 = int(max(0, np.floor(box_xyxy[1])))
    x2 = min(w, x1 + crop.shape[1])
    y2 = min(h, y1 + crop.shape[0])
    full[y1:y2, x1:x2] = crop[: y2 - y1, : x2 - x1].astype(np.float32) / 255.0
    return full
