from __future__ import annotations

import gzip
import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


# スキーマを変えたら必ず上げる。読み込み時に不一致なら明確なエラーにする。
#
# 例外: **既定値を持つ省略可能なフィールドの追加は上げない**。dataclass の既定値がクラス属性に
# なるため、古い pickle（`__dict__` にそのキーを持たない）を読んでも既定値が返り、静かに壊れる
# ことがない。80MB のキャッシュを再抽出（27分）させてまで版を上げる理由が無いため。
# 版を上げるのは、既存フィールドの意味・形・単位が変わるときに限る。
CACHE_VERSION = 3


@dataclass
class TrackFrame:
    track_id: int
    box_xyxy: np.ndarray  # (4,) float32, 作業解像度でのピクセル座標
    keypoints: np.ndarray  # (17, 2) float32, One-Euro 平滑化「後」。描画用（低遅延・非対称フィルタ）
    keypoints_raw: np.ndarray  # (17, 2) float32, 平滑化「前」の生値。計測用（ゼロ位相フィルタを別途かける）
    keypoint_scores: np.ndarray  # (17,) float32, ※ヒートマップのピーク値であり確率ではない（1.0 を超え得る）
    mask_png: bytes | None  # box_xyxy でクロップした人物マスクの PNG バイト列
    akaze_points: np.ndarray  # (K, 2) float32
    akaze_residual: np.ndarray  # (K, 2) float32
    akaze_residual_mag: np.ndarray  # (K,) float32
    akaze_point_ids: np.ndarray  # (K,) int64, トラック内で永続する特徴点 ID
    det_score: float = 1.0  # 検出スコア（低スコア救済フレームでは threshold 未満になる）
    depth_rank: int = 0  # 0 = 最も手前。重なりが無いフレームは前回値を維持（ヒステリシス）
    depth_score: float = 0.0
    occluded: bool = False  # このフレームで他人物と一定以上重なっているか
    recovered: bool = False  # low_threshold 帯の検出でトラックが継続されたフレームか
    # (17, 3) float32, **H36M-17 の関節順**（`keypoints` の COCO-17 とは並びが違う）。
    # `pose-viz lift3d` を走らせるまで None。root 相対で、単位は正規化空間（角度算出にのみ使う）。
    keypoints_3d: np.ndarray | None = None


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
    # (frame_count, 2, 3) float32。i 行目は「フレーム i-1 → フレーム i」のカメラ相似変換。
    # 先頭フレームと推定失敗フレームは NaN（識別できるよう単位行列で埋めない）。camera.enabled=false なら None。
    camera_affine: np.ndarray | None = None
    #: 音楽の拍。`pose-viz beats` を走らせるまで None。時刻は **この抽出区間の先頭を 0 秒**
    #: とした秒数なので、`beat_times * fps` がそのまま frame_idx に対応する。
    tempo_bpm: float | None = None
    beat_times: np.ndarray | None = None
    version: int = CACHE_VERSION

    def by_track(self) -> dict[int, list[tuple[int, "TrackFrame"]]]:
        """track_id ごとに (frame_idx, TrackFrame) の時系列を返す（frame_idx 昇順）。

        `frames` は frame_idx をキーにした構造なので、1トラック分の時系列を取り出すには
        全フレーム走査が要る。特徴量計算はトラック単位で行うため、ここで一度だけ転置する。
        欠損フレームは「その frame_idx が現れない」ことで表現される（穴埋めはしない）。
        """
        out: dict[int, list[tuple[int, TrackFrame]]] = {}
        for frame_idx in sorted(self.frames):
            for tf in self.frames[frame_idx]:
                out.setdefault(tf.track_id, []).append((frame_idx, tf))
        return out

    def save(self, path: Path | str) -> None:
        """一時ファイルに書いてから差し替える（途中で失敗しても既存キャッシュを壊さない）。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with gzip.open(tmp, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)

    @staticmethod
    def load(path: Path | str) -> "ExtractCache":
        with gzip.open(path, "rb") as f:
            cache = pickle.load(f)
        # `version` はデータクラスのフィールド既定値としてクラス属性にもなるため、
        # 旧キャッシュ（pickle 復元時に __dict__ に 'version' が無い）を正しく検出するには
        # getattr ではなくインスタンス辞書を直接見る必要がある。
        cache_version = cache.__dict__.get("version", 1)
        if cache_version != CACHE_VERSION:
            raise ValueError(
                f"キャッシュのスキーマバージョン({cache_version})が現在のコード({CACHE_VERSION})と"
                f" 一致しません。'{path}' を extract からやり直してください。"
            )
        return cache

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


def mask_crop_origin(box_xyxy: np.ndarray) -> tuple[int, int]:
    """`encode_mask_crop` が使ったクロップ左上原点（フル解像度座標）を返す。"""
    return int(max(0, np.floor(box_xyxy[0]))), int(max(0, np.floor(box_xyxy[1])))


def decode_mask_crop_only(mask_png: bytes) -> np.ndarray | None:
    """PNG をクロップのまま（フルフレームに配置せずに）二値で復元する。

    面積や差分だけが必要な用途では、フルフレームのキャンバスを毎回確保するより桁違いに軽い。
    """
    crop = cv2.imdecode(np.frombuffer(mask_png, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    return None if crop is None else crop > 127


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
