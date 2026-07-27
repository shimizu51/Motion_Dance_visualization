from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """xyxy 形式の box 集合同士の IoU 行列 (len(a), len(b)) を返す。"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

    ix1 = np.maximum(ax1, bx1)
    iy1 = np.maximum(ay1, by1)
    ix2 = np.minimum(ax2, bx2)
    iy2 = np.minimum(ay2, by2)
    iw = np.clip(ix2 - ix1, 0, None)
    ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih

    area_a = np.clip(ax2 - ax1, 0, None) * np.clip(ay2 - ay1, 0, None)
    area_b = np.clip(bx2 - bx1, 0, None) * np.clip(by2 - by1, 0, None)
    union = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


@dataclass
class _Track:
    track_id: int
    box: np.ndarray
    age: int = 0  # 直近でマッチしてから経過したフレーム数
    hits: int = 1  # 連続でマッチした回数
    confirmed: bool = False


class IoUTracker:
    """IoU ベースの貪欲マッチングによる軽量トラッカ。

    残差の連続性は track_id の安定性に依存するため、min_hits で確定するまでは
    ID を発行しない（誤検出のちらつきを防ぐ）。
    """

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 15, min_hits: int = 3):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self._tracks: dict[int, _Track] = {}
        self._next_id = 0

    def update(self, boxes_xyxy: np.ndarray) -> list[int | None]:
        """現在フレームの box 集合を既存トラックにマッチさせる。

        戻り値は boxes_xyxy と同じ長さのリストで、確定済みトラックには track_id、
        未確定（min_hits 未達）には None を返す。
        """
        track_ids = list(self._tracks.keys())
        track_boxes = np.array([self._tracks[t].box for t in track_ids]) if track_ids else np.zeros((0, 4))
        iou = _iou_matrix(track_boxes, boxes_xyxy)

        assigned_track_for_det: dict[int, int] = {}  # det_idx -> track列インデックス
        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()

        # 貪欲マッチ: IoU が高い組み合わせから確定させる
        pairs = [
            (iou[ti, di], ti, di)
            for ti in range(len(track_ids))
            for di in range(len(boxes_xyxy))
            if iou[ti, di] >= self.iou_threshold
        ]
        pairs.sort(key=lambda x: x[0], reverse=True)
        for _, ti, di in pairs:
            if ti in matched_tracks or di in matched_dets:
                continue
            matched_tracks.add(ti)
            matched_dets.add(di)
            assigned_track_for_det[di] = ti

        result: list[int | None] = [None] * len(boxes_xyxy)

        for di, ti in assigned_track_for_det.items():
            t = self._tracks[track_ids[ti]]
            t.box = boxes_xyxy[di]
            t.age = 0
            t.hits += 1
            if not t.confirmed and t.hits >= self.min_hits:
                t.confirmed = True
            if t.confirmed:
                result[di] = t.track_id

        # 未マッチの既存トラック: 年齢を進め、max_age を超えたら破棄
        for ti, tid in enumerate(track_ids):
            if ti in matched_tracks:
                continue
            t = self._tracks[tid]
            t.age += 1
            if t.age > self.max_age:
                del self._tracks[tid]

        # 未マッチの検出: 新規トラックとして登録（この時点ではまだ ID を公開しない）
        for di in range(len(boxes_xyxy)):
            if di in matched_dets:
                continue
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = _Track(track_id=tid, box=boxes_xyxy[di])
            if self._tracks[tid].hits >= self.min_hits:
                self._tracks[tid].confirmed = True
                result[di] = tid

        return result

    def active_track_ids(self) -> set[int]:
        """現在保持している（確定・未確定を問わない）トラック ID の集合。"""
        return set(self._tracks.keys())
