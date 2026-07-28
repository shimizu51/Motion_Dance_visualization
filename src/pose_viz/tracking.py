from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

_INF_COST = 1e6  # ハンガリアン割当でこの組を選ばせないための擬似無限大（np.inf だと数値的に不安定なため）


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


def _containment_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """xyxy 形式の box 集合同士の包含率 inter / min(area_a, area_b) 行列を返す。

    IoU は「小さい box が大きい box にすっぽり収まる」オクルージョンを過小評価する
    （面積比の分だけ IoU が下がる）ため、前後判定・オクルージョン検出には代わりにこちらを使う。
    """
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
    min_area = np.minimum(area_a, area_b)
    return np.where(min_area > 0, inter / min_area, 0.0).astype(np.float32)


def _shift_box(box: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    """中心を velocity 分だけ動かし、大きさは維持したまま 1 フレーム先の box を予測する。"""
    return np.array(
        [box[0] + velocity[0], box[1] + velocity[1], box[2] + velocity[0], box[3] + velocity[1]],
        dtype=np.float32,
    )


@dataclass
class _Track:
    track_id: int
    box: np.ndarray
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    time_since_update: int = 0  # 直近でマッチしてから経過したフレーム数
    hits: int = 1  # 累計マッチ回数
    confirmed: bool = False
    occluded: bool = False  # 未マッチ中、他トラックに重なられて消えたと推定されるか（寿命延長に使う）
    ref_area: float = 0.0  # 非オクルージョン時の面積の EMA（depth 判定の基準）
    depth_rank: int = 0  # 0 = 最も手前
    depth_score: float = 0.0
    last_det_score: float = 1.0


@dataclass
class TrackMatch:
    """1検出分のマッチ結果。confirmed でない検出には割り当てられない（呼び出し側は None を受け取る）。"""

    track_id: int
    det_score: float
    depth_rank: int
    depth_score: float
    occluded: bool  # このフレームで他人物と一定以上重なっているか（マスク抑制・描画順に使う）
    recovered: bool  # low_threshold 帯の検出で救済されたフレームか


class IoUTracker:
    """予測 + 2段階（高スコア/低スコア）ハンガリアン割当による追跡と、前後関係（depth）の推定を行う。

    オクルージョンでスコアが `score_threshold` を下回った検出（`iou_threshold_low` 以上の box）は
    「既存トラックの継続にのみ」使い、新規トラックの生成には使わない（ByteTrack の BYTE 方式）。
    重なった box 同士は面積・足元位置から前後を判定し、`depth_rank` として返す。
    """

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        score_threshold: float = 0.5,
        iou_threshold: float = 0.3,
        iou_threshold_low: float = 0.25,
        max_age: int = 15,
        max_age_occluded: int = 45,
        min_hits: int = 3,
        w_iou: float = 1.0,
        w_scale: float = 0.5,
        w_center: float = 0.3,
        center_gate: float = 0.15,
        occlusion_containment: float = 0.3,
        depth_overlap_threshold: float = 0.3,
        depth_w_area: float = 1.0,
        depth_w_foot: float = 0.5,
        depth_ref_area_ema: float = 0.9,
        depth_hysteresis: float = 0.1,
    ):
        self.frame_height = float(frame_height)
        self.frame_diag = float(np.hypot(frame_width, frame_height))
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.iou_threshold_low = iou_threshold_low
        self.max_age = max_age
        self.max_age_occluded = max_age_occluded
        self.min_hits = min_hits
        self.w_iou = w_iou
        self.w_scale = w_scale
        self.w_center = w_center
        self.center_gate = center_gate
        self.occlusion_containment = occlusion_containment
        self.depth_overlap_threshold = depth_overlap_threshold
        self.depth_w_area = depth_w_area
        self.depth_w_foot = depth_w_foot
        self.depth_ref_area_ema = depth_ref_area_ema
        self.depth_hysteresis = depth_hysteresis

        self._tracks: dict[int, _Track] = {}
        self._next_id = 0

    def _cost_matrix(self, track_boxes: np.ndarray, det_boxes: np.ndarray) -> np.ndarray:
        iou = _iou_matrix(track_boxes, det_boxes)

        t_area = np.clip((track_boxes[:, 2] - track_boxes[:, 0]) * (track_boxes[:, 3] - track_boxes[:, 1]), 1e-6, None)
        d_area = np.clip((det_boxes[:, 2] - det_boxes[:, 0]) * (det_boxes[:, 3] - det_boxes[:, 1]), 1e-6, None)
        scale_cost = np.abs(np.log(d_area[None, :] / t_area[:, None]))

        t_cx = (track_boxes[:, 0] + track_boxes[:, 2]) / 2
        t_cy = (track_boxes[:, 1] + track_boxes[:, 3]) / 2
        d_cx = (det_boxes[:, 0] + det_boxes[:, 2]) / 2
        d_cy = (det_boxes[:, 1] + det_boxes[:, 3]) / 2
        center_dist = np.sqrt(
            (t_cx[:, None] - d_cx[None, :]) ** 2 + (t_cy[:, None] - d_cy[None, :]) ** 2
        ) / self.frame_diag

        cost = self.w_iou * (1 - iou) + self.w_scale * scale_cost + self.w_center * center_dist
        gate = (iou < self.iou_threshold) & (center_dist > self.center_gate)
        cost = np.where(gate, _INF_COST, cost)
        return cost.astype(np.float64)

    def update(self, boxes_xyxy: np.ndarray, scores: np.ndarray) -> list[TrackMatch | None]:
        """現在フレームの box 集合を既存トラックにマッチさせ、前後関係を更新する。

        戻り値は boxes_xyxy と同じ長さのリストで、確定済みトラックには TrackMatch、
        未確定（min_hits 未達）・低スコアで救済されなかった検出には None を返す。
        """
        n_det = len(boxes_xyxy)
        track_ids = list(self._tracks.keys())
        n_track = len(track_ids)

        pred_boxes = np.zeros((n_track, 4), dtype=np.float32)
        for i, tid in enumerate(track_ids):
            t = self._tracks[tid]
            pred_boxes[i] = _shift_box(t.box, t.velocity)

        if n_det > 0:
            high_det_idx = np.where(scores >= self.score_threshold)[0]
            low_det_idx = np.where(scores < self.score_threshold)[0]
        else:
            high_det_idx = np.zeros(0, dtype=np.int64)
            low_det_idx = np.zeros(0, dtype=np.int64)

        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        assigned_track_for_det: dict[int, int] = {}
        recovered_dets: set[int] = set()

        # Stage 1: 高スコア検出 <-> 全トラック（追跡中・ロスト中問わず）をハンガリアン割当
        if n_track > 0 and len(high_det_idx) > 0:
            cost = self._cost_matrix(pred_boxes, boxes_xyxy[high_det_idx])
            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] >= _INF_COST:
                    continue
                di = int(high_det_idx[c])
                matched_tracks.add(int(r))
                matched_dets.add(di)
                assigned_track_for_det[di] = int(r)

        # Stage 2: 低スコア検出による既存トラックの救済。新規トラックはここでは絶対に作らない。
        remaining_tracks = [ti for ti in range(n_track) if ti not in matched_tracks]
        remaining_low = [int(di) for di in low_det_idx.tolist() if di not in matched_dets]
        if remaining_tracks and remaining_low:
            t_boxes = pred_boxes[remaining_tracks]
            d_boxes = boxes_xyxy[remaining_low]
            iou = _iou_matrix(t_boxes, d_boxes)
            cost = np.where(iou >= self.iou_threshold_low, 1.0 - iou, _INF_COST).astype(np.float64)
            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] >= _INF_COST:
                    continue
                ti = remaining_tracks[r]
                di = remaining_low[c]
                matched_tracks.add(ti)
                matched_dets.add(di)
                assigned_track_for_det[di] = ti
                recovered_dets.add(di)

        result: list[TrackMatch | None] = [None] * n_det

        for di, ti in assigned_track_for_det.items():
            t = self._tracks[track_ids[ti]]
            old_cx, old_cy = (t.box[0] + t.box[2]) / 2, (t.box[1] + t.box[3]) / 2
            new_box = boxes_xyxy[di].astype(np.float32)
            new_cx, new_cy = (new_box[0] + new_box[2]) / 2, (new_box[1] + new_box[3]) / 2
            new_velocity = np.array([new_cx - old_cx, new_cy - old_cy], dtype=np.float32)
            t.velocity = 0.7 * new_velocity + 0.3 * t.velocity
            t.box = new_box
            t.time_since_update = 0
            t.occluded = False
            t.hits += 1
            t.last_det_score = float(scores[di])
            if not t.confirmed and t.hits >= self.min_hits:
                t.confirmed = True
            if t.confirmed:
                result[di] = TrackMatch(
                    track_id=t.track_id,
                    det_score=float(scores[di]),
                    depth_rank=t.depth_rank,
                    depth_score=t.depth_score,
                    occluded=False,
                    recovered=di in recovered_dets,
                )

        # 未マッチの既存トラック: age を進める。他トラックの現在 box と重なっていれば
        # オクルージョン中とみなし、破棄までの猶予を max_age_occluded まで延ばす。
        visible_now = [boxes_xyxy[di].astype(np.float32) for di in assigned_track_for_det]
        visible_boxes_now = np.array(visible_now, dtype=np.float32) if visible_now else np.zeros((0, 4), dtype=np.float32)

        for ti, tid in enumerate(track_ids):
            if ti in matched_tracks:
                continue
            t = self._tracks[tid]
            if len(visible_boxes_now) > 0:
                containment = _containment_matrix(pred_boxes[ti : ti + 1], visible_boxes_now)[0]
                if containment.max() > self.occlusion_containment:
                    t.occluded = True
            t.time_since_update += 1
            limit = self.max_age_occluded if t.occluded else self.max_age
            if t.time_since_update > limit:
                del self._tracks[tid]

        # 未マッチの高スコア検出: 新規トラックとして登録（低スコア検出からは新規トラックを作らない）
        for di in high_det_idx.tolist():
            if di in matched_dets:
                continue
            box = boxes_xyxy[di].astype(np.float32)
            area = max(float((box[2] - box[0]) * (box[3] - box[1])), 1e-6)
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = _Track(track_id=tid, box=box, ref_area=area, last_det_score=float(scores[di]))
            # min_hits(既定3) > 1 のため、新規トラックはこのフレームでは未確定（result は None のまま）

        self._update_depth(result, boxes_xyxy)

        return result

    def _update_depth(self, result: list[TrackMatch | None], boxes_xyxy: np.ndarray) -> None:
        """このフレームで確定・可視の人物について、面積と足元位置から前後関係を更新する。"""
        visible = [(di, m) for di, m in enumerate(result) if m is not None]
        if not visible:
            return
        idxs = [di for di, _ in visible]
        matches = [m for _, m in visible]
        boxes = boxes_xyxy[idxs].astype(np.float32)
        tracks = [self._tracks[m.track_id] for m in matches]

        containment = _containment_matrix(boxes, boxes)
        np.fill_diagonal(containment, 0.0)
        max_containment = containment.max(axis=1) if len(idxs) > 1 else np.zeros(len(idxs), dtype=np.float32)
        is_overlapping = max_containment > self.depth_overlap_threshold

        areas = np.clip((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), 1e-6, None)
        bottoms = boxes[:, 3]

        for k, t in enumerate(tracks):
            matches[k].occluded = bool(is_overlapping[k])
            if not is_overlapping[k]:
                ema = self.depth_ref_area_ema
                t.ref_area = float(areas[k]) if t.ref_area <= 0 else ema * t.ref_area + (1 - ema) * float(areas[k])

        if not is_overlapping.any():
            return  # 重なりが無いフレームは前フレームの順位を維持する（ヒステリシス）

        ref_areas = np.array([max(t.ref_area, 1e-6) for t in tracks], dtype=np.float32)
        depth_scores = self.depth_w_area * np.log(ref_areas) + self.depth_w_foot * (bottoms / self.frame_height)
        prev_ranks = np.array([t.depth_rank for t in tracks], dtype=np.float32)
        # 前フレームで手前（rank が小さい）だったトラックを少しだけ優遇し、僅差でのランク反転を抑える
        adjusted = depth_scores - self.depth_hysteresis * prev_ranks
        order = np.argsort(-adjusted)  # 降順: 0番目が最も手前

        for rank, pos in enumerate(order):
            tracks[pos].depth_rank = rank
            tracks[pos].depth_score = float(depth_scores[pos])
            matches[pos].depth_rank = rank
            matches[pos].depth_score = float(depth_scores[pos])

    def active_track_ids(self) -> set[int]:
        """現在保持している（確定・未確定を問わない）トラック ID の集合。"""
        return set(self._tracks.keys())
