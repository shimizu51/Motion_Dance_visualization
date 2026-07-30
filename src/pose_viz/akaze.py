from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class AkazeFrameResult:
    points: np.ndarray  # (K, 2) float32, フル解像度のフレーム座標
    residual: np.ndarray  # (K, 2) float32, 大域運動を差し引いた残差ベクトル
    residual_mag: np.ndarray  # (K,) float32
    point_ids: np.ndarray  # (K,) int64, フレームをまたいだ永続 ID


class AkazeResidualTracker:
    """1トラック分の AKAZE 特徴点をフレーム間で対応付け、大域運動からの残差を計算する。

    人物マスクで切り出した ROI 内にのみ特徴点を探すことで、背景の特徴点を残差として
    拾わないようにする。
    """

    def __init__(
        self,
        ratio_test: float = 0.75,
        max_displacement: float = 60.0,
        max_points: int = 400,
    ):
        self.ratio_test = ratio_test
        self.max_displacement = max_displacement
        self.max_points = max_points
        self._detector = cv2.AKAZE_create()
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self._prev_pts_full: np.ndarray | None = None  # (N,2) フル解像度座標
        self._prev_desc: np.ndarray | None = None
        self._prev_ids: np.ndarray | None = None
        self._next_id = 0

    def reset(self) -> None:
        self._prev_pts_full = None
        self._prev_desc = None
        self._prev_ids = None

    def update(
        self,
        gray_frame: np.ndarray,
        box_xyxy: np.ndarray,
        mask_full: np.ndarray | None = None,
    ) -> AkazeFrameResult:
        h, w = gray_frame.shape[:2]
        x1 = int(max(0, math.floor(box_xyxy[0])))
        y1 = int(max(0, math.floor(box_xyxy[1])))
        x2 = int(min(w, math.ceil(box_xyxy[2])))
        y2 = int(min(h, math.ceil(box_xyxy[3])))
        if x2 <= x1 or y2 <= y1:
            return self._empty_and_reset()

        roi = gray_frame[y1:y2, x1:x2]
        mask_roi = None
        if mask_full is not None:
            mask_roi = mask_full[y1:y2, x1:x2].astype(np.uint8) * 255

        kps, desc = self._detector.detectAndCompute(roi, mask_roi)
        if not kps or desc is None:
            self._prev_pts_full = None
            self._prev_desc = None
            self._prev_ids = None
            return self._empty()

        if len(kps) > self.max_points:
            order = np.argsort([-kp.response for kp in kps])[: self.max_points]
            kps = [kps[i] for i in order]
            desc = desc[order]

        cur_pts_local = np.array([kp.pt for kp in kps], dtype=np.float32)
        cur_pts_full = cur_pts_local + np.array([x1, y1], dtype=np.float32)

        cur_ids = np.full(len(cur_pts_full), -1, dtype=np.int64)
        residual = np.zeros_like(cur_pts_full)

        matched = False
        if self._prev_desc is not None and len(self._prev_desc) >= 2 and len(desc) >= 2:
            knn = self._matcher.knnMatch(desc, self._prev_desc, k=2)
            good = [m for m, n in knn if m.distance < self.ratio_test * n.distance]
            if len(good) >= 3:
                cur_idx = np.array([m.queryIdx for m in good])
                prev_idx = np.array([m.trainIdx for m in good])
                cur_matched = cur_pts_full[cur_idx]
                prev_matched = self._prev_pts_full[prev_idx]

                disp = np.linalg.norm(cur_matched - prev_matched, axis=1)
                keep = disp <= self.max_displacement
                if keep.sum() >= 3:
                    cur_idx, prev_idx = cur_idx[keep], prev_idx[keep]
                    cur_matched, prev_matched = cur_matched[keep], prev_matched[keep]

                    M, inliers = cv2.estimateAffinePartial2D(
                        prev_matched, cur_matched, method=cv2.RANSAC, ransacReprojThreshold=3.0
                    )
                    if M is not None:
                        inliers = inliers.ravel().astype(bool)
                        predicted = (M[:, :2] @ prev_matched.T).T + M[:, 2]
                        res = cur_matched - predicted
                        for k, ci in enumerate(cur_idx):
                            cur_ids[ci] = self._prev_ids[prev_idx[k]]
                            residual[ci] = res[k]
                        matched = True

        # マッチしなかった点（新規出現）には新しい永続 ID を振る
        for i in range(len(cur_ids)):
            if cur_ids[i] == -1:
                cur_ids[i] = self._next_id
                self._next_id += 1

        residual_mag = np.linalg.norm(residual, axis=1).astype(np.float32)

        self._prev_pts_full = cur_pts_full
        self._prev_desc = desc
        self._prev_ids = cur_ids

        return AkazeFrameResult(
            points=cur_pts_full, residual=residual, residual_mag=residual_mag, point_ids=cur_ids
        )

    def _empty(self) -> AkazeFrameResult:
        z2 = np.zeros((0, 2), dtype=np.float32)
        return AkazeFrameResult(
            points=z2, residual=z2.copy(), residual_mag=np.zeros(0, dtype=np.float32), point_ids=np.zeros(0, dtype=np.int64)
        )

    def _empty_and_reset(self) -> AkazeFrameResult:
        self.reset()
        return self._empty()


class CameraMotionEstimator:
    """人物を除いた背景の AKAZE 特徴点から、フレーム間のカメラ大域運動を推定する。

    人物の絶対速度にはカメラのパン・ズーム・手ぶれが混入する。ここで推定した相似変換を
    差し引くことで、特徴量側でカメラ運動に依存しない速度を出せるようにする。
    **計測専用であり、描画（render）では一切使わない。**

    フル解像度で毎フレーム AKAZE を掛けると重いため `downscale` で縮小して推定し、
    平行移動成分だけを元の解像度にスケールし直す（回転・スケール成分は解像度に依らない）。
    """

    def __init__(
        self,
        ratio_test: float = 0.75,
        max_points: int = 1000,
        ransac_threshold: float = 3.0,
        downscale: int = 2,
        mask_dilate: int = 15,
    ):
        self.ratio_test = ratio_test
        self.max_points = max_points
        self.ransac_threshold = ransac_threshold
        self.downscale = max(1, int(downscale))
        self.mask_dilate = mask_dilate
        self._detector = cv2.AKAZE_create()
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self._prev_pts: np.ndarray | None = None
        self._prev_desc: np.ndarray | None = None

    def _background_mask(self, person_mask_union: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray | None:
        """人物マスクを膨張させて反転した「背景だけ」のマスクを縮小解像度で作る。"""
        if person_mask_union is None:
            return None
        pm = person_mask_union.astype(np.uint8)
        if pm.shape[:2] != shape:
            pm = cv2.resize(pm, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        if self.mask_dilate > 0:
            # 輪郭のすぐ外側は人物の動きを拾ってしまうため、縮小後の画素数に換算して膨張させる
            k = max(1, int(round(self.mask_dilate / self.downscale)))
            k = k if k % 2 == 1 else k + 1
            pm = cv2.dilate(pm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        return ((1 - pm) * 255).astype(np.uint8)

    def update(self, gray_frame: np.ndarray, person_mask_union: np.ndarray | None = None) -> np.ndarray | None:
        """前フレーム→現フレームの相似変換 (2,3) を返す。推定できなければ None。"""
        d = self.downscale
        if d > 1:
            small = cv2.resize(gray_frame, None, fx=1.0 / d, fy=1.0 / d, interpolation=cv2.INTER_AREA)
        else:
            small = gray_frame

        bg_mask = self._background_mask(person_mask_union, small.shape[:2])
        kps, desc = self._detector.detectAndCompute(small, bg_mask)
        if not kps or desc is None:
            self._prev_pts = None
            self._prev_desc = None
            return None

        if len(kps) > self.max_points:
            order = np.argsort([-kp.response for kp in kps])[: self.max_points]
            kps = [kps[i] for i in order]
            desc = desc[order]
        cur_pts = np.array([kp.pt for kp in kps], dtype=np.float32)

        affine: np.ndarray | None = None
        if self._prev_desc is not None and len(self._prev_desc) >= 2 and len(desc) >= 2:
            knn = self._matcher.knnMatch(desc, self._prev_desc, k=2)
            good = [m for m, n in knn if m.distance < self.ratio_test * n.distance]
            if len(good) >= 3:
                cur_matched = cur_pts[[m.queryIdx for m in good]]
                prev_matched = self._prev_pts[[m.trainIdx for m in good]]
                M, _ = cv2.estimateAffinePartial2D(
                    prev_matched, cur_matched, method=cv2.RANSAC, ransacReprojThreshold=self.ransac_threshold
                )
                if M is not None:
                    affine = M.astype(np.float32)
                    # 縮小画像で推定したので、平行移動だけ元の解像度に戻す
                    affine[:, 2] *= d

        self._prev_pts = cur_pts
        self._prev_desc = desc
        return affine
