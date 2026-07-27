from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# RF-DETR (COCO) の person クラス ID。rfdetr.assets.coco_classes で確認済み。
PERSON_CLASS_ID = 1


@dataclass
class DetectionResult:
    boxes_xyxy: np.ndarray  # (N, 4) float32
    scores: np.ndarray  # (N,) float32
    masks: np.ndarray | None  # (N, H, W) bool、モデルがマスクを返さない場合は None


class PersonDetector:
    """RF-DETR Seg Small で人物の box とマスクを1パスで取得する。"""

    def __init__(self, threshold: float = 0.5, min_area_ratio: float = 0.002):
        from rfdetr import RFDETRSegSmall

        self._model = RFDETRSegSmall()
        self.threshold = threshold
        self.min_area_ratio = min_area_ratio

    def detect(self, frame_bgr: np.ndarray) -> DetectionResult:
        from PIL import Image

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        det = self._model.predict(image, threshold=self.threshold)

        keep = det.class_id == PERSON_CLASS_ID
        boxes = det.xyxy[keep].astype(np.float32)
        scores = det.confidence[keep].astype(np.float32)
        masks = det.mask[keep] if getattr(det, "mask", None) is not None else None

        if len(boxes) == 0:
            return DetectionResult(boxes, scores, masks)

        area_ratio = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])) / (h * w)
        keep2 = area_ratio >= self.min_area_ratio
        boxes = boxes[keep2]
        scores = scores[keep2]
        masks = masks[keep2] if masks is not None else None
        return DetectionResult(boxes, scores, masks)
