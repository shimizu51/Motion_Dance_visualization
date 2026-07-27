from __future__ import annotations

import colorsys

_GOLDEN_RATIO_CONJUGATE = 0.61803398875


def hue_for_track(track_id: int) -> float:
    """黄金比刻みで track_id に決定的な色相 [0,1) を割り当てる（隣接 ID でも色が離れる）。"""
    return (track_id * _GOLDEN_RATIO_CONJUGATE) % 1.0


def hue_to_bgr(hue: float, saturation: float = 0.85, value: float = 1.0) -> tuple[float, float, float]:
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return (b * 255.0, g * 255.0, r * 255.0)
