from __future__ import annotations

import colorsys

_GOLDEN_RATIO_CONJUGATE = 0.61803398875


def hue_for_track(track_id: int) -> float:
    """黄金比刻みで track_id に決定的な色相 [0,1) を割り当てる（隣接 ID でも色が離れる）。"""
    return (track_id * _GOLDEN_RATIO_CONJUGATE) % 1.0


def hue_to_bgr(hue: float, saturation: float = 0.85, value: float = 1.0) -> tuple[float, float, float]:
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return (b * 255.0, g * 255.0, r * 255.0)


def heat_core(color: tuple[float, float, float], weight: float) -> tuple[float, float, float]:
    """重み 0..1 に応じて、トラック色から白熱側へ寄せた「芯」の色を返す。

    色相は人物 ID を表す情報なので潰さない。重みは **彩度を抜いて白へ寄せる** 方向だけに
    使い、色相のグローは周囲に残す。こうすると「誰か」と「どこに負荷が出ているか」を
    同じ骨格の上で同時に読める。
    """
    w = min(max(weight, 0.0), 1.0)
    return tuple(c + (255.0 - c) * w for c in color)
