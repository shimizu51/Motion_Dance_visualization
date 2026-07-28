from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class VideoConfig:
    width: int = 1920
    fps: float | None = None
    start: float = 0.0
    duration: float | None = None


@dataclass
class DetectConfig:
    model: str = "RFDETRSegSmall"
    threshold: float = 0.5
    low_threshold: float = 0.15
    min_area_ratio: float = 0.002


@dataclass
class TrackConfig:
    iou_threshold: float = 0.3
    iou_threshold_low: float = 0.25
    max_age: int = 15
    max_age_occluded: int = 45
    min_hits: int = 3
    w_iou: float = 1.0
    w_scale: float = 0.5
    w_center: float = 0.3
    center_gate: float = 0.15
    occlusion_containment: float = 0.3
    akaze_reset_gap: int = 3


@dataclass
class DepthConfig:
    overlap_threshold: float = 0.3
    w_area: float = 1.0
    w_foot: float = 0.5
    ref_area_ema: float = 0.9
    hysteresis: float = 0.1


@dataclass
class OneEuroConfig:
    min_cutoff: float = 1.0
    beta: float = 0.01


@dataclass
class PoseConfig:
    model: str = "usyd-community/vitpose-base-simple"
    keypoint_score_threshold: float = 0.3
    mask_suppress_alpha: float = 0.8
    oneeuro: OneEuroConfig = field(default_factory=OneEuroConfig)


@dataclass
class AkazeConfig:
    trail_mode: str = "akaze"  # "akaze" | "akaze_lk"
    ratio_test: float = 0.75
    max_displacement: float = 60.0
    max_points_per_person: int = 400


@dataclass
class ResidualConfig:
    lifetime_sec: float = 0.8
    gamma: float = 1.6
    lifetime_boost_k: float = 1.0
    joint_trail_len: int = 20
    norm_scale: float = 8.0
    max_trail_len: int = 40


@dataclass
class RenderConfig:
    ghost_alpha: float = 0.15
    ghost_blur: int = 9
    ghost_desaturate: float = 0.8
    particle_glow: int = 11
    skeleton_bloom: tuple[int, int] = (5, 25)
    bone_thickness: int = 3
    vignette: float = 0.2
    grain: float = 0.02
    depth_body_occlude: bool = False


@dataclass
class Config:
    video: VideoConfig = field(default_factory=VideoConfig)
    detect: DetectConfig = field(default_factory=DetectConfig)
    track: TrackConfig = field(default_factory=TrackConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    akaze: AkazeConfig = field(default_factory=AkazeConfig)
    residual: ResidualConfig = field(default_factory=ResidualConfig)
    render: RenderConfig = field(default_factory=RenderConfig)

    @staticmethod
    def load(*paths: Path | str | None) -> "Config":
        merged: dict[str, Any] = {}
        for p in paths:
            if p is None:
                continue
            p = Path(p)
            with open(p, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            merged = _deep_merge(merged, data)
        return _dataclass_from_dict(Config, merged)

    def extract_hash(self) -> str:
        """extract 段階の結果を左右する設定だけを対象にしたハッシュ。

        render 側のパラメータ（配色・寿命の見た目調整など）を変えてもキャッシュは
        無効化しない。`video.start`/`duration` はどの区間を切り出すかの指定であって
        設定内容そのものではない（`cache.start`/`cache.duration` で別途管理する）ため、
        ここには含めない。
        """
        payload = {
            "video_width": self.video.width,
            "video_fps": self.video.fps,
            "detect": asdict(self.detect),
            "track": asdict(self.track),
            "depth": asdict(self.depth),
            "pose": asdict(self.pose),
            "akaze": asdict(self.akaze),
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _dataclass_from_dict(cls: type, data: dict) -> Any:
    if not is_dataclass(cls):
        return data
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        field_type = f.type
        if isinstance(field_type, str):
            # postponed annotations (from __future__ import annotations) 対応
            field_type = _resolve_type(cls, f.name)
        if is_dataclass(field_type) and isinstance(value, dict):
            kwargs[f.name] = _dataclass_from_dict(field_type, value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


def _resolve_type(cls: type, field_name: str) -> Any:
    hints = _type_hints_cache.get(cls)
    if hints is None:
        import typing

        hints = typing.get_type_hints(cls)
        _type_hints_cache[cls] = hints
    return hints.get(field_name)


_type_hints_cache: dict[type, dict[str, Any]] = {}
