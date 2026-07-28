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
    trail_mode: str = "akaze"  # "akaze" | "akaze_lk"（※ akaze_lk は未実装。現在この値は読まれていない）
    ratio_test: float = 0.75
    max_displacement: float = 60.0
    max_points_per_person: int = 400


@dataclass
class CameraConfig:
    """背景特徴点によるカメラ大域運動（パン・ズーム・手ぶれ）の推定。

    人物の絶対速度からカメラ自身の動きを差し引くための **計測用** の情報であり、描画には使わない。
    """

    enabled: bool = True
    downscale: int = 2  # 縮小してから検出する（フル解像度の毎フレーム AKAZE は重いため）
    ratio_test: float = 0.75
    max_points: int = 1000
    ransac_threshold: float = 3.0
    mask_dilate: int = 15  # 人物マスクを膨張させ、輪郭付近の特徴点を背景から除外する（フル解像度の画素数）


@dataclass
class Lift3DConfig:
    """単眼 2D→3D リフティング（MotionBERT）。`pose-viz lift3d` でのみ使う。

    2D 関節角度は面外回転で系統的に歪むため、その計測誤差を潰す目的でのみ深層学習を使う。
    出力は依然として「関節角度」という説明可能な量のまま。
    """

    clip_len: int = 243  # モデルの上限。これを超える値は指定できない
    stride: int | None = None  # 窓の移動量（既定は clip_len の半分＝5割重ねる）
    flip_augment: bool = True  # 左右反転を平均するテスト時拡張。精度が上がる代わりに 2 倍の時間
    device: str | None = None  # 既定は mps があれば mps


@dataclass
class FeatureConfig:
    """解釈可能な動作特徴量の算出パラメータ。

    キャッシュ済みの抽出結果から計算する **分析側** の設定なので、`extract_hash()` には含めない
    （ここを変えても再 extract は不要）。
    """

    source: str = "raw"  # "raw"（平滑化前・計測用）| "smoothed"（One-Euro 後・比較用）
    #: 関節角度をどの座標から出すか。"auto" = キャッシュに 3D があれば 3D、無ければ 2D。
    #: "2d" / "3d" は明示的に固定する（2D と 3D を突き合わせて検証したいときに使う）。
    angle_source: str = "auto"
    min_score: float = 0.3  # これ未満のキーポイントは欠損として扱う（スコアは確率ではない点に注意）
    max_gap_sec: float = 0.2  # これ以下の欠損は線形補間する。超えるとセグメントを分割する
    scale_window_sec: float = 2.0  # 体幹長の移動中央値の窓（面外回転による瞬間的短縮を均す）
    deriv_window_sec: float = 0.25  # Savitzky-Golay の窓（秒指定なので fps に依存しない）
    deriv_polyorder: int = 3
    foreshorten_threshold: float = 0.6  # セグメント長がこの比率を下回る区間は関節角度を信用しない
    flip_min_sec: float = 0.2  # これより短い体の向きの反転は L/R 取り違えとみなす
    sparc_window_sec: float = 2.0
    symmetry_window_sec: float = 2.0
    rhythm_min_period_sec: float = 0.25
    rhythm_max_period_sec: float = 4.0
    load_percentile: float = 95.0  # 負荷代理指標の正規化基準（動画内相対）
    load_weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)  # 角速度・角加速度・ROM逸脱・制動


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
    #: 特徴量で骨格を変調する。"off" のときは特徴量を一切計算せず、従来と同一の出力になる。
    #: "load" = 関節負荷の代理指標、"speed" = 関節速度。
    feature_modulation: str = "off"
    feature_thickness_gain: float = 4.0  # 重み 1.0 のときに増える線幅（px）
    feature_highlight: float = 1.0  # 白熱コアの強さ（0 で無効）
    #: AKAZE 残差の「向き」で粒子の先頭を伸ばす倍率。0 で無効（従来の描画のまま）。
    residual_streak_gain: float = 0.0


@dataclass
class Config:
    video: VideoConfig = field(default_factory=VideoConfig)
    detect: DetectConfig = field(default_factory=DetectConfig)
    track: TrackConfig = field(default_factory=TrackConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    akaze: AkazeConfig = field(default_factory=AkazeConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    lift3d: Lift3DConfig = field(default_factory=Lift3DConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
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

        render 側のパラメータ（配色・寿命の見た目調整など）と feature 側のパラメータ
        （特徴量の窓幅・閾値など）を変えてもキャッシュは無効化しない。`video.start`/`duration` は
        どの区間を切り出すかの指定であって設定内容そのものではない（`cache.start`/`cache.duration`
        で別途管理する）ため、ここには含めない。
        """
        payload = {
            "video_width": self.video.width,
            "video_fps": self.video.fps,
            "detect": asdict(self.detect),
            "track": asdict(self.track),
            "depth": asdict(self.depth),
            "pose": asdict(self.pose),
            "akaze": asdict(self.akaze),
            "camera": asdict(self.camera),
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
