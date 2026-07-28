from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from pose_viz.akaze import AkazeResidualTracker, CameraMotionEstimator
from pose_viz.cache import ExtractCache, TrackFrame, encode_mask_crop
from pose_viz.config import Config
from pose_viz.detect import PersonDetector
from pose_viz.pose import PoseEstimator, PoseSmoother
from pose_viz.residual import ParticleSystem
from pose_viz.render import render_frame
from pose_viz.tracking import IoUTracker
from pose_viz.video_io import FrameReader, FrameWriter, mux_audio

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


def _default_cache_path(video_path: Path) -> Path:
    return REPO_ROOT / "data" / "cache" / f"{video_path.stem}.pkl.gz"


def cmd_extract(args: argparse.Namespace) -> None:
    from tqdm import tqdm

    cfg = Config.load(DEFAULT_CONFIG_PATH, args.config)
    if args.start is not None:
        cfg.video.start = args.start
    if args.duration is not None:
        cfg.video.duration = args.duration

    video_path = Path(args.video)
    out_path = Path(args.out) if args.out else _default_cache_path(video_path)

    print(f"loading models (RF-DETR Seg Small / ViTPose)...")
    detector = PersonDetector(
        threshold=cfg.detect.threshold, low_threshold=cfg.detect.low_threshold, min_area_ratio=cfg.detect.min_area_ratio
    )
    pose_estimator = PoseEstimator(model_name=cfg.pose.model)
    pose_smoother = PoseSmoother(cfg.pose.oneeuro.min_cutoff, cfg.pose.oneeuro.beta)
    akaze_trackers: dict[int, AkazeResidualTracker] = {}
    last_seen_frame: dict[int, int] = {}
    camera_estimator = (
        CameraMotionEstimator(
            ratio_test=cfg.camera.ratio_test,
            max_points=cfg.camera.max_points,
            ransac_threshold=cfg.camera.ransac_threshold,
            downscale=cfg.camera.downscale,
            mask_dilate=cfg.camera.mask_dilate,
        )
        if cfg.camera.enabled
        else None
    )
    camera_affine_rows: list[np.ndarray] = []

    frames: dict[int, list[TrackFrame]] = {}
    frame_idx = -1

    with FrameReader(
        video_path, width=cfg.video.width, start=cfg.video.start, duration=cfg.video.duration, fps=cfg.video.fps
    ) as reader:
        fps = reader.fps
        width, height = reader.width, reader.height
        tracker = IoUTracker(
            frame_width=width,
            frame_height=height,
            score_threshold=cfg.detect.threshold,
            iou_threshold=cfg.track.iou_threshold,
            iou_threshold_low=cfg.track.iou_threshold_low,
            max_age=cfg.track.max_age,
            max_age_occluded=cfg.track.max_age_occluded,
            min_hits=cfg.track.min_hits,
            w_iou=cfg.track.w_iou,
            w_scale=cfg.track.w_scale,
            w_center=cfg.track.w_center,
            center_gate=cfg.track.center_gate,
            occlusion_containment=cfg.track.occlusion_containment,
            depth_overlap_threshold=cfg.depth.overlap_threshold,
            depth_w_area=cfg.depth.w_area,
            depth_w_foot=cfg.depth.w_foot,
            depth_ref_area_ema=cfg.depth.ref_area_ema,
            depth_hysteresis=cfg.depth.hysteresis,
        )
        for frame_idx, frame_bgr in enumerate(tqdm(reader, desc="extract")):
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            det = detector.detect(frame_bgr)

            if camera_estimator is not None:
                # 確定トラックだけでなく検出された全人物を除外する（未確定の人物も背景ではない）
                person_union = np.any(det.masks, axis=0) if det.masks is not None and len(det.masks) else None
                affine = camera_estimator.update(gray, person_union)
                camera_affine_rows.append(
                    affine if affine is not None else np.full((2, 3), np.nan, dtype=np.float32)
                )

            matches = tracker.update(det.boxes_xyxy, det.scores)

            confirmed_idx = [i for i, m in enumerate(matches) if m is not None]
            track_frame_list: list[TrackFrame] = []

            if confirmed_idx:
                boxes_conf = det.boxes_xyxy[confirmed_idx]
                masks_conf = [det.masks[i] if det.masks is not None else None for i in confirmed_idx]
                occluded_conf = [matches[i].occluded for i in confirmed_idx]
                pose_out = pose_estimator.estimate(
                    frame_bgr,
                    boxes_conf,
                    masks_full=masks_conf,
                    occluded=occluded_conf,
                    mask_suppress_alpha=cfg.pose.mask_suppress_alpha,
                )
                t = frame_idx / fps
                for local_i, det_i in enumerate(confirmed_idx):
                    m = matches[det_i]
                    tid = m.track_id
                    kp_raw, sc = pose_out[local_i]
                    # One-Euro は描画用の平滑化。計測用に生値も別途持たせる（初回フレームは
                    # smooth() が入力と同一オブジェクトを返すため、明示的にコピーして切り離す）
                    kp_raw = np.array(kp_raw, dtype=np.float32, copy=True)
                    kp = pose_smoother.smooth(tid, t, kp_raw)
                    box = boxes_conf[local_i]
                    mask_full = det.masks[det_i] if det.masks is not None else None
                    mask_png = encode_mask_crop(mask_full, box) if mask_full is not None else None

                    ak_tracker = akaze_trackers.get(tid)
                    if ak_tracker is None:
                        ak_tracker = AkazeResidualTracker(
                            cfg.akaze.ratio_test, cfg.akaze.max_displacement, cfg.akaze.max_points_per_person
                        )
                        akaze_trackers[tid] = ak_tracker
                    elif frame_idx - last_seen_frame.get(tid, frame_idx) > cfg.track.akaze_reset_gap:
                        # 長時間のオクルージョン明けは特徴点の対応付けが破綻するため張り直す
                        ak_tracker.reset()
                    ak_res = ak_tracker.update(gray, box, mask_full)
                    last_seen_frame[tid] = frame_idx

                    track_frame_list.append(
                        TrackFrame(
                            track_id=tid,
                            box_xyxy=box.astype(np.float32),
                            keypoints=kp,
                            keypoints_raw=kp_raw,
                            keypoint_scores=sc,
                            mask_png=mask_png,
                            akaze_points=ak_res.points,
                            akaze_residual=ak_res.residual,
                            akaze_residual_mag=ak_res.residual_mag,
                            akaze_point_ids=ak_res.point_ids,
                            det_score=m.det_score,
                            depth_rank=m.depth_rank,
                            depth_score=m.depth_score,
                            occluded=m.occluded,
                            recovered=m.recovered,
                        )
                    )

            frames[frame_idx] = track_frame_list

            active_ids = tracker.active_track_ids()
            for tid in list(akaze_trackers.keys()):
                if tid not in active_ids:
                    del akaze_trackers[tid]
                    pose_smoother.drop(tid)
                    last_seen_frame.pop(tid, None)

    frame_count = frame_idx + 1
    camera_affine = np.stack(camera_affine_rows).astype(np.float32) if camera_affine_rows else None
    cache = ExtractCache(
        video_path=str(video_path),
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        config_hash=cfg.extract_hash(),
        edges=pose_estimator.edges,
        start=cfg.video.start,
        duration=cfg.video.duration,
        frames=frames,
        camera_affine=camera_affine,
    )
    cache.save(out_path)
    print(f"saved cache: {out_path} ({frame_count} frames, {width}x{height} @ {fps:.3f}fps)")
    if camera_affine is not None:
        ok = int(np.isfinite(camera_affine).all(axis=(1, 2)).sum())
        print(f"  camera motion: {ok}/{frame_count} frames estimated")


def cmd_render(args: argparse.Namespace) -> None:
    from tqdm import tqdm

    cfg = Config.load(DEFAULT_CONFIG_PATH, args.config)
    if getattr(args, "feature_modulation", None):
        cfg.render.feature_modulation = args.feature_modulation
    cache_path = Path(args.cache)
    cache = ExtractCache.load(cache_path)
    cache.check_hash(cfg.extract_hash())

    video_path = Path(args.video) if args.video else Path(cache.video_path)
    out_path = Path(args.out)

    particle_system = ParticleSystem(
        fps=cache.fps,
        lifetime_sec=cfg.residual.lifetime_sec,
        gamma=cfg.residual.gamma,
        lifetime_boost_k=cfg.residual.lifetime_boost_k,
        norm_scale=cfg.residual.norm_scale,
        max_trail_len=cfg.residual.max_trail_len,
    )

    # 特徴量による変調は "off" のとき一切計算しない（従来と同一の出力を保証するため）。
    # render.py は特徴量パッケージを import せず、関節ごとの 0..1 スカラーだけを受け取る。
    joint_weights: dict[int, dict[int, np.ndarray]] = {}
    if cfg.render.feature_modulation != "off":
        from pose_viz.features.export import compute_features
        from pose_viz.features.modulation import build_joint_weights

        print(f"computing features for modulation ({cfg.render.feature_modulation})...")
        joint_weights = build_joint_weights(compute_features(cache, cfg.feature), cfg.render.feature_modulation)

    tmp_out = out_path if args.no_audio or not args.audio else out_path.with_suffix(".noaudio.mp4")

    # 抽出時と同じ区間（start/duration）を読まないと、cache の frame_idx がずれてしまう。
    with FrameReader(
        video_path, width=cache.width, start=cache.start, duration=cache.duration, fps=cache.fps
    ) as reader, FrameWriter(tmp_out, fps=cache.fps, width=cache.width, height=cache.height) as writer:
        for frame_idx, frame_bgr in enumerate(tqdm(reader, desc="render")):
            track_frames = cache.frames.get(frame_idx, [])
            particle_system.update(frame_idx, track_frames)
            out_frame = render_frame(
                frame_bgr,
                frame_idx,
                track_frames,
                cache,
                cache.edges,
                cfg.render,
                cfg.residual,
                particle_system,
                debug_overlay=args.debug_overlay,
                joint_weights=joint_weights.get(frame_idx) or None,
            )
            writer.write(out_frame)

    if args.audio and not args.no_audio:
        mux_audio(tmp_out, video_path, out_path, start=cache.start, duration=cache.duration)
        tmp_out.unlink()
        print(f"saved (with audio): {out_path}")
    else:
        print(f"saved: {out_path}")


def _default_features_path(cache_path: Path) -> Path:
    stem = cache_path.name.removesuffix(".pkl.gz")
    return REPO_ROOT / "data" / "features" / f"{stem}.csv"


def cmd_features(args: argparse.Namespace) -> None:
    """キャッシュから解釈可能な動作特徴量を計算して CSV に書き出す（モデル推論なし）。"""
    from pose_viz.features.export import compute_features, write_summary_csv, write_timeseries_csv

    cfg = Config.load(DEFAULT_CONFIG_PATH, args.config)
    cache_path = Path(args.cache)
    cache = ExtractCache.load(cache_path)
    cache.check_hash(cfg.extract_hash())

    out_path = Path(args.out) if args.out else _default_features_path(cache_path)
    summary_path = out_path.with_name(f"{out_path.stem}_summary.csv")

    print(f"computing features ({len(cache.by_track())} tracks, source={cfg.feature.source})...")
    features = compute_features(cache, cfg.feature)

    rows = write_timeseries_csv(out_path, features)
    write_summary_csv(summary_path, features, cache)
    print(f"saved timeseries: {out_path} ({rows} rows)")
    print(f"saved summary:    {summary_path} ({len(features)} tracks)")

    if args.plot:
        from pose_viz.features.plots import plot_tracks

        plot_dir = Path(args.plot_dir) if args.plot_dir else out_path.parent / f"{out_path.stem}_plots"
        paths = plot_tracks(features, plot_dir, top_n=args.plot_top)
        print(f"saved plots:      {plot_dir} ({len(paths)} figures)")


def cmd_run(args: argparse.Namespace) -> None:
    video_path = Path(args.video)
    cache_path = Path(args.cache) if args.cache else _default_cache_path(video_path)

    extract_args = argparse.Namespace(
        video=str(video_path), out=str(cache_path), config=args.config, start=args.start, duration=args.duration
    )
    cmd_extract(extract_args)

    render_args = argparse.Namespace(
        cache=str(cache_path),
        video=str(video_path),
        out=args.out,
        config=args.config,
        debug_overlay=args.debug_overlay,
        feature_modulation=args.feature_modulation,
        audio=args.audio,
        no_audio=not args.audio,
    )
    cmd_render(render_args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pose-viz")
    sub = parser.add_subparsers(dest="command", required=True)

    p_extract = sub.add_parser("extract", help="動画から骨格・追跡・AKAZE残差を抽出しキャッシュに保存する")
    p_extract.add_argument("video", help="入力動画パス")
    p_extract.add_argument("--out", help="キャッシュの出力先（既定: data/cache/<動画名>.pkl.gz）")
    p_extract.add_argument("--config", help="上書き設定 YAML")
    p_extract.add_argument("--start", type=float, default=None, help="開始秒（試作用に一部だけ抽出する）")
    p_extract.add_argument("--duration", type=float, default=None, help="抽出する長さ（秒）")
    p_extract.set_defaults(func=cmd_extract)

    p_render = sub.add_parser("render", help="キャッシュから可視化映像を合成する（モデル推論なし）")
    p_render.add_argument("--cache", required=True, help="extract で作成したキャッシュ")
    p_render.add_argument("--video", help="元動画パス（省略時はキャッシュ内のパスを使う）")
    p_render.add_argument("--out", required=True, help="出力動画パス")
    p_render.add_argument("--config", help="上書き設定 YAML")
    p_render.add_argument("--debug-overlay", action="store_true", help="黒背景ではなく元映像に検出結果を重ねて確認する")
    p_render.add_argument(
        "--feature-modulation",
        choices=["off", "load", "speed"],
        help="骨格を特徴量で変調する（設定ファイルの render.feature_modulation を上書き）",
    )
    p_render.add_argument("--audio", action="store_true", help="元動画の音声をミックスする")
    p_render.add_argument("--no-audio", action="store_true", help="(内部用) 音声を付けない")
    p_render.set_defaults(func=cmd_render)

    p_feat = sub.add_parser("features", help="キャッシュから解釈可能な動作特徴量を計算し CSV に出力する")
    p_feat.add_argument("--cache", required=True, help="extract で作成したキャッシュ")
    p_feat.add_argument("--out", help="時系列 CSV の出力先（既定: data/features/<キャッシュ名>.csv）")
    p_feat.add_argument("--config", help="上書き設定 YAML")
    p_feat.add_argument("--plot", action="store_true", help="検証用のグラフ（PNG）も出力する")
    p_feat.add_argument("--plot-dir", help="グラフの出力先ディレクトリ")
    p_feat.add_argument("--plot-top", type=int, default=3, help="グラフ化するトラック数（長い順）")
    p_feat.set_defaults(func=cmd_features)

    p_run = sub.add_parser("run", help="extract と render を通しで実行する")
    p_run.add_argument("video", help="入力動画パス")
    p_run.add_argument("--cache", help="キャッシュの保存先（既定: data/cache/<動画名>.pkl.gz）")
    p_run.add_argument("--out", required=True, help="出力動画パス")
    p_run.add_argument("--config", help="上書き設定 YAML")
    p_run.add_argument("--start", type=float, default=None)
    p_run.add_argument("--duration", type=float, default=None)
    p_run.add_argument("--debug-overlay", action="store_true")
    p_run.add_argument("--feature-modulation", choices=["off", "load", "speed"])
    p_run.add_argument("--audio", action="store_true")
    p_run.set_defaults(func=cmd_run)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
