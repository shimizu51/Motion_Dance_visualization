from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Iterator

import numpy as np


class VideoProbeError(RuntimeError):
    pass


def probe(path: Path | str) -> dict:
    """ffprobe で幅・高さ・fps・フレーム数を取得する。"""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,nb_frames",
        "-of",
        "json",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise VideoProbeError(f"ffprobe failed for {path}: {proc.stderr}")
    data = json.loads(proc.stdout)
    streams = data.get("streams") or []
    if not streams:
        raise VideoProbeError(f"no video stream found in {path}")
    s = streams[0]
    num, den = s["r_frame_rate"].split("/")
    fps = float(num) / float(den) if float(den) != 0 else float(num)
    return {
        "width": int(s["width"]),
        "height": int(s["height"]),
        "fps": fps,
        "nb_frames": int(s["nb_frames"]) if s.get("nb_frames", "N/A").isdigit() else None,
    }


def _scaled_size(orig_w: int, orig_h: int, target_w: int) -> tuple[int, int]:
    """アスペクト比を維持し、高さは偶数に丸める（ffmpeg の scale=-2 と同じ規則）。"""
    h = round(orig_h * target_w / orig_w)
    h = h if h % 2 == 0 else h + 1
    return target_w, h


class FrameReader:
    """ffmpeg を rawvideo(bgr24) パイプで実行し、フレームを1枚ずつ yield する。"""

    def __init__(
        self,
        path: Path | str,
        width: int = 1920,
        start: float = 0.0,
        duration: float | None = None,
        fps: float | None = None,
    ):
        self.path = str(path)
        info = probe(self.path)
        self.width, self.height = _scaled_size(info["width"], info["height"], width)
        self.fps = fps or info["fps"]
        self.start = start
        self.duration = duration
        self._frame_bytes = self.width * self.height * 3
        self._proc: subprocess.Popen | None = None

    def __enter__(self) -> "FrameReader":
        cmd = ["ffmpeg", "-v", "error"]
        if self.start:
            cmd += ["-ss", str(self.start)]
        cmd += ["-i", self.path]
        if self.duration is not None:
            cmd += ["-t", str(self.duration)]
        cmd += [
            "-vf",
            f"scale={self.width}:{self.height}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-an",
            "-",
        ]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __iter__(self) -> Iterator[np.ndarray]:
        assert self._proc is not None and self._proc.stdout is not None
        while True:
            buf = self._proc.stdout.read(self._frame_bytes)
            if len(buf) < self._frame_bytes:
                break
            yield np.frombuffer(buf, dtype=np.uint8).reshape(self.height, self.width, 3)

    def close(self) -> None:
        if self._proc is None:
            return
        if self._proc.stdout:
            self._proc.stdout.close()
        self._proc.wait()
        self._proc = None


class FrameWriter:
    """rawvideo(bgr24) フレームを受け取り、ffmpeg で H.264 にエンコードする。"""

    def __init__(self, path: Path | str, fps: float, width: int, height: int, crf: int = 16):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.width = width
        self.height = height
        cmd = [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            str(self.path),
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def write(self, frame_bgr: np.ndarray) -> None:
        assert self._proc.stdin is not None
        assert frame_bgr.shape == (self.height, self.width, 3), (
            f"expected frame shape {(self.height, self.width, 3)}, got {frame_bgr.shape}"
        )
        self._proc.stdin.write(np.ascontiguousarray(frame_bgr, dtype=np.uint8).tobytes())

    def close(self) -> None:
        if self._proc.stdin:
            self._proc.stdin.close()
        self._proc.wait()

    def __enter__(self) -> "FrameWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def mux_audio(
    video_no_audio: Path | str,
    audio_source: Path | str,
    out_path: Path | str,
    start: float = 0.0,
    duration: float | None = None,
) -> None:
    """`video_no_audio` の映像に、`audio_source` の該当区間の音声を合わせて `out_path` に出力する。"""
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(video_no_audio)]
    if start:
        cmd += ["-ss", str(start)]
    cmd += ["-i", str(audio_source)]
    if duration is not None:
        cmd += ["-t", str(duration)]
    cmd += [
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
