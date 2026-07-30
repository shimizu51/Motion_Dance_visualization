"""単眼 2D → 3D リフティング（MotionBERT）。

**深層学習を「表現層」ではなく「計測改善層」として使う。** ここで得るのは説明不能な埋め込みでは
なく、依然として「膝の屈曲角」という説明可能な量である。2D 関節角度は面外回転（カメラ面から
外れた方向への屈曲）で系統的に歪むという計測上の弱点があり、それを潰すためだけに使う。
出力の解釈可能性は一切犠牲にしない。

モデル定義は `pose_viz.vendor.motionbert`（Apache-2.0）に固定してあり、重みは実行時に
HuggingFace から取得する。extract 側の処理なので `render.py` からは決して呼ばれない。
"""

from __future__ import annotations

import numpy as np

# --- Human3.6M 17 関節（MotionBERT の入出力フォーマット）------------------------
H_HIP = 0  # 骨盤（ルート）
H_RHIP, H_RKNEE, H_RANKLE = 1, 2, 3
H_LHIP, H_LKNEE, H_LANKLE = 4, 5, 6
H_SPINE, H_THORAX, H_NOSE, H_HEAD = 7, 8, 9, 10
H_LSHOULDER, H_LELBOW, H_LWRIST = 11, 12, 13
H_RSHOULDER, H_RELBOW, H_RWRIST = 14, 15, 16

H36M_NAMES = [
    "hip", "r_hip", "r_knee", "r_ankle", "l_hip", "l_knee", "l_ankle",
    "spine", "thorax", "nose", "head", "l_shoulder", "l_elbow", "l_wrist",
    "r_shoulder", "r_elbow", "r_wrist",
]

#: 左右反転（テスト時拡張）で入れ替える関節。MotionBERT の `flip_data` と同じ定義。
_FLIP_LEFT = [4, 5, 6, 11, 12, 13]
_FLIP_RIGHT = [1, 2, 3, 14, 15, 16]

HF_REPO = "walterzhu/MotionBERT"
#: root 相対で 3D を出す本命の重み。lite/global 版より精度が高く、角度の算出には root 相対で足りる。
HF_CHECKPOINT = "checkpoint/pose3d/FT_MB_release_MB_ft_h36m/best_epoch.bin"
#: 上流 configs/pose3d/MB_ft_h36m.yaml の値
_ARCH = dict(dim_in=3, dim_out=3, dim_feat=512, dim_rep=512, depth=5, num_heads=8, mlp_ratio=2, num_joints=17)
MAX_CLIP_LEN = 243  # モデルが扱える最大フレーム数（temp_embed のサイズ）


def coco2h36m(x: np.ndarray) -> np.ndarray:
    """COCO-17 (T,17,C) を H36M-17 (T,17,C) に並べ替える。

    COCO には骨盤・脊椎・胸郭・頭頂に対応する点が無いため、左右の股関節・肩から合成する。
    上流の `halpe2h36m` と同じ関節定義に合わせてある。
    """
    y = np.zeros_like(x)
    y[:, H_HIP] = (x[:, 11] + x[:, 12]) * 0.5  # 左右股関節の中点
    y[:, H_RHIP] = x[:, 12]
    y[:, H_RKNEE] = x[:, 14]
    y[:, H_RANKLE] = x[:, 16]
    y[:, H_LHIP] = x[:, 11]
    y[:, H_LKNEE] = x[:, 13]
    y[:, H_LANKLE] = x[:, 15]
    y[:, H_THORAX] = (x[:, 5] + x[:, 6]) * 0.5  # 左右肩の中点
    y[:, H_SPINE] = (y[:, H_HIP] + y[:, H_THORAX]) * 0.5
    y[:, H_NOSE] = x[:, 0]
    y[:, H_HEAD] = (x[:, 1] + x[:, 2]) * 0.5  # 両目の中点を頭部の代用にする
    y[:, H_LSHOULDER] = x[:, 5]
    y[:, H_LELBOW] = x[:, 7]
    y[:, H_LWRIST] = x[:, 9]
    y[:, H_RSHOULDER] = x[:, 6]
    y[:, H_RELBOW] = x[:, 8]
    y[:, H_RWRIST] = x[:, 10]
    return y


def normalize_sequence(motion: np.ndarray) -> np.ndarray:
    """系列全体の外接矩形を [-1, 1] に正規化する（上流 `crop_scale` の scale_range=[1,1] と同一）。

    信頼度 0 の点は外接矩形の計算から外す。クリップごとではなく **トラック全体** で正規化するのは、
    窓をまたいでスケールが揃っていないと、重なり部分をブレンドしたときに段差が出るため。
    """
    result = motion.copy()
    valid = motion[..., 2] != 0
    coords = motion[valid][:, :2]
    if len(coords) < 4:
        return np.zeros_like(motion)

    xmin, xmax = float(coords[:, 0].min()), float(coords[:, 0].max())
    ymin, ymax = float(coords[:, 1].min()), float(coords[:, 1].max())
    scale = max(xmax - xmin, ymax - ymin)
    if scale == 0:
        return np.zeros_like(motion)

    xs = (xmin + xmax - scale) / 2
    ys = (ymin + ymax - scale) / 2
    result[..., :2] = (motion[..., :2] - np.array([xs, ys], dtype=motion.dtype)) / scale
    result[..., :2] = (result[..., :2] - 0.5) * 2
    return np.clip(result, -1, 1)


def flip_motion(x: np.ndarray) -> np.ndarray:
    """左右反転（x 符号を反転し、左右の関節を入れ替える）。"""
    y = x.copy()
    y[..., 0] *= -1
    y[..., _FLIP_LEFT + _FLIP_RIGHT, :] = y[..., _FLIP_RIGHT + _FLIP_LEFT, :]
    return y


def plan_windows(n: int, clip_len: int, stride: int) -> list[tuple[int, int]]:
    """系列を重なりありの窓に分割する。

    重なりを持たせるのは、窓ごとに独立して推論すると **境界で 3D が不連続になり**、
    そこから角度を微分したときにスパイクを生むため（Phase 1 で SavGol の端に対して
    行ったのと同じ配慮）。
    """
    if n <= clip_len:
        return [(0, n)]
    starts = list(range(0, n - clip_len + 1, max(1, stride)))
    if starts[-1] + clip_len < n:
        starts.append(n - clip_len)
    return [(s, s + clip_len) for s in starts]


def _blend_weights(length: int) -> np.ndarray:
    """窓の中央ほど重い三角重み（両端でも 0 にはしない）。"""
    idx = np.arange(1, length + 1, dtype=np.float64)
    return np.minimum(idx, idx[::-1])


class Pose3DLifter:
    """MotionBERT (DSTformer) のラッパ。COCO-17 の 2D 系列を H36M-17 の 3D 系列に変換する。"""

    def __init__(
        self,
        clip_len: int = MAX_CLIP_LEN,
        stride: int | None = None,
        flip_augment: bool = True,
        device: str | None = None,
    ):
        import torch
        import torch.nn as nn
        from functools import partial

        from huggingface_hub import hf_hub_download

        from pose_viz.vendor.motionbert.DSTformer import DSTformer

        if clip_len > MAX_CLIP_LEN:
            raise ValueError(f"clip_len は {MAX_CLIP_LEN} 以下（モデルの temp_embed の上限）: {clip_len}")

        self._torch = torch
        self.clip_len = clip_len
        self.stride = stride if stride is not None else max(1, clip_len // 2)
        self.flip_augment = flip_augment
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")

        ckpt_path = hf_hub_download(HF_REPO, HF_CHECKPOINT)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model_pos"]
        # 上流は DataParallel でラップした状態で保存しているので "module." を剥がす
        state = {k.removeprefix("module."): v for k, v in state.items()}

        self.model = DSTformer(norm_layer=partial(nn.LayerNorm, eps=1e-6), maxlen=MAX_CLIP_LEN, **_ARCH)
        self.model.load_state_dict(state, strict=True)
        self.model = self.model.to(self.device).eval()

    def _forward(self, clip: np.ndarray) -> np.ndarray:
        """(F,17,3) の正規化済み 2D を (F,17,3) の 3D にする。"""
        torch = self._torch
        x = torch.from_numpy(clip[None].astype(np.float32)).to(self.device)
        with torch.no_grad():
            out = self.model(x)
            if self.flip_augment:
                flipped = torch.from_numpy(flip_motion(clip)[None].astype(np.float32)).to(self.device)
                out_flip = self.model(flipped)
                out = (out + torch.from_numpy(flip_motion(out_flip.cpu().numpy())).to(self.device)) / 2.0
        return out[0].cpu().numpy()

    def lift(self, keypoints_coco: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """COCO-17 の 2D 系列を H36M-17 の root 相対 3D 系列に変換する。

        `keypoints_coco` は (T,17,2) のピクセル座標（NaN は欠損）、`scores` は (T,17)。
        戻り値は (T,17,3) float32。入力が短すぎる等で推論できない場合は NaN で埋めた配列。
        """
        n = len(keypoints_coco)
        out = np.full((n, 17, 3), np.nan, dtype=np.float32)
        if n == 0:
            return out

        # 欠損は「信頼度 0」で表現する。座標は前後の有効値で埋めておく（モデルは全フレームに
        # 値を要求するが、信頼度 0 の点は正規化の外接矩形からも除かれ、モデルも低く重み付けする）
        xy = np.asarray(keypoints_coco, dtype=np.float64).copy()
        conf = np.clip(np.nan_to_num(np.asarray(scores, dtype=np.float64), nan=0.0), 0.0, 1.0)
        missing = ~np.isfinite(xy).all(axis=2)
        conf[missing] = 0.0
        for j in range(xy.shape[1]):
            for c in range(2):
                col = xy[:, j, c]
                valid = np.isfinite(col)
                if not valid.any():
                    col[:] = 0.0
                elif not valid.all():
                    idx = np.arange(n)
                    col[~valid] = np.interp(idx[~valid], idx[valid], col[valid])

        motion = np.concatenate([xy, conf[..., None]], axis=2)
        motion = coco2h36m(motion)
        # coco2h36m の合成点（骨盤・脊椎・胸郭・頭部）は平均で信頼度も混ざるので下限を切る
        motion[..., 2] = np.clip(motion[..., 2], 0.0, 1.0)
        motion = normalize_sequence(motion)
        if not np.any(motion):
            return out

        acc = np.zeros((n, 17, 3), dtype=np.float64)
        wsum = np.zeros(n, dtype=np.float64)
        for s, e in plan_windows(n, self.clip_len, self.stride):
            pred = self._forward(motion[s:e])
            w = _blend_weights(e - s)
            acc[s:e] += pred * w[:, None, None]
            wsum[s:e] += w

        good = wsum > 0
        out[good] = (acc[good] / wsum[good, None, None]).astype(np.float32)
        out[:, H_HIP, :] = 0.0  # root 相対（上流 rootrel: True と同じ扱い）
        # 元から一度も観測されなかったフレームは 3D も持たせない
        out[~np.isfinite(keypoints_coco).any(axis=(1, 2))] = np.nan
        return out
