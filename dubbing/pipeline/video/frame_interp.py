"""
GPU frame-rate conversion for lip-synced shots.

LatentSync renders at 25 fps. Repeating frames to reach 30 fps stutters, and
ffmpeg's motion-compensated minterpolate (CPU, mostly single-threaded) took
minutes per minute of 720p video. Here each in-between frame is warped from
its two neighbours with RAFT optical flow (torchvision, BSD) on the GPU,
the approach of Super SloMo without its learned refinement:

    F_t→0 ≈ −(1−t)·t·F01 + t²·F10      F_t→1 ≈ (1−t)²·F01 − t·(1−t)·F10
    I_t   = (1−t)·warp(I0, F_t→0) + t·warp(I1, F_t→1)

Flow is estimated at reduced resolution (≤ _FLOW_W wide) and upsampled.
convert() returns False when it cannot run (no GPU, too little memory, an
error); assemble then falls back to ffmpeg frame blending.
"""

import os
import subprocess

import numpy as np

_GPU_NEED_GB = float(os.environ.get("INTERP_GPU_NEED_GB", "3"))
_FLOW_W = 640
_model = None


def gpu_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available() and torch.cuda.mem_get_info()[0] / 1e9 >= _GPU_NEED_GB
    except Exception:
        return False


def _raft():
    global _model
    if _model is None:
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

        _model = raft_large(weights=Raft_Large_Weights.DEFAULT).eval().cuda()
    return _model


def convert(src: str, dst: str, fps_out: float, fps_in: float = 25.0) -> bool:
    """Write `src` re-timed to fps_out (same duration) as `dst`."""
    if not gpu_available():
        return False
    try:
        _convert(src, dst, fps_out, fps_in)
        return True
    except Exception as exc:
        print(f"  ⚠ GPU frame interpolation failed ({exc}) — using frame blending")
        return False


def _convert(src: str, dst: str, fps_out: float, fps_in: float) -> None:
    import cv2
    import torch
    import torch.nn.functional as F

    cap = cv2.VideoCapture(src)
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    if len(frames) < 2:
        raise RuntimeError("clip too short")
    h, w = frames[0].shape[:2]
    fw = min(_FLOW_W, w) // 8 * 8
    fh = max(8, int(round(h * fw / w / 8)) * 8)

    def tens(fr):
        return torch.from_numpy(fr).cuda().permute(2, 0, 1).float().unsqueeze(0) / 255.0

    def flow(a, b):
        # RAFT wants [-1, 1] at a size divisible by 8.
        a8 = F.interpolate(a, size=(fh, fw), mode="bilinear", align_corners=False) * 2 - 1
        b8 = F.interpolate(b, size=(fh, fw), mode="bilinear", align_corners=False) * 2 - 1
        f = _raft()(a8, b8)[-1]
        f = F.interpolate(f, size=(h, w), mode="bilinear", align_corners=False)
        return f * torch.tensor([w / fw, h / fh], device=f.device).view(1, 2, 1, 1)

    ys, xs = torch.meshgrid(
        torch.arange(h, device="cuda"), torch.arange(w, device="cuda"), indexing="ij"
    )
    base = torch.stack([xs, ys], 0).float().unsqueeze(0)

    def warp(img, fl):
        g = base + fl
        gx = 2 * g[:, 0] / (w - 1) - 1
        gy = 2 * g[:, 1] / (h - 1) - 1
        return F.grid_sample(
            img,
            torch.stack([gx, gy], -1),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

    proc = subprocess.Popen(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{w}x{h}",
            "-r",
            f"{fps_out:.6f}",
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "16",
            "-pix_fmt",
            "yuv420p",
            dst,
        ],
        stdin=subprocess.PIPE,
    )
    n_out = int(np.floor((len(frames) - 1) / fps_in * fps_out)) + 1
    cache: dict[int, tuple] = {}
    with torch.no_grad():
        for k in range(n_out):
            pos = k * fps_in / fps_out
            i = min(int(pos), len(frames) - 2)
            t = pos - i
            if t < 0.02 or t > 0.98:
                out = frames[i if t < 0.5 else i + 1]
            else:
                if i not in cache:
                    cache.clear()
                    a, b = tens(frames[i]), tens(frames[i + 1])
                    cache[i] = (a, b, flow(a, b), flow(b, a))
                a, b, f01, f10 = cache[i]
                ft0 = -(1 - t) * t * f01 + t * t * f10
                ft1 = (1 - t) ** 2 * f01 - t * (1 - t) * f10
                img = (1 - t) * warp(a, ft0) + t * warp(b, ft1)
                out = (img[0].permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()
            proc.stdin.write(np.ascontiguousarray(out).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg encode failed")
