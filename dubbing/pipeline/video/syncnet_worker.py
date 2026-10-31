"""
Scores how well each face's lips move with the original voice (SyncNet),
for smart lip sync (smart_sync.py).

Runs inside LatentSync/.venv-ls, with LatentSync/ as the working directory.

  python syncnet_worker.py <jobs.json> <out.json>

jobs.json: {"video": path, "audio": path (the original voice track),
            "jobs": [{"key": str, "a": s, "b": s, "track": [[t, x1, y1, x2, y2], ...]}]}

For every job (one face during one shot) the face is cropped frame by frame
from the original video, paired with the original audio of the same time,
and scored with SyncNet (LatentSync's checkpoints/auxiliary/syncnet_v2):
how well the lips move with that audio. Writes {key: [offset, confidence]}.
A face that is really saying the line scores high at an offset near 0; a
listener, a photo or a reaction shot scores low.
"""

import json
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from eval.syncnet import SyncNetEval  # noqa: E402

FPS = 25
cfg = json.load(open(sys.argv[1]))
dev = "cuda" if torch.cuda.is_available() else "cpu"
net = SyncNetEval(device=dev)
net.loadParameters("checkpoints/auxiliary/syncnet_v2.model")

out = {}
with tempfile.TemporaryDirectory() as tmp:
    for job in cfg["jobs"]:
        a, b = float(job["a"]), float(job["b"])
        tr = np.array(job["track"], dtype=np.float32)
        frames = os.path.join(tmp, "f")
        os.makedirs(frames, exist_ok=True)
        for f in os.listdir(frames):
            os.remove(os.path.join(frames, f))
        subprocess.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{a:.3f}",
                "-to",
                f"{b:.3f}",
                "-i",
                cfg["video"],
                "-vf",
                f"fps={FPS}",
                os.path.join(frames, "%05d.jpg"),
            ],
            check=True,
        )
        names = sorted(os.listdir(frames))
        if len(names) < 12 or len(tr) == 0:
            out[job["key"]] = [99, 0.0]
            continue
        crop_v = os.path.join(tmp, "crop.mp4")
        vw = cv2.VideoWriter(crop_v, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (224, 224))
        for i, n in enumerate(names):
            t = a + i / FPS
            box = [np.interp(t, tr[:, 0], tr[:, k]) for k in range(1, 5)]
            img = cv2.imread(os.path.join(frames, n))
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            half = max(box[2] - box[0], box[3] - box[1]) * 0.7  # face plus a margin
            x1, y1 = int(max(0, cx - half)), int(max(0, cy - half))
            x2, y2 = int(min(img.shape[1], cx + half)), int(min(img.shape[0], cy + half))
            face = img[y1:y2, x1:x2]
            vw.write(cv2.resize(face if face.size else img, (224, 224)))
        vw.release()
        clip = os.path.join(tmp, "clip.mp4")
        subprocess.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-i",
                crop_v,
                "-ss",
                f"{a:.3f}",
                "-to",
                f"{b:.3f}",
                "-i",
                cfg["audio"],
                "-map",
                "0:v",
                "-map",
                "1:a",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                clip,
            ],
            check=True,
        )
        vshift = max(2, min(15, (len(names) - 6) // 2))
        try:
            off, _, conf = net.evaluate(
                video_path=clip, temp_dir=os.path.join(tmp, "e"), vshift=vshift
            )
            out[job["key"]] = [int(off), float(conf)]
        except Exception as exc:  # too short, no audio…
            print(f"syncnet: {job['key']}: {exc}", file=sys.stderr)
            out[job["key"]] = [99, 0.0]
json.dump(out, open(sys.argv[2], "w"))
