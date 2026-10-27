"""
LatentSync batch worker.

Runs inside LatentSync's own venv (dubbing/LatentSync/.venv-ls), with the
LatentSync checkout as the working directory. LatentSync pins old library
versions that clash with the main pipeline, so it is kept out of process.

The model loads once and every segment is processed in the same process —
loading the UNet, VAE and Whisper encoder takes far longer than syncing a
short line.

    python latentsync_worker.py jobs.json results.json

jobs.json:    [{"video": ..., "audio": ..., "out": ..., "cx": ..., "cy": ...}, ...]
              cx, cy (optional): position of the face to sync, in pixels
results.json: {out: "ok" | "error: ..."} — rewritten after every job, so a
              crash part-way still reports what finished.
"""

import json
import os
import sys
import time


def main(jobs_path: str, results_path: str) -> None:
    sys.path.insert(0, os.getcwd())
    import torch
    from omegaconf import OmegaConf
    from diffusers import AutoencoderKL, DDIMScheduler
    from accelerate.utils import set_seed
    from latentsync.models.unet import UNet3DConditionModel
    from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
    from latentsync.whisper.audio2feature import Audio2Feature

    config = OmegaConf.load(os.environ.get("LATENTSYNC_CONFIG", "configs/unet/stage2_512.yaml"))
    steps = int(os.environ.get("LATENTSYNC_STEPS", "20"))
    guidance = float(os.environ.get("LATENTSYNC_GUIDANCE", "1.5"))
    dtype = torch.float16

    whisper_path = ("checkpoints/whisper/small.pt" if config.model.cross_attention_dim == 768
                    else "checkpoints/whisper/tiny.pt")
    audio_encoder = Audio2Feature(
        model_path=whisper_path, device="cuda",
        num_frames=config.data.num_frames, audio_feat_length=config.data.audio_feat_length,
    )
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype)
    vae.enable_slicing()                 # decode frames one by one: ~8 GB peak, not ~17
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0
    unet, _ = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(config.model), "checkpoints/latentsync_unet.pt", device="cpu")
    pipeline = LipsyncPipeline(
        vae=vae, audio_encoder=audio_encoder, unet=unet.to(dtype=dtype),
        scheduler=DDIMScheduler.from_pretrained("configs"),
    ).to("cuda")
    target = _target_face_patch()
    _tolerant_faces()
    tmp_root = _private_temp_dirs()
    if os.environ.get("LATENTSYNC_DEEPCACHE", "1") == "1":
        from DeepCache import DeepCacheSDHelper
        helper = DeepCacheSDHelper(pipe=pipeline)
        helper.set_params(cache_interval=3, cache_branch_id=0)
        helper.enable()

    with open(jobs_path, encoding="utf-8") as f:
        jobs = json.load(f)
    results: dict[str, str] = {}
    for k, job in enumerate(jobs, 1):
        t0 = time.time()
        target["xy"] = (job["cx"], job["cy"]) if "cx" in job else None
        try:
            set_seed(1247)
            pipeline(
                video_path=job["video"], audio_path=job["audio"], video_out_path=job["out"],
                num_frames=config.data.num_frames, num_inference_steps=steps,
                guidance_scale=guidance, weight_dtype=dtype,
                width=config.data.resolution, height=config.data.resolution,
                mask_image_path=config.data.mask_image_path,
                temp_dir=os.path.join(tmp_root, "out"),
            )
            results[job["out"]] = "ok" if os.path.exists(job["out"]) else "error: no output"
        except Exception as exc:          # e.g. "Face not detected" on a cutaway shot
            results[job["out"]] = f"error: {type(exc).__name__}: {exc}"[:300]
        print(f"[latentsync] {k}/{len(jobs)} {results[job['out']][:60]} "
              f"({time.time() - t0:.1f}s)", flush=True)
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f)
    import shutil
    shutil.rmtree(tmp_root, ignore_errors=True)


def _tolerant_faces(max_missing: float = 0.4) -> None:
    """
    LatentSync drops a whole shot when the face is missed in a single frame
    (motion blur, a hand, a quick turn): 14 of 20 trailer shots failed that
    way. A frame without a detected face reuses the alignment of the nearest
    frame that has one (the face is tracked through the gap); only a shot
    missing the face in more than `max_missing` of its frames is dropped.
    """
    import cv2
    import kornia
    import numpy as np
    import torch
    from einops import rearrange
    import latentsync.pipelines.lipsync_pipeline as lp

    def affine_transform_video(self, video_frames):
        ip = self.image_processor
        rs = ip.restorer
        res = [None] * len(video_frames)
        for i, frame in enumerate(video_frames):
            try:
                res[i] = ip.affine_transform(frame)
            except RuntimeError:
                pass
        good = [i for i, r in enumerate(res) if r is not None]
        if not good or 1 - len(good) / len(video_frames) > max_missing:
            raise RuntimeError("Face not detected")
        for i, r in enumerate(res):
            if r is not None:
                continue
            _, box, m = res[min(good, key=lambda g: abs(g - i))]
            img = rearrange(torch.from_numpy(video_frames[i]).to(device=rs.device, dtype=rs.dtype),
                            "h w c -> c h w").unsqueeze(0)
            face = kornia.geometry.transform.warp_affine(
                img, m, (rs.face_size[1], rs.face_size[0]), mode="bilinear",
                padding_mode="fill", fill_value=rs.fill_value)
            face = rearrange(face.squeeze(0), "c h w -> h w c").cpu().numpy().astype(np.uint8)
            face = cv2.resize(face, (ip.resolution, ip.resolution), interpolation=cv2.INTER_LANCZOS4)
            res[i] = (rearrange(torch.from_numpy(face), "h w c -> c h w"), box, m)
        if len(good) < len(video_frames):
            print(f"[latentsync] face tracked through {len(video_frames) - len(good)} "
                  f"of {len(video_frames)} frames", flush=True)
        return torch.stack([r[0] for r in res]), [r[1] for r in res], [r[2] for r in res]

    lp.LipsyncPipeline.affine_transform_video = affine_transform_video


def _private_temp_dirs() -> str:
    """
    LatentSync writes intermediate files to a fixed ./temp folder (and
    deletes it first), so workers running at once would destroy each
    other's frames. Give this process its own folder instead.
    """
    import subprocess
    import latentsync.pipelines.lipsync_pipeline as lp
    from latentsync.utils import util

    root = os.path.abspath(f"temp_w{os.getpid()}")
    os.makedirs(root, exist_ok=True)

    def read_video(video_path, change_fps=True, use_decord=True):
        if change_fps:
            out = os.path.join(root, "in_25fps.mp4")
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-nostdin", "-i", video_path,
                            "-r", "25", "-crf", "18", out], check=True)
            video_path = out
        return util.read_video_decord(video_path) if use_decord else util.read_video_cv2(video_path)

    lp.read_video = read_video
    return root


def _target_face_patch() -> dict:
    """
    LatentSync syncs the LARGEST face in each frame. In an over-the-shoulder
    shot that is the listener's blurred head in the foreground, so when a job
    names a face position, offer the detector only the face nearest to it.
    A frame where nothing is near that position counts as "no face", so the
    shot keeps its original frames rather than syncing the wrong person.
    """
    from latentsync.utils.face_detector import FaceDetector
    target = {"xy": None}
    original = FaceDetector.__call__

    def call(self, frame, threshold=0.5):
        if target["xy"] is None:
            return original(self, frame, threshold)
        get = self.app.get
        tx, ty = target["xy"]
        limit = 0.2 * frame.shape[1]

        def nearest(img):
            faces = get(img)
            if not faces:
                return faces
            def dist(f):
                return ((f.bbox[0] + f.bbox[2]) / 2 - tx) ** 2 + ((f.bbox[1] + f.bbox[3]) / 2 - ty) ** 2
            f = min(faces, key=dist)
            return [f] if dist(f) ** 0.5 <= limit else []

        self.app.get = nearest
        try:
            return original(self, frame, threshold)
        finally:
            self.app.get = get

    FaceDetector.__call__ = call
    return target


def analyze(video_path: str, out_path: str) -> None:
    """
    Who is on screen, and whose mouth is moving, frame by frame; plus the
    camera cuts. lipsync.py uses this to sync only shots that show the
    speaker's own face — in shot/reverse-shot editing the camera is often on
    the listener, and syncing whichever face is largest would put one
    person's words in the other's mouth.

    out_path (.npz): t, bbox[4], score, frontal, mouth_motion, emb[512] per
    detected face, and cuts (seconds).
    """
    import cv2
    import numpy as np
    from insightface.app import FaceAnalysis
    from scenedetect import detect, ContentDetector

    step = int(os.environ.get("FACE_SAMPLE_EVERY", "2"))      # every Nth frame
    cuts = [s[0].get_seconds() for s in detect(video_path, ContentDetector())][1:]

    app = FaceAnalysis(name="buffalo_l", root="checkpoints/auxiliary",
                       allowed_modules=["detection", "recognition", "genderage"],
                       providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    rows = {"t": [], "bbox": [], "score": [], "frontal": [], "yaw": [], "mouth": [], "emb": [],
            "male": []}
    prev_mouths: list[tuple[np.ndarray, np.ndarray]] = []   # (emb, mouth crop) last sample
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step:
            i += 1
            continue
        t = i / fps
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cur_mouths = []
        for f in app.get(frame):
            kps = f.kps                      # eyes, nose, mouth corners
            eye_d = np.linalg.norm(kps[1] - kps[0]) + 1e-6
            yaw = abs(kps[2][0] - (kps[0][0] + kps[1][0]) / 2) / eye_d
            # Mouth box from the mouth corners, normalised so motion is
            # comparable between a close-up and a wide shot.
            (mx1, my1), (mx2, my2) = kps[3], kps[4]
            w = max(abs(mx2 - mx1), eye_d * 0.6)
            cx, cy = (mx1 + mx2) / 2, (my1 + my2) / 2
            x1, x2 = int(cx - w * 0.75), int(cx + w * 0.75)
            y1, y2 = int(cy - w * 0.6), int(cy + w * 0.6)
            crop = gray[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            mouth = (cv2.resize(crop, (32, 24)).astype(np.float32) / 255.0
                     if crop.size else None)
            emb = f.normed_embedding.astype(np.float32)
            motion = np.nan
            if mouth is not None and prev_mouths:
                sims = [float(emb @ e) for e, _ in prev_mouths]
                k = int(np.argmax(sims))
                if sims[k] > 0.4 and prev_mouths[k][1] is not None:
                    motion = float(np.abs(mouth - prev_mouths[k][1]).mean())
            cur_mouths.append((emb, mouth))
            rows["t"].append(t)
            rows["bbox"].append(f.bbox.astype(np.float32))
            rows["score"].append(float(f.det_score))
            rows["frontal"].append(bool(yaw < 0.35))
            rows["yaw"].append(float(yaw))
            rows["mouth"].append(motion)
            rows["emb"].append(emb)
            # Face gender: reliable for singers, where voice-based gender
            # classifiers hear a man singing at 260 Hz as a woman.
            rows["male"].append(1.0 if getattr(f, "sex", None) == "M" else
                                0.0 if getattr(f, "sex", None) == "F" else np.nan)
        prev_mouths = cur_mouths
        i += 1
    cap.release()
    np.savez_compressed(
        out_path, t=np.array(rows["t"], np.float32),
        bbox=np.array(rows["bbox"], np.float32).reshape(-1, 4),
        score=np.array(rows["score"], np.float32), frontal=np.array(rows["frontal"]),
        yaw=np.array(rows["yaw"], np.float32),
        mouth=np.array(rows["mouth"], np.float32),
        emb=np.array(rows["emb"], np.float32).reshape(-1, 512),
        male=np.array(rows["male"], np.float32),
        cuts=np.array(cuts, np.float32), fps=np.float32(fps), step=np.int32(step))
    print(f"[analyze] {len(rows['t'])} faces in {i} frames, {len(cuts)} cuts", flush=True)


if __name__ == "__main__":
    # Return freed blocks to the GPU instead of holding them in PyTorch's
    # cache, so other workers fit alongside this one.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if sys.argv[1] == "--analyze":
        sys.path.insert(0, os.getcwd())
        analyze(sys.argv[2], sys.argv[3])
    else:
        main(sys.argv[1], sys.argv[2])
