"""
Seed-VC batch worker.

Runs inside Seed-VC's own venv (dubbing/SeedVC/.venv-vc), with the Seed-VC
checkout as the working directory — like the LatentSync worker, it is kept
out of process (Seed-VC is GPL-3 and pins its own libraries).

The model loads once and every line is converted in the same process.

    python seedvc_worker.py jobs.json results.json

jobs.json:    [{"source": ..., "target": ..., "out": ...}, ...]
              source: the TTS line; target: the original speaker's reference
results.json: {out: "ok" | "error: ..."} — rewritten after every job.
"""

import json
import os
import shutil
import sys
import tempfile
import time


def main(jobs_path: str, results_path: str) -> None:
    sys.path.insert(0, os.getcwd())
    import torch
    import inference as sv

    args = sv.argparse.Namespace(
        checkpoint=None, config=None, fp16=torch.cuda.is_available(), f0_condition=False, auto_f0_adjust=False,
        semi_tone_shift=0, length_adjust=1.0,
        diffusion_steps=int(os.environ.get("SEEDVC_STEPS", "30")),
        inference_cfg_rate=float(os.environ.get("SEEDVC_CFG", "0.7")),
    )
    models = sv.load_models(args)
    sv.load_models = lambda _args: models          # main() reloads otherwise

    with open(jobs_path, encoding="utf-8") as f:
        jobs = json.load(f)
    results: dict[str, str] = {}
    tmp = tempfile.mkdtemp(prefix="seedvc_")
    for k, job in enumerate(jobs, 1):
        t0 = time.time()
        try:
            torch.manual_seed(1247)                # same line → same take
            args.source, args.target, args.output = job["source"], job["target"], tmp
            sv.main(args)
            made = [os.path.join(tmp, f) for f in os.listdir(tmp)]
            if not made:
                raise RuntimeError("no output")
            shutil.move(made[0], job["out"])
            results[job["out"]] = "ok"
        except Exception as exc:
            results[job["out"]] = f"error: {type(exc).__name__}: {exc}"[:300]
        for f in os.listdir(tmp):
            os.remove(os.path.join(tmp, f))
        print(f"[seedvc] {k}/{len(jobs)} {results[job['out']][:60]} ({time.time() - t0:.1f}s)",
              flush=True)
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f)
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main(sys.argv[1], sys.argv[2])
