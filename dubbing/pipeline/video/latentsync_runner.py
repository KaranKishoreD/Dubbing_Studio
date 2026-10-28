"""
Running LatentSync from the main pipeline.

LatentSync needs its own Python environment (dubbing/LatentSync/.venv-ls),
so it runs as a separate process (latentsync_worker.py). This module starts
those workers: as many in parallel as free GPU memory allows, waits when
another job is using the GPU, and retries with one worker after an
out-of-memory error.
"""

import os
import subprocess
from pathlib import Path

LATENTSYNC_DIR = Path(os.environ.get("LATENTSYNC_DIR", "LatentSync"))


_LS_PYTHON = LATENTSYNC_DIR / ".venv-ls" / "bin" / "python"


_LS_CKPT = LATENTSYNC_DIR / "checkpoints" / "latentsync_unet.pt"


_WORKER = Path(__file__).resolve().parent / "latentsync_worker.py"


# Measured peak per LatentSync worker with VAE slicing: ~8.3 GB. Workers are
# sized from FREE memory at the moment lip sync starts, with headroom kept.
_WORKER_GB = float(os.environ.get("LATENTSYNC_WORKER_GB", "8.5"))


_GPU_RESERVE_GB = float(os.environ.get("LATENTSYNC_RESERVE_GB", "1.5"))


_MAX_WORKERS = int(os.environ.get("LATENTSYNC_MAX_WORKERS", "4"))


def free_gpu_gb() -> float:
    """Free memory on GPU 0 right now, as the driver reports it (0 if unknown)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
        return float(out.strip().splitlines()[0]) / 1024
    except Exception:
        return 0.0


_WAIT_SECS = float(os.environ.get("LATENTSYNC_WAIT_SECS", "600"))


def wait_for_gpu(need_gb: float) -> float:
    """Poll free GPU memory every 15 s until `need_gb` is free or the wait
    runs out. Returns the last free amount."""
    import time

    deadline = time.time() + _WAIT_SECS
    free = free_gpu_gb()
    if free < need_gb:
        print(
            f"  LatentSync: waiting for GPU memory ({free:.1f} GB free, need "
            f"{need_gb:.1f}) — another job is using the GPU",
            flush=True,
        )
    while free < need_gb and time.time() < deadline:
        time.sleep(15)
        free = free_gpu_gb()
    return free


def run_parallel(jobs: list[dict], pending: list, cache_dir: Path) -> tuple[dict, int]:
    """
    Render with as many LatentSync workers as free GPU memory allows (at
    least 1, at most _MAX_WORKERS). Shots are dealt longest-first to the
    least-loaded worker so all workers finish at about the same time.
    """
    import json

    # The GPU is shared with other users' jobs on this server: wait (up to
    # LATENTSYNC_WAIT_SECS) until one worker fits instead of failing every shot.
    free = wait_for_gpu(_WORKER_GB + _GPU_RESERVE_GB)
    if free < _WORKER_GB + _GPU_RESERVE_GB:
        print(
            f"  ⚠ LatentSync: GPU busy with another job ({free:.1f} GB free, need "
            f"{_WORKER_GB + _GPU_RESERVE_GB:.1f}) — lip sync skipped, original frames kept",
            flush=True,
        )
        return {j["out"]: "error: GPU busy with another job" for j in jobs}, 0
    n = max(1, min(_MAX_WORKERS, int((free - _GPU_RESERVE_GB) // _WORKER_GB), len(jobs)))
    print(
        f"  LatentSync: {free:.1f} GB GPU free → {n} worker(s) for {len(jobs)} shot(s)", flush=True
    )

    dur = [p[1]["end"] - p[1]["start"] for p in pending]
    buckets: list[list[dict]] = [[] for _ in range(n)]
    load = [0.0] * n
    for k in sorted(range(len(jobs)), key=lambda k: -dur[k]):
        w = load.index(min(load))
        buckets[w].append(jobs[k])
        load[w] += dur[k]

    procs = []
    for w, bucket in enumerate(buckets):
        jp = cache_dir / f"latentsync_jobs_{w}.json"
        rp = cache_dir / f"latentsync_results_{w}.json"
        jp.write_text(json.dumps(bucket), encoding="utf-8")
        rp.unlink(missing_ok=True)
        log = open(cache_dir / ("latentsync.log" if w == 0 else f"latentsync_{w}.log"), "w")
        procs.append(
            (
                subprocess.Popen(
                    [
                        str(_LS_PYTHON.absolute()),
                        str(_WORKER),
                        str(jp.resolve()),
                        str(rp.resolve()),
                    ],
                    cwd=str(LATENTSYNC_DIR),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                ),
                log,
                rp,
            )
        )
    results: dict = {}
    for proc, log, rp in procs:
        proc.wait()
        log.close()
        try:
            results.update(json.loads(rp.read_text(encoding="utf-8")))
        except Exception:
            pass

    # GPU memory can be taken by something else after the workers were sized
    # (e.g. the studio loading its models). Shots that failed for lack of
    # memory are retried once, one at a time, instead of silently keeping
    # their original frames.
    oom = [
        j
        for j in jobs
        if any(
            k in str(results.get(j["out"], "missing"))
            for k in ("Failed to al", "out of memory", "CUDA error", "missing")
        )
    ]
    if oom and wait_for_gpu(_WORKER_GB + _GPU_RESERVE_GB) >= _WORKER_GB + _GPU_RESERVE_GB:
        print(
            f"  LatentSync: retrying {len(oom)} shot(s) that ran out of GPU memory, one worker",
            flush=True,
        )
        jp, rp = (
            cache_dir / "latentsync_jobs_retry.json",
            cache_dir / "latentsync_results_retry.json",
        )
        jp.write_text(json.dumps(oom), encoding="utf-8")
        rp.unlink(missing_ok=True)
        with open(cache_dir / "latentsync_retry.log", "w") as log:
            subprocess.run(
                [str(_LS_PYTHON.absolute()), str(_WORKER), str(jp.resolve()), str(rp.resolve())],
                cwd=str(LATENTSYNC_DIR),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        try:
            results.update(json.loads(rp.read_text(encoding="utf-8")))
        except Exception:
            pass
    return results, n


def run_worker(args: list[str], log_path: Path) -> None:
    with open(log_path, "w") as log:
        subprocess.run(
            # absolute(), not resolve(): resolving follows the venv symlink to
            # the base interpreter, which has none of LatentSync's packages.
            [str(_LS_PYTHON.absolute()), str(_WORKER), *args],
            cwd=str(LATENTSYNC_DIR),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
