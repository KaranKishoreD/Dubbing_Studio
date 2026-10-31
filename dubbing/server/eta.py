"""
Time estimates for jobs, learned from past runs.

Every finished stage is appended to cache/_history.jsonl with what drove its
time. A stage's time is modelled as

    seconds = fixed + rate × amount

where `amount` is the length of the whole video for stages that always
process all of it (download, separation, transcription, speakers, mixing),
and the speech that has to be redone for the others (translation, voices,
timing, lip sync). That is why an edit of one line is estimated in seconds
while a new dub of the same video takes minutes.

`rate` comes from the median of past runs of the same stage in the same
mode, and from rough starting values (_PRIORS) until there is history. The
mode captures GPU contention: with too little free GPU memory, voice cloning
runs on the CPU and lip sync waits for memory, then skips.
"""

import json
import statistics
import time

from pipeline.cache import CACHE_ROOT

HISTORY = CACHE_ROOT / "_history.jsonl"

# Stages whose time depends on the whole video; the rest scale with the
# speech that has to be redone.
WHOLE_VIDEO = {"download", "separate", "transcribe", "diarize", "refine", "assemble"}

# Rough starting values, (fixed seconds, seconds per second of video/speech),
# by stage and mode. Replaced by measurements as runs are recorded.
_PRIORS = {
    ("lipsync", "off"): (5, 0.0),  # fast mode: finished shots are only reused
    ("download", "-"): (5, 0.05),
    ("download", "cached"): (2, 0.0),
    ("separate", "cached"): (3, 0.0),
    ("transcribe", "cached"): (8, 0.0),
    ("separate", "-"): (10, 0.25),
    ("transcribe", "-"): (25, 0.35),
    ("diarize", "-"): (30, 0.5),
    ("refine", "-"): (10, 0.15),
    ("translate", "-"): (10, 0.4),
    ("synthesize", "-"): (5, 0.5),
    ("align", "gpu"): (20, 0.4),
    ("align", "cpu"): (20, 6.0),
    ("lipsync", "gpu"): (60, 6.0),
    ("lipsync", "wait"): (610, 0.0),
    ("assemble", "-"): (10, 0.25),
}

# Free GPU memory below which voice cloning moves to the CPU and lip sync
# has to wait (voice/voice_clone.py, video/latentsync_runner.py).
_CLONE_GPU_GB = 4.0
_LIPSYNC_GPU_GB = 10.0
_LIPSYNC_WAIT_MIN = 10


def modes(free_gpu_gb: float) -> dict[str, str]:
    return {
        "align": "gpu" if free_gpu_gb >= _CLONE_GPU_GB else "cpu",
        "lipsync": "gpu" if free_gpu_gb >= _LIPSYNC_GPU_GB else "wait",
    }


def notes(free_gpu_gb: float, stages: set) -> list[str]:
    """Plain-words warnings about a busy GPU, for the stages this job runs."""
    m = modes(free_gpu_gb)
    out = []
    if "align" in stages and m["align"] == "cpu":
        out.append(
            f"The GPU is busy ({free_gpu_gb:.0f} GB free), so matching each line to "
            "the speaker's voice runs on the CPU — the “Fit timing” step will be slow."
        )
    if "lipsync" in stages and m["lipsync"] == "wait":
        out.append(
            f"Lip sync needs about {_LIPSYNC_GPU_GB:.0f} GB of GPU memory. It waits up to "
            f"{_LIPSYNC_WAIT_MIN} min for it, then keeps the original frames."
        )
    return out


def _history() -> list[dict]:
    try:
        lines = HISTORY.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    rows = []
    for line in lines[-2000:]:
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def _mode(stage: str, job_modes: dict) -> str:
    return job_modes.get(stage, "-")


def _amount(stage: str, video_s: float, work_s: float) -> float:
    return video_s if stage in WHOLE_VIDEO else work_s


def estimate(stages: list[str], video_s: float, work_s: float, job_modes: dict) -> dict[str, float]:
    """Expected seconds per stage."""
    rows = _history()
    out = {}
    for stage in stages:
        mode = _mode(stage, job_modes)
        fixed, rate = _PRIORS.get((stage, mode), _PRIORS.get((stage, "-"), (10, 0.3)))
        past = [
            r
            for r in rows
            if r["stage"] == stage
            and r.get("mode", "-") == mode
            and r["amount"] > 0
            # a stage that had nothing to do (an edit with no new lines to
            # lip-sync) says nothing about its speed
            and (stage in WHOLE_VIDEO or r["seconds"] >= 2)
        ]
        if past:
            measured = statistics.median(
                max(0.0, (r["seconds"] - fixed) / r["amount"]) for r in past[-30:]
            )
            # Few runs: lean on the starting value too; many: trust the data.
            weight = min(1.0, len(past) / 3)
            rate = weight * measured + (1 - weight) * rate
        out[stage] = round(fixed + rate * _amount(stage, video_s, work_s), 1)
    return out


def record(
    stage: str, seconds: float, video_s: float, work_s: float, job_modes: dict, kind: str
) -> None:
    """Append one finished stage to the history."""
    row = {
        "stage": stage,
        "seconds": round(seconds, 1),
        "amount": round(_amount(stage, video_s, work_s), 1),
        "mode": _mode(stage, job_modes),
        "kind": kind,
        "at": int(time.time()),
    }
    try:
        with open(HISTORY, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
    except OSError:
        pass  # estimates are a nicety; never fail a job over them


def remaining(estimates: dict, timings: dict, current: str | None, in_stage_s: float) -> float:
    """
    Seconds left. Stages already done tell how fast this machine is running
    today: if they took longer than estimated, the rest is scaled the same way.
    """
    done = [s for s in timings if s in estimates and estimates[s] > 5]
    drift = 1.0
    if done:
        drift = sum(timings[s] for s in done) / sum(estimates[s] for s in done)
        drift = min(3.0, max(0.5, drift))
    left = 0.0
    for stage, est in estimates.items():
        if stage in timings:
            continue
        expected = est * drift
        if stage == current:
            # Already running longer than expected: assume it is nearly done
            # rather than showing a negative or frozen time.
            left += max(expected - in_stage_s, 0.15 * expected)
        else:
            left += expected
    return round(left)
