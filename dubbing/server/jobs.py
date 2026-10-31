"""
Pipeline jobs: one at a time, in the background, with live progress.

The GPU can only run one dub at a time, so jobs wait in a queue and a single
worker thread runs them in order. The web page follows a job through
/api/jobs/<id>/events, which sends its state whenever it changes.

When a job finishes, the dub's final state is saved (projects.save_project),
so it shows up in the library and can be edited later.
"""

import itertools
import queue
import threading
import time
from dataclasses import dataclass, field

from pipeline.cache import CACHE_ROOT
from pipeline.video.latentsync_runner import free_gpu_gb

from . import eta, versions
from .projects import save_project
from .runner import stream_run


@dataclass
class Job:
    id: int
    kind: str  # "dub" | "revoice" | "edit"
    project: str
    lang: str
    title: str
    stages: list  # [(stage key, label)]
    graph: object
    state: dict
    status: str = "queued"  # queued | running | done | error | stopped
    current: str | None = None
    timings: dict = field(default_factory=dict)
    log: list = field(default_factory=list)
    message: str = ""
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    before: list | None = None  # segments before an edit, to describe what changed
    work_s: float | None = None  # seconds of speech to redo; None = the whole video
    version: int = 0  # bumped on every change, so watchers know to send an update
    # Time estimate (server/eta.py), filled in when the job starts.
    video_s: float = 0.0
    modes: dict = field(default_factory=dict)
    estimates: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    stage_start: float = 0.0
    latest: dict = field(default_factory=dict)  # pipeline state after the last finished stage
    # Set by Stop: the run ends after the stage in progress (server/runner.py, stream_run).
    stop: threading.Event = field(default_factory=threading.Event)
    # What was asked, so a stopped job can be corrected and started again:
    # the edit plan, or the New dub form.
    plan: dict | None = None
    request: dict | None = None

    def public(self, position: int | None = None) -> dict:
        """What the web page sees (no pipeline internals)."""
        now = self.finished or time.time()
        return {
            "id": self.id,
            "kind": self.kind,
            "project": self.project,
            "lang": self.lang,
            "title": self.title,
            "status": self.status,
            "position": position,
            "stages": [{"key": k, "label": label} for k, label in self.stages],
            "current": self.current,
            "timings": self.timings,
            "elapsed": round(now - self.started, 1) if self.started else 0,
            "message": self.message,
            "log": self.log[-40:],
            "notes": self.notes,
            "estimates": self.estimates,
            "remaining": self.remaining(),
            "stopping": self.stop.is_set() and self.status == "running",
            "plan": self.plan,
            "request": self.request,
        }

    def remaining(self) -> float | None:
        """Seconds left, or None when there is nothing sensible to say."""
        if self.status not in ("queued", "running") or not self.estimates:
            return None
        in_stage = time.time() - self.stage_start if self.stage_start else 0
        return eta.remaining(self.estimates, self.timings, self.current, in_stage)

    def plan_estimates(self) -> None:
        """(Re)compute the per-stage estimate from what is known about the video."""
        self.video_s = _video_seconds(self.project, self.state) or self.video_s
        work = self.video_s if self.work_s is None else self.work_s
        keys = [k for k, _ in self.stages]
        self.estimates = eta.estimate(keys, self.video_s, work, self.modes)


class JobQueue:
    def __init__(self):
        self._jobs: dict[int, Job] = {}
        self._waiting: queue.Queue = queue.Queue()
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        threading.Thread(target=self._worker, daemon=True).start()

    # ── Used by the API ──────────────────────────────────────────────────────
    def submit(self, **kwargs) -> Job:
        job = Job(id=next(self._ids), **kwargs)
        with self._lock:
            self._jobs[job.id] = job
        self._waiting.put(job.id)
        return job

    def get(self, job_id: int) -> Job | None:
        return self._jobs.get(job_id)

    def position(self, job: Job) -> int | None:
        """1 = next to run; None once it has started."""
        if job.status != "queued":
            return None
        queued = sorted(j.id for j in self._jobs.values() if j.status == "queued")
        return queued.index(job.id) + 1

    def active(self) -> list[Job]:
        """Jobs still queued or running, oldest first."""
        return sorted(
            (j for j in self._jobs.values() if j.status in ("queued", "running")),
            key=lambda j: j.id,
        )

    def cancel(self, job: Job) -> None:
        """Stop a job: a waiting one at once, a running one after its current stage."""
        with self._lock:
            if job.status == "queued":
                job.status, job.finished = "stopped", time.time()
                job.message = "Stopped before it started — nothing was changed."
            elif job.status == "running":
                job.stop.set()
            job.version += 1

    def busy_with(self, project: str) -> bool:
        return any(j.project == project for j in self.active())

    # ── The worker ───────────────────────────────────────────────────────────
    def _worker(self):
        while True:
            job = self._jobs[self._waiting.get()]
            if job.status == "queued":  # not stopped while it waited
                self._run(job)

    def _run(self, job: Job):
        job.status, job.started = "running", time.time()
        job.current = job.stages[0][0]
        job.stage_start = job.started
        # How busy the GPU is decides how fast voice cloning and lip sync run.
        free = free_gpu_gb()
        keys = {k for k, _ in job.stages}
        job.modes = eta.modes(free)
        job.modes.update(_cached_modes(job.project, keys))
        if job.state.get("lipsync") is False:
            job.modes["lipsync"] = "off"  # fast mode: nothing is rendered
        job.notes = eta.notes(free, keys)
        job.plan_estimates()
        # Keep the dub as it is before changing it, so the change can be undone.
        versions.ensure_first(job.project, job.lang)
        job.version += 1
        print(f"\n=== Job {job.id}: {job.kind} {job.project} → {job.lang}", flush=True)
        for info in stream_run(job.graph, job.state, job.stages, job.stop):
            for stage, secs in info["timings"].items():
                if stage not in job.timings:  # a stage just finished
                    work = job.video_s if job.work_s is None else job.work_s
                    eta.record(stage, secs, job.video_s, work, job.modes, job.kind)
                    job.stage_start = time.time()
                    job.latest = info["final"]
                    if stage == "download":  # the video's real length is known now
                        job.state = info["final"]
                        job.plan_estimates()
            job.current = info["current"]
            job.timings = info["timings"]
            job.log = info["log"].splitlines()
            if info["kind"] == "stopped":
                stage = dict(job.stages).get(info["message"], info["message"])
                job.message = (
                    f"Stopped after “{stage}” — nothing was changed. Finished steps are kept, "
                    "so starting again is quicker."
                )
                job.status = "stopped"
            elif info["kind"] == "error":
                job.status, job.message = "error", _explain(info["message"])
            elif info["kind"] == "done":
                try:
                    save_project(job.project, job.lang, info["final"])
                    job.latest = info["final"]
                    # Message before status: watchers treat "done" as final.
                    job.message = _summary(job, info["final"])
                    versions.snapshot(
                        job.project,
                        job.lang,
                        _version_title(job),
                        job.kind,
                        job.message if job.kind == "edit" else "",
                    )
                    job.status = "done"
                except Exception as exc:
                    job.status, job.message = "error", f"Could not save the result: {exc}"
            job.version += 1
        if job.status == "running":  # the stream ended without a verdict
            job.status, job.message = "error", "The run stopped unexpectedly."
        job.finished = time.time()
        job.version += 1


def _video_seconds(project: str, state: dict) -> float:
    """Length of the video being dubbed, from whatever is known so far."""
    path = state.get("audio_path")
    if path:
        try:
            import soundfile as sf

            return sf.info(path).duration
        except Exception:
            pass
    a, b = state.get("clip_start"), state.get("clip_end")
    if b is not None:
        return b - (a or 0)
    # The whole video, before it is downloaded: its length from the link
    # check (or the upload), minus a start if only "From" was given.
    if state.get("source_duration"):
        return max(0.0, state["source_duration"] - (a or 0))
    segs = state.get("segments") or []
    if segs:
        return max(s["end"] for s in segs)
    try:
        import json

        meta = json.loads((CACHE_ROOT / project / "meta.json").read_text(encoding="utf-8"))
        return float(meta.get("duration") or 0)
    except Exception:
        return 0.0


def _cached_modes(project: str, stages: set) -> dict:
    """The first stages are near-instant when this video was processed before."""
    d = CACHE_ROOT / project
    done = {
        "download": (d / "video.mp4").exists(),
        "separate": any((d / "separated").glob("*_vocals.wav"))
        if (d / "separated").exists()
        else False,
        "transcribe": (d / "transcript.json").exists(),
    }
    return {s: "cached" for s, ok in done.items() if ok and s in stages}


def _version_title(job: Job) -> str:
    if job.kind != "dub":
        return job.title
    return "Dubbed again" if versions.list_versions(job.project, job.lang) else "First dub"


def _explain(error: str) -> str:
    """Plain words for errors a user can act on; anything else as it came."""
    if "out of memory" in error.lower():
        return (
            "The GPU is full — another program is using its memory. Try again when "
            "it is free (check with nvidia-smi). Finished steps are kept."
        )
    return error


def _summary(job: Job, final: dict) -> str:
    """One line for the finished job; for edits, what actually changed."""
    if job.kind == "edit" and job.before is not None:
        from .revise import render_diff

        return render_diff(job.before, final.get("segments") or [])
    took = time.time() - (job.started or time.time())
    return f"Finished in {took / 60:.1f} min"


JOBS = JobQueue()
