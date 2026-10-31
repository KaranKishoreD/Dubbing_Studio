"""
One gate per Bodhan endpoint (indic-speak, indic-translate,
indic-transcribe), shared by every thread in the process.

Bodhan's limits are per key: a requests-per-minute allowance and a ceiling
on concurrent requests. Every response carries both, in the headers
x-ratelimit-limit-requests and x-ratelimit-limit-parallel-requests, and
Bodhan asks clients to read them there rather than hard-code them
(https://api.bodhan.ai, "Limits"). Over the allowance a call gets HTTP 429
with a retry-after header, and retries count against the limit.

So the gate:
  - starts from a cautious guess (below), then adopts the limits the
    responses report, and saves them in cache/_bodhan_limits.json so the
    next run starts from them;
  - lets at most `parallel` calls run at once, and at most
    requests-per-minute calls start in any 60 s (a sliding window, so a short
    job goes out at once and a long one never goes over);
  - on a 429, pauses the whole endpoint for retry-after (else a growing
    back-off, 2…30 s) and retries; other errors are raised at once;
  - after `trip` failures in a row, stops calling for a minute and fails
    fast with a clear message, instead of piling up failures.

BODHAN_<NAME>_CONCURRENT / _RPM override everything, e.g.
BODHAN_SPEAK_CONCURRENT=1. Clients must be made with bodhan_client(), which
turns the OpenAI library's own retries off and reports response headers
to the gate.
"""

import json
import os
import random
import threading
import time
from pathlib import Path

import httpx
from openai import OpenAI

_BACKOFF = [2, 4, 8, 16, 30]
_SAVED = Path("cache") / "_bodhan_limits.json"


class BodhanBusy(RuntimeError):
    """Bodhan kept refusing: the gate stopped calling it for a while."""


def _saved() -> dict:
    try:
        return json.loads(_SAVED.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


class Gate:
    def __init__(self, name: str, parallel: int, rpm: int, trip: int = 8):
        env = name.upper()
        self.name = name
        self.fixed = f"BODHAN_{env}_CONCURRENT" in os.environ or f"BODHAN_{env}_RPM" in os.environ
        known = _saved().get(name, {})
        self.parallel = int(
            os.environ.get(f"BODHAN_{env}_CONCURRENT", known.get("parallel", parallel))
        )
        self.rpm = int(os.environ.get(f"BODHAN_{env}_RPM", known.get("rpm", rpm)))
        self.trip = trip
        self.cond = threading.Condition()
        self.inflight = 0
        self.starts: list[float] = []  # call starts in the last 60 s (sliding window)
        self.next_start = 0.0  # no call may start before this (spacing, 429 pause)
        self.fails = 0  # failures in a row
        self.closed_until = 0.0
        self.stats = {"calls": 0, "rate_limited": 0, "failed": 0}

    # ── limits reported by Bodhan ──────────────────────────────────────────
    def observe(self, headers) -> None:
        """Adopt the limits a response reports (unless overridden by env)."""
        if self.fixed:
            return
        # The docs name x-ratelimit-limit-parallel-requests / -requests; the
        # translate endpoint actually sends x-ratelimit-api_key-limit-
        # max_parallel_requests / -requests. Speak and transcribe send none.
        try:
            par = int(
                headers.get("x-ratelimit-limit-parallel-requests")
                or headers.get("x-ratelimit-api_key-limit-max_parallel_requests")
                or 0
            )
            rpm = int(
                headers.get("x-ratelimit-limit-requests")
                or headers.get("x-ratelimit-api_key-limit-requests")
                or 0
            )
        except ValueError:
            return
        if not par and not rpm:
            return
        with self.cond:
            changed = (par and par != self.parallel) or (rpm and rpm != self.rpm)
            self.parallel = par or self.parallel
            self.rpm = rpm or self.rpm
            self.cond.notify_all()
        if changed:
            print(
                f"  ✓ Bodhan {self.name}: key allows {self.parallel} at once, {self.rpm} per minute"
            )
            data = _saved()
            data[self.name] = {"parallel": self.parallel, "rpm": self.rpm, "at": time.time()}
            try:
                _SAVED.parent.mkdir(parents=True, exist_ok=True)
                _SAVED.write_text(json.dumps(data, indent=1), encoding="utf-8")
            except OSError:
                pass

    # ── admission ──────────────────────────────────────────────────────────
    def _enter(self) -> None:
        with self.cond:
            while True:
                now = time.monotonic()
                if now < self.closed_until:
                    raise BodhanBusy(
                        f"Bodhan {self.name} keeps refusing requests; paused for "
                        f"{self.closed_until - now:.0f}s more"
                    )
                self.starts = [t for t in self.starts if now - t < 60.0]
                room = self.inflight < max(1, self.parallel) and len(self.starts) < max(1, self.rpm)
                if room and now >= self.next_start:
                    self.inflight += 1
                    self.starts.append(now)
                    return
                # wait for a 429 pause to end, the oldest start to leave the
                # window, or a call to finish
                waits = [self.next_start - now] if now < self.next_start else []
                if len(self.starts) >= self.rpm:
                    waits.append(60.0 - (now - self.starts[0]) + 0.05)
                self.cond.wait(timeout=max(0.05, min(waits)) if waits else 1.0)

    def _leave(self) -> None:
        with self.cond:
            self.inflight -= 1
            self.cond.notify_all()

    def _pause(self, seconds: float) -> None:
        with self.cond:
            self.next_start = max(self.next_start, time.monotonic() + seconds)

    def _failed(self) -> None:
        with self.cond:
            self.fails += 1
            self.stats["failed"] += 1
            if self.fails >= self.trip:
                self.closed_until = time.monotonic() + 60
                self.fails = 0
                print(f"  ⚠ Bodhan {self.name}: {self.trip} failures in a row — pausing 60 s")

    def call(self, fn, *args, **kwargs):
        """fn(*args, **kwargs) under the limits, retrying 429s only."""
        for attempt in range(len(_BACKOFF) + 1):
            self._enter()
            try:
                out = fn(*args, **kwargs)
            except Exception as exc:
                if not _is_429(exc) or attempt == len(_BACKOFF):
                    self._failed()
                    raise
                self.stats["rate_limited"] += 1
                wait = _retry_after(exc) or _BACKOFF[attempt] * random.uniform(0.8, 1.25)
                print(f"  … Bodhan {self.name} rate limit — all calls wait {wait:.0f}s")
                self._pause(wait)
                continue
            finally:
                self._leave()
            with self.cond:
                self.fails = 0
                self.stats["calls"] += 1
            return out


def _is_429(exc: Exception) -> bool:
    return getattr(exc, "status_code", None) == 429 or "429" in str(exc)


def _retry_after(exc: Exception) -> float | None:
    resp = getattr(exc, "response", None)
    try:
        v = float(resp.headers.get("retry-after"))
        return min(max(v, 1.0), 120.0)
    except Exception:
        return None


# The keys' limits, checked on 2026-10-31. Translate reports them in its
# headers (10 at once, 120 per minute). Speak and transcribe report nothing,
# so they were measured with tiny requests: speak refused a 4th call at once
# and a 9th within a minute; transcribe took 8 at once and refused the 17th
# within a minute.
SPEAK = Gate("speak", parallel=3, rpm=8)
TRANSLATE = Gate("translate", parallel=10, rpm=120)
TRANSCRIBE = Gate("transcribe", parallel=8, rpm=16)
_GATE_OF_KEY = {
    "BODHAN_SPEAK_KEY": SPEAK,
    "BODHAN_TRANSLATE_KEY": TRANSLATE,
    "BODHAN_TRANSCRIBE_KEY": TRANSCRIBE,
}


def bodhan_client(key_env: str, timeout: float = 120) -> OpenAI:
    """An OpenAI-style client for Bodhan. The library's own retries are off
    (the gates retry, and every retry counts against the key's limit), and
    every response's rate-limit headers are passed to the key's gate."""
    gate = _GATE_OF_KEY.get(key_env)
    hooks = {"response": [lambda r: gate.observe(r.headers)]} if gate else {}
    return OpenAI(
        base_url="https://api.bodhan.ai/v1",
        api_key=os.environ[key_env],
        max_retries=0,
        timeout=timeout,
        http_client=httpx.Client(timeout=timeout, event_hooks=hooks),
    )
