"""
Per-run LLM spending guard.

Every OpenRouter chat call made through client() is billed back into a
running total (OpenRouter returns the exact cost when usage is requested).
Optional steps ask allow() before calling; once a run has spent
RUN_BUDGET_USD they are skipped, so no single dub can run away with the
budget (one review prompt once produced 37K output tokens per call).
"""

import os
import threading

RUN_BUDGET_USD = float(os.environ.get("RUN_BUDGET_USD", "0.80"))
_lock = threading.Lock()
_spent = 0.0
_calls = 0


def reset() -> None:
    global _spent, _calls
    with _lock:
        _spent, _calls = 0.0, 0


def spent() -> float:
    return _spent


def calls() -> int:
    return _calls


def allow(reserve: float = 0.05) -> bool:
    """True if there is budget left for another call of ~`reserve` dollars."""
    return _spent + reserve <= RUN_BUDGET_USD


def client():
    """OpenRouter client whose chat calls record their exact cost."""
    from openai import OpenAI
    c = OpenAI(base_url="https://openrouter.ai/api/v1",
               api_key=os.environ["OPENROUTER_API_KEY"])
    create = c.chat.completions.create

    def tracked(**kw):
        global _spent, _calls
        kw.setdefault("extra_body", {}).setdefault("usage", {"include": True})
        r = create(**kw)
        cost = getattr(getattr(r, "usage", None), "cost", None) or 0.0
        with _lock:
            _spent += float(cost)
            _calls += 1
        return r

    c.chat.completions.create = tracked
    return c
