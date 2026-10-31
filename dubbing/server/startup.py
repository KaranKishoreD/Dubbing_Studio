"""
Start-up for the studio. Imported by server/app.py before the pipeline,
because pipeline modules read their settings from the environment when they
are imported.

  1. Put the dubbing/ folder on the import path.
  2. Load API keys and settings from the project's .env file.
  3. Stop with a clear message if a required key is missing.
  4. Copy everything printed (and every traceback) to logs/ui_debug.log,
     so a run can be looked into afterwards.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load .env from project root (one level above dubbing/)
ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# Fail fast with a clear message if keys are missing
for _key in ("BODHAN_TRANSLATE_KEY", "BODHAN_SPEAK_KEY", "OPENROUTER_API_KEY"):
    if not os.environ.get(_key):
        raise EnvironmentError(f"{_key} not set. Add it to your .env file at: {ENV_PATH}")

# Everything the pipeline prints (and every traceback) also goes to a log
# file, however the studio was launched.
LOG_FILE = Path(__file__).resolve().parent.parent.parent / "logs" / "ui_debug.log"
LOG_FILE.parent.mkdir(exist_ok=True)


class _Tee:
    def __init__(self, stream, fh):
        self.stream, self.fh = stream, fh

    def write(self, data):
        self.stream.write(data)
        self.fh.write(data)
        self.fh.flush()
        return len(data)

    def flush(self):
        self.stream.flush()
        self.fh.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


_log_fh = open(LOG_FILE, "a", encoding="utf-8", buffering=1)
sys.stdout = _Tee(sys.stdout, _log_fh)
sys.stderr = _Tee(sys.stderr, _log_fh)
