"""Reading the .env file the setup instructions tell people to create.

``config.Settings`` reads ``os.environ``, and nothing was putting the file
there, so a filled-in .env had no effect and the run silently used defaults
and an empty API key.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_ENV_FILE = Path(".env")


def load_env_file(path: str | Path = DEFAULT_ENV_FILE) -> bool:
    """Load KEY=VALUE lines into the environment. Returns whether a file was read.

    An already-exported variable wins: choosing it in the shell is the more
    deliberate of the two, and it keeps a stale .env from overriding a
    one-off run.
    """
    target = Path(path)
    if not target.exists():
        return False

    for number, raw in enumerate(target.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise SystemExit(f"{target}:{number}: KEY=VALUE の形式ではありません: {line}")
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
    return True
