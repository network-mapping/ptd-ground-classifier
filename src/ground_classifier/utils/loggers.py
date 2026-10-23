"""Process-wide logging: atomic prints, or a per-window capture buffer."""
from __future__ import annotations

import threading

# Per-thread: `buffer` captures a window's messages, `prefix` tags output lines.
_LOG_STATE = threading.local()
_PRINT_LOCK = threading.Lock()


def log(message: str) -> None:
    """Print atomically, or capture a complete window log in its worker."""
    buffer = getattr(_LOG_STATE, "buffer", None)
    if buffer is not None:
        buffer.append(message)
        return
    prefix = getattr(_LOG_STATE, "prefix", "")
    with _PRINT_LOCK:
        print(f"{prefix}{message}", flush=True)
