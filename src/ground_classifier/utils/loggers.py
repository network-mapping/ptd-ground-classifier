"""Process-wide logging: atomic prints, or a per-window capture buffer."""
from __future__ import annotations

import logging
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


class _LogBridge(logging.Handler):
    """Forwards standard `logging` records to `log()`, so they read like the rest of the output."""

    def emit(self, record: logging.LogRecord) -> None:
        log(f"{record.levelname.capitalize()}: {record.getMessage()}")


def install_logging_bridge(name: str = "ground_classifier.las") -> None:
    """Send the standard-library log records of package `name` (the `las` package) to `log()`. Safe to repeat."""
    logger = logging.getLogger(name)
    if any(isinstance(handler, _LogBridge) for handler in logger.handlers):
        return
    logger.addHandler(_LogBridge(level=logging.WARNING))
    logger.propagate = False
