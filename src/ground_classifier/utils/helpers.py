import os
from pathlib import Path


def normalize_path(path_value):
    return Path(path_value).expanduser().resolve()


def format_memory(byte_count: int) -> str:
    return f"{byte_count / 1024**3:.1f} GB"


def env_default(*names, fallback=None):
    """First environment variable in `names` that is set and not empty, else `fallback`."""
    for name in names:
        value = os.environ.get(name)
        if value:
            return value

    return fallback
