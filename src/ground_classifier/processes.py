"""Helper-process health: why a worker process died, and whether this machine can run worker processes at all.

Some machines (security software, job or policy limits, a broken Python installation) stop Python helper processes. The
scheduler uses this module to say why a helper died and to carry on without helpers when none can run. The verdict is a
per-process flag: `helpers_usable()` is True until `disable_helpers()` is called.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

# Set False for the rest of the run when this machine turns out to be unable to keep helper processes alive (see
# helper_processes_work()). The file-classification stage then stays in the main process as well.
_HELPERS_USABLE = True


def helpers_usable() -> bool:
    return _HELPERS_USABLE


def disable_helpers() -> None:
    global _HELPERS_USABLE
    _HELPERS_USABLE = False


def enable_helpers() -> None:
    """Forget an earlier verdict (a new run, or a test)."""
    global _HELPERS_USABLE
    _HELPERS_USABLE = True


# Windows process exit codes that explain an abrupt helper death.
_HELPER_EXIT_HINTS = {
    0x00000001: "generic failure, or ended by another program",
    0x00000003: "abort() called inside a compiled library",
    0xC0000005: "access violation: a crash inside a compiled library",
    0xC0000017: "not enough memory to start the process",
    0xC00000FD: "stack overflow",
    0xC000012D: "commit limit reached: paging file too small",
    0xC0000135: "a required DLL was not found",
    0xC000013A: "interrupted: Ctrl+C or the console was closed",
    0xC0000142: "a DLL failed to initialise: the process could not start",
    0xC0000374: "heap corruption inside a compiled library",
    0xC0000409: "fail-fast abort inside a compiled library",
    0xC0000417: "invalid C runtime parameter inside a compiled library",
}


def describe_exit_code(code: int) -> str:
    """Human-readable form of a helper process exit code."""
    if code < 0:
        return f"killed by signal {-code}"
    unsigned = code & 0xFFFFFFFF
    text = f"0x{unsigned:08X}" if unsigned > 255 else str(unsigned)
    hint = _HELPER_EXIT_HINTS.get(unsigned)
    return f"{text} ({hint})" if hint else text


def pool_processes(executor: ProcessPoolExecutor) -> list:
    """Best-effort handles to a pool's helper processes (diagnostics only)."""
    try:
        return list((getattr(executor, "_processes", None) or {}).values())
    except Exception:
        return []


def dead_helper_exit_codes(processes: list) -> list[str]:
    """Exit codes of helpers that died on their own, never raising."""
    described: list[str] = []
    for process in processes:
        try:
            process.join(timeout=10)
            code = process.exitcode
        except Exception:
            continue
        # None: still running. 0: clean exit. -15: stopped by the pool itself
        # while it shut down after the first helper died.
        if code is None or code == 0 or code == -15:
            continue
        described.append(describe_exit_code(int(code)))
    return described


def helper_process_probe() -> int:
    """Trivial task: can a helper process start and answer at all?"""
    return os.getpid()


def helper_processes_work() -> bool:
    """True when a brand-new helper process starts and runs a trivial task.

    This separates the two families of BrokenProcessPool causes: if even
    this fails, no input file is involved and the machine itself is stopping
    Python helper processes; if it passes, the death is tied to the work the
    helper was doing.
    """
    executor = ProcessPoolExecutor(max_workers=1)
    working = False
    try:
        working = executor.submit(helper_process_probe).result(timeout=300) > 0
    except KeyboardInterrupt:
        raise
    except BaseException:
        working = False
    finally:
        executor.shutdown(wait=working)
    return working
