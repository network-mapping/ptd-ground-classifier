"""Checks of the neighbour halo caches (halo.py) and helper-process health (processes.py). usage: python tests/test_halo.py

The GOLDEN values were recorded from the first implementation (the same functions in pipeline.py, before they moved) on
exactly the tile `make_source` writes, so they pin the cache contents, the density grid and the points loaded for a
target across the move."""
import hashlib
import io
import shutil
import sys
import tempfile
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
from contextlib import redirect_stdout
from pathlib import Path

import laspy
import numpy as np

from ground_classifier import halo, processes

fails = 0
def ok(name, cond, note=""):
    global fails
    print(("ok   " if cond else "FAIL ") + name + (" - " + note if note else ""), flush=True)
    fails += 0 if cond else 1


def make_source(path, n=6000, seed=11, extent=(0.0, 0.0, 400.0, 300.0)):
    rng = np.random.default_rng(seed)
    xmin, ymin, xmax, ymax = extent
    header = laspy.LasHeader(point_format=3, version="1.4")
    header.scales = (0.001, 0.001, 0.001)
    header.offsets = (xmin, ymin, 0.0)
    las = laspy.LasData(header)
    las.x = rng.uniform(xmin, xmax, n)
    las.y = rng.uniform(ymin, ymax, n)
    las.z = rng.uniform(0.0, 40.0, n)
    las.classification = rng.choice(np.array([1, 2, 3, 5, 6], np.uint8), n)
    path.parent.mkdir(parents=True, exist_ok=True)
    las.write(path)
    return path


def digest(*arrays):
    h = hashlib.sha1()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:12]


BOUNDS = {
    "A": ((50.0, 50.0, 150.0, 120.0),),
    "B": ((50.0, 50.0, 150.0, 120.0), (300.0, 200.0, 500.0, 400.0)),
    "C": ((1000.0, 1000.0, 1100.0, 1100.0),),
}
RECTS = [(0.0, 0.0, 400.0, 300.0), (10.0, 10.0, 20.0, 20.0), (-50.0, -50.0, 5.0, 5.0), (390.0, 290.0, 900.0, 900.0), (500.0, 500.0, 600.0, 600.0),
         (100.0, 100.0, 100.0, 100.0), (-10.0, 120.0, 410.0, 130.0), (199.9, 0.0, 200.1, 300.0)]

GOLDEN_CACHE = {"A": (365, "e2981c0e3f4f"), "B": (851, "6b59af4e23f2"), "C": (0, "da39a3ee5e6b")}
GOLDEN_DENSITY = ((0.076, 0.025, 399.902, 299.986), 64, 6000, "9f0740ea2810")
GOLDEN_RECTS = [6000, 14, 3, 10, 0, 3, 260, 174]
GOLDEN_EMPTY = (0, "da39a3ee5e6b")
GOLDEN_LOAD = {  # (halo, caches) -> (points, digest)
    (0.0, "A"): (100, "e96592e6e9a8"), (0.0, "AB"): (200, "5a30e5864afb"), (0.0, "C"): GOLDEN_EMPTY, (0.0, ""): GOLDEN_EMPTY,
    (10.0, "A"): (154, "8d3a036ac3ea"), (10.0, "AB"): (308, "d52c7295020e"), (10.0, "C"): GOLDEN_EMPTY, (10.0, ""): GOLDEN_EMPTY,
    (40.0, "A"): (337, "e6216a0e9352"), (40.0, "AB"): (674, "303974ba5b71"), (40.0, "C"): GOLDEN_EMPTY, (40.0, ""): GOLDEN_EMPTY,
}


def quiet(call):
    text = io.StringIO()
    with redirect_stdout(text):
        value = call()
    return value, text.getvalue()


def raised(exc, call):
    try:
        call()
    except exc:
        return True
    except Exception:
        return False
    return False


class BrokenPool:
    """A process pool whose helpers all die: every job fails with BrokenProcessPool."""

    def __init__(self, max_workers=None):
        self._processes = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def submit(self, fn, *args):
        future = Future()
        future.set_exception(BrokenProcessPool("a process in the pool was terminated abruptly"))
        return future

    def shutdown(self, wait=True):
        pass


class FakeProcess:
    def __init__(self, exitcode, fail_join=False):
        self.exitcode, self.fail_join = exitcode, fail_join

    def join(self, timeout=None):
        if self.fail_join:
            raise OSError("no such process")


def main():
    tmp = Path(tempfile.mkdtemp(prefix="halo_"))
    try:
        source = make_source(tmp / "s.las")
        infos = {}
        for name, bounds in BOUNDS.items():
            info = halo.create_halo_cache_worker(str(source), bounds, str(tmp / f"{name}.bin"))
            infos[name] = info
        ok("cache: the points inside the requested rectangles, as the first implementation wrote them (A, B, none)",
           {k: (i.point_count, digest(np.fromfile(i.path, dtype=np.uint8))) for k, i in infos.items()} == GOLDEN_CACHE,
           str({k: (i.point_count, digest(np.fromfile(i.path, dtype=np.uint8))) for k, i in infos.items()}))
        density = infos["A"].density
        ok("density: the grid of the whole file (not just the requested part) is as the first implementation made it",
           (density.bounds, density.cells, sum(density.counts), digest(np.array(density.counts))) == GOLDEN_DENSITY)
        ok("density: the same grid whatever is requested", all(infos[k].density == density for k in "BC"))
        ok("density: the upper bound in a rectangle is as the first implementation gave", [halo.density_points_in_rect(density, r) for r in RECTS] == GOLDEN_RECTS)

        las = laspy.read(source)
        x, y = np.asarray(las.x), np.asarray(las.y)
        rng = np.random.default_rng(5)
        never_low = True
        for _ in range(200):
            x0, y0 = rng.uniform(-20, 420), rng.uniform(-20, 320)
            rect = (x0, y0, x0 + rng.uniform(0, 200), y0 + rng.uniform(0, 200))
            true = int(((x >= rect[0]) & (x <= rect[2]) & (y >= rect[1]) & (y <= rect[3])).sum())
            never_low &= halo.density_points_in_rect(density, rect) >= true
        ok("density: over any rectangle the bound is never below the true count (200 random rectangles)", never_low)

        records = np.fromfile(infos["B"].path, dtype=halo.HALO_CACHE_DTYPE)
        inside = np.zeros(x.size, bool)
        for xmin, ymin, xmax, ymax in BOUNDS["B"]:
            inside |= (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
        ok("cache: exactly the points inside the rectangles, in file order, with x, y, z and class", records.size == int(inside.sum())
           and np.array_equal(records["x"], x[inside]) and np.array_equal(records["y"], y[inside])
           and np.array_equal(records["z"], np.asarray(las.z)[inside]) and np.array_equal(records["classification"], np.asarray(las.classification)[inside]))

        got = {}
        for halo_distance in (0.0, 10.0, 40.0):
            for names in ("A", "AB", "C", ""):
                out = halo.load_halo_points(tuple(infos[n] for n in names), (100.0, 80.0, 200.0, 160.0), halo_distance)
                got[(halo_distance, names)] = (out[0].size, digest(*out))
        ok("load: the cached points within the halo of a target, for halos of 0, 10 and 40 and several caches, as before", got == GOLDEN_LOAD)
        out = halo.load_halo_points((infos["A"],), (100.0, 80.0, 200.0, 160.0), 10.0)
        ok("load: x, y, z are float64 and the class is uint8, also when nothing is found",
           [a.dtype.str for a in out] == ["<f8", "<f8", "<f8", "|u1"] and [a.dtype.str for a in halo.load_halo_points((), (0, 0, 1, 1), 5.0)] == ["<f8", "<f8", "<f8", "|u1"])
        ok("load: every point lies within the halo of the target", bool(((out[0] >= 90) & (out[0] <= 210) & (out[1] >= 70) & (out[1] <= 170)).all()))
        ok("expand_bounds: grown by the halo on every side", halo.expand_bounds((1.0, 2.0, 3.0, 4.0), 0.5) == (0.5, 1.5, 3.5, 4.5) and halo.expand_bounds((1.0, 2.0, 3.0, 4.0), 0.0) == (1.0, 2.0, 3.0, 4.0))

        # building the caches of several sources
        sources = {}
        for i, seed in enumerate((11, 12, 13)):
            sources[make_source(tmp / f"src{i}.las", n=3000, seed=seed)] = {BOUNDS["A"][0], (200.0, 100.0, 330.0, 280.0)}
        one, text = quiet(lambda: halo.build_halo_caches(sources, tmp / "one", 1))
        two, _ = quiet(lambda: halo.build_halo_caches(sources, tmp / "two", 2))
        view = lambda caches: {p.name: (i.point_count, digest(np.fromfile(i.path, dtype=np.uint8)), i.density) for p, i in caches.items()}
        ok("build: a cache for each source, named in path order", sorted(i.path.name for i in one.values()) == ["halo_000000.bin", "halo_000001.bin", "halo_000002.bin"]
           and [one[p].path.name for p in sorted(sources, key=str)] == ["halo_000000.bin", "halo_000001.bin", "halo_000002.bin"])
        ok("build: it says how many sources and workers", "from 3 source file(s) with 1 worker(s)" in text)
        ok("build: two helper processes make caches identical to one worker's, density grids included", view(one) == view(two))
        ok("build: nothing requested, nothing built", halo.build_halo_caches({}, tmp / "none", 3) == {} and not (tmp / "none").exists())
        missing = {tmp / "absent.las": {BOUNDS["A"][0]}}
        ok("build: a file that cannot be read is an error naming it (one worker)", raised(RuntimeError, lambda: quiet(lambda: halo.build_halo_caches(missing, tmp / "m1", 1))))
        try:
            quiet(lambda: halo.build_halo_caches({**sources, tmp / "absent.las": {BOUNDS["A"][0]}}, tmp / "m2", 2))
            message = ""
        except RuntimeError as exc:
            message = str(exc)
        ok("build: ... and with helper processes", "absent.las" in message and "Could not read neighbouring tile" in message, message[:120])

        # a helper that dies: the unfinished files are read in this process
        real_pool, real_works = halo.ProcessPoolExecutor, halo.helper_processes_work
        try:
            halo.ProcessPoolExecutor = BrokenPool
            halo.helper_processes_work = lambda: True
            healthy, text = quiet(lambda: halo.build_halo_caches(sources, tmp / "dead_a", 2))
            ok("recovery: helpers die but start again: the files are read in this process, with the same caches", view(healthy) == view(one) and processes.helpers_usable()
               and "a helper process died while reading neighbouring tiles; 3 of 3" in text and "a fresh helper process starts and runs normally" in text and text.count("Reading neighbour context") == 3)
            ok("recovery: ... the retry caches are named apart from the dead pool's", all(i.path.name.endswith("_retry.bin") for i in healthy.values()))
            halo.helper_processes_work = lambda: False
            broken, text = quiet(lambda: halo.build_halo_caches(sources, tmp / "dead_b", 2))
            ok("recovery: helpers cannot run at all: the caches are read on threads, identical, and helpers are switched off for the run",
               view(broken) == view(one) and not processes.helpers_usable() and "cannot run even a trivial task" in text)
        finally:
            halo.ProcessPoolExecutor, halo.helper_processes_work = real_pool, real_works
            processes.enable_helpers()
        ok("processes: the verdict can be reset", processes.helpers_usable())

        # processes.py
        ok("exit codes: small codes are plain (or explained), signals say so", processes.describe_exit_code(42) == "42" and processes.describe_exit_code(3) == "3 (abort() called inside a compiled library)"
           and processes.describe_exit_code(-9) == "killed by signal 9")
        ok("exit codes: Windows codes are in hex with an explanation",
           processes.describe_exit_code(0xC0000005) == "0xC0000005 (access violation: a crash inside a compiled library)"
           and processes.describe_exit_code(0xC0000005 - 2**32).startswith("killed by signal"))
        ok("exit codes: an unknown large code is just hex", processes.describe_exit_code(0x12345678) == "0x12345678")
        codes = processes.dead_helper_exit_codes([FakeProcess(None), FakeProcess(0), FakeProcess(-15), FakeProcess(-11), FakeProcess(3), FakeProcess(7, fail_join=True)])
        ok("dead helpers: running, clean and pool-stopped helpers are not reported; a signal and a failure code are; a helper that cannot be joined is skipped",
           codes == ["killed by signal 11", "3 (abort() called inside a compiled library)"], str(codes))
        ok("pool_processes: the pool's processes, or nothing", processes.pool_processes(BrokenPool()) == [] and processes.pool_processes(object()) == [])
        ok("probe: the probe returns the helper's process id", processes.helper_process_probe() > 0)
        ok("health: a fresh helper process starts and answers", processes.helper_processes_work() is True)

        class NoHelpers:
            def __init__(self, max_workers=None):
                pass
            def submit(self, fn):
                raise OSError("cannot start a process")
            def shutdown(self, wait=True):
                pass
        real_pool = processes.ProcessPoolExecutor
        processes.ProcessPoolExecutor = NoHelpers
        try:
            ok("health: a machine that cannot start helpers is reported as such", processes.helper_processes_work() is False)
        finally:
            processes.ProcessPoolExecutor = real_pool
        processes.disable_helpers()
        off = not processes.helpers_usable()
        processes.enable_helpers()
        ok("health: disable and enable switch the flag", off and processes.helpers_usable())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("all passed" if not fails else f"{fails} failed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
