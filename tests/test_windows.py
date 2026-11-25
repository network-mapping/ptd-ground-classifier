"""Checks of the window layout, the buffer and core of a window, the per-file exports and the window runner.
The layouts in GOLDEN were recorded from the first implementation (pipeline.window_cores_for_extent, before it moved to
windows.py), as (window count, sha1 of repr of the cores). usage: python tests/test_windows.py"""
import hashlib
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from ground_classifier import window_runner, windows
from ground_classifier.exports import DtmExport, TinExport, default_dtm_cell, file_exports
from ground_classifier.reclassify import InsufficientGroundSupport
from ground_classifier.utils.params import RunSettings
from ground_classifier.window_runner import process_file_windows

fails = 0
def ok(name, cond, note=""):
    global fails
    print(("ok   " if cond else "FAIL ") + name + (" - " + note if note else ""), flush=True)
    fails += 0 if cond else 1


# (extent, points, block size) and what the first implementation made of it
SCENARIOS = [
    ((0, 0, 300, 300), 1000, 5000), ((0, 0, 300, 300), 1000, 150), ((0, 0, 5000, 200), 1000, 5000), ((0, 0, 2000, 2000), 1000, 5000),
    ((0, 0, 2000, 2000), 1000, 500), ((0, 0, 100, 100), 500_000_000, 5000), ((0, 0, 10000, 300), 1000, 5000), ((0, 0, 1000, 1000), 1000, 100),
    ((5, 5, 5, 5), 1, 5000), ((0, 0, 600, 600), 1000, 250), ((100.5, -40.25, 900.75, 1500.5), 50_000, 400), ((0, 0, 300, 10000), 1000, 2500),
    ((0, 0, 5000, 5000), 300_000_000, 5000), ((0, 0, 500, 500), 1000, 125), ((0, 0, 440, 440), 1000, 220), ((0, 0, 449, 100), 1000, 100),
]
GOLDEN = [
    (1, "b961c122e514"), (4, "addd15c13848"), (1, "9ff81280311e"), (1, "2bbaaa9d43fb"), (16, "354e5ca6f151"), (1, "602be2dec797"),
    (9, "1c8fd2115966"), (100, "3f8eba118744"), (1, "bb837197778f"), (9, "6da919c79d40"), (12, "8042961a0301"), (4, "e0ced49bda89"),
    (4, "8f34b31f05be"), (16, "6e160ace9948"), (4, "ab2a298ff855"), (5, "5ad7fa5da41c"),
]
# (overlap, core, buffered point count, sum of their indices, first index) for the points made by `cloud()`
GOLDEN_BUFFERS = [
    (0.0, (0, 0, 500, 300), 972, 1915133, 4), (0.0, (500, 300, 1000, 600), 982, 2017984, 0), (0.0, (250, 100, 260, 110), 3, 7866, 188),
    (40.0, (0, 0, 500, 300), 1203, 2370883, 4), (40.0, (500, 300, 1000, 600), 1196, 2439594, 0), (40.0, (250, 100, 260, 110), 66, 136338, 11),
    (80.0, (0, 0, 500, 300), 1434, 2829520, 3), (80.0, (500, 300, 1000, 600), 1445, 2923633, 0), (80.0, (250, 100, 260, 110), 224, 445908, 11),
]


def cloud(n=4000):
    rng = np.random.default_rng(7)
    return rng.uniform(0, 1000, n), rng.uniform(0, 600, n)


def layout(extent, points, block_size):
    return [(n, tuple(round(float(v), 9) for v in core)) for n, core in windows.window_cores_for_extent(*map(float, extent), points, block_size)]


def raised(exc, call):
    try:
        call()
    except exc:
        return True
    except Exception:
        return False
    return False


def fake_block(local_x, local_y, local_z, local_class, settings, faces, max_edge):
    """Stands in for classify_block: every point ground (2), HAG = z."""
    return np.full(local_x.size, 2, np.uint8), local_z.astype(np.float32), np.arange(3)


def run_windows(x, y, z, block_size, workers, budget=10**12, allow_sparse=False, block=fake_block, settings=None):
    settings = settings or RunSettings(block_size=block_size, block_overlap=20.0)
    cores = list(windows.processing_windows(x, y, block_size))
    cls_out = np.zeros(x.size, np.uint8)
    hag_out = np.full(x.size, np.nan, np.float32)
    real = window_runner.classify_block
    window_runner.classify_block = block
    text = io.StringIO()
    try:
        with redirect_stdout(text):
            process_file_windows(cores, x, y, z, np.full(x.size, 1, np.uint8), cls_out, hag_out, workers, budget, settings, None, allow_sparse)
    finally:
        window_runner.classify_block = real
    return cores, cls_out, hag_out, text.getvalue()


def main():
    got = [(len(v), hashlib.sha1(repr(v).encode()).hexdigest()[:12]) for v in (layout(*s) for s in SCENARIOS)]
    ok("layout: the 16 scenarios give the layouts the first implementation made (window count and every core)", got == GOLDEN,
       str([i for i, (a, b) in enumerate(zip(got, GOLDEN)) if a != b]))
    ok("layout: a 300 m tile is one window at the default block size and four at 150 m",
       [c for _, c in layout((0, 0, 300, 300), 1000, 5000)] == [(0.0, 0.0, 300.0, 300.0)] and len(layout((0, 0, 300, 300), 1000, 150)) == 4)
    ok("layout: the block size is a hard cut, the point count adds windows up to 8",
       len(layout((0, 0, 100, 100), 500_000_000, 5000)) == 1 and len(layout((0, 0, 5000, 5000), 300_000_000, 5000)) == 4)
    ok("layout: processing_windows is the layout of the points' extent", list(windows.processing_windows(np.array([0.0, 300.0]), np.array([0.0, 300.0]), 150.0))
       == list(windows.window_cores_for_extent(0.0, 0.0, 300.0, 300.0, 2, 150.0)))

    # every point is owned by exactly one core
    x, y = cloud()
    owners_ok = True
    for block_size in (100.0, 250.0, 400.0, 5000.0):
        cores = [core for _, core in windows.processing_windows(x, y, block_size)]
        owned = sum(windows.core_point_mask(x, y, core, float(x.max()), float(y.max())).astype(int) for core in cores)
        owners_ok &= bool((owned == 1).all())
    ok("cores: at every block size each point, including those on the upper edges, falls in exactly one core", owners_ok)
    edge = windows.core_point_mask(np.array([5.0, 10.0, 10.0]), np.array([5.0, 5.0, 10.0]), (0.0, 0.0, 5.0, 10.0), 99.0, 99.0)
    ok("cores: half-open: the upper edge is out unless it is the file's own edge", edge.tolist() == [False, False, False])
    edge = windows.core_point_mask(np.array([5.0, 10.0]), np.array([5.0, 10.0]), (0.0, 0.0, 10.0, 10.0), 10.0, 10.0)
    ok("cores: ... and in at the file's own edge", edge.tolist() == [True, True])

    # the buffer around a core
    order = np.argsort(x, kind="stable")
    result = []
    same = True
    for overlap in (0.0, 40.0, 80.0):
        for core in ((0, 0, 500, 300), (500, 300, 1000, 600), (250, 100, 260, 110)):
            a = windows.buffered_indices_for_window(x, y, core, None, None, overlap)
            b = windows.buffered_indices_for_window(x, y, core, order, x[order], overlap)
            same &= bool((a == b).all())
            result.append((overlap, core, int(a.size), int(a.sum()), int(a[0]) if a.size else -1))
    ok("buffer: the points of the core plus the overlap, as the first implementation selected them", result == GOLDEN_BUFFERS)
    ok("buffer: the scan and the x-sorted search agree", same)
    ok("memory: a window's estimate is 256 MB plus 64 bytes a point", windows.estimate_window_memory(0) == 256 * 1024**2 and windows.estimate_window_memory(1000) == 256 * 1024**2 + 64000
       and windows.estimate_window_memory(-5) == 256 * 1024**2)

    # the exports a file is given
    cores = [(0.0, 0.0, 100.0, 100.0), (100.0, 0.0, 250.0, 120.0)]
    none = file_exports(RunSettings(), "tile", 1.0, cores)
    both = file_exports(RunSettings(tin_export_dir=Path("/t"), tin_max_edge=5.0, dtm_export_dir=Path("/d")), "tile", 0.3048, cores)
    ok("exports: none asked for, none made", none == (None, None))
    ok("exports: the TIN export carries the directory, stem, unit and longest edge", both[0] == TinExport(Path("/t"), "tile", 0.3048, 5.0))
    ok("exports: the DTM export covers all the windows, with the default cell of the unit",
       both[1] == DtmExport(Path("/d"), "tile", 0.3048, 3.0, (0.0, 0.0, 250.0, 120.0)))
    ok("exports: a cell on the settings wins; metres default to 1 m",
       file_exports(RunSettings(dtm_export_dir=Path("/d"), dtm_cell=7.0), "t", 1.0, cores)[1].cell == 7.0
       and file_exports(RunSettings(dtm_export_dir=Path("/d")), "t", 1.0, cores)[1].cell == default_dtm_cell(1.0) == 1.0)

    # the runner, with the classifier replaced by one with a known answer
    rng = np.random.default_rng(3)
    n = 3000
    px, py, pz = rng.uniform(0, 600, n), rng.uniform(0, 400, n), rng.uniform(0, 30, n)
    cores_1, cls_1, hag_1, text_1 = run_windows(px, py, pz, 200.0, 1)
    ok("runner: 200 m windows over 600 x 400 m give 6 windows, every point classified once", len(cores_1) == 6 and (cls_1 == 2).all()
       and np.allclose(hag_1, pz.astype(np.float32)))
    ok("runner: each window's messages are logged, in order, in the sequential run",
       [l.strip().split(":")[0] for l in text_1.splitlines() if l.strip().startswith("Window")] == [f"Window {i}/6" for i in range(1, 7)])
    _, cls_4, hag_4, text_4 = run_windows(px, py, pz, 200.0, 4)
    ok("runner: four threads give the same result as one", (cls_1 == cls_4).all() and np.array_equal(hag_1, hag_4) and "Parallel windows: up to 4" in text_4)
    _, cls_b, hag_b, text_b = run_windows(px, py, pz, 200.0, 4, budget=1)
    ok("runner: a budget too small for any window runs them one at a time, with the same result",
       (cls_1 == cls_b).all() and np.array_equal(hag_1, hag_b) and "exceeds the parallel RAM budget" in text_b)

    seen = []
    def recording(local_x, local_y, local_z, local_class, settings, faces, max_edge):
        seen.append((settings, local_x.size))
        return fake_block(local_x, local_y, local_z, local_class, settings, faces, max_edge)
    settings = RunSettings(block_size=200.0, block_overlap=20.0)
    run_windows(px, py, pz, 200.0, 2, block=recording, settings=settings)
    ok("runner: the classifier is given the run's settings object", len(seen) == 6 and all(s is settings for s, _ in seen))
    ok("runner: the overlap of the settings widens each window's buffer (more points than the cores hold)", sum(size for _, size in seen) > n)

    def insufficient(*args):
        raise InsufficientGroundSupport("no triangle")
    _, cls_s, hag_s, text_s = run_windows(px, py, pz, 200.0, 1, block=insufficient)
    ok("runner: a sparse window keeps its input classes and has no HAG, and the run goes on", (cls_s == 1).all() and np.isnan(hag_s).all()
       and text_s.count("Sparse-window fallback") == 6)
    big = 12_000
    bx, by, bz = rng.uniform(0, 100, big), rng.uniform(0, 100, big), rng.uniform(0, 5, big)
    ok("runner: a window over 10,000 points that cannot be built is an error, naming the window",
       raised(RuntimeError, lambda: run_windows(bx, by, bz, 5000.0, 1, block=insufficient)))
    _, cls_a, _, _ = run_windows(bx, by, bz, 5000.0, 1, allow_sparse=True, block=insufficient)
    ok("runner: ... unless the caller says the whole file is small (allow_sparse_fallback)", (cls_a == 1).all())

    print("all passed" if not fails else f"{fails} failed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
