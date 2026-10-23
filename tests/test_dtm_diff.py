"""Checks of the DTM subtraction: dtm.subtract() and `ground-classify dtm-diff`. usage: python tests/test_dtm_diff.py"""
import sys, tempfile, shutil, io, contextlib
from pathlib import Path
import numpy as np
from ground_classifier import cli, dtm

fails = 0
def ok(name, cond, note=""):
    global fails
    print(("ok   " if cond else "FAIL ") + name + (" - " + note if note else ""), flush=True)
    fails += 0 if cond else 1


def run(*argv):
    """(exit text, stdout) of the command; a SystemExit carries its message."""
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            cli.main(["dtm-diff", *map(str, argv)])
        return None, out.getvalue()
    except SystemExit as exc:
        return str(exc.code), out.getvalue()


def make(path, rng, cells, cell=3.0, unit="usft", offset=0.0, holes=0.0):
    rows, columns = cells[3] - cells[2] + 1, cells[1] - cells[0] + 1
    grid = (100 + rng.uniform(0, 5, (rows, columns)) + offset).astype(np.float32)
    grid[rng.random(grid.shape) < holes] = dtm.NODATA
    dtm.write_dtm(path, grid, cells, cell, unit)
    return grid


def main():
    rng = np.random.default_rng(31)
    tmp = Path(tempfile.mkdtemp(prefix="dtm_diff_"))

    # 1. subtract()
    cells = (10, 19, 5, 9)
    a = rng.uniform(0, 10, (5, 10)).astype(np.float32); b = rng.uniform(0, 10, (5, 10)).astype(np.float32)
    d, dc = dtm.subtract(a, cells, b, cells)
    ok("subtract: B - A on the same cells", dc == cells and d.dtype == np.float32 and np.allclose(d, b - a, atol=1e-6))
    a2 = a.copy(); a2[1, 2] = dtm.NODATA; b2 = b.copy(); b2[3, 4] = dtm.NODATA
    d, _ = dtm.subtract(a2, cells, b2, cells)
    ok("subtract: a cell with no value in either raster is nodata, never a difference", d[1, 2] == dtm.NODATA and d[3, 4] == dtm.NODATA and (d != dtm.NODATA).sum() == 48)
    # rasters that cover different areas, on the same grid
    a_cells, b_cells = (0, 9, 0, 4), (5, 14, 2, 8)
    ga = rng.uniform(0, 10, (5, 10)).astype(np.float32); gb = rng.uniform(0, 10, (7, 10)).astype(np.float32)
    d, dc = dtm.subtract(ga, a_cells, gb, b_cells)
    ok("subtract: covers both rasters", dc == (0, 14, 0, 8) and d.shape == (9, 15))
    overlap = d[2:5, 5:10]
    ok("subtract: a value only where both rasters have one, at the right cells", np.allclose(overlap, gb[0:3, 0:5] - ga[2:5, 5:10], atol=1e-6)
       and (d != dtm.NODATA).sum() == 15)
    ok("subtract: the same raster from itself is zero", np.all(dtm.subtract(a, cells, a, cells)[0] == 0.0))
    d, _ = dtm.subtract(a, cells, a + np.float32(0.25), cells)
    ok("subtract: a constant offset is that offset in every cell", np.allclose(d, 0.25, atol=1e-5))

    # 2. read_dtm_info
    path = tmp / "i.tif"
    dtm.write_dtm(path, a, cells, 3.0, "usft", ("x",))
    ok("info: the header without the raster", dtm.read_dtm_info(path) == (cells, 3.0, "usft", ["x"]))

    # 3. the command: same names, differences, output
    A, B, O = tmp / "A", tmp / "B", tmp / "out"
    A.mkdir(); B.mkdir()
    cells1, cells2 = (100, 139, 50, 79), (140, 179, 50, 79)
    ga1 = make(A / "tile_w01.tif", rng, cells1, holes=0.05); gb1 = make(B / "tile_w01.tif", rng, cells1, offset=0.04, holes=0.05)
    ga2 = make(A / "tile_w02.tif", rng, cells2); gb2 = make(B / "tile_w02.tif", rng, (130, 179, 50, 79))   # a different layout of the same tile
    message, text = run(A, B, O)
    ok("dtm-diff: runs", message is None, str(message))
    ok("dtm-diff: one raster per name, same names", sorted(p.name for p in O.glob("*.tif")) == ["tile_w01.tif", "tile_w02.tif"])
    d1, c1, cell1, unit1, comments1 = dtm.read_dtm(O / "tile_w01.tif")
    expected, _ = dtm.subtract(ga1, cells1, gb1, cells1)
    ok("dtm-diff: the height is B - A, nodata where either has none", np.array_equal(d1, expected) and c1 == cells1)
    ok("dtm-diff: cell size and unit carried through, the header says what it is", cell1 == 3.0 and unit1 == "usft" and "ground-classifier" in comments1[0] and "B - A" in comments1[0])
    d2, c2, *_ = dtm.read_dtm(O / "tile_w02.tif")
    ok("dtm-diff: rasters of different extent: the result covers both", c2 == (130, 179, 50, 79) and (d2[:, :10] == dtm.NODATA).all() and (d2[:, 10:] != dtm.NODATA).all())
    mean = float((gb1 - ga1)[(ga1 != dtm.NODATA) & (gb1 != dtm.NODATA)].mean())
    ok("dtm-diff: reports each file's mean (B - A), RMS, percentile and largest, with the unit",
       "tile_w01.tif: " in text and f"mean {mean:+.4f}" in text and "RMS" in text and "95% within" in text and "(usft)" in text, text.splitlines()[-3])
    ok("dtm-diff: the inputs are not changed", np.array_equal(dtm.read_dtm(A / "tile_w01.tif")[0], ga1) and np.array_equal(dtm.read_dtm(B / "tile_w01.tif")[0], gb1))
    swapped = tmp / "swapped"
    run(B, A, swapped)
    s1 = dtm.read_dtm(swapped / "tile_w01.tif")[0]
    both = d1 != dtm.NODATA
    ok("dtm-diff: swapping the folders swaps the sign", np.allclose(s1[both], -d1[both], atol=1e-6) and np.array_equal(s1 != dtm.NODATA, both))

    # 4. errors, raised before anything is written
    for label, fix in (
        ("a file only in A", lambda: make(A / "only_a.tif", rng, cells1)),
        ("a file only in B", lambda: make(B / "only_b.tif", rng, cells1)),
    ):
        fix()
        o = tmp / ("o_" + label.replace(" ", "_"))
        message, _ = run(A, B, o)
        name = "only_a.tif" if "A" in label else "only_b.tif"
        ok(f"error: {label} is an error naming it, and nothing is written", message is not None and name in message and not o.exists(), str(message)[:90])
        (A / name if "A" in label else B / name).unlink()
    make(B / "tile_w02.tif", rng, cells2, cell=1.5)
    o = tmp / "o_cell"
    message, _ = run(A, B, o)
    ok("error: a different cell size is an error, and nothing is written", message is not None and "cell size" in message and "tile_w02.tif" in message and not o.exists(), str(message)[:100])
    make(B / "tile_w02.tif", rng, cells2, unit="metres")
    o = tmp / "o_unit"
    message, _ = run(A, B, o)
    ok("error: a different unit is an error, and nothing is written", message is not None and "unit" in message and not o.exists(), str(message)[:100])
    make(B / "tile_w02.tif", rng, cells2)
    (B / "tile_w02.tif").write_bytes(b"not a tiff")
    message, _ = run(A, B, tmp / "o_bad")
    ok("error: a file that is not a raster of ours is an error", message is not None and "tile_w02.tif" in message)
    make(B / "tile_w02.tif", rng, cells2)
    ok("error: the output folder may not be an input", run(A, B, A)[0] is not None and run(A, B, B)[0] is not None)
    ok("error: a folder that is missing or empty is an error", run(tmp / "none", B, tmp / "o1")[0] is not None and run(A, tmp / "empty", tmp / "o2")[0] is not None)
    (tmp / "empty").mkdir()
    ok("error: an empty folder is an error", "No .tif" in (run(A, tmp / "empty", tmp / "o3")[0] or ""))

    # 5. end to end: two exports of the same data through the pipeline differ by exactly zero
    import laspy
    scale = 0.3048006096012192
    n = 40_000
    ex = (rng.uniform(0, 200, n) + 468500) / scale; ey = (rng.uniform(0, 200, n) + 4966000) / scale
    ez = (4 * np.sin((ex * scale - 468500) / 50) + 100) / scale
    header = laspy.LasHeader(point_format=3, version="1.4")
    header.scales = (0.01, 0.01, 0.01); header.offsets = (np.floor(ex.min()), np.floor(ey.min()), np.floor(ez.min()))
    las = laspy.LasData(header); las.x, las.y, las.z = ex, ey, ez; las.classification = np.full(n, 2, np.uint8)
    (tmp / "in").mkdir(); las.write(tmp / "in" / "tile.las")
    with contextlib.redirect_stdout(io.StringIO()):
        for name in ("one", "two"):
            cli.main([str(tmp / "in"), "--units", "usft", "--output-dir", str(tmp / f"o_{name}"), "--export-dtm", str(tmp / f"dtm_{name}")])
        cli.main([str(tmp / "in"), "--units", "usft", "--output-dir", str(tmp / "o_three"), "--export-dtm", str(tmp / "dtm_three"), "--dtm-cell", "6"])
    message, text = run(tmp / "dtm_one", tmp / "dtm_two", tmp / "dtm_zero")
    zero, *_ = dtm.read_dtm(tmp / "dtm_zero" / "tile_w01.tif")
    valid = zero != dtm.NODATA
    ok("pipeline: two identical runs differ by exactly zero in every cell", message is None and valid.sum() > 1000 and (zero[valid] == 0).all(), f"{int(valid.sum()):,} cells")
    ok("pipeline: runs with different cell sizes cannot be subtracted", run(tmp / "dtm_one", tmp / "dtm_three", tmp / "dtm_bad")[0] is not None)

    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{fails} failed" if fails else "\nall passed")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
