"""Checks of the DTM export: the raster module, the pipeline's per-window export and the command line.
usage: python tests/test_dtm.py"""
import sys, tempfile, shutil
from pathlib import Path
import numpy as np
from ground_classifier import dtm, pipeline

fails = 0
def ok(name, cond, note=""):
    global fails
    print(("ok   " if cond else "FAIL ") + name + (" - " + note if note else ""), flush=True)
    fails += 0 if cond else 1


def brute_lowest(x, y, z, cell, cells):
    """The same raster the slow way, one cell at a time."""
    i0, i1, j0, j1 = cells
    out = np.full((j1 - j0 + 1, i1 - i0 + 1), dtm.NODATA, np.float32)
    for j in range(j0, j1 + 1):
        for i in range(i0, i1 + 1):
            m = (x >= i * cell) & (x < (i + 1) * cell) & (y >= j * cell) & (y < (j + 1) * cell)
            if m.any():
                out[j - j0, i - i0] = z[m].min()
    return out


def main():
    rng = np.random.default_rng(21)
    tmp = Path(tempfile.mkdtemp(prefix="dtm_"))

    # 1. which cells a block owns
    cell = 3.0
    xs = np.sort(np.concatenate(([100.4, 700.9], rng.uniform(100.4, 700.9, 5))))   # core edges that are not on cell edges
    owned = []
    for a, b in zip(xs[:-1], xs[1:]):
        owned.append(dtm.owned_cells((a, 0.0, b, 30.0), cell, first_x=a == xs[0], last_x=b == xs[-1], first_y=True, last_y=True))
    columns = [set(range(c[0], c[1] + 1)) for c in owned]
    union = set().union(*columns)
    ok("cells: blocks side by side share no cell", sum(len(c) for c in columns) == len(union))
    ok("cells: and leave none out, up to the cells that hold the edges of the run",
       union == set(range(int(np.floor(xs[0] / cell)), int(np.floor(xs[-1] / cell)) + 1)))
    ok("cells: rows follow the same rule", owned[0][2:] == (0, 10), str(owned[0][2:]))
    none = dtm.owned_cells((10.1, 0.0, 10.4, 30.0), cell)
    ok("cells: a core smaller than a cell can own none", none[1] < none[0])

    # 2. lowest point of each cell
    x = rng.uniform(95, 140, 20_000); y = rng.uniform(-20, 40, 20_000); z = rng.uniform(0, 50, 20_000)
    cells = dtm.owned_cells((99.0, -9.0, 130.0, 30.0), cell, True, True, True, True)
    grid = dtm.lowest_ground(x, y, z, cell, cells)
    ok("raster: float32, shape (rows, columns)", grid.dtype == np.float32 and grid.shape == (cells[3] - cells[2] + 1, cells[1] - cells[0] + 1))
    ok("raster: the lowest point of each cell, row 0 the lowest y", np.allclose(grid, brute_lowest(x, y, z, cell, cells), atol=1e-4))
    sparse = dtm.lowest_ground(np.array([1.0, 100.0]), np.array([1.0, 100.0]), np.array([5.0, 6.0]), cell, (0, 3, 0, 3))
    ok("raster: cells without a point are nodata, points outside the cells are ignored", (sparse != dtm.NODATA).sum() == 1 and sparse[0, 0] == 5.0)
    edge = dtm.lowest_ground(np.array([3.0, 2.999999]), np.array([0.0, 0.0]), np.array([1.0, 2.0]), cell, (0, 1, 0, 0))
    ok("raster: a cell holds [i * cell, (i + 1) * cell): a point on the edge is in the next cell", edge.tolist() == [[2.0, 1.0]], str(edge.tolist()))
    ok("raster: no points give an all-nodata raster; no cells give an empty one",
       (dtm.lowest_ground(np.array([]), np.array([]), np.array([]), cell, (0, 2, 0, 1)) == dtm.NODATA).all()
       and dtm.lowest_ground(x, y, z, cell, (5, 4, 0, 1)).shape == (2, 0))

    # 3. file
    path = tmp / "a.tif"
    ok("file: writes columns and rows", dtm.write_dtm(path, grid, cells, cell, "usft", ("one line",)) == (grid.shape[1], grid.shape[0]))
    back, bcells, bcell, unit, comments = dtm.read_dtm(path)
    ok("file: round trip is exact, with the cells, cell size, unit and comments", np.array_equal(back, grid) and bcells == cells and bcell == cell and unit == "usft" and comments == ["one line"])
    import tifffile
    with tifffile.TiffFile(path) as t:
        tags = t.pages[0].tags
        arr = t.pages[0].asarray()
        compression = t.pages[0].compression.name
        scale, tie, nodata = tags[33550].value, tags[33922].value, tags[42113].value
    ok("file: GeoTIFF tags tie the top left corner to its coordinates, with the cell size and nodata",
       tuple(scale[:2]) == (cell, cell) and tuple(tie[3:5]) == (cells[0] * cell, (cells[3] + 1) * cell) and nodata.strip("\x00") == f"{dtm.NODATA:g}")
    ok("file: north up (the file's first row is the highest y) and compressed", np.array_equal(arr[0], grid[-1]) and compression != "NONE", compression)
    big = np.full((300, 300), 12.5, np.float32)
    dtm.write_dtm(tmp / "big.tif", big, (0, 299, 0, 299), 1.0)
    ok("file: compressed well below its raw size", (tmp / "big.tif").stat().st_size < big.nbytes / 4, f"{(tmp / 'big.tif').stat().st_size:,} B for {big.nbytes:,}")
    for label, call in (("an empty raster", lambda: dtm.write_dtm(tmp / "e.tif", np.zeros((0, 3), np.float32), (0, 2, 0, -1), cell)),
                        ("a multi-line comment", lambda: dtm.write_dtm(tmp / "e.tif", grid, cells, cell, "", ("a\nb",))),
                        ("another tiff", lambda: (tifffile.imwrite(tmp / "o.tif", np.zeros((4, 4), np.float32)), dtm.read_dtm(tmp / "o.tif")))):
        try:
            call(); raised = False
        except ValueError:
            raised = True
        ok(f"file: {label} is an error", raised)

    # 4. the pipeline's per-window export: two windows write one raster between them, in the file's unit
    unit_scale = 0.3048006096012192
    n = 80_000
    fx = rng.uniform(468500.0, 468800.0, n).round(2); fy = rng.uniform(4966000.0, 4966300.0, n).round(2); fz = (100 + 0.02 * (fx - 468500) + rng.uniform(0, 3, n)).round(2)
    ground = rng.random(n) < 0.5
    mx, my, mz = fx * unit_scale, fy * unit_scale, fz * unit_scale           # the pipeline works in metres
    extent = (mx.min(), my.min(), mx.max(), my.max())
    split = (extent[0] + extent[2]) / 2 + 0.123
    export = pipeline.DtmExport(tmp / "dtm", "tile", unit_scale, 3.0, extent)
    w1 = pipeline.export_window_dtm(export, 1, (extent[0], extent[1], split, extent[3]), mx, my, mz, ground)
    w2 = pipeline.export_window_dtm(export, 2, (split, extent[1], extent[2], extent[3]), mx, my, mz, ground)
    ok("export: one raster per window, named by tile and window", [p.name for p in (w1, w2)] == ["tile_w01.tif", "tile_w02.tif"])
    g1, c1, _, u1, _ = dtm.read_dtm(w1); g2, c2, _, _, _ = dtm.read_dtm(w2)
    ok("export: in the input's unit (usft), on a grid fixed to the coordinates", u1 == "usft" and c1[2:] == c2[2:] and c2[0] == c1[1] + 1)
    whole_cells = (c1[0], c2[1], c1[2], c1[3])
    expected = brute_lowest(fx[ground], fy[ground], fz[ground], 3.0, whole_cells)
    both = np.concatenate((g1, g2), axis=1)
    ok("export: the two windows together are the lowest ground point of every cell of the file, each cell once", np.array_equal(both, expected),
       f"{int((both != expected).sum())} cells differ of {both.size:,}")
    ok("export: the file's outermost cells are written too (every ground point has a cell)",
       int((both != dtm.NODATA).sum()) == len({(int(a // 3), int(b // 3)) for a, b in zip(fx[ground], fy[ground])}))
    ok("export: no ground point in the core = no file", pipeline.export_window_dtm(export, 3, (extent[0], extent[1], split, extent[3]), mx, my, mz, np.zeros(n, bool)) is None
       and not (tmp / "dtm" / "tile_w03.tif").exists())
    ok("export: the default cell is 1 m for metres and 3 ft for feet", pipeline.default_dtm_cell(1.0) == 1.0 and pipeline.default_dtm_cell(unit_scale) == 3.0)
    # a point whose feet value is exactly a cell edge must land in the cell that starts there
    edge_export = pipeline.DtmExport(tmp / "edge", "edge", unit_scale, 3.0, (0.0, 0.0, 400.0, 400.0))
    p = pipeline.export_window_dtm(edge_export, 1, (0.0, 0.0, 400.0, 400.0), np.array([300.0 * unit_scale]), np.array([6.0 * unit_scale]), np.array([50.0 * unit_scale]), np.array([True]))
    ge, ce = dtm.read_dtm(p)[:2]
    found = np.argwhere(ge != dtm.NODATA)
    ok("export: the round trip through metres does not move an edge point into the cell before",
       len(found) == 1 and found[0][1] + ce[0] == 100 and found[0][0] + ce[2] == 2, str(found.tolist()))

    # 5. the command line
    try:
        import laspy
        from ground_classifier import cli
    except ImportError:
        laspy = None
    if laspy is not None:
        ex = (rng.uniform(0, 300, 60_000) + 468500) / unit_scale
        ey = (rng.uniform(0, 300, 60_000) + 4966000) / unit_scale
        ez = (0.05 * (ex * unit_scale - 468500) + 0.5 * np.sin(ey * unit_scale / 30) + 100) / unit_scale
        header = laspy.LasHeader(point_format=3, version="1.4")
        header.scales = (0.01, 0.01, 0.01); header.offsets = (np.floor(ex.min()), np.floor(ey.min()), np.floor(ez.min()))
        las = laspy.LasData(header); las.x, las.y, las.z = ex, ey, ez; las.classification = np.full(ex.size, 2, np.uint8)
        (tmp / "in").mkdir(); las.write(tmp / "in" / "tile.las")
        args = [str(tmp / "in"), "--units", "usft"]
        code = cli.main(args + ["--output-dir", str(tmp / "out"), "--export-dtm", str(tmp / "cli_dtm"), "--workers", "2"])
        files = sorted((tmp / "cli_dtm").glob("*.tif"))
        ok("cli: runs and writes one raster for the tile's window", code == 0 and [f.name for f in files] == ["tile_w01.tif"], str([f.name for f in files]))
        if files:
            grid, cells, cell_size, unit, _ = dtm.read_dtm(files[0])
            out = laspy.read(tmp / "out" / "tile_AdaptiveGround.las")
            g = np.asarray(out.classification) == 2
            ox, oy, oz = (np.asarray(v, dtype=np.float64)[g] for v in (out.x, out.y, out.z))
            ok("cli: default cell for a feet file is 3 ft, in the file's unit", cell_size == 3.0 and unit == "usft")
            expected = brute_lowest(ox, oy, oz, cell_size, cells)
            ok("cli: the raster is the lowest point of the file's final class 2 in each cell", np.allclose(grid, expected, atol=1e-3) and (grid != dtm.NODATA).any(),
               f"{int((np.abs(grid - expected) > 1e-3).sum())} cells differ of {grid.size:,}")
        cli.main(args + ["--output-dir", str(tmp / "out2"), "--export-dtm", str(tmp / "cell_dtm"), "--dtm-cell", "10"])
        ok("cli: --dtm-cell sets the cell size", dtm.read_dtm(tmp / "cell_dtm" / "tile_w01.tif")[2] == 10.0)
        # several windows on threads: each window writes its own raster, with no cell in two of them
        saved = (pipeline.BLOCK_SIZE, pipeline.DTM_EXPORT_DIR, pipeline.DTM_CELL)
        pipeline.BLOCK_SIZE = 150.0
        pipeline.configure_dtm_export(tmp / "windows", 0.0)
        try:
            pipeline.classify_file(tmp / "in" / "tile.las", tmp / "w_out.las", window_workers=2, kdtree_workers=2, input_unit_scale=unit_scale)
        finally:
            pipeline.BLOCK_SIZE = saved[0]; pipeline.configure_dtm_export(saved[1], saved[2])
        win = sorted((tmp / "windows").glob("tile_w*.tif"))
        ok("windows: one raster per window", [w.name for w in win] == [f"tile_w0{i}.tif" for i in range(1, 5)], str([w.name for w in win]))
        if len(win) == 4:
            seen = {}; duplicates = 0
            for w in win:
                grid, c, _, _, _ = dtm.read_dtm(w)
                for (r, k), value in np.ndenumerate(grid):
                    if value != dtm.NODATA:
                        key = (c[0] + k, c[2] + r)
                        duplicates += key in seen
                        seen[key] = value
            ok("windows: no cell is written by two windows", duplicates == 0, f"{len(seen):,} cells")
            whole, wc, _, _, _ = dtm.read_dtm(tmp / "cli_dtm" / "tile_w01.tif")
            cover = {(wc[0] + k, wc[2] + r): v for (r, k), v in np.ndenumerate(whole) if v != dtm.NODATA}
            same = sum(1 for key in seen.keys() & cover.keys() if abs(seen[key] - cover[key]) < 1e-3)
            different = len(seen.keys() ^ cover.keys())
            ok("windows: the same cells and heights as the single-window raster almost everywhere (they differ near window edges)",
               different <= 0.01 * len(cover) and same >= 0.97 * len(cover), f"{different} cells differ in coverage, {same:,} of {len(cover):,} equal")
        # several files on worker processes: the setting reaches them
        (tmp / "in2").mkdir()
        for name in ("tile_a", "tile_b"):
            las.write(tmp / "in2" / f"{name}.las")
        code = cli.main([str(tmp / "in2"), "--output-dir", str(tmp / "out3"), "--units", "usft", "--export-dtm", str(tmp / "pool_dtm"), "--dtm-cell", "5"])
        names = sorted(f.name for f in (tmp / "pool_dtm").glob("*.tif"))
        ok("pool: every file's raster is written, with the cell size", code == 0 and names == ["tile_a_w01.tif", "tile_b_w01.tif"]
           and dtm.read_dtm(tmp / "pool_dtm" / names[0])[2] == 5.0, str(names))
        cli.main(args + ["--output-dir", str(tmp / "out4")])
        ok("cli: no export option, no raster", not list((tmp / "out4").glob("*.tif")))

    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{fails} failed" if fails else "\nall passed")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
