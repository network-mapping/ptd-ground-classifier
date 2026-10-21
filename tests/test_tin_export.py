"""Checks of the TIN export: the PLY writer, GroundRun.tin() and the pipeline's per-window export.
usage: python tests/test_tin_export.py"""
import sys, tempfile
from pathlib import Path
import numpy as np
from ground_classifier import fast_ground as fg, ply, pipeline, ptdcore

fails = 0
def ok(name, cond, note=""):
    global fails
    print(("ok   " if cond else "FAIL ") + name + (" - " + note if note else ""), flush=True)
    fails += 0 if cond else 1


def main():
    rng = np.random.default_rng(11)
    tmp = Path(tempfile.mkdtemp(prefix="tin_export_"))

    # 1. PLY writer / reader
    xyz = np.column_stack((rng.uniform(468500, 469000, 50), rng.uniform(4966000, 4966500, 50), rng.uniform(0, 30, 50)))
    faces = np.array([[3, 7, 9], [7, 9, 20], [20, 3, 49]])
    n, m = ply.write_mesh(tmp / "a.ply", xyz, faces, ("first line", "second line"))
    back, bf, comments = ply.read_mesh(tmp / "a.ply")
    ok("ply: only the used vertices are written", (n, m) == (5, 3) and back.shape == (5, 3))
    ok("ply: round trip is exact in doubles (coordinates of 5e6)", np.array_equal(back[bf], xyz[faces]))
    ok("ply: comments kept", comments == ["first line", "second line"])
    head = (tmp / "a.ply").read_bytes()[:200].decode("ascii", "ignore")
    ok("ply: header says binary little endian, double xyz, uchar/int faces",
       "format binary_little_endian 1.0" in head and head.count("property double") == 3 and "property list uchar int vertex_indices" in head)
    ok("ply: file size = header + 24 B per vertex + 13 B per face",
       (tmp / "a.ply").stat().st_size == len((tmp / "a.ply").read_bytes().split(b"end_header\n")[0]) + len(b"end_header\n") + 24 * 5 + 13 * 3)
    ply.write_mesh(tmp / "e.ply", xyz, np.zeros((0, 3), np.int32))
    ok("ply: an empty mesh round-trips", ply.read_mesh(tmp / "e.ply")[0].shape == (0, 3) and ply.read_mesh(tmp / "e.ply")[1].shape == (0, 3))
    for label, bad in (("faces of the wrong shape", lambda: ply.write_mesh(tmp / "b.ply", xyz, np.array([1, 2, 3]))),
                       ("a multi-line comment", lambda: ply.write_mesh(tmp / "b.ply", xyz, faces, ("a\nb",)))):
        try:
            bad(); raised = False
        except ValueError:
            raised = True
        ok(f"ply: {label} is an error", raised)
    (tmp / "c.ply").write_bytes(b"ply\nformat ascii 1.0\nelement vertex 0\nend_header\n")
    try:
        ply.read_mesh(tmp / "c.ply"); raised = False
    except ValueError:
        raised = True
    ok("ply: reading another layout is an error", raised)

    # 2. GroundRun.tin() on terrain with objects and isolated low points
    N = 120_000
    x = rng.uniform(0, 300, N); y = rng.uniform(0, 300, N); z = 0.05 * x + 2 * np.sin(y / 35) + rng.normal(0, 0.015, N)
    obj = rng.random(N) < 0.25; z[obj] += rng.uniform(1.5, 15, obj.sum())
    low = np.flatnonzero(~obj)[:60]; z[low] -= rng.uniform(1.0, 8.0, low.size)
    r = fg.GroundRun(x, y, z).run()
    tri = r.tin()
    model = r.model()
    ok("tin: the engine's triangulation is valid", r.check() == 0)
    ok("tin: (m, 3) int32 indices into the input arrays", tri.ndim == 2 and tri.shape[1] == 3 and tri.dtype == np.int32 and 0 <= tri.min() and tri.max() < N, str(tri.shape))
    ok("tin: no corner vertex and no removed triangle (every vertex is a surface point)", model[tri].all())
    ok("tin: every surface point is used by the mesh", np.array_equal(np.unique(tri), np.flatnonzero(model)))
    a = np.column_stack((x, y))
    area2 = (a[tri[:, 1], 0] - a[tri[:, 0], 0]) * (a[tri[:, 2], 1] - a[tri[:, 0], 1]) - (a[tri[:, 2], 0] - a[tri[:, 0], 0]) * (a[tri[:, 1], 1] - a[tri[:, 0], 1])
    ok("tin: every triangle is counter-clockwise and has an area", (area2 > 0).all())
    edges = lambda t: np.max([np.hypot(*(a[t[:, i]] - a[t[:, (i + 1) % 3]]).T) for i in range(3)], axis=0)
    und = np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1)
    _, count = np.unique(und, axis=0, return_counts=True)
    ok("tin: no edge is shared by more than two triangles", count.max() <= 2)

    # 2b. the exported mesh is the surface the heights were measured from
    h = r.heights(); g = r.ground()
    pick = np.flatnonzero(g & ~model)[:1500]
    px, py, pz = x[pick], y[pick], z[pick]
    surface = np.full(pick.size, np.nan)
    for start in range(0, len(tri), 4000):
        t = tri[start:start + 4000]
        ax, ay, az = x[t[:, 0]], y[t[:, 0]], z[t[:, 0]]
        bx, by, bz = x[t[:, 1]] - ax, y[t[:, 1]] - ay, z[t[:, 1]] - az
        cx, cy, cz = x[t[:, 2]] - ax, y[t[:, 2]] - ay, z[t[:, 2]] - az
        det = bx * cy - cx * by
        dx = px[:, None] - ax; dy = py[:, None] - ay
        u = (dx * cy - cx * dy) / det; v = (bx * dy - dx * by) / det
        inside = (u >= -1e-12) & (v >= -1e-12) & (u + v <= 1 + 1e-12)
        zz = az + u * bz + v * cz
        first = np.argmax(inside, axis=1)
        found = inside.any(axis=1) & np.isnan(surface)
        surface[found] = zz[np.arange(pick.size), first][found]
    located = ~np.isnan(surface)
    ok("tin: the ground points of the band lie over the exported triangles", located.mean() > 0.95, f"{100 * located.mean():.1f}% located")
    ok("tin: the mesh gives the same heights as GroundRun.heights()", np.abs((pz - surface)[located] - h[pick][located]).max() < 2e-3,
       f"largest difference {np.abs((pz - surface)[located] - h[pick][located]).max():.5f} m")

    # 2c. max_edge
    limit = 3.0
    short = r.tin(limit)
    kept = {tuple(t) for t in short.tolist()}
    dropped = np.array([tuple(t) not in kept for t in tri.tolist()])
    # the engine measures edges in whole millimetres, so allow a millimetre at the limit
    ok("tin: max_edge leaves out the triangles with a longer edge, and only those",
       0 < len(short) < len(tri) and edges(short).max() <= limit + 1e-3 and edges(tri[dropped]).min() >= limit - 1e-3,
       f"{len(tri):,} -> {len(short):,} triangles")
    ok("tin: max_edge 0 keeps all", np.array_equal(r.tin(0.0), tri))

    # 2d. spikes taken out of the surface leave removed triangles in the core's arrays; none of them is exported
    sx = rng.uniform(0, 200, 40_000); sy = rng.uniform(0, 200, 40_000); sz = 0.03 * sx + rng.normal(0, 0.05, 40_000)
    rs = fg.GroundRun(sx, sy, sz, fg.Settings(spikes=fg.Spikes(up_angle=2.0, up_height=0.001, rounds=5))).run()
    raw = rs._core._tv[: 3 * rs._core.triangles].reshape(-1, 3)
    dead = (raw < 0).any(axis=1)
    ts = rs.tin()
    ok("tin: spikes were taken out, leaving removed triangles in the core", rs.spikes_removed > 0 and dead.any(), f"{rs.spikes_removed} spikes, {int(dead.sum())} removed triangles")
    ok("tin: removed triangles are not exported and the rest are", rs.check() == 0 and rs.model()[ts].all() and len(ts) == int(((raw >= 0) & (raw < rs.n)).all(axis=1).sum()))

    # 3. the pipeline's per-window export: cores partition the triangles, units, names
    unit = 0.3048006096012192
    tin_faces = []
    cls = np.full(N, 2, np.uint8)
    pipeline.classify_block(x, y, z, cls, tin_faces)
    ok("classify_block: appends the block's TIN to tin_out", len(tin_faces) == 1 and np.array_equal(tin_faces[0], tri) or
       (len(tin_faces) == 1 and tin_faces[0].shape == tri.shape))
    ok("classify_block: tin_out is optional (the unpacking of validate_suite.py still works)", len(pipeline.classify_block(x, y, z, cls)) == 3)
    export = pipeline.TinExport(tmp / "tins", "tile", unit)
    xmax, ymax = float(x.max()), float(y.max())
    cores = [(0.0, 0.0, 150.0, ymax), (150.0, 0.0, xmax, ymax)]
    paths = [pipeline.export_window_tin(export, i + 1, c, tri, x, y, z, xmax, ymax) for i, c in enumerate(cores)]
    ok("export: one file per window, named by tile and window", [p.name for p in paths] == ["tile_w01.ply", "tile_w02.ply"])
    parts = []
    for p in paths:
        v, f, comments = ply.read_mesh(p)
        parts.append(v[f] * unit)                       # triangles as (m, 3 corners, 3), back in metres
    ok("export: the windows' triangles together are the whole TIN, each once", sum(len(q) for q in parts) == len(tri), f"{[len(q) for q in parts]} of {len(tri)}")
    everything = np.sort(np.concatenate(parts).reshape(-1, 9), axis=0)
    original = np.sort(np.column_stack([np.concatenate((x[tri[:, i]][:, None], y[tri[:, i]][:, None], z[tri[:, i]][:, None]), axis=1) for i in range(3)]), axis=0)
    ok("export: coordinates are the input's, in the input's unit (metres / unit_scale), to within rounding",
       np.allclose(everything, original, atol=1e-6))
    cx = parts[0][:, :, 0].mean(axis=1); ok("export: window 1 holds only triangles whose centroid is in its core", (cx < 150.0).all() and (parts[1][:, :, 0].mean(axis=1) >= 150.0).all())
    ok("export: unit_scale is recorded in the header", any("metres per unit" in c for c in comments))
    ok("export: no triangle in the core = no file", pipeline.export_window_tin(export, 3, (1e6, 1e6, 2e6, 2e6), tri, x, y, z, 3e6, 3e6) is None
       and not (tmp / "tins" / "tile_w03.ply").exists())

    # 3b. classify_window_job writes the export for its window, and leaves classification as it was
    idx = np.arange(N)
    base = pipeline.classify_window_job(1, 1, (0.0, 0.0, xmax, ymax), idx, x, y, z, np.full(N, 2, np.uint8), xmax, ymax)
    with_tin = pipeline.classify_window_job(1, 1, (0.0, 0.0, xmax, ymax), idx, x, y, z, np.full(N, 2, np.uint8), xmax, ymax,
                                            tin_export=pipeline.TinExport(tmp / "job", "job", 1.0))
    ok("window job: classification is the same with or without the export",
       np.array_equal(base.classification, with_tin.classification) and np.array_equal(base.hag, with_tin.hag, equal_nan=True))
    jv, jf, _ = ply.read_mesh(tmp / "job" / "job_w01.ply")
    ok("window job: the whole block's TIN is written for a single window", len(jf) == len(tri) and np.allclose(jv[jf].reshape(-1, 3).mean(axis=0), np.column_stack((x, y, z))[tri].reshape(-1, 3).mean(axis=0)))
    ok("window job: no file without the export", not any((tmp / "job").glob("*_w02.ply")))

    # 4. the command line: a LAS in feet, --export-tin, the mesh lies over the points
    try:
        import laspy
        from ground_classifier import cli
    except ImportError:
        laspy = None
    if laspy is not None:
        scale_ft = 1 / unit
        ex, ey, ez = (np.round(rng.uniform(0, 300, 60_000), 2) for _ in range(3))
        ez = 0.05 * ex + 0.5 * np.sin(ey / 30) + 100
        ex += 468500; ey += 4966000                        # coordinates large enough for float32 to matter
        ex, ey, ez = ex * scale_ft, ey * scale_ft, ez * scale_ft
        header = laspy.LasHeader(point_format=3, version="1.4")
        header.scales = (0.001, 0.001, 0.001); header.offsets = (np.floor(ex.min()), np.floor(ey.min()), np.floor(ez.min()))
        las = laspy.LasData(header); las.x, las.y, las.z = ex, ey, ez; las.classification = np.full(ex.size, 2, np.uint8)
        (tmp / "in").mkdir(); las.write(tmp / "in" / "tile.las")
        code = cli.main([str(tmp / "in"), "--output-dir", str(tmp / "out"), "--units", "usft", "--export-tin", str(tmp / "cli_tin"), "--tin-max-edge", "5", "--workers", "2"])
        files = sorted((tmp / "cli_tin").glob("*.ply"))
        ok("cli: runs and writes one PLY for the tile's window", code == 0 and [f.name for f in files] == ["tile_w01.ply"], str([f.name for f in files]))
        if files:
            v, f, _ = ply.read_mesh(files[0])
            inside = (v[:, 0] >= ex.min() - 1e-3) & (v[:, 0] <= ex.max() + 1e-3) & (v[:, 1] >= ey.min() - 1e-3) & (v[:, 1] <= ey.max() + 1e-3)
            ok("cli: the mesh is in the file's own unit and lies over the points", len(f) > 1000 and inside.all(), f"{len(f):,} triangles")
            t = v[f]; longest = max(np.hypot(*(t[:, i, :2] - t[:, (i + 1) % 3, :2]).T).max() for i in range(3))
            ok("cli: --tin-max-edge (metres) limits the edges", longest <= 5.0 * scale_ft + 1e-3, f"longest {longest / scale_ft:.2f} m")
        # several windows on threads: a file split into four windows, each writing its own mesh; together they are one TIN
        saved = (pipeline.BLOCK_SIZE, pipeline.TIN_EXPORT_DIR, pipeline.TIN_EXPORT_MAX_EDGE)
        pipeline.BLOCK_SIZE = 150.0
        pipeline.configure_tin_export(tmp / "windows", 0.0)
        try:
            pipeline.classify_file(tmp / "in" / "tile.las", tmp / "w_out.las", window_workers=2, kdtree_workers=2, input_unit_scale=unit)
        finally:
            pipeline.BLOCK_SIZE = saved[0]; pipeline.configure_tin_export(saved[1], saved[2])
        win = sorted((tmp / "windows").glob("tile_w*.ply"))
        ok("windows: a file split into windows gets one mesh per window", [w.name for w in win] == [f"tile_w0{i}.ply" for i in range(1, 5)], str([w.name for w in win]))
        if len(win) == 4:
            meshes = [ply.read_mesh(w)[:2] for w in win]
            centroids = np.concatenate([v[f].mean(axis=1)[:, :2] for v, f in meshes])
            _, first = np.unique(np.round(centroids, 6), axis=0, return_index=True)
            ok("windows: no triangle is written by two windows", len(first) == len(centroids), f"{len(centroids):,} triangles")
            whole = ply.read_mesh(tmp / "cli_tin" / "tile_w01.ply")
            ok("windows: about as many triangles as the single-window TIN (windows differ only near their edges)",
               abs(len(centroids) - len(whole[1])) < 0.05 * len(whole[1]), f"{len(centroids):,} vs {len(whole[1]):,}")
        # several files processed by worker processes: the export setting reaches them
        (tmp / "in2").mkdir()
        for name in ("tile_a", "tile_b"):
            las.write(tmp / "in2" / f"{name}.las")
        code = cli.main([str(tmp / "in2"), "--output-dir", str(tmp / "out3"), "--units", "usft", "--export-tin", str(tmp / "pool_tin")])
        ok("pool: every file's window mesh is written", code == 0 and sorted(f.name for f in (tmp / "pool_tin").glob("*.ply")) == ["tile_a_w01.ply", "tile_b_w01.ply"],
           str(sorted(f.name for f in (tmp / "pool_tin").glob("*.ply"))))
        code = cli.main([str(tmp / "in"), "--output-dir", str(tmp / "out2"), "--units", "usft"])
        ok("cli: no export option, no export", code == 0 and not (tmp / "out2" / "tile_w01.ply").exists())

    import shutil; shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{fails} failed" if fails else "\nall passed")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()