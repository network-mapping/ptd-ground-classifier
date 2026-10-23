"""Checks of the anchored seeds (seeds.anchor / --anchor-seeds): the squares the surface starts from are cut from a
fixed grid, so they do not depend on where a block starts. usage: python tests/test_seed_anchor.py"""
import sys, tempfile, shutil, io, json, contextlib
from dataclasses import replace
from pathlib import Path
import numpy as np
from ground_classifier import fast_ground as fg, ptdcore

fails = 0
def ok(name, cond, note=""):
    global fails
    print(("ok   " if cond else "FAIL ") + name + (" - " + note if note else ""), flush=True)
    fails += 0 if cond else 1


def settings(anchor, **seeds):
    base = fg.Settings()
    return replace(base, seeds=replace(base.seeds, anchor=anchor, **seeds))


def scene(rng, n=250_000, x0=468500.0, y0=4966000.0, size=400.0):
    """Rolling terrain with buildings and trees, in large projected coordinates."""
    x = rng.uniform(x0, x0 + size, n); y = rng.uniform(y0, y0 + size, n)
    z = 4 * np.sin((x - x0) / 70) + 3 * np.cos((y - y0) / 90) + rng.normal(0, 0.02, n)
    obj = rng.random(n) < 0.35
    z[obj] += rng.uniform(1.5, 25, obj.sum())
    return x, y, z


def crop(x, y, z, box):
    idx = np.flatnonzero((x >= box[0]) & (x <= box[1]) & (y >= box[2]) & (y <= box[3]))
    return idx, x[idx], y[idx], z[idx]


def main():
    rng = np.random.default_rng(7)
    sq = ptdcore.length_to_steps(fg.Seeds().max_building_size)         # a starting square, in steps

    # 1. the setting
    ok("setting: off by default, so nothing changes until it is asked for", fg.Settings().seeds.anchor is False and fg.Settings().changed() == {})
    ok("setting: shows among the settings that differ from the defaults", settings(True).changed() == {"seeds.anchor": True})
    x, y, z = scene(rng, 20_000, size=200.0)
    ok("setting: off, the origin is the block's corner and no anchor is recorded", fg.GroundRun(x, y, z, settings(False)).origin is None)

    # 2. the steps
    x, y, z = scene(rng, 5000)
    xi, yi, (ox, oy) = fg.GroundRun._anchored_steps(x, y, settings(True))
    ok("steps: the origin is a multiple of the starting square, at or below the corner", ox % sq == 0 and oy % sq == 0 and 0 <= xi.min() < sq and 0 <= yi.min() < sq, f"origin {ox}, {oy}; smallest step {xi.min()}")
    left = x < x.mean()
    xl, yl, (oxl, oyl) = fg.GroundRun._anchored_steps(x[left], y[left], settings(True))
    ok("steps: a point's steps plus the origin are the same whichever block it is in", np.array_equal(xl + oxl, xi[left] + ox) and np.array_equal(yl + oyl, yi[left] + oy))
    ok("steps: they are the point's own rounded coordinate, from the metre 0", np.array_equal(xi + ox, ptdcore.metres_to_steps(x)))
    wide = fg.GroundRun._anchored_steps(x, y, settings(True, max_building_size=45.0))[2]
    ok("steps: the origin follows max_building_size", wide[0] % ptdcore.length_to_steps(45.0) == 0 and wide[1] % ptdcore.length_to_steps(45.0) == 0)
    cell = fg.GroundRun._anchored_steps(x, y, replace(settings(True), low_points=replace(fg.LowPoints(), cell=2.0)))[2]
    ok("steps: and rounds it to whole low-point squares, as choose_seeds does", cell[0] % ptdcore.length_to_steps(60.0) == 0)
    neg = fg.GroundRun._anchored_steps(x - 468700.0, y - 4966300.0, settings(True))
    ok("steps: negative coordinates round down to the multiple below, and steps stay non-negative", neg[0].min() >= 0 and neg[2][0] % sq == 0 and neg[2][0] <= ptdcore.metres_to_steps(x - 468700.0).min())

    # 3. the same ground gives the same starting points whichever block it is cut into
    x, y, z = scene(rng)
    box_a = (468500 + 13.7, 468500 + 400, 4966000 + 9.1, 4966000 + 400)       # two blocks that start at different, arbitrary corners
    box_b = (468500 + 101.3, 468500 + 400, 4966000 + 3.3, 4966000 + 390)
    inner = (468500 + 250, 468500 + 330, 4966000 + 250, 4966000 + 330)         # more than two squares from every edge of both
    in_inner = (x >= inner[0]) & (x <= inner[1]) & (y >= inner[2]) & (y <= inner[3])
    seeds = {}
    for anchor in (False, True):
        found = []
        for box in (box_a, box_b):
            idx, bx, by, bz = crop(x, y, z, box)
            run = fg.GroundRun(bx, by, bz, settings(anchor)).run()
            ok(f"run (anchor={anchor}): the triangulation is valid", run.check() == 0)
            found.append({int(i) for i in idx[run.order[run.seeds]] if in_inner[i]})
        seeds[anchor] = found
    ok("seeds: not anchored, two blocks pick different starting points on the same ground", seeds[False][0] != seeds[False][1], f"{len(seeds[False][0])} vs {len(seeds[False][1])}")
    ok("seeds: anchored, two blocks pick exactly the same starting points on the same ground", seeds[True][0] == seeds[True][1] and len(seeds[True][0]) > 0, f"{len(seeds[True][0])} starting points")

    # 4. shifting a block by a whole starting square changes nothing; so does a block that already starts on the grid
    idx, bx, by, bz = crop(x, y, z, box_a)
    base = fg.GroundRun(bx, by, bz, settings(True)).run()
    moved = fg.GroundRun(bx + 60.0, by - 120.0, bz, settings(True)).run()
    ok("anchored: the block moved by whole squares gives the same ground and heights", np.array_equal(base.ground(), moved.ground()) and np.array_equal(base.heights(), moved.heights()))
    cx = bx - bx.min() + 60.0 * 7808; cy = by - by.min() + 60.0 * 82767; cz = bz                                  # the block's corner on the grid
    ok("anchored: a block whose corner is on the grid is cut as before",
       np.array_equal(fg.GroundRun(cx, cy, cz, settings(True)).run().ground(), fg.GroundRun(cx, cy, cz, settings(False)).run().ground()))
    ok("anchored: the mm steps still fit the core for a block this far from 0", fg.GroundRun(bx, by, bz, settings(True)).origin[0] > 4e8)

    # 5. the command line
    import laspy
    from ground_classifier import cli
    tmp = Path(tempfile.mkdtemp(prefix="anchor_"))
    scale = 0.3048006096012192
    ex, ey, ez = scene(rng, 60_000, 468500 / scale, 4966000 / scale, 300 / scale)
    ez = (4 * np.sin((ex * scale - 468500) / 70) + 100) / scale
    header = laspy.LasHeader(point_format=3, version="1.4")
    header.scales = (0.01, 0.01, 0.01); header.offsets = (np.floor(ex.min()), np.floor(ey.min()), np.floor(ez.min()))
    las = laspy.LasData(header); las.x, las.y, las.z = ex, ey, ez; las.classification = np.full(ex.size, 2, np.uint8)
    (tmp / "in").mkdir(); las.write(tmp / "in" / "tile.las")
    manifests = {}
    logs = {}
    for flag in (False, True):
        out = tmp / f"out_{flag}"
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = cli.main([str(tmp / "in"), "--units", "usft", "--output-dir", str(out)] + (["--anchor-seeds"] if flag else []))
        manifests[flag] = json.loads((out / "manifest.json").read_text()); logs[flag] = buffer.getvalue()
        ok(f"cli: --anchor-seeds={flag}: runs", code == 0)
    ok("cli: the flag is recorded in the manifest's command line", manifests[True]["parameters"]["cli"]["anchor_seeds"] is True and manifests[False]["parameters"]["cli"]["anchor_seeds"] is False)
    ok("cli: the run says which engine settings it used", "seeds.anchor=1" in logs[True] and "seeds.anchor" not in logs[False])
    shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{fails} failed" if fails else "\nall passed")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
