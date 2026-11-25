"""Checks of neighbour discovery: which tiles lie within the neighbour halo of each tile, and the tile footprint rule
(a numeric file name such as 468500_4966000 is a nominal square tile). usage: python tests/test_neighbours.py

The expected sets in GOLDEN were recorded from the first implementation (pipeline.find_neighbour_sources) on exactly
these tiles, before it moved to LasDataset.neighbours, so they pin the behaviour across the move."""
import shutil
import sys
import tempfile
from pathlib import Path

import laspy
import numpy as np

FEET = 1200 / 3937

fails = 0
def ok(name, cond, note=""):
    global fails
    print(("ok   " if cond else "FAIL ") + name + (" - " + note if note else ""), flush=True)
    fails += 0 if cond else 1


def write_tile(path, extent, points=5):
    """A LAS tile whose points reach exactly `extent` = (xmin, ymin, xmax, ymax); points=0 is an empty tile."""
    xmin, ymin, xmax, ymax = extent
    header = laspy.LasHeader(point_format=3, version="1.4")
    header.scales = (0.001, 0.001, 0.001)
    header.offsets = (xmin, ymin, 0.0)
    tile = laspy.LasData(header)
    if points:
        xs = [xmin, xmax, xmin, xmax, (xmin + xmax) / 2][:points]
        ys = [ymin, ymin, ymax, ymax, (ymin + ymax) / 2][:points]
        tile.x = np.array(xs, float); tile.y = np.array(ys, float); tile.z = np.zeros(len(xs))
        tile.classification = np.full(len(xs), 2, np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    tile.write(path)
    return path


def scenarios(root: Path, unit: str):
    """(name, halo in metres, the tile files, the target names or None for all), in the unit's own coordinates."""
    size = 500.0 if unit == "metres" else 1500.0           # the nominal tile of a numeric name
    scale = 1.0 if unit == "metres" else FEET
    ox, oy = (468500.0, 4966000.0) if unit == "metres" else (1537000.0, 16293000.0)
    gap = lambda metres: metres / scale                  # a gap of so many metres, in the unit
    base = root / unit
    out = []

    def named(folder, col, row, inset=1.0, points=5):
        x, y = ox + col * size, oy + row * size
        return write_tile(base / folder / f"{x:.0f}_{y:.0f}.las", (x + inset, y + inset, x + size - inset, y + size - inset), points)

    # numeric names: full tiles of a 3 x 3 block, and one two tiles to the side
    block = [named("grid", c, r) for c in range(3) for r in range(3)]
    far = named("grid", 5, 1)
    out.append(("grid: edge and corner neighbours, halo 10 m", 10.0, block + [far], None))
    out.append(("grid: no halo, no neighbours", 0.0, block, None))
    out.append(("grid: only the middle tile is a target", 10.0, block + [far], [block[4].name]))
    # unnamed tiles: the header extent, with gaps of 5, 15 m
    a = write_tile(base / "gaps" / "a.las", (1000.0, 2000.0, 1000.0 + gap(100), 2000.0 + gap(100)))
    b = write_tile(base / "gaps" / "b.las", (1000.0 + gap(105), 2000.0, 1000.0 + gap(205), 2000.0 + gap(100)))
    c = write_tile(base / "gaps" / "c.las", (1000.0 + gap(220), 2000.0, 1000.0 + gap(320), 2000.0 + gap(100)))
    out.append(("gaps: 5 m is inside a 10 m halo, 15 m is not", 10.0, [a, b, c], None))
    out.append(("gaps: a 20 m halo reaches 15 m", 20.0, [a, b, c], None))
    # a sparse sliver of a numeric tile is still found through the nominal footprint
    sliver = write_tile(base / "sliver" / f"{ox:.0f}_{oy:.0f}.las", (ox + 5, oy + 5, ox + 25, oy + 25))
    beside = named("sliver", 1, 0)
    out.append(("sliver: the nominal footprint makes a 20 m sliver a neighbour of the full tile next to it", 10.0, [sliver, beside], None))
    # a numeric name that its data does not fit is not trusted
    stray = write_tile(base / "stray" / "100_100.las", (ox + 4000, oy, ox + 4100, oy + 100))
    near = write_tile(base / "stray" / "near.las", (ox + 4105, oy, ox + 4200, oy + 100))
    out.append(("stray: a numeric name that the data does not fit uses the header extent", 10.0, [stray, near], None))
    # empty tiles
    full = named("empty", 0, 0)
    empty_named = named("empty", 1, 0, points=0)
    empty_plain = write_tile(base / "empty" / "e.las", (0, 0, 0, 0), points=0)
    out.append(("empty: a named empty tile has its nominal footprint, an unnamed one none to speak of", 10.0, [full, empty_named, empty_plain], None))
    out.append(("single: one file has no neighbours", 10.0, [named("single", 0, 0)], None))
    return out


GOLDEN = {
    ("metres", "grid: edge and corner neighbours, halo 10 m"): [[1, 3, 4], [0, 2, 3, 4, 5], [1, 4, 5], [0, 1, 4, 6, 7], [0, 1, 2, 3, 5, 6, 7, 8], [1, 2, 4, 7, 8], [3, 4, 7], [3, 4, 5, 6, 8], [4, 5, 7], []],
    ("metres", "grid: no halo, no neighbours"): [[], [], [], [], [], [], [], [], []],
    ("metres", "grid: only the middle tile is a target"): [[0, 1, 2, 3, 5, 6, 7, 8]],
    ("metres", "gaps: 5 m is inside a 10 m halo, 15 m is not"): [[1], [0], []],
    ("metres", "gaps: a 20 m halo reaches 15 m"): [[1], [0, 2], [1]],
    ("metres", "sliver: the nominal footprint makes a 20 m sliver a neighbour of the full tile next to it"): [[1], [0]],
    ("metres", "stray: a numeric name that the data does not fit uses the header extent"): [[1], [0]],
    ("metres", "empty: a named empty tile has its nominal footprint, an unnamed one none to speak of"): [[1], [0], []],
    ("metres", "single: one file has no neighbours"): [[]],
    ("usft", "grid: edge and corner neighbours, halo 10 m"): [[1, 3, 4], [0, 2, 3, 4, 5], [1, 4, 5], [0, 1, 4, 6, 7], [0, 1, 2, 3, 5, 6, 7, 8], [1, 2, 4, 7, 8], [3, 4, 7], [3, 4, 5, 6, 8], [4, 5, 7], []],
    ("usft", "grid: no halo, no neighbours"): [[], [], [], [], [], [], [], [], []],
    ("usft", "grid: only the middle tile is a target"): [[0, 1, 2, 3, 5, 6, 7, 8]],
    ("usft", "gaps: 5 m is inside a 10 m halo, 15 m is not"): [[1], [0], []],
    ("usft", "gaps: a 20 m halo reaches 15 m"): [[1], [0, 2], [1]],
    ("usft", "sliver: the nominal footprint makes a 20 m sliver a neighbour of the full tile next to it"): [[1], [0]],
    ("usft", "stray: a numeric name that the data does not fit uses the header extent"): [[1], [0]],
    ("usft", "empty: a named empty tile has its nominal footprint, an unnamed one none to speak of"): [[1], [0], []],
    ("usft", "single: one file has no neighbours"): [[]],
}


def main():
    from ground_classifier import tiles
    from ground_classifier.las import LasDataset, LasFile

    tmp = Path(tempfile.mkdtemp(prefix="neighbours_"))
    for unit in ("metres", "usft"):
        scale = 1.0 if unit == "metres" else FEET
        for name, halo_m, files, targets in scenarios(tmp, unit):
            dataset = LasDataset(sorted(files), units=unit)
            index = {source.path: i for i, source in enumerate(dataset.sources)}
            halo = tiles.halo_in_input_units(halo_m, scale)
            footprint = tiles.footprint_function(dataset, halo)
            chosen = list(dataset.sources) if targets is None else [dataset.file(files[0].parent / t) for t in targets]
            found = dataset.neighbours(halo, targets=chosen, footprint=footprint)
            got = [[index[f.path] for f in found[c.path]] for c in chosen]
            ok(f"{unit} {name}", got == GOLDEN[(unit, name)], str(got) if got != GOLDEN[(unit, name)] else "")

    # the API
    unit_files = scenarios(tmp, "metres")
    block = unit_files[0][2]
    dataset = LasDataset(sorted(block), units="metres")
    found = dataset.neighbours(10.0)
    ok("api: by default the header extents, for every source; the result is keyed by path",
       list(found) == [s.path for s in dataset.sources] and all(isinstance(f, LasFile) for v in found.values() for f in v))
    ok("api: full tiles inset by 1 m, a 10 m halo: touching tiles and diagonals", len(found[dataset.sources[4].path]) == 8)
    # the tiles are 2 m apart. The search is pre-filtered by bounding circles (as before the move), so a corner-only
    # neighbour needs a little more than the gap: 4 edge neighbours at 2.5 m, all 8 at 3 m, none at 1.5 m
    mid_counts = [len(dataset.neighbours(d)[dataset.sources[4].path]) for d in (1.5, 2.5, 3.0)]
    ok("api: 1.5 m halo reaches none of the tiles 2 m away, 2.5 m the 4 edges, 3 m all 8", mid_counts == [0, 4, 8], str(mid_counts))
    pair = LasDataset([write_tile(tmp / "diag" / "a.las", (0.0, 0.0, 100.0, 100.0)), write_tile(tmp / "diag" / "b.las", (108.0, 108.0, 208.0, 208.0)),
                       write_tile(tmp / "diag" / "c.las", (115.0, 115.0, 215.0, 215.0))], units="metres")
    near = pair.neighbours(12.0)
    ok("api: a diagonal neighbour 8 m off in both axes is within a 12 m halo, one 15 m off is not",
       [f.path.name for f in near[pair.sources[0].path]] == ["b.las"] and [f.path.name for f in near[pair.sources[1].path]] == ["a.las", "c.las"],
       str({k.name: [f.path.name for f in v] for k, v in near.items()}))
    ok("api: neighbours are sorted by path", all([str(f.path) for f in v] == sorted(str(f.path) for f in v) for v in found.values()))
    ok("api: a target may be given as a path or a member", dataset.neighbours(10.0, targets=[dataset.sources[0].path, dataset.sources[1]]).keys() == {dataset.sources[0].path, dataset.sources[1].path})
    ok("api: a target that is not a member is a KeyError", raised(KeyError, lambda: dataset.neighbours(10.0, targets=[tmp / "nope.las"])))
    ok("api: a distance of zero or less finds nothing, for every target", all(v == () for d in (0.0, -5.0) for v in dataset.neighbours(d).values()))
    counted = []
    def counting(source):
        counted.append(source.path)
        return source.header.extent_xy
    dataset.neighbours(10.0, footprint=counting)
    ok("api: a footprint is evaluated once per source", sorted(counted) == sorted(s.path for s in dataset.sources))

    # tiles
    ok("tiles: the nominal tile is 500 for metres and 1500 for any feet", [tiles.nominal_tile_size(s) for s in (1.0, FEET, 0.3048)] == [500.0, 1500.0, 1500.0])
    ok("tiles: a halo in the files' unit", abs(tiles.halo_in_input_units(25.0, FEET) - 25.0 / FEET) < 1e-12 and tiles.halo_in_input_units(25.0, 1.0) == 25.0 and tiles.halo_in_input_units(0.0, FEET) == 0.0)
    ok("tiles: numeric names", [bool(tiles.NUMERIC_TILE_NAME.match(n)) for n in ("468500_4966000", "-5_2.5_x", "tile", "1_2_3_4", "12_a")] == [True, True, False, True, False])
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{fails} failed" if fails else "\nall passed")
    sys.exit(1 if fails else 0)


def raised(exception, call):
    try:
        call()
    except exception:
        return True
    return False


if __name__ == "__main__":
    main()
