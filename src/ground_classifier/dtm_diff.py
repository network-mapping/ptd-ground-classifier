"""Subtract the DTM rasters of one folder from those of another.

    ground-classify dtm-diff <A> <B> <output>

<A> and <B> are folders of GeoTIFFs written by --export-dtm (for example the same tiles classified two ways).
<output> is a new folder that receives one raster per file, with the same name, whose height is B - A: positive where
B is higher. Swap A and B for the other sign.

The two folders must hold files of exactly the same names, and every pair must be on the same grid: the same cell size
and the same unit (the heights are never converted). Anything else is an error, reported before any file is written.
A pair need not cover the same area: the result covers both, and a cell has a value only where both have one (a cell
where either has none is nodata).

For every file it reports the number of cells compared and the difference (mean, RMS, 95th percentile and largest of
its size) in the unit of the rasters. Nothing in <A> or <B> is changed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from . import __version__, dtm


def collect(folder: Path) -> dict[str, Path]:
    """The rasters of a folder by file name (not looking into sub-folders)."""
    if not folder.is_dir():
        raise SystemExit(f"Not a folder: {folder}")
    found = {p.name: p for p in sorted(folder.glob("*.tif"))}
    if not found:
        raise SystemExit(f"No .tif rasters in {folder}")
    return found


def validate(a_files: dict[str, Path], b_files: dict[str, Path], a_folder: Path, b_folder: Path) -> None:
    """Raise SystemExit, naming every problem, unless the folders match name for name and grid for grid."""
    problems = []
    only_a = sorted(set(a_files) - set(b_files))
    only_b = sorted(set(b_files) - set(a_files))
    if only_a:
        problems.append(f"only in {a_folder}: {', '.join(only_a)}")
    if only_b:
        problems.append(f"only in {b_folder}: {', '.join(only_b)}")
    if not problems:
        for name in a_files:
            try:
                _, a_cell, a_unit, _ = dtm.read_dtm_info(a_files[name])
                _, b_cell, b_unit, _ = dtm.read_dtm_info(b_files[name])
            except ValueError as exc:
                problems.append(str(exc))
                continue
            if a_cell != b_cell:
                problems.append(f"{name}: cell size {a_cell:g} in {a_folder} but {b_cell:g} in {b_folder}")
            if a_unit != b_unit:
                problems.append(f"{name}: unit {a_unit!r} in {a_folder} but {b_unit!r} in {b_folder}")
    if problems:
        raise SystemExit("The two folders cannot be subtracted:\n  " + "\n  ".join(problems))


def describe(diff: np.ndarray, unit: str) -> str:
    valid = diff != dtm.NODATA
    if not valid.any():
        return "no cell has a value in both"
    d = diff[valid].astype(np.float64)
    size = np.abs(d)
    return (f"{d.size:,} cells compared: mean {d.mean():+.4f}, RMS {np.sqrt(np.mean(d ** 2)):.4f}, "
            f"95% within {np.percentile(size, 95):.4f}, largest {size.max():.4f} ({unit})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ground-classify dtm-diff",
        description="Write B - A for every DTM raster that two folders hold under the same name.",
    )
    parser.add_argument("a", type=Path, help="Folder of rasters to subtract (the 'before').")
    parser.add_argument("b", type=Path, help="Folder of rasters to subtract from (the 'after'): the result is B - A.")
    parser.add_argument("output", type=Path, help="New folder for the difference rasters; must not be A or B.")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    a_folder, b_folder, out = args.a.resolve(), args.b.resolve(), args.output.resolve()
    if out in (a_folder, b_folder):
        raise SystemExit(f"The output folder must be neither A nor B: {out}")
    a_files, b_files = collect(a_folder), collect(b_folder)
    validate(a_files, b_files, a_folder, b_folder)

    out.mkdir(parents=True, exist_ok=True)
    print(f"difference = B - A\nA = {a_folder}\nB = {b_folder}\noutput = {out}")
    for name in a_files:
        a, a_cells, cell, unit, _ = dtm.read_dtm(a_files[name])
        b, b_cells, _, _, _ = dtm.read_dtm(b_files[name])
        diff, cells = dtm.subtract(a, a_cells, b, b_cells)
        dtm.write_dtm(
            out / name, diff, cells, cell, unit,
            (f"ground-classifier {__version__} difference: height = B - A",
             f"A = {a_files[name]}", f"B = {b_files[name]}"),
        )
        print(f"{name}: {describe(diff, unit)}")
    print(f"{len(a_files)} raster(s) written.")
    return 0
