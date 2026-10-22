"""Ground rasters (DTM): lowest_ground(), write_dtm() and read_dtm().

A cell's value is the lowest ground point in it; a cell with no ground point holds `NODATA`. Nothing is interpolated.
The grid is fixed to the coordinates themselves: cell (i, j) covers x in [i * cell, (i + 1) * cell) and y in
[j * cell, (j + 1) * cell), whatever the extent of the data. Rasters of different blocks, tiles or runs therefore line
up cell for cell, and can be subtracted without resampling.

Everything is in the unit of the coordinates the caller gives (feet stay feet): the cell size, the origin and the
heights. Nothing here converts units.

Files are float32 GeoTIFFs (deflate-compressed, north up) with the georeferencing tags and no coordinate system, which
CloudCompare, QGIS and GDAL read. The cell size and unit are also kept in the image description, for `read_dtm`.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

try:
    import tifffile
except ImportError as exc:
    raise SystemExit("Missing Python package: tifffile\nInstall dependencies with: pip install -r requirements.txt") from exc

NODATA = -9999.0

# GeoTIFF tags: pixel size, the corner the image is tied to, the GeoKey directory, GDAL's nodata value.
_PIXEL_SCALE, _TIEPOINT, _GEO_KEYS, _GDAL_NODATA = 33550, 33922, 34735, 42113
_PIXEL_IS_AREA = 1


def owned_cells(core, cell, first_x=False, first_y=False, last_x=False, last_y=False):
    """The cells a block of a larger run writes: those whose centre lies in its core.

    Cores are half-open (lower edge in, upper edge out), so neighbouring blocks share no cell and leave none out. At
    the outer edge of the whole run (`first_*`, `last_*`) the cell that holds the edge is taken even when its centre
    lies beyond it, so that no point of the run is left without a cell.

    Parameters
    ----------
    core : (xmin, ymin, xmax, ymax)
        The block's core, in the unit of the coordinates.
    cell : float
        Cell size in the same unit.
    first_x, first_y, last_x, last_y : bool
        The core's lower / upper edge in x / y is an edge of the whole run.

    Returns
    -------
    (i0, i1, j0, j1) : cell columns i0..i1 and rows j0..j1, both inclusive; i1 < i0 or j1 < j0 if there are none."""
    xmin, ymin, xmax, ymax = core
    # The same expressions on both sides of a shared edge, so the cells on either side never overlap or leave a gap.
    def lower(edge, first):
        return math.floor(edge / cell) if first else math.ceil(edge / cell - 0.5)
    def upper(edge, last):
        return math.floor(edge / cell) if last else math.ceil(edge / cell - 0.5) - 1
    return lower(xmin, first_x), upper(xmax, last_x), lower(ymin, first_y), upper(ymax, last_y)


def lowest_ground(x, y, z, cell, cells) -> np.ndarray:
    """The lowest point of each cell of `cells` (from `owned_cells`), as a float32 array of shape (rows, columns) with
    row 0 the lowest y. Cells without a point hold NODATA. Points outside the cells are ignored."""
    i0, i1, j0, j1 = cells
    columns, rows = max(0, i1 - i0 + 1), max(0, j1 - j0 + 1)
    if not columns or not rows or not len(x):
        return np.full((rows, columns), NODATA, np.float32)
    i = np.floor(np.asarray(x) / cell).astype(np.int64) - i0
    j = np.floor(np.asarray(y) / cell).astype(np.int64) - j0
    inside = (i >= 0) & (i < columns) & (j >= 0) & (j < rows)
    lowest = np.full(rows * columns, np.inf)
    np.minimum.at(lowest, j[inside] * columns + i[inside], np.asarray(z, dtype=np.float64)[inside])
    lowest[np.isinf(lowest)] = NODATA
    return lowest.reshape(rows, columns).astype(np.float32)


def write_dtm(path, grid, cells, cell, unit="", comments=()) -> tuple[int, int]:
    """Write a raster made by `lowest_ground` as a GeoTIFF.

    Parameters
    ----------
    grid : (rows, columns) float32 array, row 0 the lowest y
    cells : (i0, i1, j0, j1) the cells it holds, as given to `lowest_ground`
    cell : float cell size
    unit : name of the unit of the coordinates and heights ("metres", "usft"), recorded in the file
    comments : strings recorded in the image description (one line each)

    Returns
    -------
    (columns, rows) written."""
    i0, _, j0, j1 = cells
    rows, columns = grid.shape
    if not rows or not columns:
        raise ValueError("an empty raster cannot be written")
    for text in comments:
        if "\n" in text or "\r" in text:
            raise ValueError("a comment must be a single line")
    description = "\n".join(["ground-classifier DTM", f"cell {cell!r}", f"unit {unit}", f"nodata {NODATA:g}",
                             *(f"comment {c}" for c in comments)])
    left, top = i0 * cell, (j1 + 1) * cell
    tags = [
        (_PIXEL_SCALE, "d", 3, (cell, cell, 0.0), True),
        (_TIEPOINT, "d", 6, (0.0, 0.0, 0.0, left, top, 0.0), True),
        (_GEO_KEYS, "H", 8, (1, 1, 0, 1, 1025, 0, 1, _PIXEL_IS_AREA), True),
        (_GDAL_NODATA, "s", 0, f"{NODATA:g}", True),
    ]
    tifffile.imwrite(path, np.ascontiguousarray(grid[::-1], dtype=np.float32), compression="zlib", description=description,
                     photometric="minisblack", extratags=tags, metadata=None)
    return columns, rows


def read_dtm(path) -> tuple[np.ndarray, tuple[int, int, int, int], float, str, list[str]]:
    """Read a file written by `write_dtm`: (grid, cells, cell, unit, comments), the grid with row 0 the lowest y and
    NODATA in the cells without a point.

    Raises ValueError for any other file."""
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        description = page.description or ""
        lines = description.split("\n")
        if lines[0] != "ground-classifier DTM":
            raise ValueError("not a file written by write_dtm")
        values = {k: v for k, _, v in (line.partition(" ") for line in lines[1:] if not line.startswith("comment "))}
        cell = float(values["cell"])
        scale = page.tags[_PIXEL_SCALE].value
        tie = page.tags[_TIEPOINT].value
        if not math.isclose(scale[0], cell) or not math.isclose(scale[1], cell):
            raise ValueError("pixel size and cell size disagree")
        grid = page.asarray()[::-1].copy()
    rows, columns = grid.shape
    i0 = round(tie[3] / cell)
    j1 = round(tie[4] / cell) - 1
    comments = [line[len("comment "):] for line in lines[1:] if line.startswith("comment ")]
    return grid, (i0, i0 + columns - 1, j1 - rows + 1, j1), cell, values.get("unit", ""), comments
