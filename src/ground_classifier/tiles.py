"""Tile footprints: the area a tile is taken to cover when looking for the tiles around it.

A file whose name is numeric (468500_4966000) is, by the convention of the data this tool was built for, the lower-left
corner of a nominal square tile: 1500 ft (US survey or international) for a feet project, 500 m for a metric one. Sparse
and sliver tiles often cover only a small part of their square, so using the nominal footprint keeps the neighbours
within the halo of the whole tile discoverable. The name is trusted only when the header extent fits that square;
otherwise, and for any other name, the footprint is the header extent.

These are the footprints `LasDataset.neighbours` is given. Everything here is in the files' own coordinate unit.
"""
from __future__ import annotations

import math
import re

from .las import LasDataset, LasFile

NUMERIC_TILE_NAME = re.compile(
    r"^(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)(?:$|_)"
)


def nominal_tile_size(metres_per_unit: float) -> float:
    """500 for a metric project, 1500 for a feet project, in the project's own unit."""
    return 500.0 if abs(metres_per_unit - 1.0) < 1.0e-6 else 1500.0


def halo_in_input_units(halo_m: float, metres_per_unit: float) -> float:
    """A distance in metres, in the files' own unit."""
    return halo_m / max(metres_per_unit, 1.0e-9)


def tile_footprint(
    source: LasFile, tile_size: float, tolerance: float
) -> tuple[float, float, float, float]:
    """The validated filename-derived tile footprint when there is one, else the header extent (xmin, ymin, xmax, ymax).

    `tolerance` is how far the data may reach outside the nominal square, in the files' unit."""
    data_bounds = source.header.extent_xy
    match = NUMERIC_TILE_NAME.match(source.path.stem)
    if match is None:
        return data_bounds

    origin_x = float(match.group(1))
    origin_y = float(match.group(2))
    nominal = (
        origin_x,
        origin_y,
        origin_x + tile_size,
        origin_y + tile_size,
    )
    if source.header.point_count == 0:
        return nominal

    # Only trust the naming convention when the header bounds actually fit
    # that square.  This avoids treating unrelated numeric filenames as map
    # tile origins.
    xmin, ymin, xmax, ymax = data_bounds
    finite = all(math.isfinite(value) for value in data_bounds)
    contained = (
        xmin >= nominal[0] - tolerance
        and ymin >= nominal[1] - tolerance
        and xmax <= nominal[2] + tolerance
        and ymax <= nominal[3] + tolerance
    )
    return nominal if finite and contained else data_bounds


def footprint_function(dataset: LasDataset, halo: float):
    """`LasFile -> footprint` for the files of `dataset`, with `halo` (in the dataset's unit) as the tolerance."""
    tile_size = nominal_tile_size(dataset.metres_per_unit)
    tolerance = max(1.0, halo)

    def footprint(source: LasFile) -> tuple[float, float, float, float]:
        return tile_footprint(source, tile_size, tolerance)

    return footprint
