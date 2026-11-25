"""Per-window TIN (PLY) and ground-raster (GeoTIFF) exports, and the settings that say where and how they are written."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import __version__, dtm, ply
from .las import UNIT_OPTION_NAMES, unit_for_scale
from .utils.params import RunSettings
from .windows import core_point_mask


def default_dtm_cell(unit_scale: float) -> float:
    """1 m for a metric file, 3 ft for a feet file (in the file's own unit)."""
    return 1.0 if abs(unit_scale - 1.0) < 1.0e-9 else 3.0


@dataclass(frozen=True)
class DtmExport:
    """Where and how one file's window rasters are written."""

    directory: Path
    stem: str               # file name (without extension) of the input
    unit_scale: float       # metres per input unit: rasters are written in the input's own unit
    cell: float             # cell size in the input unit
    extent: tuple[float, float, float, float]  # the file's windows together, in metres


@dataclass(frozen=True)
class TinExport:
    """Where and how one file's window TINs are written."""

    directory: Path
    stem: str               # file name (without extension) of the input
    unit_scale: float       # metres per input unit: meshes are written in the input's own unit
    max_edge: float = 0.0   # metres; triangles with a longer edge are left out (0 = keep all)


def file_exports(
    settings: RunSettings,
    stem: str,
    unit_scale: float,
    cores: list[tuple[float, float, float, float]],
) -> tuple[TinExport | None, DtmExport | None]:
    """The TIN and DTM exports a run asks for, for the file `stem` whose windows have these `cores` (metres)."""
    tin_export = (
        TinExport(settings.tin_export_dir, stem, unit_scale, settings.tin_max_edge)
        if settings.tin_export_dir is not None
        else None
    )
    dtm_export = (
        DtmExport(
            settings.dtm_export_dir,
            stem,
            unit_scale,
            settings.dtm_cell or default_dtm_cell(unit_scale),
            (
                min(core[0] for core in cores),
                min(core[1] for core in cores),
                max(core[2] for core in cores),
                max(core[3] for core in cores),
            ),
        )
        if settings.dtm_export_dir is not None
        else None
    )
    return tin_export, dtm_export


def export_window_tin(
    export: TinExport,
    window_number: int,
    core: tuple[float, float, float, float],
    faces: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    overall_xmax: float,
    overall_ymax: float,
) -> Path | None:
    """Write the part of a window's TIN that lies in its core as a PLY mesh.

    faces index x, y, z (the window's points, in metres). A triangle belongs
    to the window whose core holds its centroid, so neighbouring windows
    write each triangle once and no vertex is added at the boundary. The
    mesh is written in the input's own unit. Returns the path, or None if
    no triangle is left.
    """
    if faces.size:
        centre_x = x[faces].mean(axis=1)
        centre_y = y[faces].mean(axis=1)
        faces = faces[
            core_point_mask(centre_x, centre_y, core, overall_xmax, overall_ymax)
        ]
    if not faces.size:
        return None
    scale = export.unit_scale
    xyz = np.column_stack((x / scale, y / scale, z / scale))
    export.directory.mkdir(parents=True, exist_ok=True)
    path = export.directory / f"{export.stem}_w{window_number:02d}.ply"
    ply.write_mesh(
        path,
        xyz,
        faces,
        (
            f"ground-classifier {__version__} TIN of {export.stem}, window {window_number}",
            "triangles whose centroid is in the window core; surface vertices only",
            f"coordinates in the input unit ({scale:.10g} metres per unit)",
        ),
    )
    return path


def export_window_dtm(
    export: DtmExport,
    window_number: int,
    core: tuple[float, float, float, float],
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    ground: np.ndarray,
) -> Path | None:
    """Write the ground raster of a window's core: the lowest ground point
    (final ground class) of each cell, nodata where a cell has none.

    x, y, z are the window's points in metres and ground flags the ground
    ones. The raster is written in the input's own unit on a grid fixed to
    the coordinates (see dtm.py). A cell belongs to the window whose core
    holds its centre, so windows write no cell twice; the cell's points
    come from the whole buffered window. Returns the path, or None if the
    core has no ground point.
    """
    scale = export.unit_scale
    # Back to the file's unit. Rounding removes the error of the round trip
    # through metres, which could put a point on a cell edge in the next cell.
    gx, gy, gz = (np.round(v[ground] / scale, 6) for v in (x, y, z))
    xmin, ymin, xmax, ymax = export.extent
    cxmin, cymin, cxmax, cymax = core
    cells = dtm.owned_cells(
        tuple(c / scale for c in core),
        export.cell,
        first_x=cxmin <= xmin,
        first_y=cymin <= ymin,
        last_x=cxmax >= xmax,
        last_y=cymax >= ymax,
    )
    grid = dtm.lowest_ground(gx, gy, gz, export.cell, cells)
    if not (grid != dtm.NODATA).any():
        return None
    export.directory.mkdir(parents=True, exist_ok=True)
    path = export.directory / f"{export.stem}_w{window_number:02d}.tif"
    dtm.write_dtm(
        path,
        grid,
        cells,
        export.cell,
        UNIT_OPTION_NAMES[unit_for_scale(scale)],
        (
            f"ground-classifier {__version__} DTM of {export.stem}, window {window_number}",
            "lowest ground point of each cell; cells owned by this window's core",
        ),
    )
    return path
