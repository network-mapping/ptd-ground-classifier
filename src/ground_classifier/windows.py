"""How a tile is cut into windows: the layout of the cores, the buffer read around each, and which points a core owns.

Pure geometry on arrays (metres). The block size and overlap are arguments (the `windows` section of the config file,
`RunSettings.block_size` and `block_overlap`), so nothing here depends on a run being configured.

Every window reprocesses a `block_overlap`-wide buffer on each side, so the area a window computes is
(core + 2 * block_overlap)^2, not core^2. With an overlap of 80 m a core near 120 m is buffered out to ~280 m per side,
about 5x its own area. That overhead was measured to make small-window splits SLOWER than one window despite running in
parallel, so TURBO_MIN_CORE_SIZE is well above the overlap and keeps the buffered-to-core area ratio at roughly <= 3x.
The overlap should stay larger than the 60 m initial seed grid. The window count is always limited by the real tile
geometry via TURBO_MIN_CORE_SIZE.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

TURBO_DENSE_WINDOWS = True
TURBO_TARGET_CORE_POINTS = 120_000_000
TURBO_MAX_INTERNAL_WINDOWS = 8
TURBO_MIN_CORE_SIZE = 220.0

# Conservative peak-memory estimate of one window, used only to decide concurrency. It does not allocate memory and does
# not alter classification.
WINDOW_FIXED_MEMORY_MB = 256.0
WINDOW_BYTES_PER_POINT = 64


def estimate_window_memory(point_count: int) -> int:
    """Conservative peak working-memory estimate for one buffered window."""
    return int(
        WINDOW_FIXED_MEMORY_MB * 1024**2
        + max(0, int(point_count)) * WINDOW_BYTES_PER_POINT
    )


def processing_windows(
    x: np.ndarray,
    y: np.ndarray,
    block_size: float,
) -> Iterable[tuple[int, tuple[float, float, float, float]]]:
    """Yield non-overlapping core windows using extent AND point count."""
    yield from window_cores_for_extent(
        float(np.min(x)),
        float(np.min(y)),
        float(np.max(x)),
        float(np.max(y)),
        int(x.size),
        block_size,
    )


def window_cores_for_extent(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    point_count: int,
    block_size: float,
) -> Iterable[tuple[int, tuple[float, float, float, float]]]:
    """The window layout itself, from an XY extent (metres) and point count.

    Separated from processing_windows() only so the scheduler can ask what
    layout a file WILL get from its header, before the file is read.
    """
    width = max(xmax - xmin, 1.0e-9)
    height = max(ymax - ymin, 1.0e-9)

    physical_nx = max(1, int(math.ceil(width / block_size)))
    physical_ny = max(1, int(math.ceil(height / block_size)))
    physical_windows = physical_nx * physical_ny

    point_windows = (
        max(
            1,
            int(math.ceil(
                point_count / max(TURBO_TARGET_CORE_POINTS, 1)
            )),
        )
        if TURBO_DENSE_WINDOWS
        else 1
    )
    desired_windows = min(
        max(physical_windows, point_windows),
        TURBO_MAX_INTERNAL_WINDOWS,
    )

    if desired_windows <= 1:
        yield 1, (xmin, ymin, xmax, ymax)
        return

    aspect = width / max(height, 1.0e-9)
    nx = max(
        physical_nx,
        int(math.ceil(math.sqrt(desired_windows * aspect))),
    )
    ny = max(
        physical_ny,
        int(math.ceil(desired_windows / max(nx, 1))),
    )

    while nx > physical_nx and width / nx < TURBO_MIN_CORE_SIZE:
        nx -= 1
    while ny > physical_ny and height / ny < TURBO_MIN_CORE_SIZE:
        ny -= 1

    while (
        nx * ny < desired_windows
        and nx * ny < TURBO_MAX_INTERNAL_WINDOWS
    ):
        can_x = width / (nx + 1) >= TURBO_MIN_CORE_SIZE
        can_y = height / (ny + 1) >= TURBO_MIN_CORE_SIZE
        if can_x and (not can_y or width / nx >= height / ny):
            nx += 1
        elif can_y:
            ny += 1
        else:
            break

    x_step = width / nx
    y_step = height / ny
    number = 0

    for iy in range(ny):
        core_ymin = ymin + iy * y_step
        core_ymax = (
            ymax if iy == ny - 1
            else ymin + (iy + 1) * y_step
        )
        for ix in range(nx):
            core_xmin = xmin + ix * x_step
            core_xmax = (
                xmax if ix == nx - 1
                else xmin + (ix + 1) * x_step
            )
            number += 1
            yield number, (
                core_xmin,
                core_ymin,
                core_xmax,
                core_ymax,
            )


def buffered_indices_for_window(
    x: np.ndarray,
    y: np.ndarray,
    core: tuple[float, float, float, float],
    x_order: np.ndarray | None,
    sorted_x: np.ndarray | None,
    block_overlap: float,
) -> np.ndarray:
    """Return stable source-order indices for one overlap-buffered window."""
    cxmin, cymin, cxmax, cymax = core
    bxmin = cxmin - block_overlap
    bymin = cymin - block_overlap
    bxmax = cxmax + block_overlap
    bymax = cymax + block_overlap

    if x_order is None or sorted_x is None:
        return np.flatnonzero(
            (x >= bxmin)
            & (x <= bxmax)
            & (y >= bymin)
            & (y <= bymax)
        )

    left = np.searchsorted(sorted_x, bxmin, side="left")
    right = np.searchsorted(sorted_x, bxmax, side="right")
    x_candidates = x_order[left:right]
    return np.sort(
        x_candidates[
            (y[x_candidates] >= bymin)
            & (y[x_candidates] <= bymax)
        ]
    )


def core_point_mask(
    px: np.ndarray,
    py: np.ndarray,
    core: tuple[float, float, float, float],
    overall_xmax: float,
    overall_ymax: float,
) -> np.ndarray:
    """Which of the positions lie in the window's core. Cores are half-open
    (lower edge in, upper edge out) except at the file's own upper edge, so
    every position belongs to exactly one window."""
    cxmin, cymin, cxmax, cymax = core
    x_upper = px <= cxmax if cxmax >= overall_xmax else px < cxmax
    y_upper = py <= cymax if cymax >= overall_ymax else py < cymax
    return (px >= cxmin) & x_upper & (py >= cymin) & y_upper
