"""Neighbouring-tile context: the halo caches.

For every target tile the points of the tiles around it that lie within the halo are needed. Each neighbour file is read
once per run, in full, into a temporary raw cache holding just the points any target asked for, so a tile that borders
several targets is not decompressed several times. A cache is memory-mapped and filtered again for each target.

While a file is streamed it is also counted on a coarse XY grid (`SourceDensity`). That is for scheduling only: the
scheduler uses it to bound how many points a window will hold. Nothing here changes a classification.

Everything is in the files' own coordinate unit; the halo is a distance in that unit (see tiles.halo_in_input_units).
"""
from __future__ import annotations

import math
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .las import LasFile
from .processes import dead_helper_exit_codes, disable_helpers, helper_processes_work, pool_processes
from .utils.loggers import log

HALO_CACHE_CHUNK_POINTS = 1_000_000
HALO_CACHE_BYTES_PER_POINT = 25

# Scheduling-only point-density grid. While a file is streamed for the
# neighbour cache its points are also counted on a coarse XY grid, so the
# scheduler knows how many points each processing window of that file will
# really hold instead of assuming the worst case. Counts are taken over whole
# grid cells (an over-count, never an under-count) and a further margin is
# added. None of this is used by classification.
DENSITY_GRID_CELLS = 64
DENSITY_ESTIMATE_MARGIN = 1.15

Bounds = tuple[float, float, float, float]


@dataclass(frozen=True)
class SourceDensity:
    """Coarse XY point counts of one file, in the file's native unit.

    ``counts`` is row-major: index ``ix * cells + iy``."""

    bounds: Bounds
    cells: int
    counts: tuple[int, ...]


@dataclass(frozen=True)
class HaloCacheInfo:
    path: Path
    point_count: int
    density: SourceDensity | None = None


HALO_CACHE_DTYPE = np.dtype(
    [
        ("x", "<f8"),
        ("y", "<f8"),
        ("z", "<f8"),
        ("classification", "u1"),
    ]
)


def expand_bounds(bounds: Bounds, halo: float) -> Bounds:
    """`bounds` grown by `halo` on every side."""
    xmin, ymin, xmax, ymax = bounds
    return (xmin - halo, ymin - halo, xmax + halo, ymax + halo)


def create_halo_cache_worker(
    source_path_text: str,
    requested_bounds: tuple[Bounds, ...],
    cache_path_text: str,
) -> HaloCacheInfo:
    """Stream requested halo regions from one source into a raw cache."""
    source = LasFile.from_file(source_path_text)
    cache_path = Path(cache_path_text)
    point_count = 0
    cells = DENSITY_GRID_CELLS
    density_counts: np.ndarray | None = None
    density_bounds: Bounds | None = None
    with cache_path.open("wb") as output_stream:
        try:
            gxmin, gymin, gxmax, gymax = source.header.extent_xy
            if (
                all(
                    math.isfinite(value)
                    for value in (gxmin, gymin, gxmax, gymax)
                )
                and gxmax > gxmin
                and gymax > gymin
            ):
                density_bounds = (gxmin, gymin, gxmax, gymax)
                density_counts = np.zeros(cells * cells, dtype=np.int64)
        except Exception:
            density_counts = None
        for points in source.iter_record_batches(HALO_CACHE_CHUNK_POINTS):
            px = np.asarray(points.x, dtype=np.float64)
            py = np.asarray(points.y, dtype=np.float64)
            if density_counts is not None and density_bounds is not None:
                # Scheduling aid only; see DENSITY_GRID_CELLS.
                cell_x = np.clip(
                    np.floor(
                        (px - density_bounds[0])
                        * (cells / (density_bounds[2] - density_bounds[0]))
                    ),
                    0,
                    cells - 1,
                ).astype(np.int64)
                cell_y = np.clip(
                    np.floor(
                        (py - density_bounds[1])
                        * (cells / (density_bounds[3] - density_bounds[1]))
                    ),
                    0,
                    cells - 1,
                ).astype(np.int64)
                density_counts += np.bincount(
                    cell_x * cells + cell_y,
                    minlength=cells * cells,
                )
            selected = np.zeros(px.size, dtype=bool)
            for xmin, ymin, xmax, ymax in requested_bounds:
                selected |= (
                    (px >= xmin)
                    & (px <= xmax)
                    & (py >= ymin)
                    & (py <= ymax)
                )
            count = int(np.count_nonzero(selected))
            if count == 0:
                continue
            records = np.empty(count, dtype=HALO_CACHE_DTYPE)
            records["x"] = px[selected]
            records["y"] = py[selected]
            records["z"] = np.asarray(
                points.z, dtype=np.float64
            )[selected]
            records["classification"] = np.asarray(
                points.classification, dtype=np.uint8
            )[selected]
            records.tofile(output_stream)
            point_count += count
    density = (
        SourceDensity(
            density_bounds,
            cells,
            tuple(int(value) for value in density_counts),
        )
        if density_counts is not None and density_bounds is not None
        else None
    )
    return HaloCacheInfo(cache_path, point_count, density)


def read_halo_source_named(
    source: Path,
    bounds: tuple[Bounds, ...],
    cache_path: Path,
    announce: bool = False,
) -> HaloCacheInfo:
    """In-process halo read whose errors always name the file involved."""
    if announce:
        log(f"  Reading neighbour context: {source.name}")
    try:
        return create_halo_cache_worker(str(source), bounds, str(cache_path))
    except Exception as exc:
        raise RuntimeError(
            f"Could not read neighbouring tile {source}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def finish_halo_caches_in_process(
    unfinished: list,
    total_jobs: int,
    crashed_processes: list,
    workers: int,
) -> dict[Path, HaloCacheInfo]:
    """Recover after a helper process died while building halo caches.

    Reports why the helper died as far as the operating system will say,
    works out whether helper processes can run on this machine at all, and
    reads the remaining files inside the main process so that a file-level
    problem is named instead of surfacing as an anonymous broken pool.
    """
    codes = dead_helper_exit_codes(crashed_processes)
    log(
        f"WARNING: a helper process died while reading neighbouring tiles; "
        f"{len(unfinished):,} of {total_jobs:,} file(s) were not read. "
        + (
            f"Helper exit code(s): {', '.join(codes)}."
            if codes
            else "The operating system reported no exit code."
        )
    )

    # A different cache name: the dead pool may have left a partial file.
    retry_jobs = [
        (source, bounds, cache_path.with_name(f"{cache_path.stem}_retry.bin"))
        for source, bounds, cache_path in unfinished
    ]
    caches: dict[Path, HaloCacheInfo] = {}

    if helper_processes_work():
        log(
            "  DIAGNOSIS: a fresh helper process starts and runs normally, "
            "so the failure is tied to reading these files (or to several "
            "helpers running at once). Re-reading the remaining files one "
            "at a time in the main process; if the run stops again, the "
            "last file named below is the one that was being read."
        )
        for source, bounds, cache_path in retry_jobs:
            caches[source] = read_halo_source_named(
                source, bounds, cache_path, announce=True
            )
        return caches

    disable_helpers()
    log(
        "  DIAGNOSIS: a fresh helper process cannot run even a trivial task "
        "on this machine, so no input file is involved. Something on the "
        "machine is stopping Python helper processes (security software, a "
        "job or policy limit, or the Python installation itself). "
        "Continuing without helper processes: slower, identical results."
    )
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = {
            executor.submit(
                read_halo_source_named, source, bounds, cache_path, True
            ): source
            for source, bounds, cache_path in retry_jobs
        }
        try:
            while pending:
                completed, _ = wait(
                    tuple(pending), return_when=FIRST_COMPLETED
                )
                for future in completed:
                    source = pending.pop(future)
                    caches[source] = future.result()
        except BaseException:
            for future in pending:
                future.cancel()
            raise
    return caches


def build_halo_caches(
    source_requests: dict[Path, set[Bounds]],
    cache_directory: Path,
    worker_count: int,
) -> dict[Path, HaloCacheInfo]:
    """Build each reusable neighbouring-tile cache once per run."""
    ordered_sources = sorted(source_requests, key=str)
    if not ordered_sources:
        return {}

    cache_directory.mkdir(parents=True, exist_ok=True)
    jobs = [
        (
            source,
            tuple(sorted(source_requests[source])),
            cache_directory / f"halo_{position:06d}.bin",
        )
        for position, source in enumerate(ordered_sources)
    ]
    caches: dict[Path, HaloCacheInfo] = {}
    workers = max(1, min(worker_count, len(jobs)))
    log(
        f"Preparing 10 m-equivalent neighbouring-tile context from "
        f"{len(jobs)} source file(s) with {workers} worker(s)..."
    )

    if workers == 1:
        for source, bounds, cache_path in jobs:
            caches[source] = read_halo_source_named(source, bounds, cache_path)
        return caches

    # A helper that dies outright (as opposed to raising a Python error)
    # breaks the whole pool and fails every unfinished job with an anonymous
    # BrokenProcessPool. Collect those jobs and finish them in-process
    # instead of aborting the batch.
    unfinished: list = []
    crashed_processes: list = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        pending = {
            executor.submit(
                create_halo_cache_worker,
                str(source),
                bounds,
                str(cache_path),
            ): (source, bounds, cache_path)
            for source, bounds, cache_path in jobs
        }
        try:
            while pending:
                completed, _ = wait(
                    tuple(pending), return_when=FIRST_COMPLETED
                )
                for future in completed:
                    job = pending.pop(future)
                    source = job[0]
                    try:
                        caches[source] = future.result()
                    except BrokenProcessPool:
                        if not crashed_processes:
                            crashed_processes = pool_processes(executor)
                        unfinished.append(job)
                    except Exception as exc:
                        raise RuntimeError(
                            f"Could not read neighbouring tile {source}: "
                            f"{type(exc).__name__}: {exc}"
                        ) from exc
        except BaseException:
            # Do not make the user wait for every queued file before the
            # error is shown.
            for future in pending:
                future.cancel()
            raise

    if unfinished:
        caches.update(
            finish_halo_caches_in_process(
                unfinished, len(jobs), crashed_processes, workers
            )
        )
    return caches


def load_halo_points(
    caches: tuple[HaloCacheInfo, ...],
    target_bounds: Bounds,
    halo: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load only cached returns within `halo` (the file's unit) of this target file."""
    expanded = expand_bounds(target_bounds, halo)
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    z_parts: list[np.ndarray] = []
    class_parts: list[np.ndarray] = []
    for info in caches:
        if info.point_count == 0:
            continue
        records = np.memmap(
            info.path,
            dtype=HALO_CACHE_DTYPE,
            mode="r",
            shape=(info.point_count,),
        )
        selected = (
            (records["x"] >= expanded[0])
            & (records["x"] <= expanded[2])
            & (records["y"] >= expanded[1])
            & (records["y"] <= expanded[3])
        )
        if np.any(selected):
            x_parts.append(np.asarray(records["x"][selected]))
            y_parts.append(np.asarray(records["y"][selected]))
            z_parts.append(np.asarray(records["z"][selected]))
            class_parts.append(
                np.asarray(records["classification"][selected])
            )
        del records

    if not x_parts:
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.uint8),
        )
    return (
        np.concatenate(x_parts),
        np.concatenate(y_parts),
        np.concatenate(z_parts),
        np.concatenate(class_parts).astype(np.uint8, copy=False),
    )


def density_points_in_rect(
    density: SourceDensity,
    rect: Bounds,
) -> int:
    """Upper bound on a file's points inside ``rect`` (native units).

    Every grid cell the rectangle touches is counted in full, so the result
    can exceed the true count but never fall below it."""
    gxmin, gymin, gxmax, gymax = density.bounds
    xmin, ymin, xmax, ymax = rect
    if xmax < gxmin or xmin > gxmax or ymax < gymin or ymin > gymax:
        return 0
    cells = density.cells
    scale_x = cells / (gxmax - gxmin)
    scale_y = cells / (gymax - gymin)
    first_x = min(cells - 1, max(0, int(math.floor((xmin - gxmin) * scale_x))))
    last_x = min(cells - 1, max(0, int(math.floor((xmax - gxmin) * scale_x))))
    first_y = min(cells - 1, max(0, int(math.floor((ymin - gymin) * scale_y))))
    last_y = min(cells - 1, max(0, int(math.floor((ymax - gymin) * scale_y))))
    counts = density.counts
    total = 0
    for cell_x in range(first_x, last_x + 1):
        row = cell_x * cells
        total += sum(counts[row + first_y : row + last_y + 1])
    return int(total)
