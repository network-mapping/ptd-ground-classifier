"""Batch pipeline: reads LAS/LAZ tiles, classifies them with the compiled
progressive-TIN core (see fast_ground.py) and writes them back.

Input classes 2, 3, 4, 5 and 13 are reset and re-labelled; every other class is
kept. HAG = height above the finished ground surface, in metres:
    2  Ground
    3  Low vegetation:    HAG <= low_veg_max
    4  Medium vegetation: low_veg_max < HAG < medium_veg_max
    5  High vegetation:   medium_veg_max <= HAG <= high_veg_max
    7  Isolated low points, and HAG > high_veg_max

Every threshold is in metres. Input X/Y/Z are converted to metres once, in
memory, in classify_file(), using the file's resolved unit (see units.py); the
LAS coordinates on disk are never rewritten.
"""

from __future__ import annotations

import faulthandler
import math
import os
import re
import sys
import tempfile
import time
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# If a compiled library crashes the interpreter, print where it happened.
# Module-level so spawned helper processes get it too.
try:
    faulthandler.enable()
except Exception:  # no usable stderr; diagnostics only, never fatal
    pass

# Small batched solves gain nothing from a BLAS thread pool, and several files
# or windows run at once; cKDTree parallelism is controlled explicitly below.
for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

try:
    import laspy
    import numpy as np
    from scipy.spatial import cKDTree
except ImportError as exc:
    missing = getattr(exc, "name", "a required package")
    raise SystemExit(
        f"Missing Python package: {missing}\n"
        "Install dependencies with: pip install -r requirements.txt"
    ) from exc

from . import __version__, dtm, fast_ground, ply
from .units import METRES_PER_US_SURVEY_FOOT, metres_to_usft
from .utils.loggers import _LOG_STATE, log
from .utils.params import (
    DEFAULT_HIGH_VEG_MAX,
    DEFAULT_LOW_VEG_MAX,
    DEFAULT_MEDIUM_VEG_MAX,
)

# Written into every output header so a file shows which version made it.
GENERATING_SOFTWARE = f"ground-classifier {__version__}"

# Settings handed to fast_ground.GroundRun (metres); set per process by
# configure_engine_settings() because spawned helpers do not inherit them.
FAST_ENGINE_SETTINGS = fast_ground.Settings()


def configure_engine_settings(settings: fast_ground.Settings) -> None:
    global FAST_ENGINE_SETTINGS
    FAST_ENGINE_SETTINGS = settings
# A point the low-point filter kept out of the surface is labelled noise (7)
# only if it ends up at least this far below the finished ground (metres).
FAST_ENGINE_LOW_NOISE_DEPTH = 0.5



LOW_VEG_MAX = DEFAULT_LOW_VEG_MAX
MEDIUM_VEG_MAX = DEFAULT_MEDIUM_VEG_MAX
HIGH_VEG_MAX = DEFAULT_HIGH_VEG_MAX

def configure_hag_bands(low: float, medium: float, high: float) -> None:
    """Set the process-local HAG thresholds. Must be re-called at the start
    of every worker process (Windows multiprocessing spawns fresh
    interpreters that do not inherit the parent's mutated globals) -
    mirrors the existing configure_runtime() pattern."""
    global LOW_VEG_MAX, MEDIUM_VEG_MAX, HIGH_VEG_MAX
    LOW_VEG_MAX = low
    MEDIUM_VEG_MAX = medium
    HIGH_VEG_MAX = high

# Optional export of each window's TIN as a PLY mesh (see export_window_tin()).
# Set per process by configure_tin_export() because spawned helpers do not
# inherit it. None = no export; the edge limit is in metres, 0 = no limit.
TIN_EXPORT_DIR: Path | None = None
TIN_EXPORT_MAX_EDGE = 0.0


def configure_tin_export(directory: Path | None, max_edge: float = 0.0) -> None:
    global TIN_EXPORT_DIR, TIN_EXPORT_MAX_EDGE
    TIN_EXPORT_DIR = directory
    TIN_EXPORT_MAX_EDGE = max_edge


# Optional export of each window's ground raster (see export_window_dtm()).
# Set per process like the TIN export. The cell size is in the input unit;
# 0 = the default for the file's unit (see default_dtm_cell()).
DTM_EXPORT_DIR: Path | None = None
DTM_CELL = 0.0


def configure_dtm_export(directory: Path | None, cell: float = 0.0) -> None:
    global DTM_EXPORT_DIR, DTM_CELL
    DTM_EXPORT_DIR = directory
    DTM_CELL = cell


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


# Large-file block processing. The overlap is deliberately larger than the
# 60 m initial seed grid.
BLOCK_SIZE = 5000.0
BLOCK_OVERLAP = 80.0
# FPL performance profile.
#
# These control how many pieces a single tile is split into for internal
# window-level parallelism (see processing_windows()). Actual window count
# is always self-limited by real tile geometry via TURBO_MIN_CORE_SIZE.
#
# IMPORTANT: every window reprocesses an BLOCK_OVERLAP-wide buffer on each
# side, so the "buffered" area a window actually computes is
# (core + 2*BLOCK_OVERLAP)^2, not core^2. With BLOCK_OVERLAP=80 m, a core
# near the old 120 m floor is buffered out to ~280 m per side - about 5x its
# own area reprocessed. That overhead was measured directly (see test notes)
# to make small-window splits SLOWER than a single window despite running in
# parallel, not faster. TURBO_MIN_CORE_SIZE is therefore raised well above
# the overlap width so the buffered-to-core area ratio stays roughly <=3x.
# This is a moderate, conservative step (up from a hard 4-window cap) rather
# than a maximal one - true wall-clock benefit depends on real hardware and
# should be confirmed on an actual tile before pushing further.
TURBO_DENSE_WINDOWS = True
TURBO_TARGET_CORE_POINTS = 120_000_000
TURBO_MAX_INTERNAL_WINDOWS = 8
TURBO_MIN_CORE_SIZE = 220.0

OUTPUT_SUFFIX = "_AdaptiveGround"

PERFORMANCE_STAGE_TIMINGS = True

# Numeric tile names such as 468500_4966000 are interpreted as the lower-left
# corner of a nominal square tile in this build - 1500 US survey feet for a
# feet project, 500 m for a metric project (both conventions the user
# described directly; see configure_nominal_tile_size(), which sets the
# real value once the project's unit is resolved). Sparse/sliver LAS files
# often cover only a small part of a tile; using the nominal footprint keeps
# their 10 m-equivalent adjacent-tile context discoverable.
NUMERIC_TILE_NAME = re.compile(
    r"^(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)(?:$|_)"
)

# A progressive TIN is undefined for fewer than three usable, non-collinear
# points.  Such tiny/sliver tiles retain their incoming classification rather
# than aborting an otherwise successful large batch.  The limit prevents a
# genuine algorithm failure on a normal tile from being silently hidden.
SPARSE_FALLBACK_MAX_BUFFERED_POINTS = 10_000

# Resource-aware parallel processing. Independent input files use processes;
# independent windows of one long file use shared-memory threads. The two
# levels are never nested. Automatic limits deliberately leave CPU and RAM for
# Windows, TerraScan and other applications.
AUTO_PARALLEL = True
# Automatic mode may use up to 80% of logical CPUs, leaving the remainder for
# Windows, decompression, storage and interactive work. Six files is a ceiling,
# not a forced count: the scheduler still reduces concurrency for CPU and RAM.
CPU_BUDGET_FRACTION = 0.80
# Ceiling on concurrent file processes. The real number is whatever fits the
# CPU budget and the RAM budget (see the scheduler in main()); this is only a
# hard stop below the Windows limit of 61 workers per process pool.
MAX_PARALLEL_FILES = 60
# Concurrent readers while building neighbouring-tile caches. This step is
# disk-bound and a small share of the run, so it stays at its earlier value.
MAX_PARALLEL_HALO_READERS = 6
# Raised from a hard 4, but deliberately matched to TURBO_MAX_INTERNAL_WINDOWS
# (8) above rather than pushed to the full CPU budget - see the overlap-ratio
# note there. window_workers is still clamped to the real window count for
# this tile, so smaller/sparser tiles are unaffected.
MAX_PARALLEL_WINDOWS = 8
MAX_KD_TREE_WORKERS = 6
RAM_BUDGET_FRACTION = 0.55
MIN_FREE_RAM_GB = 4.0

# Conservative peak-memory estimates used only to decide concurrency. They do
# not allocate memory and do not alter classification. If a block is larger
# than the safe budget it still runs, but by itself.
WINDOW_FIXED_MEMORY_MB = 256.0
WINDOW_BYTES_PER_POINT = 64
FILE_FIXED_MEMORY_MB = 384.0
FULL_FILE_BYTES_PER_POINT = 96

# TerraScan-style neighbouring-tile context. For every target LAS/LAZ, points
# from spatially adjacent files are loaded only where they fall within this
# halo around the target bounds. Temporary halo caches mean each neighbour
# file is decompressed once per run rather than once for every adjacent tile.
NEIGHBOUR_HALO = metres_to_usft(10.0)  # 32.808333... USft = 10 m physical.
NEIGHBOUR_HALO_METRES = 10.0
# Re-derived per run by configure_neighbour_halo() once the project's real
# input unit is known (see detect_input_unit_scale) - the fixed feet value
# above is only the pre-configuration default, kept for any direct caller
# that never resolves a project unit.


def configure_neighbour_halo(
    metres_per_input_unit: float, halo_metres: float = 10.0
) -> None:
    """Set NEIGHBOUR_HALO to `halo_metres` (the config file's
    input.neighbour_halo, 10 by default) expressed in the project's real
    input unit. Must be called once the input unit is known, before any
    neighbour-tile discovery or halo-buffered classify_file work runs, and
    re-called at the start of any worker process that does not inherit this
    process's mutated globals (Windows multiprocessing spawn)."""
    global NEIGHBOUR_HALO, NEIGHBOUR_HALO_METRES
    NEIGHBOUR_HALO_METRES = float(halo_metres)
    NEIGHBOUR_HALO = NEIGHBOUR_HALO_METRES / max(metres_per_input_unit, 1.0e-9)


# Numeric tile filenames encode a project's standard tile footprint in its
# own native unit - 1500x1500 US survey feet for a feet project, 500x500 m
# for a metric project (both conventions described directly by the user).
# The value below is only the pre-configuration default (feet); like
# NEIGHBOUR_HALO, configure_nominal_tile_size() sets the real value once the
# project's unit is known, and must be re-called in any worker process that
# does not inherit this process's mutated globals.
NOMINAL_TILE_SIZE_USFT = 1500.0


def configure_nominal_tile_size(metres_per_input_unit: float) -> None:
    global NOMINAL_TILE_SIZE_USFT
    # metres_per_input_unit == 1.0 for a metres project (native tile = 500),
    # METRES_PER_US_SURVEY_FOOT for a feet project (native tile = 1500).
    NOMINAL_TILE_SIZE_USFT = (
        500.0 if abs(metres_per_input_unit - 1.0) < 1.0e-6 else 1500.0
    )


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
# Scheduling only: how much work N windows of one file get done when they
# run on parallel threads of one process, relative to one window alone
# (N ** 0.95: four windows count as about 3.7). Separate files in separate
# processes count in full. Two windows on two threads measured at very
# nearly twice the speed of one after the other (the heavy steps release
# Python's interpreter lock), hence a value close to 1. Used solely to
# choose between "more files at once" and "more windows at once" - it has
# no effect on any result.
WINDOW_THREAD_SCALING = 0.95

# scipy's cKDTree query is deterministic with multiple workers. Runtime setup
# divides the total CPU budget among active outer workers, avoiding nested
# thread storms while still accelerating the large neighbour searches.
KD_TREE_WORKERS = 1
@dataclass(frozen=True)
class WindowResult:
    """Core-only output returned by one buffered processing window."""

    number: int
    core_indices: np.ndarray
    classification: np.ndarray
    hag: np.ndarray
    model_count: int
    messages: tuple[str, ...]


@dataclass(frozen=True)
class FileResult:
    """What one classified file produced (for the run manifest)."""

    elapsed: float
    point_count: int
    class_counts: dict


@dataclass(frozen=True)
class MemorySnapshot:
    """Physical and currently available memory, in bytes."""

    total: int
    available: int


@dataclass(frozen=True)
class InputMetadata:
    path: Path
    point_count: int
    record_size: int
    bounds: tuple[float, float, float, float]
    neighbour_bounds: tuple[float, float, float, float]


class InsufficientGroundSupport(RuntimeError):
    """The available geometry cannot support a progressive ground TIN."""


@dataclass(frozen=True)
class SourceDensity:
    """Coarse XY point counts of one file, in the file's native unit.

    ``counts`` is row-major: index ``ix * cells + iy``."""

    bounds: tuple[float, float, float, float]
    cells: int
    counts: tuple[int, ...]


@dataclass(frozen=True)
class FileMemoryPlan:
    """Scheduling estimate for one target file (bytes; never allocated).

    resident: the file itself plus its neighbour context, held for as long
        as the file is being processed.
    window:   working memory of its largest processing window, as an upper
        bound (used wherever files have to fit beside each other).
    windows:  how many processing windows it will be split into.
    window_likely: the same window without the safety margin - the best
        guess, used only to predict how many windows a file run by itself
        will manage at once (that run is governed by exact window sizes
        once the file is loaded, not by this figure).
    """

    resident: int
    window: int
    windows: int
    window_likely: int = 0


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


def configure_runtime(kdtree_workers: int, prefix: str = "") -> None:
    """Set process-local runtime limits without changing classifications."""
    global KD_TREE_WORKERS
    KD_TREE_WORKERS = max(1, int(kdtree_workers))
    _LOG_STATE.prefix = prefix


def memory_snapshot() -> MemorySnapshot:
    """Return total/available RAM without adding a psutil dependency."""
    if sys.platform == "win32":
        try:
            import ctypes

            class MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong),
                    ("memory_load", ctypes.c_ulong),
                    ("total_physical", ctypes.c_ulonglong),
                    ("available_physical", ctypes.c_ulonglong),
                    ("total_page_file", ctypes.c_ulonglong),
                    ("available_page_file", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("available_virtual", ctypes.c_ulonglong),
                    ("available_extended_virtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatusEx()
            status.length = ctypes.sizeof(MemoryStatusEx)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(
                ctypes.byref(status)
            ):
                return MemorySnapshot(
                    int(status.total_physical),
                    int(status.available_physical),
                )
        except (AttributeError, OSError, ValueError):
            pass

    try:
        values: dict[str, int] = {}
        with open("/proc/meminfo", encoding="ascii") as stream:
            for line in stream:
                name, value = line.split(":", 1)
                values[name] = int(value.split()[0]) * 1024
        total = values.get("MemTotal", 0)
        available = values.get(
            "MemAvailable",
            values.get("MemFree", 0),
        )
        if total and available:
            return MemorySnapshot(total, available)
    except (OSError, ValueError, IndexError):
        pass

    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        total = int(os.sysconf("SC_PHYS_PAGES")) * page_size
        available = int(os.sysconf("SC_AVPHYS_PAGES")) * page_size
        if total and available:
            return MemorySnapshot(total, available)
    except (AttributeError, OSError, ValueError):
        pass

    # A deliberately cautious fallback for uncommon platforms.
    fallback = 8 * 1024**3
    return MemorySnapshot(fallback, fallback // 2)


def safe_parallel_memory_budget(user_limit_gb: float = 0.0) -> int:
    """Return a guarded budget for concurrently active classification jobs."""
    snapshot = memory_snapshot()
    reserve = int(MIN_FREE_RAM_GB * 1024**3)
    after_reserve = max(0, snapshot.available - reserve)
    automatic = min(
        after_reserve,
        int(snapshot.total * RAM_BUDGET_FRACTION),
    )
    if user_limit_gb > 0:
        # An explicit request replaces the automatic share of total RAM, in
        # either direction, but can never reach into the free-memory
        # reserve: on a machine dedicated to this run, asking for more than
        # the automatic 55% lets more files run at once.
        requested = int(user_limit_gb * 1024**3)
        automatic = min(after_reserve, requested)
    return max(256 * 1024**2, automatic)


def cpu_worker_budget(requested_workers: int = 0) -> int:
    """Return a bounded total worker budget for this invocation."""
    logical = max(1, os.cpu_count() or 1)
    if requested_workers > 0:
        return max(1, min(int(requested_workers), logical))
    return max(1, int(math.floor(logical * CPU_BUDGET_FRACTION)))


def estimate_window_memory(point_count: int) -> int:
    """Conservative peak working-memory estimate for one buffered window."""
    return int(
        WINDOW_FIXED_MEMORY_MB * 1024**2
        + max(0, int(point_count)) * WINDOW_BYTES_PER_POINT
    )


def format_memory(byte_count: int) -> str:
    return f"{byte_count / 1024**3:.1f} GB"


def reset_input_class_mask(classification: np.ndarray) -> np.ndarray:
    """Return the five-class reset mask without np.isin's setup overhead."""
    return (
        (classification == 2)
        | (classification == 3)
        | (classification == 4)
        | (classification == 5)
        | (classification == 13)
    )


# Output-header policy. TerraScan and some upstream workflows may leave a
# temporary Extra Bytes dimension named "Distance" in the input. It is not
# required by this classifier and should not be propagated to delivery files.
# All other input Extra Bytes dimensions and VLRs are preserved.
STRIP_OUTPUT_EXTRA_DIMENSIONS = ("Distance",)


def strip_unwanted_output_extra_dimensions(las: "laspy.LasData") -> tuple[str, ...]:
    """Remove only explicitly configured temporary Extra Bytes dimensions.

    Matching is case-insensitive, but the exact stored dimension name is passed
    back to laspy. ``remove_extra_dim`` updates both the point-record layout and
    the LAS Extra Bytes VLR, so a lone int32 Distance field is removed cleanly
    rather than merely hiding its descriptor.
    """
    configured = {name.casefold() for name in STRIP_OUTPUT_EXTRA_DIMENSIONS}
    present = [str(name) for name in las.point_format.extra_dimension_names]
    removed: list[str] = []
    for name in present:
        if name.casefold() not in configured:
            continue
        las.remove_extra_dim(name)
        removed.append(name)
    return tuple(removed)


def assign_hag_classes(
    classification: np.ndarray,
    remaining: np.ndarray,
    hag: np.ndarray,
) -> None:
    """Assign the three vegetation classes (3/4/5) from final HAG, using
    the project's configured HAG bands (see config.load_run_config() /
    configure_hag_bands()). HAG is always in metres regardless of the
    input file's own coordinate unit."""
    classification[remaining & (hag <= LOW_VEG_MAX)] = 3
    classification[
        remaining
        & (hag > LOW_VEG_MAX)
        & (hag < MEDIUM_VEG_MAX)
    ] = 4
    classification[
        remaining
        & (hag >= MEDIUM_VEG_MAX)
        & (hag <= HIGH_VEG_MAX)
    ] = 5


def classify_block(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    initial_classification: np.ndarray,
    tin_out: list[np.ndarray] | None = None,
    tin_max_edge: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classify one block with the compiled core.

    The five source classes are reset and re-labelled 2 / 3 / 4 / 5 / 7,
    every other class is left alone, and the result is (classification,
    height above ground, ground point indices).

    If tin_out is a list, the block's TIN is appended to it as an (m, 3)
    array of indices into x, y, z (see GroundRun.tin(); tin_max_edge is
    its max_edge, in metres).
    """
    classification = initial_classification.copy()
    source = np.flatnonzero(reset_input_class_mask(classification))
    if source.size < 3:
        raise InsufficientGroundSupport(
            "Fewer than three eligible source-class candidates."
        )
    try:
        run = fast_ground.GroundRun(
            x[source], y[source], z[source], FAST_ENGINE_SETTINGS
        ).run()
    except ValueError as exc:
        raise InsufficientGroundSupport(str(exc)) from exc
    ground = run.ground()
    height = run.heights()
    low = run.low()

    labels = np.zeros(source.size, dtype=classification.dtype)
    labels[ground] = 2
    open_points = ~ground
    labels[open_points & (height > HIGH_VEG_MAX)] = 7
    low_noise = open_points & low & (height < -FAST_ENGINE_LOW_NOISE_DEPTH)
    labels[low_noise] = 7
    assign_hag_classes(labels, labels == 0, height)
    classification[source] = labels

    hag = np.full(x.size, np.nan, dtype=np.float32)
    hag[source] = height
    model = run.model()
    if tin_out is not None:
        tin_out.append(source[run.tin(tin_max_edge)])
    surface_count = int(np.count_nonzero(model))
    ground_count = int(np.count_nonzero(ground))
    log(
        f"      Fast engine: {ground_count:,} ground points of "
        f"{source.size:,} candidates ({surface_count:,} on the surface, "
        f"{ground_count - surface_count:,} within "
        f"{100 * float(run.settings.ground_band.width):g} cm of it); surface "
        f"followed past its edge +{run.followed:,}, corners turned "
        f"+{run.turned:,}, spikes taken out {run.spikes_removed:,}; "
        f"isolated low points kept out: {int(np.count_nonzero(low)):,} "
        f"({int(np.count_nonzero(low_noise)):,} labelled class 7)"
    )
    if run.seed_square < float(FAST_ENGINE_SETTINGS.seeds.max_building_size):
        log(
            "      Small block: starting squares reduced to "
            f"{run.seed_square:g} m so the surface has at least three "
            "starting points."
        )
    return classification, hag, source[model]


def processing_windows(
    x: np.ndarray,
    y: np.ndarray,
) -> Iterable[tuple[int, tuple[float, float, float, float]]]:
    """Yield non-overlapping core windows using extent AND point count."""
    yield from window_cores_for_extent(
        float(np.min(x)),
        float(np.min(y)),
        float(np.max(x)),
        float(np.max(y)),
        int(x.size),
    )


def window_cores_for_extent(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    point_count: int,
) -> Iterable[tuple[int, tuple[float, float, float, float]]]:
    """The window layout itself, from an XY extent (metres) and point count.

    Separated from processing_windows() only so the scheduler can ask what
    layout a file WILL get from its header, before the file is read. The
    layout rules below are unchanged.
    """
    width = max(xmax - xmin, 1.0e-9)
    height = max(ymax - ymin, 1.0e-9)

    physical_nx = max(1, int(math.ceil(width / BLOCK_SIZE)))
    physical_ny = max(1, int(math.ceil(height / BLOCK_SIZE)))
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
) -> np.ndarray:
    """Return stable source-order indices for one overlap-buffered window."""
    cxmin, cymin, cxmax, cymax = core
    bxmin = cxmin - BLOCK_OVERLAP
    bymin = cymin - BLOCK_OVERLAP
    bxmax = cxmax + BLOCK_OVERLAP
    bymax = cymax + BLOCK_OVERLAP

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
    (final class 2) of each cell, nodata where a cell has none.

    x, y, z are the window's points in metres and ground flags the class 2
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
        "metres" if abs(scale - 1.0) < 1.0e-9 else "usft",
        (
            f"ground-classifier {__version__} DTM of {export.stem}, window {window_number}",
            "lowest class 2 point of each cell; cells owned by this window's core",
        ),
    )
    return path


def classify_window_job(
    window_number: int,
    window_count: int,
    core: tuple[float, float, float, float],
    buffered_indices: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    classification_in: np.ndarray,
    overall_xmax: float,
    overall_ymax: float,
    allow_sparse_fallback: bool = False,
    tin_export: TinExport | None = None,
    dtm_export: DtmExport | None = None,
) -> WindowResult:
    """Classify one window and return only its non-overlapping core."""
    messages: list[str] = []
    previous_buffer = getattr(_LOG_STATE, "buffer", None)
    _LOG_STATE.buffer = messages
    try:
        log(
            f"  Window {window_number}/{window_count}: "
            f"{buffered_indices.size:,} buffered points"
        )
        local_x = x[buffered_indices]
        local_y = y[buffered_indices]
        local_z = z[buffered_indices]
        local_input_class = classification_in[buffered_indices]

        core_local = core_point_mask(
            local_x, local_y, core, overall_xmax, overall_ymax
        )

        try:
            block_started = time.perf_counter()
            faces: list[np.ndarray] = []
            local_class, local_hag, model = classify_block(
                local_x, local_y, local_z, local_input_class,
                faces if tin_export is not None else None,
                tin_export.max_edge if tin_export is not None else 0.0,
            )
            if PERFORMANCE_STAGE_TIMINGS:
                log(
                    "      TIMING classify_block: "
                    f"{time.perf_counter() - block_started:.1f} s"
                )
        except InsufficientGroundSupport as exc:
            # allow_sparse_fallback (from the caller) reflects whether the
            # WHOLE FILE is small. A window inside an otherwise large, dense
            # file can still legitimately end up with very few or zero
            # points - an irregular tile boundary, a coverage void, or the
            # windowing grid splitting awkwardly relative to the real data
            # extent - and that is exactly what this exception means: not
            # enough geometry for a TIN, nothing pathological. Falling back
            # here for a tiny/empty WINDOW specifically (regardless of the
            # file's overall size) turns that into a graceful "keep this
            # sliver's original classification" instead of aborting every
            # other file still queued in the batch.
            window_is_sparse = (
                buffered_indices.size <= SPARSE_FALLBACK_MAX_BUFFERED_POINTS
            )
            if not (allow_sparse_fallback or window_is_sparse):
                raise
            log(
                "      Sparse-window fallback: insufficient geometry for "
                f"a TIN in this window ({exc}). Original classes "
                "preserved for this window only; HAG is unavailable here."
            )
            return WindowResult(
                window_number,
                buffered_indices[core_local].copy(),
                local_input_class[core_local].copy(),
                np.full(
                    int(np.count_nonzero(core_local)),
                    np.nan,
                    dtype=np.float32,
                ),
                0,
                tuple(messages),
            )
        log(f"      Final TIN model points: {model.size:,}")
        if dtm_export is not None:
            dtm_path = export_window_dtm(
                dtm_export, window_number, core,
                local_x, local_y, local_z, local_class == 2,
            )
            log(
                f"      DTM written: {dtm_path}"
                if dtm_path else "      DTM: no ground points in this core"
            )
        if tin_export is not None:
            tin_path = export_window_tin(
                tin_export, window_number, core, faces[0],
                local_x, local_y, local_z, overall_xmax, overall_ymax,
            )
            log(
                f"      TIN written: {tin_path}"
                if tin_path else "      TIN: no triangles in this core"
            )
        return WindowResult(
            window_number,
            buffered_indices[core_local].copy(),
            local_class[core_local].copy(),
            local_hag[core_local].copy(),
            int(model.size),
            tuple(messages),
        )
    except Exception as exc:
        detail = "\n".join(messages)
        raise RuntimeError(
            f"Window {window_number}/{window_count} failed.\n{detail}\n"
            f"Cause: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        if previous_buffer is None:
            delattr(_LOG_STATE, "buffer")
        else:
            _LOG_STATE.buffer = previous_buffer


def apply_window_result(
    result: WindowResult,
    classification_out: np.ndarray,
    hag_out: np.ndarray,
) -> None:
    for message in result.messages:
        log(message)
    classification_out[result.core_indices] = result.classification
    hag_out[result.core_indices] = result.hag


def process_file_windows(
    windows: list[tuple[int, tuple[float, float, float, float]]],
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    classification_in: np.ndarray,
    classification_out: np.ndarray,
    hag_out: np.ndarray,
    requested_workers: int,
    memory_budget: int,
    core_maxima: tuple[float, float] | None = None,
    allow_sparse_fallback: bool = False,
    tin_export: TinExport | None = None,
    dtm_export: DtmExport | None = None,
) -> None:
    """Process windows sequentially or with bounded shared-memory workers."""
    window_count = len(windows)
    if core_maxima is None:
        overall_xmax = float(np.max(x))
        overall_ymax = float(np.max(y))
    else:
        overall_xmax, overall_ymax = core_maxima

    # Avoid an O(points * windows) full-cloud mask scan on long corridors.
    # Stable X sorting and final source-order sorting keep results deterministic.
    if window_count > 1:
        x_order = np.argsort(x, kind="stable")
        sorted_x = x[x_order]
    else:
        x_order = None
        sorted_x = None

    workers = max(1, min(requested_workers, window_count))
    if workers == 1:
        for window_number, core in windows:
            buffered_indices = buffered_indices_for_window(
                x, y, core, x_order, sorted_x
            )
            result = classify_window_job(
                window_number,
                window_count,
                core,
                buffered_indices,
                x,
                y,
                z,
                classification_in,
                overall_xmax,
                overall_ymax,
                allow_sparse_fallback,
                tin_export,
                dtm_export,
            )
            apply_window_result(result, classification_out, hag_out)
        return

    log(
        f"Parallel windows: up to {workers}; guarded working-memory "
        f"budget {format_memory(memory_budget)}"
    )
    pending: dict[object, int] = {}
    in_flight_memory = 0

    def collect_one_or_more() -> None:
        nonlocal in_flight_memory
        completed, _ = wait(
            tuple(pending),
            return_when=FIRST_COMPLETED,
        )
        for future in completed:
            estimate = pending.pop(future)
            in_flight_memory -= estimate
            result = future.result()
            apply_window_result(result, classification_out, hag_out)

    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="ground-window",
    ) as executor:
        for window_number, core in windows:
            buffered_indices = buffered_indices_for_window(
                x, y, core, x_order, sorted_x
            )
            estimate = estimate_window_memory(buffered_indices.size)
            while pending and (
                len(pending) >= workers
                or in_flight_memory + estimate > memory_budget
            ):
                collect_one_or_more()

            # A single oversized window is permitted, but no peer window runs
            # beside it. This preserves the existing ability to process it.
            if estimate > memory_budget:
                log(
                    f"  Window {window_number} exceeds the parallel RAM "
                    "budget; running it alone."
                )
            future = executor.submit(
                classify_window_job,
                window_number,
                window_count,
                core,
                buffered_indices,
                x,
                y,
                z,
                classification_in,
                overall_xmax,
                overall_ymax,
                allow_sparse_fallback,
                tin_export,
                dtm_export,
            )
            pending[future] = estimate
            in_flight_memory += estimate

        while pending:
            collect_one_or_more()


def classify_file(
    input_path: Path,
    output_path: Path,
    window_workers: int = 1,
    memory_budget: int | None = None,
    kdtree_workers: int = 1,
    neighbour_caches: tuple[HaloCacheInfo, ...] = (),
    target_bounds: tuple[float, float, float, float] | None = None,
    neighbour_bounds: tuple[float, float, float, float] | None = None,
    input_unit_scale: float = METRES_PER_US_SURVEY_FOOT,
) -> FileResult:
    started = time.perf_counter()
    log(f"\nReading: {input_path}")
    read_started = time.perf_counter()
    las = laspy.read(input_path)
    if PERFORMANCE_STAGE_TIMINGS:
        log(
            f"TIMING LAZ read: "
            f"{time.perf_counter() - read_started:.1f} s"
        )

    # Read this file's raw, native-unit coordinates (feet or metres,
    # whichever the file actually uses - see unit_scale below). These
    # arrays are NEVER written back to the LAS; only temporary
    # metre-normalized copies are used for classification. Variable names
    # below keep their historical "_usft" suffix for now (from before this
    # tool supported metric input) - the values themselves are genuinely
    # native-unit, not always feet.
    target_x_usft = np.asarray(las.x, dtype=np.float64)
    target_y_usft = np.asarray(las.y, dtype=np.float64)
    target_z_usft = np.asarray(las.z, dtype=np.float64)
    target_classification = np.asarray(
        las.classification, dtype=np.uint8
    )
    target_point_count = target_x_usft.size
    if target_point_count == 0:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        removed_extra = strip_unwanted_output_extra_dimensions(las)
        if removed_extra:
            log(
                "Removed temporary output Extra Bytes dimension(s): "
                + ", ".join(removed_extra)
            )
        log("Empty input tile: writing output with classification/header policy applied.")
        las.header.generating_software = GENERATING_SOFTWARE
        las.write(output_path)
        return FileResult(time.perf_counter() - started, 0, {})

    # target_bounds / neighbour_bounds stay in the file's native input unit
    # (feet or metres) because halo discovery and cache filtering happen
    # before algorithm normalization to metres.
    if target_bounds is None:
        target_bounds = (
            float(np.min(target_x_usft)),
            float(np.min(target_y_usft)),
            float(np.max(target_x_usft)),
            float(np.max(target_y_usft)),
        )
    if neighbour_bounds is None:
        neighbour_bounds = target_bounds

    halo_x_usft, halo_y_usft, halo_z_usft, halo_classification = load_halo_points(
        neighbour_caches, neighbour_bounds
    )
    if halo_x_usft.size:
        all_x_usft = np.concatenate((target_x_usft, halo_x_usft))
        all_y_usft = np.concatenate((target_y_usft, halo_y_usft))
        all_z_usft = np.concatenate((target_z_usft, halo_z_usft))
        classification_in = np.concatenate(
            (target_classification, halo_classification)
        )
        log(
            f"Neighbour context: {halo_x_usft.size:,} points from "
            f"{len(neighbour_caches)} adjacent file(s), "
            f"{NEIGHBOUR_HALO:.3f} native-unit halo "
            f"({NEIGHBOUR_HALO_METRES:g} m physical)"
        )
    else:
        all_x_usft = target_x_usft
        all_y_usft = target_y_usft
        all_z_usft = target_z_usft
        classification_in = target_classification

    # ---- CRITICAL UNIT NORMALIZATION ----
    # Uniformly scale X, Y and Z to metres in memory. The original LAS point
    # records/header remain untouched; only the output classification byte is
    # changed. This avoids a fragile manual conversion of hundreds of distance,
    # RMSE, curvature and regression-conditioning constants.
    #
    # unit_scale is metres-per-input-unit: METRES_PER_US_SURVEY_FOOT for a
    # feet project, 1.0 for a metres project (see detect_input_unit_scale /
    # resolve_input_unit_scale). Callers that already resolved this per file
    # pass it in; falling back to feet here only matters for direct calls
    # from outside main()'s per-file resolution, kept only for backward
    # compatibility with any external script calling classify_file directly.
    unit_scale = (
        input_unit_scale if input_unit_scale is not None
        else METRES_PER_US_SURVEY_FOOT
    )
    x = all_x_usft * unit_scale
    y = all_y_usft * unit_scale
    z = all_z_usft * unit_scale
    target_x = x[:target_point_count]
    target_y = y[:target_point_count]

    # Native-unit floating-point XYZ copies are no longer required once halo
    # selection is complete. Releasing them keeps peak memory close to what
    # a metres-only project would use, regardless of the input file's unit.
    del all_x_usft, all_y_usft, all_z_usft
    del target_x_usft, target_y_usft, target_z_usft
    if halo_x_usft.size:
        del halo_x_usft, halo_y_usft, halo_z_usft

    classification_out = classification_in.copy()
    reset_global = reset_input_class_mask(classification_out)
    classification_out[reset_global] = 0
    hag_out = np.full(x.size, np.nan, dtype=np.float32)

    windows = list(processing_windows(target_x, target_y))
    log(
        f"Target points: {target_point_count:,}; context points: "
        f"{x.size - target_point_count:,}; processing window(s): "
        f"{len(windows)}"
    )
    actual_window_workers = max(1, min(window_workers, len(windows)))
    tree_workers_per_window = max(
        1,
        min(
            MAX_KD_TREE_WORKERS,
            kdtree_workers // actual_window_workers,
        ),
    )
    configure_runtime(
        tree_workers_per_window,
        getattr(_LOG_STATE, "prefix", ""),
    )
    if memory_budget is None:
        memory_budget = safe_parallel_memory_budget()
    else:
        # Recheck after LAS decompression and full-cloud arrays are resident.
        memory_budget = min(
            memory_budget,
            safe_parallel_memory_budget(),
        )
    tin_export = (
        TinExport(
            TIN_EXPORT_DIR, input_path.stem, unit_scale, TIN_EXPORT_MAX_EDGE
        )
        if TIN_EXPORT_DIR is not None
        else None
    )
    dtm_export = (
        DtmExport(
            DTM_EXPORT_DIR,
            input_path.stem,
            unit_scale,
            DTM_CELL or default_dtm_cell(unit_scale),
            (
                min(core[0] for _, core in windows),
                min(core[1] for _, core in windows),
                max(core[2] for _, core in windows),
                max(core[3] for _, core in windows),
            ),
        )
        if DTM_EXPORT_DIR is not None
        else None
    )
    processing_started = time.perf_counter()
    process_file_windows(
        windows,
        x,
        y,
        z,
        classification_in,
        classification_out,
        hag_out,
        actual_window_workers,
        memory_budget,
        (target_bounds[2] * unit_scale, target_bounds[3] * unit_scale),
        target_point_count <= SPARSE_FALLBACK_MAX_BUFFERED_POINTS,
        tin_export,
        dtm_export,
    )
    if PERFORMANCE_STAGE_TIMINGS:
        log(
            f"TIMING classification windows: "
            f"{time.perf_counter() - processing_started:.1f} s"
        )

    las.classification = classification_out[:target_point_count]

    # Do not propagate TerraScan's temporary Distance Extra Byte into the
    # classified output. This changes only the point-record layout/header; XYZ,
    # standard LAS dimensions, classifications and all other Extra Bytes/VLRs
    # remain intact.
    removed_extra = strip_unwanted_output_extra_dimensions(las)
    if removed_extra:
        log(
            "Removed temporary output Extra Bytes dimension(s): "
            + ", ".join(removed_extra)
        )

    las.header.generating_software = GENERATING_SOFTWARE
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"Writing: {output_path}")
    write_started = time.perf_counter()
    las.write(output_path)
    if PERFORMANCE_STAGE_TIMINGS:
        log(
            f"TIMING LAZ write: "
            f"{time.perf_counter() - write_started:.1f} s"
        )

    elapsed = time.perf_counter() - started
    counts = {
        int(code): int(
            np.count_nonzero(
                classification_out[:target_point_count] == code
            )
        )
        for code in (0, 2, 3, 4, 5, 7)
    }
    log(
        "Class counts: "
        + ", ".join(f"{code}={count:,}" for code, count in counts.items())
    )
    log(f"Completed in {elapsed / 60.0:.1f} minutes")
    return FileResult(elapsed, int(target_point_count), counts)


def discover_inputs(folder: Path) -> list[Path]:
    inputs = sorted(folder.glob("*.las"))
    inputs.extend(sorted(folder.glob("*.laz")))
    return inputs


def nominal_neighbour_bounds(
    input_path: Path,
    data_bounds: tuple[float, float, float, float],
    point_count: int,
) -> tuple[float, float, float, float]:
    """Return a validated filename-derived tile footprint when available."""
    match = NUMERIC_TILE_NAME.match(input_path.stem)
    if match is None:
        return data_bounds

    origin_x = float(match.group(1))
    origin_y = float(match.group(2))
    nominal = (
        origin_x,
        origin_y,
        origin_x + NOMINAL_TILE_SIZE_USFT,
        origin_y + NOMINAL_TILE_SIZE_USFT,
    )
    if point_count == 0:
        return nominal

    # Only trust the naming convention when the header bounds actually fit
    # that square.  This avoids treating unrelated numeric filenames as map
    # tile origins.
    tolerance = max(1.0, NEIGHBOUR_HALO)
    xmin, ymin, xmax, ymax = data_bounds
    finite = all(math.isfinite(value) for value in data_bounds)
    contained = (
        xmin >= nominal[0] - tolerance
        and ymin >= nominal[1] - tolerance
        and xmax <= nominal[2] + tolerance
        and ymax <= nominal[3] + tolerance
    )
    return nominal if finite and contained else data_bounds


def read_input_metadata(input_path: Path) -> InputMetadata:
    """Read only the LAS/LAZ header needed for spatial scheduling."""
    with laspy.open(input_path) as reader:
        header = reader.header
        point_count = int(header.point_count)
        bounds = (
            float(header.mins[0]),
            float(header.mins[1]),
            float(header.maxs[0]),
            float(header.maxs[1]),
        )
        return InputMetadata(
            input_path,
            point_count,
            int(header.point_format.size),
            bounds,
            nominal_neighbour_bounds(input_path, bounds, point_count),
        )


def find_neighbour_sources(
    targets: list[InputMetadata],
    candidates: list[InputMetadata],
) -> dict[Path, tuple[Path, ...]]:
    """Find files intersecting each target's expanded (by the neighbour halo) bounds."""
    result = {target.path: tuple() for target in targets}
    if NEIGHBOUR_HALO <= 0 or len(candidates) < 2:
        return result

    centers = np.array(
        [
            [(item.neighbour_bounds[0] + item.neighbour_bounds[2]) * 0.5,
             (item.neighbour_bounds[1] + item.neighbour_bounds[3]) * 0.5]
            for item in candidates
        ],
        dtype=np.float64,
    )
    half_diagonal = np.array(
        [
            0.5
            * math.hypot(
                item.neighbour_bounds[2] - item.neighbour_bounds[0],
                item.neighbour_bounds[3] - item.neighbour_bounds[1],
            )
            for item in candidates
        ],
        dtype=np.float64,
    )
    tree = cKDTree(centers)
    largest_half_diagonal = float(np.max(half_diagonal))
    candidate_position = {
        item.path: position
        for position, item in enumerate(candidates)
    }

    for target in targets:
        target_position = candidate_position[target.path]
        search_radius = (
            half_diagonal[target_position]
            + largest_half_diagonal
            + NEIGHBOUR_HALO
        )
        positions = tree.query_ball_point(
            centers[target_position], search_radius
        )
        txmin, tymin, txmax, tymax = target.neighbour_bounds
        expanded = (
            txmin - NEIGHBOUR_HALO,
            tymin - NEIGHBOUR_HALO,
            txmax + NEIGHBOUR_HALO,
            tymax + NEIGHBOUR_HALO,
        )
        neighbours: list[Path] = []
        for position in positions:
            candidate = candidates[int(position)]
            if candidate.path == target.path:
                continue
            xmin, ymin, xmax, ymax = candidate.neighbour_bounds
            if (
                xmax >= expanded[0]
                and xmin <= expanded[2]
                and ymax >= expanded[1]
                and ymin <= expanded[3]
            ):
                neighbours.append(candidate.path)
        result[target.path] = tuple(sorted(neighbours, key=str))
    return result


def create_halo_cache_worker(
    source_path_text: str,
    requested_bounds: tuple[tuple[float, float, float, float], ...],
    cache_path_text: str,
) -> HaloCacheInfo:
    """Stream requested halo regions from one source into a raw cache."""
    source_path = Path(source_path_text)
    cache_path = Path(cache_path_text)
    point_count = 0
    cells = DENSITY_GRID_CELLS
    density_counts: np.ndarray | None = None
    density_bounds: tuple[float, float, float, float] | None = None
    with cache_path.open("wb") as output_stream:
        with laspy.open(source_path) as reader:
            try:
                gxmin = float(reader.header.mins[0])
                gymin = float(reader.header.mins[1])
                gxmax = float(reader.header.maxs[0])
                gymax = float(reader.header.maxs[1])
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
            for points in reader.chunk_iterator(HALO_CACHE_CHUNK_POINTS):
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


# Set to False for the rest of the run when this machine turns out to be
# unable to keep helper processes alive (see helper_processes_work()). The
# file-classification stage then stays in the main process as well.
HELPER_PROCESSES_USABLE = True

# Windows process exit codes that explain an abrupt helper death.
_HELPER_EXIT_HINTS = {
    0x00000001: "generic failure, or ended by another program",
    0x00000003: "abort() called inside a compiled library",
    0xC0000005: "access violation: a crash inside a compiled library",
    0xC0000017: "not enough memory to start the process",
    0xC00000FD: "stack overflow",
    0xC000012D: "commit limit reached: paging file too small",
    0xC0000135: "a required DLL was not found",
    0xC000013A: "interrupted: Ctrl+C or the console was closed",
    0xC0000142: "a DLL failed to initialise: the process could not start",
    0xC0000374: "heap corruption inside a compiled library",
    0xC0000409: "fail-fast abort inside a compiled library",
    0xC0000417: "invalid C runtime parameter inside a compiled library",
}


def describe_exit_code(code: int) -> str:
    """Human-readable form of a helper process exit code."""
    if code < 0:
        return f"killed by signal {-code}"
    unsigned = code & 0xFFFFFFFF
    text = f"0x{unsigned:08X}" if unsigned > 255 else str(unsigned)
    hint = _HELPER_EXIT_HINTS.get(unsigned)
    return f"{text} ({hint})" if hint else text


def pool_processes(executor: ProcessPoolExecutor) -> list:
    """Best-effort handles to a pool's helper processes (diagnostics only)."""
    try:
        return list((getattr(executor, "_processes", None) or {}).values())
    except Exception:
        return []


def dead_helper_exit_codes(processes: list) -> list[str]:
    """Exit codes of helpers that died on their own, never raising."""
    described: list[str] = []
    for process in processes:
        try:
            process.join(timeout=10)
            code = process.exitcode
        except Exception:
            continue
        # None: still running. 0: clean exit. -15: stopped by the pool itself
        # while it shut down after the first helper died.
        if code is None or code == 0 or code == -15:
            continue
        described.append(describe_exit_code(int(code)))
    return described


def helper_process_probe() -> int:
    """Trivial task: can a helper process start and answer at all?"""
    return os.getpid()


def helper_processes_work() -> bool:
    """True when a brand-new helper process starts and runs a trivial task.

    This separates the two families of BrokenProcessPool causes: if even
    this fails, no input file is involved and the machine itself is stopping
    Python helper processes; if it passes, the death is tied to the work the
    helper was doing.
    """
    executor = ProcessPoolExecutor(max_workers=1)
    working = False
    try:
        working = executor.submit(helper_process_probe).result(timeout=300) > 0
    except KeyboardInterrupt:
        raise
    except BaseException:
        working = False
    finally:
        executor.shutdown(wait=working)
    return working


def read_halo_source_named(
    source: Path,
    bounds: tuple[tuple[float, float, float, float], ...],
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
    global HELPER_PROCESSES_USABLE
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

    HELPER_PROCESSES_USABLE = False
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
    source_requests: dict[
        Path, set[tuple[float, float, float, float]]
    ],
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
    target_bounds: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load only cached returns within 10 m of this target file."""
    xmin, ymin, xmax, ymax = target_bounds
    expanded = (
        xmin - NEIGHBOUR_HALO,
        ymin - NEIGHBOUR_HALO,
        xmax + NEIGHBOUR_HALO,
        ymax + NEIGHBOUR_HALO,
    )
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
    rect: tuple[float, float, float, float],
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


def estimate_target_memory(
    metadata: InputMetadata,
    neighbour_caches: tuple[HaloCacheInfo, ...],
    unit_scale: float,
    density: SourceDensity | None,
) -> FileMemoryPlan:
    """Memory estimate for one target file, for scheduling only.

    Same model as estimate_file_peak_memory() (resident file + neighbour
    context + working memory per window), but split into its parts so the
    scheduler can price "this file with N windows running at once", and
    with the two quantities the older estimate had to guess replaced by
    measured upper bounds when the neighbour-cache pass has provided
    point-density grids:

      * each processing window's point count, from the file's real window
        layout, its own density grid and its neighbours' grids, instead of
        "the whole file plus every neighbouring cache in one window";
      * the neighbour context this file will load, from the neighbours'
        grids, instead of the full size of every neighbouring cache.

    Every measured figure is an over-count (whole grid cells, plus
    DENSITY_ESTIMATE_MARGIN). Falls back to the older, more cautious
    figures whenever a grid is missing or anything looks inconsistent.
    """
    point_count = max(0, int(metadata.point_count))
    cache_points = sum(info.point_count for info in neighbour_caches)
    file_resident = int(
        FILE_FIXED_MEMORY_MB * 1024**2
        + point_count
        * max(FULL_FILE_BYTES_PER_POINT, metadata.record_size + 64)
    )
    cautious_resident = file_resident + int(
        cache_points * HALO_CACHE_BYTES_PER_POINT
    )
    cautious_window = estimate_window_memory(point_count + cache_points)
    window_count = 1
    if point_count <= 0:
        return FileMemoryPlan(
            cautious_resident, cautious_window, 1, cautious_window
        )
    try:
        bxmin, bymin, bxmax, bymax = metadata.bounds
        cores = [
            core
            for _, core in window_cores_for_extent(
                bxmin * unit_scale,
                bymin * unit_scale,
                bxmax * unit_scale,
                bymax * unit_scale,
                point_count,
            )
        ]
        window_count = max(1, len(cores))

        xmin, ymin, xmax, ymax = metadata.neighbour_bounds
        expanded = (
            xmin - NEIGHBOUR_HALO,
            ymin - NEIGHBOUR_HALO,
            xmax + NEIGHBOUR_HALO,
            ymax + NEIGHBOUR_HALO,
        )

        def neighbour_points(
            rect: tuple[float, float, float, float],
            margin: float = DENSITY_ESTIMATE_MARGIN,
        ) -> int:
            """Upper bound on loaded neighbour points inside ``rect``."""
            clipped = (
                max(rect[0], expanded[0]),
                max(rect[1], expanded[1]),
                min(rect[2], expanded[2]),
                min(rect[3], expanded[3]),
            )
            if clipped[0] > clipped[2] or clipped[1] > clipped[3]:
                return 0
            total = 0
            for info in neighbour_caches:
                if info.density is None:
                    total += info.point_count
                else:
                    total += min(
                        info.point_count,
                        density_points_in_rect(info.density, clipped),
                    )
            return min(cache_points, int(math.ceil(total * margin)))

        halo_points = neighbour_points(expanded)
        own_density = (
            density
            if density is not None and sum(density.counts) == point_count
            else None
        )
        overlap = BLOCK_OVERLAP / max(unit_scale, 1.0e-9)
        largest_window = 0
        likely_window = 0
        for core in cores:
            rect = (
                core[0] / unit_scale - overlap,
                core[1] / unit_scale - overlap,
                core[2] / unit_scale + overlap,
                core[3] / unit_scale + overlap,
            )
            own_points = (
                point_count
                if own_density is None
                else min(
                    point_count,
                    int(
                        math.ceil(
                            density_points_in_rect(own_density, rect)
                            * DENSITY_ESTIMATE_MARGIN
                        )
                    ),
                )
            )
            largest_window = max(
                largest_window, own_points + neighbour_points(rect)
            )
            likely_window = max(
                likely_window,
                (
                    point_count
                    if own_density is None
                    else min(
                        point_count,
                        density_points_in_rect(own_density, rect),
                    )
                )
                + neighbour_points(rect, 1.0),
            )
        measured_resident = file_resident + int(
            halo_points * HALO_CACHE_BYTES_PER_POINT
        )
        measured_window = estimate_window_memory(largest_window)
        likely_window_memory = estimate_window_memory(likely_window)
    except Exception:
        return FileMemoryPlan(
            cautious_resident, cautious_window, window_count, cautious_window
        )
    return FileMemoryPlan(
        min(cautious_resident, measured_resident),
        min(cautious_window, measured_window),
        window_count,
        min(cautious_window, measured_window, likely_window_memory),
    )


def plan_file_parallelism(
    plan: FileMemoryPlan,
    cpu_budget: int,
    memory_budget: int,
) -> tuple[int, int, bool]:
    """Choose how a file of this size is run: (files at once, windows in
    parallel inside each, exclusive).

    Every mix of "F files at once, W windows at once in each" that fits the
    RAM budget (F * (resident + W * window)) and the CPU budget (F * W) is
    scored by the work it gets done at once, F * W ** WINDOW_THREAD_SCALING,
    and the best is taken; ties go to more files.

    One more option is always available and is what earlier builds did for
    large files: run the file by itself with its windows in parallel, the
    windows alone held to the RAM budget from their exact sizes once the
    file is loaded. That is returned as exclusive=True when it beats every
    shared mix, so a large file is never run in a slower arrangement than
    it was before.
    """
    most_windows = max(1, min(plan.windows, cpu_budget, MAX_PARALLEL_WINDOWS))

    def work_rate(threads: int) -> float:
        """Windows really in progress at once for a file given ``threads``.

        A file's windows are worked through in rounds, so three threads on
        a four-window file finish no sooner than two threads would."""
        rounds = int(math.ceil(max(plan.windows, 1) / threads))
        return (max(plan.windows, 1) / rounds) ** WINDOW_THREAD_SCALING

    best_rate = 0.0
    best_files = 0
    best_threads = 1
    for threads in range(1, most_windows + 1):
        files = min(
            cpu_budget // threads,
            memory_budget // max(plan.resident + threads * plan.window, 1),
            MAX_PARALLEL_FILES,
        )
        if files < 1:
            continue
        rate = files * work_rate(threads)
        if rate > best_rate + 1.0e-9 or (
            abs(rate - best_rate) <= 1.0e-9 and files > best_files
        ):
            best_rate, best_files, best_threads = rate, files, threads
    likely = plan.window_likely if plan.window_likely > 0 else plan.window
    alone_threads = max(
        1, min(most_windows, memory_budget // max(likely, 1))
    )
    if best_files < 1 or work_rate(alone_threads) > best_rate + 1.0e-9:
        return 1, most_windows, True
    return int(best_files), int(best_threads), False


def classify_file_worker(
    input_path_text: str,
    output_path_text: str,
    cpu_slots: int,
    neighbour_caches: tuple[HaloCacheInfo, ...],
    target_bounds: tuple[float, float, float, float],
    neighbour_bounds: tuple[float, float, float, float],
    hag_bands: tuple[float, float, float],
    engine_settings: fast_ground.Settings,
    unit_scale: float,
    window_workers: int = 1,
    window_memory_budget: int | None = None,
    tin_export_dir_text: str | None = None,
    tin_max_edge: float = 0.0,
    dtm_export_dir_text: str | None = None,
    dtm_cell: float = 0.0,
    neighbour_halo_m: float = 10.0,
) -> FileResult:
    """Top-level process target; paths avoid transferring point arrays.

    window_workers > 1 runs this file's windows on parallel threads, held to
    window_memory_budget by classify_file's own window scheduler - the same
    mechanism a single file run in the main process has always used.
    """
    input_path = Path(input_path_text)
    configure_runtime(cpu_slots, f"[{input_path.stem}] ")
    configure_hag_bands(*hag_bands)
    configure_engine_settings(engine_settings)
    configure_tin_export(
        Path(tin_export_dir_text) if tin_export_dir_text else None,
        tin_max_edge,
    )
    configure_dtm_export(
        Path(dtm_export_dir_text) if dtm_export_dir_text else None,
        dtm_cell,
    )
    configure_neighbour_halo(unit_scale, neighbour_halo_m)
    configure_nominal_tile_size(unit_scale)
    return classify_file(
        input_path,
        Path(output_path_text),
        window_workers=max(1, int(window_workers)),
        memory_budget=window_memory_budget,
        kdtree_workers=cpu_slots,
        neighbour_caches=neighbour_caches,
        target_bounds=target_bounds,
        neighbour_bounds=neighbour_bounds,
        input_unit_scale=unit_scale,
    )


def reusable_existing_output(input_path: Path, output_path: Path) -> bool:
    """Return true for a complete output of this version with the expected layout.

    Output intentionally omits configured temporary Extra Bytes such as
    ``Distance``, so byte-for-byte point-format size equality with the input is
    no longer a valid resume test. Output from another version is rebuilt.
    """
    if not output_path.is_file() or output_path.stat().st_size == 0:
        return False
    if output_path.stat().st_mtime < input_path.stat().st_mtime:
        return False
    try:
        stripped = {name.casefold() for name in STRIP_OUTPUT_EXTRA_DIMENSIONS}
        with laspy.open(input_path) as source_reader:
            source_header = source_reader.header
            expected_extra = tuple(
                str(name)
                for name in source_header.point_format.extra_dimension_names
                if str(name).casefold() not in stripped
            )
            expected_signature = (
                int(source_header.point_count),
                int(source_header.point_format.id),
                expected_extra,
            )
        with laspy.open(output_path) as output_reader:
            output_header = output_reader.header
            if output_header.generating_software != GENERATING_SOFTWARE:
                return False
            output_signature = (
                int(output_header.point_count),
                int(output_header.point_format.id),
                tuple(str(name) for name in output_header.point_format.extra_dimension_names),
            )
        return expected_signature == output_signature
    except Exception:
        return False


def run_batch(
    tasks: list[tuple[Path, Path]],
    inputs: list[Path],
    unit_scale_by_path: dict[Path, float],
    hag_bands: tuple[float, float, float],
    engine_settings: fast_ground.Settings,
    requested_workers: int,
    requested_ram_limit_gb: float,
    tin_export_dir: Path | None = None,
    tin_max_edge: float = 0.0,
    dtm_export_dir: Path | None = None,
    dtm_cell: float = 0.0,
    neighbour_halo_m: float = 10.0,
) -> tuple[dict[Path, FileResult], list[tuple[Path, BaseException]], dict]:
    """Classify ``tasks`` (input, output) with the resource-aware scheduler.

    ``inputs`` are all files the caller asked for; neighbours of the tasks are
    taken from the same folders. Returns the per-file results, the files that
    failed, and the CPU/RAM limits the scheduler used.
    """
    configure_hag_bands(*hag_bands)
    # NEIGHBOUR_HALO is a single process-wide value used while discovering
    # which files are spatial neighbours of each other, before any one
    # file's classification begins - it needs one project-wide unit. All
    # files in a run are expected to share the same project/unit; the first
    # task's unit is used for this shared step, and every actual
    # classification still uses its own file's individually resolved unit.
    configure_neighbour_halo(unit_scale_by_path[tasks[0][0]], neighbour_halo_m)
    configure_nominal_tile_size(unit_scale_by_path[tasks[0][0]])

    unit_labels = sorted({
        "US survey feet" if abs(scale - 1.0) > 1.0e-9 else "metres"
        for scale in unit_scale_by_path.values()
    })
    unit_summary = " and ".join(unit_labels) if unit_labels else "unknown"
    low_veg, medium_veg, high_veg = hag_bands
    log(
        f"Unit mode: this run's input is {unit_summary} (resolved per file "
        "from CRS metadata, the config file or --units). Classification geometry is normalized "
        "to metres in memory; LAS XYZ is never rewritten. Output vegetation: "
        f"class 3 <={low_veg:g} m, class 4 {low_veg:g}-{medium_veg:g} m, "
        f"class 5 >={medium_veg:g} m (see the hag section of the config file)."
    )

    configure_tin_export(tin_export_dir, tin_max_edge)
    if tin_export_dir is not None:
        log(
            f"TIN export: one PLY per window in {tin_export_dir}"
            + (f", triangle edges <= {tin_max_edge:g} m" if tin_max_edge > 0 else "")
        )
    configure_dtm_export(dtm_export_dir, dtm_cell)
    if dtm_export_dir is not None:
        log(
            f"DTM export: one GeoTIFF per window in {dtm_export_dir}, cell "
            + (
                f"{dtm_cell:g} (input unit)" if dtm_cell > 0
                else "1 m / 3 ft (default for the file's unit)"
            )
        )
    configure_engine_settings(engine_settings)
    changed = engine_settings.changed()
    log(
        "Engine settings (metres): "
        + (
            ", ".join(f"{k}={v:g}" for k, v in changed.items())
            if changed else "all defaults"
        )
        + " (see the engine section of the config file)"
    )
    cpu_budget = cpu_worker_budget(requested_workers)
    memory_budget = safe_parallel_memory_budget(requested_ram_limit_gb)
    logical_cpus = max(1, os.cpu_count() or 1)
    snapshot = memory_snapshot()
    log(
        f"Resource guard: {cpu_budget}/{logical_cpus} logical CPU slots; "
        f"parallel RAM budget {format_memory(memory_budget)}; "
        f"currently available {format_memory(snapshot.available)}"
    )

    # Explicitly supplied targets still discover spatial neighbours in their
    # folders. Those neighbours provide context but are not written unless
    # they were also supplied as targets.
    candidate_paths = set(inputs)
    for parent in {path.parent for path in inputs}:
        candidate_paths.update(
            path.resolve() for path in discover_inputs(parent)
        )
    candidate_metadata = [
        read_input_metadata(path)
        for path in sorted(candidate_paths, key=str)
    ]
    metadata_by_path = {
        metadata.path: metadata for metadata in candidate_metadata
    }
    target_metadata = [
        metadata_by_path[input_path]
        for input_path, _ in tasks
    ]
    neighbours_by_target = find_neighbour_sources(
        target_metadata, candidate_metadata
    )

    source_requests: dict[
        Path, set[tuple[float, float, float, float]]
    ] = {}
    for target in target_metadata:
        xmin, ymin, xmax, ymax = target.neighbour_bounds
        expanded = (
            xmin - NEIGHBOUR_HALO,
            ymin - NEIGHBOUR_HALO,
            xmax + NEIGHBOUR_HALO,
            ymax + NEIGHBOUR_HALO,
        )
        for source in neighbours_by_target[target.path]:
            source_requests.setdefault(source, set()).add(expanded)

    results: dict[Path, FileResult] = {}
    failures: list[tuple[Path, BaseException]] = []
    cache_workspace: tempfile.TemporaryDirectory[str] | None = None
    try:
        halo_caches: dict[Path, HaloCacheInfo] = {}
        if source_requests:
            cache_workspace = tempfile.TemporaryDirectory(
                prefix="adaptive_ground_halo_"
            )
            cache_workers = min(
                len(source_requests),
                MAX_PARALLEL_HALO_READERS,
                cpu_budget,
            )
            halo_caches = build_halo_caches(
                source_requests,
                Path(cache_workspace.name),
                cache_workers,
            )
            cached_points = sum(
                info.point_count for info in halo_caches.values()
            )
            log(
                f"Neighbour cache ready: {cached_points:,} context "
                "points."
            )

        # Workers only need each cache's location and size; the density
        # grids stay in this process, where the scheduler uses them.
        target_caches = {
            target.path: tuple(
                HaloCacheInfo(
                    halo_caches[source].path,
                    halo_caches[source].point_count,
                )
                for source in neighbours_by_target[target.path]
                if source in halo_caches
            )
            for target in target_metadata
        }

        # How the batch is run.
        #
        # Every file gets its own memory estimate and its own plan: how many
        # files of that size can run at once and how many of each file's
        # windows run in parallel (plan_file_parallelism). Large files end
        # up one at a time with all their windows in parallel, as earlier
        # builds ran them; smaller files run several at once. Files are
        # started largest-first while the running totals stay inside the
        # RAM and CPU budgets (the loop below). Earlier builds sized the
        # whole batch by its single largest file, so one large tile forced
        # every small tile to be processed one at a time as well. This
        # changes only WHEN each file and window is processed, never how.
        file_workers = 1
        memory_plans: dict[Path, FileMemoryPlan] = {}
        run_plans: dict[Path, tuple[int, int, bool]] = {}
        if AUTO_PARALLEL and len(tasks) > 1 and cpu_budget > 1:
            for input_path, _ in tasks:
                memory_plans[input_path] = estimate_target_memory(
                    metadata_by_path[input_path],
                    tuple(
                        halo_caches[source]
                        for source in neighbours_by_target[input_path]
                        if source in halo_caches
                    ),
                    unit_scale_by_path[input_path],
                    (
                        halo_caches[input_path].density
                        if input_path in halo_caches
                        else None
                    ),
                )
                run_plans[input_path] = plan_file_parallelism(
                    memory_plans[input_path], cpu_budget, memory_budget
                )
            # Helper processes are only worth starting when at least one
            # file can share the machine with another. If every file has to
            # run by itself, the main process does the work directly (the
            # branch further down), exactly as before.
            file_workers = min(
                len(tasks),
                MAX_PARALLEL_FILES,
                cpu_budget,
                max(files for files, _, _ in run_plans.values()),
            )

        def classify_in_process(task_list: list[tuple[Path, Path]]) -> None:
            """Classify files one at a time in this process, with
            multi-threaded windows. A file that raises is recorded in
            ``failures`` and the batch continues."""
            window_workers = (
                min(MAX_PARALLEL_WINDOWS, cpu_budget)
                if AUTO_PARALLEL
                else 1
            )
            for input_path, output_path in task_list:
                configure_runtime(cpu_budget)
                configure_neighbour_halo(unit_scale_by_path[input_path], neighbour_halo_m)
                configure_nominal_tile_size(unit_scale_by_path[input_path])
                try:
                    results[input_path] = classify_file(
                        input_path,
                        output_path,
                        window_workers=window_workers,
                        memory_budget=memory_budget,
                        kdtree_workers=cpu_budget,
                        neighbour_caches=target_caches[input_path],
                        target_bounds=metadata_by_path[input_path].bounds,
                        neighbour_bounds=(
                            metadata_by_path[input_path].neighbour_bounds
                        ),
                        input_unit_scale=unit_scale_by_path[input_path],
                    )
                except Exception as exc:
                    failures.append((input_path, exc))
                    log(f"FAILED {input_path.name}: {exc}")

        if file_workers > 1 and not HELPER_PROCESSES_USABLE:
            log(
                "Helper processes cannot run on this machine (see the "
                "DIAGNOSIS above); classifying files one at a time with "
                "multi-threaded windows instead."
            )
            file_workers = 1

        if file_workers > 1:
            def footprint(input_path: Path) -> int:
                """Size measure used only to order files, largest first."""
                plan = memory_plans[input_path]
                return plan.resident + plan.window

            def run_cost(input_path: Path) -> int:
                """Estimated memory a file occupies while it runs."""
                files, threads, exclusive = run_plans[input_path]
                if exclusive:
                    return memory_budget
                plan = memory_plans[input_path]
                return plan.resident + threads * plan.window

            most = file_workers
            sizes = sorted(footprint(input_path) for input_path, _ in tasks)
            exclusive_count = sum(
                1 for _, _, exclusive in run_plans.values() if exclusive
            )
            fewest_files = min(files for files, _, _ in run_plans.values())
            most_windows = max(threads for _, threads, _ in run_plans.values())
            log(
                f"Parallel plan: {fewest_files}"
                + (f" to {most}" if most != fewest_files else "")
                + " file(s) at once, up to "
                f"{most_windows} window(s) in parallel inside a file; "
                f"estimated {format_memory(sizes[0])} to "
                f"{format_memory(sizes[-1])} per file (one window) against "
                f"a {format_memory(memory_budget)} RAM budget and "
                f"{cpu_budget} CPU slots."
            )
            if exclusive_count:
                log(
                    f"  {exclusive_count:,} large file(s) will run one at a "
                    "time with their windows in parallel; the rest share "
                    "the machine."
                )
            typical_files, typical_threads, _ = run_plans[
                sorted(run_plans, key=footprint)[len(run_plans) // 2]
            ]
            if typical_files * typical_threads < min(cpu_budget, len(tasks)):
                log(
                    "  RAM is the limit: more would run at once with a "
                    "larger RAM budget. If this machine is dedicated to "
                    "the run, ram_limit_gb in the performance section of the config file "
                    "raises it (4 GB of free RAM is always kept spare)."
                )
            # Files whose helper process died outright. One dead helper
            # breaks the pool for every unfinished file, so these are
            # finished in the main process rather than all reported FAILED.
            crashed_tasks: list[tuple[Path, Path]] = []
            crashed_processes: list = []

            # Largest first. Every file still waiting is then no larger than
            # any file already running, so whenever one finishes the next in
            # line fits: no file can be starved and the number running rises
            # naturally as the files get smaller.
            waiting = sorted(
                tasks,
                key=lambda task: (-footprint(task[0]), str(task[0])),
            )
            next_position = 0
            in_flight: dict[object, tuple[Path, Path, int, int]] = {}
            in_flight_memory = 0
            in_flight_threads = 0
            exclusive_running = False
            most_in_flight = 0
            lowest_available: int | None = None
            pool_broken = False
            free_reserve = int(MIN_FREE_RAM_GB * 1024**3)
            # No more helper processes than can ever be busy at once: an
            # idle helper still holds the imported libraries in memory.
            with ProcessPoolExecutor(max_workers=most) as executor:
                while next_position < len(waiting) or in_flight:
                    while (
                        not pool_broken
                        and next_position < len(waiting)
                        and len(in_flight) < most
                    ):
                        input_path, output_path = waiting[next_position]
                        files, threads, exclusive = run_plans[input_path]
                        estimate = run_cost(input_path)
                        # A file always starts when nothing else is running
                        # (an oversized file still runs, by itself). Beside
                        # other files it must fit the RAM and CPU budgets on
                        # paper AND the machine must really have that much
                        # RAM free right now, in case the estimates are low
                        # or another program is using memory. A file
                        # planned to run by itself waits for an empty
                        # machine and nothing starts beside it.
                        if in_flight:
                            if exclusive or exclusive_running:
                                break
                            if in_flight_memory + estimate > memory_budget:
                                break
                            if in_flight_threads + threads > cpu_budget:
                                break
                            available_now = memory_snapshot().available
                            if (
                                available_now > 0
                                and available_now - free_reserve < estimate
                            ):
                                break
                        plan = memory_plans[input_path]
                        try:
                            future = executor.submit(
                                classify_file_worker,
                                str(input_path),
                                str(output_path),
                                # CPU slots for this file: its share of the
                                # budget, split between its windows inside
                                # classify_file for neighbour searches.
                                (
                                    cpu_budget
                                    if exclusive
                                    else max(threads, cpu_budget // files)
                                ),
                                target_caches[input_path],
                                metadata_by_path[input_path].bounds,
                                metadata_by_path[input_path].neighbour_bounds,
                                hag_bands,
                                engine_settings,
                                unit_scale_by_path[input_path],
                                threads,
                                (
                                    memory_budget
                                    if exclusive
                                    else threads * plan.window
                                ),
                                (
                                    str(TIN_EXPORT_DIR)
                                    if TIN_EXPORT_DIR is not None
                                    else None
                                ),
                                TIN_EXPORT_MAX_EDGE,
                                (
                                    str(DTM_EXPORT_DIR)
                                    if DTM_EXPORT_DIR is not None
                                    else None
                                ),
                                DTM_CELL,
                                neighbour_halo_m,
                            )
                        except BrokenProcessPool:
                            pool_broken = True
                            break
                        next_position += 1
                        in_flight[future] = (
                            input_path, output_path, estimate, threads
                        )
                        in_flight_memory += estimate
                        in_flight_threads += threads
                        exclusive_running = exclusive
                        most_in_flight = max(most_in_flight, len(in_flight))

                    if not in_flight:
                        # Only reachable when the pool has broken and
                        # nothing is left running.
                        break
                    # The timeout re-checks free RAM once a minute when the
                    # next file is being held back by the live-memory guard.
                    completed, _ = wait(
                        tuple(in_flight),
                        timeout=60.0,
                        return_when=FIRST_COMPLETED,
                    )
                    available_now = memory_snapshot().available
                    if available_now > 0 and (
                        lowest_available is None
                        or available_now < lowest_available
                    ):
                        lowest_available = available_now
                    for future in completed:
                        input_path, output_path, estimate, threads = (
                            in_flight.pop(future)
                        )
                        in_flight_memory -= estimate
                        in_flight_threads -= threads
                        if not in_flight:
                            exclusive_running = False
                        try:
                            result = future.result()
                            results[input_path] = result
                            log(
                                f"Finished {input_path.name} in "
                                f"{result.elapsed / 60.0:.1f} minutes."
                            )
                        except BrokenProcessPool:
                            pool_broken = True
                            if not crashed_processes:
                                crashed_processes = pool_processes(executor)
                            crashed_tasks.append((input_path, output_path))
                        except BaseException as exc:
                            failures.append((input_path, exc))
                            log(f"FAILED {input_path.name}: {exc}")
            # Files never started because the pool broke first.
            crashed_tasks.extend(waiting[next_position:])
            log(
                f"Most files running at once: {most_in_flight}."
                + (
                    " Lowest free RAM seen during the run: "
                    f"{format_memory(lowest_available)}."
                    if lowest_available is not None
                    else ""
                )
            )
            if crashed_tasks:
                codes = dead_helper_exit_codes(crashed_processes)
                log(
                    f"WARNING: a helper process died during classification; "
                    f"{len(crashed_tasks):,} file(s) were left unfinished. "
                    + (
                        f"Helper exit code(s): {', '.join(codes)}. "
                        if codes
                        else "The operating system reported no exit code. "
                    )
                    + "Finishing them one at a time in the main process "
                    "(slower, identical results). If the run stops again, "
                    "the last file named below is the one being processed."
                )
                crashed_tasks.sort(key=lambda task: str(task[0]))
                classify_in_process(crashed_tasks)
        else:
            classify_in_process(tasks)
    finally:
        if cache_workspace is not None:
            cache_workspace.cleanup()

    resources = {
        "cpu_slots": cpu_budget,
        "ram_budget_gb": round(memory_budget / 1024**3, 1),
        "files_at_once": file_workers,
    }
    return results, failures, resources
