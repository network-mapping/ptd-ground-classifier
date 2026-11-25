"""Batch pipeline: reads LAS/LAZ tiles, classifies them with the compiled
progressive-TIN core (see fast_ground.py) and writes them back.

How a block of points gets its class codes (ground, the three vegetation
bands, noise) is in reclassify.py; this module reads the tiles, cuts them into
windows with the neighbouring tiles' context, schedules the work across
processes and threads, and writes the results.

Every threshold is in metres. Input X/Y/Z are converted to metres once, in
memory, in classify_file(), using the run's unit (see las.LasDataset.metres_per_unit); the
LAS coordinates on disk are never rewritten.
"""

from __future__ import annotations

import faulthandler
import math
import os
import sys
import tempfile
import time
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    wait,
)
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path

# If a compiled library crashes the interpreter, print where it happened.
# Module-level so spawned helper processes get it too.
try:
    faulthandler.enable()
except Exception:  # no usable stderr; diagnostics only, never fatal
    pass

# Small batched solves gain nothing from a BLAS thread pool, and several files
# or windows run at once.
for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

try:
    import numpy as np
except ImportError as exc:
    missing = getattr(exc, "name", "a required package")
    raise SystemExit(
        f"Missing Python package: {missing}\n"
        "Install dependencies with: pip install -r requirements.txt"
    ) from exc

from . import __version__
from .exports import file_exports
from .halo import (
    DENSITY_ESTIMATE_MARGIN,
    HALO_CACHE_BYTES_PER_POINT,
    HaloCacheInfo,
    SourceDensity,
    build_halo_caches,
    density_points_in_rect,
    expand_bounds,
    load_halo_points,
)
from .las import (
    UNIT_LABELS,
    US_FEET_TO_METRES,
    LasDataset,
    LasFile,
    drop_extra_dimensions,
    write_tile,
)
from .processes import dead_helper_exit_codes, helpers_usable, pool_processes
from .reclassify import (
    application_code,
    reset_input_class_mask,
)
from .tiles import footprint_function, halo_in_input_units
from .utils.helpers import format_memory
from .utils.loggers import _LOG_STATE, PERFORMANCE_STAGE_TIMINGS, log
from .utils.params import RunSettings
from .window_runner import SPARSE_FALLBACK_MAX_BUFFERED_POINTS, process_file_windows
from .windows import estimate_window_memory, processing_windows, window_cores_for_extent

# Written into every output header so a file shows which version made it.
GENERATING_SOFTWARE = f"ground-classifier {__version__}"

# The settings of the run (engine, HAG bands, feature codes, neighbour halo,
# TIN and DTM export, window size and overlap) live in this module global,
# set in every process by configure_run() because spawned helpers do not
# inherit it. The default is for a caller that never configures a run.
RUN_SETTINGS = RunSettings()


def configure_run(settings: RunSettings) -> None:
    """Apply a run's settings to this process. Must be called at the start of
    every worker process (Windows multiprocessing spawns fresh interpreters
    that do not inherit the parent's mutated globals) - mirrors the
    configure_runtime() pattern."""
    global RUN_SETTINGS
    RUN_SETTINGS = settings


OUTPUT_SUFFIX = "_AdaptiveGround"

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
# Raised from a hard 4, but deliberately matched to windows.TURBO_MAX_INTERNAL_WINDOWS
# (8) rather than pushed to the full CPU budget - see the overlap-ratio
# note in windows.py. window_workers is still clamped to the real window count for
# this tile, so smaller/sparser tiles are unaffected.
MAX_PARALLEL_WINDOWS = 8
RAM_BUDGET_FRACTION = 0.55
MIN_FREE_RAM_GB = 4.0

# Conservative peak-memory estimates used only to decide concurrency. They do
# not allocate memory and do not alter classification. If a block is larger
# than the safe budget it still runs, but by itself. (The per-window figures
# are in windows.py.)
FILE_FIXED_MEMORY_MB = 384.0
FULL_FILE_BYTES_PER_POINT = 96

# TerraScan-style neighbouring-tile context: see halo.py (the caches),
# tiles.py (the footprints) and LasDataset.neighbours (which files are
# neighbours). The halo is RunSettings.neighbour_halo_m, converted to the
# files' unit by tiles.halo_in_input_units.

# Scheduling only: how much work N windows of one file get done when they
# run on parallel threads of one process, relative to one window alone
# (N ** 0.95: four windows count as about 3.7). Separate files in separate
# processes count in full. Two windows on two threads measured at very
# nearly twice the speed of one after the other (the heavy steps release
# Python's interpreter lock), hence a value close to 1. Used solely to
# choose between "more files at once" and "more windows at once" - it has
# no effect on any result.
WINDOW_THREAD_SCALING = 0.95


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


def configure_runtime(prefix: str = "") -> None:
    """Set the process-local log-line prefix (a worker tags its lines with its file)."""
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


# Output-header policy. TerraScan and some upstream workflows may leave a
# temporary Extra Bytes dimension named "Distance" in the input. It is not
# required by this classifier and should not be propagated to delivery files.
# All other input Extra Bytes dimensions and VLRs are preserved.
STRIP_OUTPUT_EXTRA_DIMENSIONS = ("Distance",)


def classify_file(
    source: LasFile,
    output_path: Path,
    window_workers: int = 1,
    memory_budget: int | None = None,
    neighbour_caches: tuple[HaloCacheInfo, ...] = (),
    target_bounds: tuple[float, float, float, float] | None = None,
    neighbour_bounds: tuple[float, float, float, float] | None = None,
    input_unit_scale: float = US_FEET_TO_METRES,
) -> FileResult:
    started = time.perf_counter()
    input_path = source.path
    log(f"\nReading: {input_path}")
    read_started = time.perf_counter()
    tile = source.read()
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
    target_x_usft = np.asarray(tile.x, dtype=np.float64)
    target_y_usft = np.asarray(tile.y, dtype=np.float64)
    target_z_usft = np.asarray(tile.z, dtype=np.float64)
    target_classification = np.asarray(
        tile.classification, dtype=np.uint8
    )
    target_point_count = target_x_usft.size
    if target_point_count == 0:
        removed_extra = drop_extra_dimensions(tile, STRIP_OUTPUT_EXTRA_DIMENSIONS)
        if removed_extra:
            log(
                "Removed temporary output Extra Bytes dimension(s): "
                + ", ".join(removed_extra)
            )
        log("Empty input tile: writing output with classification/header policy applied.")
        write_tile(output_path, tile, generating_software=GENERATING_SOFTWARE)
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

    # unit_scale is metres-per-input-unit: US_FEET_TO_METRES for a
    # feet project, 1.0 for a metres project (las.LasDataset.metres_per_unit).
    # Callers that already resolved this pass it in; falling back to feet here
    # only matters for direct calls from outside run_batch()'s resolution, kept only
    # for backward compatibility with any external script calling classify_file
    # directly.
    unit_scale = (
        input_unit_scale if input_unit_scale is not None
        else US_FEET_TO_METRES
    )
    halo = halo_in_input_units(RUN_SETTINGS.neighbour_halo_m, unit_scale)

    halo_x_usft, halo_y_usft, halo_z_usft, halo_classification = load_halo_points(
        neighbour_caches, neighbour_bounds, halo
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
            f"{halo:.3f} native-unit halo "
            f"({RUN_SETTINGS.neighbour_halo_m:g} m physical)"
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
    reset_global = reset_input_class_mask(classification_out, RUN_SETTINGS.feature_codes)
    classification_out[reset_global] = application_code(RUN_SETTINGS.feature_codes, "unclassified")
    hag_out = np.full(x.size, np.nan, dtype=np.float32)

    windows = list(processing_windows(target_x, target_y, RUN_SETTINGS.block_size))
    log(
        f"Target points: {target_point_count:,}; context points: "
        f"{x.size - target_point_count:,}; processing window(s): "
        f"{len(windows)}"
    )
    actual_window_workers = max(1, min(window_workers, len(windows)))
    if memory_budget is None:
        memory_budget = safe_parallel_memory_budget()
    else:
        # Recheck after LAS decompression and full-cloud arrays are resident.
        memory_budget = min(
            memory_budget,
            safe_parallel_memory_budget(),
        )
    tin_export, dtm_export = file_exports(
        RUN_SETTINGS, input_path.stem, unit_scale, [core for _, core in windows]
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
        RUN_SETTINGS,
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

    tile.classification = classification_out[:target_point_count]

    # Do not propagate TerraScan's temporary Distance Extra Byte into the
    # classified output. This changes only the point-record layout/header; XYZ,
    # standard LAS dimensions, classifications and all other Extra Bytes/VLRs
    # remain intact.
    removed_extra = drop_extra_dimensions(tile, STRIP_OUTPUT_EXTRA_DIMENSIONS)
    if removed_extra:
        log(
            "Removed temporary output Extra Bytes dimension(s): "
            + ", ".join(removed_extra)
        )

    log(f"Writing: {output_path}")
    write_started = time.perf_counter()
    write_tile(output_path, tile, generating_software=GENERATING_SOFTWARE)
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
        for code in RUN_SETTINGS.feature_codes.application_codes
    }
    log(
        "Class counts: "
        + ", ".join(f"{code}={count:,}" for code, count in counts.items())
    )
    log(f"Completed in {elapsed / 60.0:.1f} minutes")
    return FileResult(elapsed, int(target_point_count), counts)


def estimate_target_memory(
    source: LasFile,
    neighbour_bounds: tuple[float, float, float, float],
    halo: float,
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
    header = source.header
    point_count = max(0, int(header.point_count))
    cache_points = sum(info.point_count for info in neighbour_caches)
    file_resident = int(
        FILE_FIXED_MEMORY_MB * 1024**2
        + point_count
        * max(FULL_FILE_BYTES_PER_POINT, header.point_format_size + 64)
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
        bxmin, bymin, bxmax, bymax = header.extent_xy
        cores = [
            core
            for _, core in window_cores_for_extent(
                bxmin * unit_scale,
                bymin * unit_scale,
                bxmax * unit_scale,
                bymax * unit_scale,
                point_count,
                RUN_SETTINGS.block_size,
            )
        ]
        window_count = max(1, len(cores))

        expanded = expand_bounds(neighbour_bounds, halo)

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
        overlap = RUN_SETTINGS.block_overlap / max(unit_scale, 1.0e-9)
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
    neighbour_caches: tuple[HaloCacheInfo, ...],
    target_bounds: tuple[float, float, float, float],
    neighbour_bounds: tuple[float, float, float, float],
    settings: RunSettings,
    unit_scale: float,
    window_workers: int = 1,
    window_memory_budget: int | None = None,
) -> FileResult:
    """Top-level process target; paths avoid transferring point arrays.

    window_workers > 1 runs this file's windows on parallel threads, held to
    window_memory_budget by classify_file's own window scheduler - the same
    mechanism a single file run in the main process has always used.
    """
    input_path = Path(input_path_text)
    configure_runtime(f"[{input_path.stem}] ")
    configure_run(settings)
    return classify_file(
        LasFile.from_file(input_path),
        Path(output_path_text),
        window_workers=max(1, int(window_workers)),
        memory_budget=window_memory_budget,
        neighbour_caches=neighbour_caches,
        target_bounds=target_bounds,
        neighbour_bounds=neighbour_bounds,
        input_unit_scale=unit_scale,
    )


def reusable_existing_output(source: LasFile, output_path: Path) -> bool:
    """Return true for a complete output of this version with the expected layout.

    Output intentionally omits configured temporary Extra Bytes such as
    ``Distance``, so byte-for-byte point-format size equality with the input is
    no longer a valid resume test. Output from another version is rebuilt.
    """
    if not output_path.is_file() or output_path.stat().st_size == 0:
        return False
    if output_path.stat().st_mtime < source.path.stat().st_mtime:
        return False
    try:
        stripped = {name.casefold() for name in STRIP_OUTPUT_EXTRA_DIMENSIONS}
        source_header = source.header
        expected_extra = tuple(
            name
            for name in source_header.extra_dimension_names
            if name.casefold() not in stripped
        )
        expected_signature = (
            int(source_header.point_count),
            int(source_header.point_format_id),
            expected_extra,
        )
        output_header = LasFile.from_file(output_path).header
        if output_header.generating_software != GENERATING_SOFTWARE:
            return False
        output_signature = (
            int(output_header.point_count),
            int(output_header.point_format_id),
            output_header.extra_dimension_names,
        )
        return expected_signature == output_signature
    except Exception:
        return False


def run_batch(
    tasks: list[tuple[Path, Path]],
    dataset: LasDataset,
    settings: RunSettings,
    requested_workers: int,
    requested_ram_limit_gb: float,
) -> tuple[dict[Path, FileResult], list[tuple[Path, BaseException]], dict]:
    """Classify ``tasks`` (input, output) with the resource-aware scheduler.

    ``dataset`` holds every file the run knows about: the tasks' inputs and
    the other tiles beside them (``LasDataset(..., siblings=True)``), whose
    points give a tile its neighbour context but are not written unless they
    are tasks too. Its units are the run's. Returns the per-file results, the
    files that failed, and the CPU/RAM limits the scheduler used.
    """
    configure_run(settings)
    unit_scale = dataset.metres_per_unit
    halo = halo_in_input_units(settings.neighbour_halo_m, unit_scale)
    footprint_of = footprint_function(dataset, halo)

    unit_summary = UNIT_LABELS[dataset.units]
    low_veg, medium_veg, high_veg = settings.hag_bands
    low_code, medium_code, high_code = (
        application_code(settings.feature_codes, name)
        for name in ("low_vegetation", "medium_vegetation", "high_vegetation")
    )
    log(
        f"Unit mode: this run's input is {unit_summary} (from CRS metadata, "
        "the config file or --units). Classification geometry is normalized "
        "to metres in memory; LAS XYZ is never rewritten. Output vegetation: "
        f"class {low_code} <={low_veg:g} m, class {medium_code} {low_veg:g}-{medium_veg:g} m, "
        f"class {high_code} >={medium_veg:g} m (see the hag section of the config file)."
    )
    log(
        "Classes: input "
        + ", ".join(str(code) for code in settings.feature_codes.mapped_input_feature_codes)
        + " re-labelled as "
        + ", ".join(f"{name} {application_code(settings.feature_codes, name)}" for name in settings.feature_codes.application_names)
        + " (see the feature_codes section of the config file)."
    )

    if settings.tin_export_dir is not None:
        log(
            f"TIN export: one PLY per window in {settings.tin_export_dir}"
            + (f", triangle edges <= {settings.tin_max_edge:g} m" if settings.tin_max_edge > 0 else "")
        )
    if settings.dtm_export_dir is not None:
        log(
            f"DTM export: one GeoTIFF per window in {settings.dtm_export_dir}, cell "
            + (
                f"{settings.dtm_cell:g} (input unit)" if settings.dtm_cell > 0
                else "1 m / 3 ft (default for the file's unit)"
            )
        )
    changed = settings.engine.changed()
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

    # The tasks still discover spatial neighbours among the dataset's other
    # files. Those neighbours provide context but are not written unless they
    # were also supplied as targets.
    target_paths = [input_path for input_path, _ in tasks]
    neighbour_bounds_by_path = {
        source.path: footprint_of(source) for source in dataset.sources
    }
    neighbours_by_target = {
        path: tuple(source.path for source in sources)
        for path, sources in dataset.neighbours(
            halo, targets=target_paths, footprint=footprint_of
        ).items()
    }

    source_requests: dict[
        Path, set[tuple[float, float, float, float]]
    ] = {}
    for target_path in target_paths:
        expanded = expand_bounds(neighbour_bounds_by_path[target_path], halo)
        for source in neighbours_by_target[target_path]:
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
            target_path: tuple(
                HaloCacheInfo(
                    halo_caches[source].path,
                    halo_caches[source].point_count,
                )
                for source in neighbours_by_target[target_path]
                if source in halo_caches
            )
            for target_path in target_paths
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
                    dataset.file(input_path),
                    neighbour_bounds_by_path[input_path],
                    halo,
                    tuple(
                        halo_caches[source]
                        for source in neighbours_by_target[input_path]
                        if source in halo_caches
                    ),
                    unit_scale,
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
                configure_runtime()
                try:
                    results[input_path] = classify_file(
                        dataset.file(input_path),
                        output_path,
                        window_workers=window_workers,
                        memory_budget=memory_budget,
                        neighbour_caches=target_caches[input_path],
                        target_bounds=dataset.file(input_path).header.extent_xy,
                        neighbour_bounds=neighbour_bounds_by_path[input_path],
                        input_unit_scale=unit_scale,
                    )
                except Exception as exc:
                    failures.append((input_path, exc))
                    log(f"FAILED {input_path.name}: {exc}")

        if file_workers > 1 and not helpers_usable():
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
                        _, threads, exclusive = run_plans[input_path]
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
                                target_caches[input_path],
                                dataset.file(input_path).header.extent_xy,
                                neighbour_bounds_by_path[input_path],
                                settings,
                                unit_scale,
                                threads,
                                (
                                    memory_budget
                                    if exclusive
                                    else threads * plan.window
                                ),
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
