"""Classifying a file's windows: one window's job, and running a file's windows in order or on bounded threads.

Everything a run configures arrives as arguments (`RunSettings`, the exports), so a window job needs no module state.
"""
from __future__ import annotations

import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass

import numpy as np

from .exports import DtmExport, TinExport, export_window_dtm, export_window_tin
from .reclassify import InsufficientGroundSupport, application_code, classify_block
from .utils.helpers import format_memory
from .utils.loggers import _LOG_STATE, PERFORMANCE_STAGE_TIMINGS, log
from .utils.params import RunSettings
from .windows import buffered_indices_for_window, core_point_mask, estimate_window_memory

# A progressive TIN is undefined for fewer than three usable, non-collinear
# points.  Such tiny/sliver tiles retain their incoming classification rather
# than aborting an otherwise successful large batch.  The limit prevents a
# genuine algorithm failure on a normal tile from being silently hidden.
SPARSE_FALLBACK_MAX_BUFFERED_POINTS = 10_000


@dataclass(frozen=True)
class WindowResult:
    """Core-only output returned by one buffered processing window."""

    number: int
    core_indices: np.ndarray
    classification: np.ndarray
    hag: np.ndarray
    model_count: int
    messages: tuple[str, ...]


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
    settings: RunSettings,
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
                settings,
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
                local_x, local_y, local_z,
                local_class == application_code(settings.feature_codes, "ground"),
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
    settings: RunSettings,
    core_maxima: tuple[float, float] | None = None,
    allow_sparse_fallback: bool = False,
    tin_export: TinExport | None = None,
    dtm_export: DtmExport | None = None,
) -> None:
    """Process windows sequentially or with bounded shared-memory workers."""
    window_count = len(windows)
    block_overlap = settings.block_overlap
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
                x, y, core, x_order, sorted_x, block_overlap
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
                settings,
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
                x, y, core, x_order, sorted_x, block_overlap
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
                settings,
                allow_sparse_fallback,
                tin_export,
                dtm_export,
            )
            pending[future] = estimate
            in_flight_memory += estimate

        while pending:
            collect_one_or_more()
