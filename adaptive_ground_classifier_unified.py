#!/usr/bin/env python3
"""
Adaptive progressive-TIN ground classification for LAS/LAZ, with
trusted-ground recovery. Input may be US survey feet or metres - see
"Input units" below; this is auto-detected per file, not fixed to one unit.

The implementation follows the documented logic of:

  Axelsson, P. (2000), "DEM Generation from Laser Scanner Data Using
  Adaptive TIN Models", ISPRS Archives, Vol. XXXIII, Part B4, pp. 110-117.

It also implements TerraScan-style controls described in the Ground routine:

  * initial low seeds selected using a maximum-building-size grid;
  * iterative distance and angle tests against the current triangle,
    slope-adaptive per TerraScan's own documented operator guidance
    (smaller angle in flat terrain, larger in mountainous terrain) -
    applied automatically per local region rather than one fixed
    project-wide value;
  * maximum terrain slope;
  * dynamic reduction of iteration angle inside small triangles;
  * a second, stricter pass that reuses current ground to fill genuine terrain;
  * a conservative unsupported-low guard for missed gross errors;
  * trusted-ground recovery that prevents unsupported vegetation/building surfaces from self-supporting;
  * thin, slope-aware progressive membership densification;
  * block processing with overlap for larger point clouds;
  * a 10 m-equivalent context halo from spatially adjacent LAS/LAZ tiles,
    expressed in whichever unit this project's input actually uses;
  * CPU/RAM-aware parallel files and processing windows.


Input classes 2, 3, 4, 5 and 13 are reset to class 0 before processing.
Other existing classes are preserved.

Final classification (HAG = height above ground, in metres - this is the
unit every internal threshold is defined in, regardless of the input
file's own unit):
    2  Ground
    3  Low vegetation:    non-ground, HAG <= low_veg_max
    4  Medium vegetation: low_veg_max < HAG < medium_veg_max
    5  High vegetation:   medium_veg_max <= HAG <= high_veg_max
    7  Existing noise, confirmed isolated low errors, and HAG > high_veg_max

low_veg_max / medium_veg_max / high_veg_max default to 0.5 / 3.0 / 200.0 m
but are project-editable via hag_config.json (auto-created beside this
script on first run) - they are not fixed constants. See load_hag_config()
and configure_hag_bands().

Class 13 is NOT produced by this build. It remains in RESET_INPUT_CLASSES so
legacy input class-13 points are reset and reprocessed rather than preserved.
Every threshold in this file - PTD distance/angle, the HAG bands above,
every recovery and completion-stage tolerance - is defined and applied in
metres. There is exactly one unit conversion in the whole pipeline: input
X/Y/Z are converted to metres, once, in memory, inside classify_file(),
using that file's own resolved input unit (see next paragraph). Nothing
downstream of that point needs its own unit awareness.

Input units: each file's coordinate unit (US survey feet or metres) is
resolved by resolve_input_unit_scale() - see units_config.json
(auto-created beside this script on first run). By default ("auto"), the
unit is read from the file's own CRS/WKT metadata; if a file has no CRS
metadata (common in practice) and no explicit override is set, the tool
refuses to guess and stops with an error naming the file, rather than risk
silently misinterpreting feet as metres (or vice versa) and corrupting
every coordinate by a factor of ~3.28x. Original XYZ coordinates on disk
are never changed regardless of unit. HAG is calculated only in memory
and is not written to the output. The classifier adds no HAG or Distance Extra
Bytes dimensions and adds no Extra Bytes VLRs. Existing input dimensions and
VLRs are preserved unchanged EXCEPT that an existing Extra Bytes dimension
named "Distance" is deliberately removed from the output.
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import math
import os
import re
import struct
import sys
import tempfile
import threading
import time
import warnings
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

# If a compiled library crashes the interpreter (an access violation inside a
# LAS/LAZ reader, for example), print where it happened instead of dying
# silently. This is module-level on purpose: Windows starts every helper
# process by re-importing this file, so the helpers get it too.
try:
    faulthandler.enable()
except Exception:  # no usable stderr; diagnostics only, never fatal
    pass

# Small batched 3x3 solves do not benefit from a large hidden BLAS thread
# pool. Keeping them single-threaded prevents oversubscription when files or
# processing windows are classified concurrently. cKDTree parallelism is
# controlled explicitly below.
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
    from scipy.spatial import Delaunay, QhullError, cKDTree
except ImportError as exc:
    missing = getattr(exc, "name", "a required package")
    raise SystemExit(
        f"Missing Python package: {missing}\n"
        "Install dependencies with:\n"
        '  py -3 -m pip install "laspy[lazrs]" numpy scipy'
    ) from exc


# Exact US survey foot definition. This script NEVER rewrites LAS XYZ; the
# factor is used only for temporary in-memory normalization and native spatial
# scheduling.
US_SURVEY_FEET_PER_METRE = 3937.0 / 1200.0
METRES_PER_US_SURVEY_FOOT = 1200.0 / 3937.0


def metres_to_usft(value: float) -> float:
    return float(value) * US_SURVEY_FEET_PER_METRE


def usft_to_metres(value: float) -> float:
    return float(value) * METRES_PER_US_SURVEY_FOOT


# =============================================================================
# INPUT UNIT DETECTION — metres vs US survey feet
#
# The classification geometry below is calibrated in metres. Earlier builds
# of this tool (and the FPL project it was originally built for) always
# assumed US survey feet input and converted unconditionally. That is wrong
# for a metric project (e.g. UTM-metre LAS files) - it silently shrinks every
# coordinate by ~3.28x and corrupts the whole classification without any
# error message. This section detects the real input unit per file, using
# (in order): an explicit per-folder override, then the file's own CRS
# metadata (WKT unit text) if present, then refuses to guess.
#
# VLR/CRS metadata in a LAS file is NEVER compressed, even in a .laz file
# (only the point records are), so this works identically for .las and .laz
# inputs and does not depend on laspy's CRS-parsing API surface (which
# varies across laspy versions) - it reads the same fixed binary layout the
# LAS spec guarantees.
# =============================================================================

UNITS_CONFIG_FILENAME = "units_config.json"

# Every "foot" appears in "international foot" text too (e.g. some legacy
# state-plane definitions), so the negative check has to come first: if the
# text says "US survey foot"/"ftUS"/"USfoot", or plain "foot"/"feet" without
# an explicit "international" qualifier, treat it as US survey feet. This
# matches the only two linear units this classifier supports (the original
# Turbo2 script only ever handled metre or US survey foot).
_METRE_UNIT_MARKERS = ("metre", "meter")
_USFT_UNIT_MARKERS = ("us survey foot", "ftus", "us foot", "foot", "feet")

_LAS_HEADER_FIELDS = [
    ("file_signature", "4s"), ("file_source_id", "H"), ("global_encoding", "H"),
    ("guid_1", "I"), ("guid_2", "H"), ("guid_3", "H"), ("guid_4", "8s"),
    ("version_major", "B"), ("version_minor", "B"),
    ("system_identifier", "32s"), ("generating_software", "32s"),
    ("file_creation_doy", "H"), ("file_creation_year", "H"),
    ("header_size", "H"), ("offset_to_point_data", "I"),
    ("num_variable_length_records", "I"),
    ("point_data_format_id", "B"), ("point_data_record_length", "H"),
    ("legacy_number_of_point_records", "I"),
    ("legacy_number_of_points_by_return", "5I"),
    ("x_scale", "d"), ("y_scale", "d"), ("z_scale", "d"),
    ("x_offset", "d"), ("y_offset", "d"), ("z_offset", "d"),
    ("max_x", "d"), ("min_x", "d"), ("max_y", "d"), ("min_y", "d"),
    ("max_z", "d"), ("min_z", "d"),
]


def _read_las_header_size_and_vlr_count(path: Path) -> tuple[int, int]:
    """Read just enough of the public header block to locate the VLRs."""
    with open(path, "rb") as f:
        raw = f.read(400)
    offset = 0
    values = {}
    for name, fmt in _LAS_HEADER_FIELDS:
        size = struct.calcsize(fmt)
        val = struct.unpack_from("<" + fmt, raw, offset)
        values[name] = val[0] if len(val) == 1 and not fmt.endswith("s") else val
        offset += size
    return int(values["header_size"]), int(values["num_variable_length_records"])


def _read_las_vlr_text(path: Path) -> str:
    """Concatenate all VLR payload bytes that look like text (WKT/GeoASCII
    projection descriptions), decoded permissively. Never raises - unit
    detection degrading to 'unknown' is safe; a crash here is not."""
    try:
        header_size, num_vlrs = _read_las_header_size_and_vlr_count(path)
        chunks = []
        with open(path, "rb") as f:
            f.seek(header_size)
            for _ in range(num_vlrs):
                vlr_header = f.read(54)
                if len(vlr_header) < 54:
                    break
                _reserved, user_id, record_id, length, _desc = struct.unpack(
                    "<H16sHH32s", vlr_header
                )
                user_id = user_id.split(b"\x00")[0].decode("ascii", "replace")
                payload = f.read(length)
                if user_id in ("LASF_Projection", "LASF_Spec") or record_id in (
                    2111, 2112, 34735, 34736, 34737,
                ):
                    chunks.append(payload.decode("ascii", "ignore"))
        return " ".join(chunks)
    except (OSError, struct.error, ValueError):
        return ""


def detect_input_unit_scale(path: Path) -> tuple[float | None, str]:
    """Return (metres_per_input_unit, description). metres_per_input_unit is
    METRES_PER_US_SURVEY_FOOT for a feet input, 1.0 for a metres input, or
    None if the file's CRS metadata does not clearly say either way.

    WKT always defines the underlying ellipsoid/datum in metres by EPSG/ISO
    convention, regardless of what unit the actual stored X/Y/Z coordinates
    are in - so a plain "does 'metre' or 'foot' appear anywhere" search
    reliably picks up the WRONG unit whenever both are present (which is
    common: a feet-projected CRS still defines its ellipsoid in metres).
    The unit that actually applies to the stored coordinates is the LAST
    LENGTHUNIT/UNIT clause in the WKT, which is always the outermost
    PROJCS/PROJCRS's own axis unit - everything nested inside (the base
    GEOGCS, the ellipsoid, individual PARAMETER units) comes earlier in the
    text. Taking the rightmost match is a reliable, version-independent way
    to get this right without a full WKT bracket-nesting parser.
    """
    text = _read_las_vlr_text(path).lower()
    if not text:
        return None, "no CRS/projection metadata found in the file"

    last_metre_pos = max(
        (text.rfind(marker) for marker in _METRE_UNIT_MARKERS), default=-1
    )
    last_usft_pos = max(
        (text.rfind(marker) for marker in _USFT_UNIT_MARKERS), default=-1
    )
    if last_metre_pos < 0 and last_usft_pos < 0:
        return None, "CRS metadata present but no recognizable unit text"
    if last_usft_pos > last_metre_pos:
        return METRES_PER_US_SURVEY_FOOT, (
            "last linear-unit clause in CRS metadata says feet"
        )
    return 1.0, "last linear-unit clause in CRS metadata says metres"


def default_units_config() -> dict:
    return {
        "_comment": (
            "Input coordinate unit for LAS/LAZ files in this folder. "
            "'auto' (default) reads each file's own CRS metadata and only "
            "proceeds when the unit is unambiguous; the tool refuses to "
            "guess rather than silently corrupt output. If a file has no "
            "CRS metadata (common - many production LAS files omit it), "
            "set this to 'usft' or 'metres' explicitly for this folder. "
            "This setting applies to every file processed from this folder "
            "in one run - keep files of different units in separate folders."
        ),
        "units": "auto",
    }


def load_units_config(script_dir: Path) -> str:
    """Return 'auto', 'usft', or 'metres'. Creates the config with 'auto' on
    first run, exactly like hag_config.json."""
    config_path = script_dir / UNITS_CONFIG_FILENAME
    if not config_path.exists():
        try:
            config_path.write_text(
                json.dumps(default_units_config(), indent=2), encoding="utf-8"
            )
            log(f"Wrote default units config: {config_path}")
        except OSError as exc:
            log(f"Could not write default units config ({exc}); using 'auto'.")
        return "auto"
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        units = str(data["units"]).strip().lower()
        if units not in ("auto", "usft", "metres", "meters"):
            raise ValueError(
                f"units must be 'auto', 'usft', or 'metres' (got {units!r})"
            )
        return "metres" if units == "meters" else units
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        log(
            f"WARNING: could not read {config_path} ({exc}); "
            "defaulting to 'auto'."
        )
        return "auto"


PERFORMANCE_CONFIG_FILENAME = "performance_config.json"
STAGES_CONFIG_FILENAME = "stages_config.json"

# Optional pipeline stages that can be switched off from stages_config.json.
# UNLIKE performance_config.json THESE CHANGE THE OUTPUT. Every one defaults
# to on, which is the standard, validated configuration.
#   key in the file  ->  module switch it controls
STAGE_SWITCHES = {
    "two_sided_bank_crest_membership": "TWO_SIDED_BREAK_MEMBERSHIP",
}


def default_stages_config() -> dict:
    return {
        "_comment": (
            "Optional stages. true = standard output (default). Setting a "
            "stage to false CHANGES THE RESULT - test on a few tiles and "
            "compare (Compare_Outputs.bat) before using it on a delivery. "
            "two_sided_bank_crest_membership: the final pass that adds "
            "extra class-2 points on banks and crests. It never changes "
            "the terrain model or any height above ground, only whether "
            "those points are labelled ground (2) or low vegetation (3). "
            "It is the most expensive stage: more than half of the "
            "processing time on dense data."
        ),
        "two_sided_bank_crest_membership": True,
    }


def apply_stages_config(script_dir: Path) -> dict[str, bool]:
    """Apply stages_config.json (if present) to this process's switches.

    Called once when the module is loaded, so every helper process - which
    on Windows starts by loading this file afresh - picks up the same
    settings without anything having to be passed to it. A missing,
    unreadable or malformed file leaves every stage on. Returns the stages
    that are switched OFF.
    """
    disabled: dict[str, bool] = {}
    config_path = script_dir / STAGES_CONFIG_FILENAME
    try:
        if not config_path.exists():
            return disabled
        data = json.loads(config_path.read_text(encoding="utf-8"))
        for key, switch in STAGE_SWITCHES.items():
            value = data.get(key, True)
            if value is False:
                globals()[switch] = False
                disabled[key] = False
            elif value is not True:
                raise ValueError(f"{key} must be true or false (got {value!r})")
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        for switch in STAGE_SWITCHES.values():
            globals()[switch] = True
        disabled = {}
        print(
            f"WARNING: could not read {config_path} ({exc}); "
            "all stages left ON.",
            flush=True,
        )
    return disabled


def ensure_stages_config(script_dir: Path) -> None:
    """Write the default stages_config.json on first run and say plainly in
    the log when any stage is switched off."""
    config_path = script_dir / STAGES_CONFIG_FILENAME
    if not config_path.exists():
        try:
            config_path.write_text(
                json.dumps(default_stages_config(), indent=2),
                encoding="utf-8",
            )
            log(f"Wrote default stages config: {config_path}")
        except OSError as exc:
            log(f"Could not write default stages config ({exc}).")
    for key in DISABLED_STAGES:
        log(
            f"STAGE SWITCHED OFF: {key} ({STAGES_CONFIG_FILENAME}). "
            "Output will differ from the standard configuration."
        )


ENGINE_CONFIG_FILENAME = "engine_config.json"

# Which ground engine classifies a block.
#   "classic"  the original pipeline (this file's own stages). Default.
#   "fast"     the compiled progressive-TIN core (fast_ground.py + ptdcore).
# The two engines are different programs: their ground is close, NOT identical.
ENGINE = "classic"
ENGINE_FALLBACK_REASON = ""
# Settings handed to the fast engine (metres). Names follow TerraScan's
# Ground routine where TerraScan has the same setting.
FAST_ENGINE_SETTINGS: dict = {}
FAST_ENGINE_KEYS = {
    "max_building_size": "max_building_size",
    "terrain_angle": "terrain_angle",
    "iteration_angle": "iteration_angle",
    "steep_iteration_angle": "steep_iteration_angle",
    "iteration_distance": "iteration_distance",
    "reduce_iteration_angle_when_edge_shorter_than": "reduce_angle_edge",
    "stop_triangulation_when_edge_shorter_than": "stop_edge",
    "ground_band": "surface_band",
    "follow_surface": None,
    "remove_spikes": None,
}
FAST_ENGINE_DEFAULTS = {
    "max_building_size": 60.0,
    "terrain_angle": 70.0,
    "iteration_angle": 6.0,
    "steep_iteration_angle": 10.0,
    "iteration_distance": 1.4,
    "reduce_iteration_angle_when_edge_shorter_than": 2.0,
    "stop_triangulation_when_edge_shorter_than": 0.0,
    "ground_band": 0.02,
    "follow_surface": True,
    "remove_spikes": True,
}
FAST_ENGINE_SWITCHES = ("follow_surface", "remove_spikes")
# engine_config.json files written before this number are from the first
# fast engine: their fast_engine values would bring its spikes back, so they
# are replaced by the defaults above (the "engine" choice is kept).
ENGINE_CONFIG_VERSION = 2
ENGINE_CONFIG_UPGRADED = False
# With the fast engine a whole tile is one window up to this many points
# (the classic engine needs small windows; the compiled core does not).
FAST_ENGINE_WINDOW_POINTS = 120_000_000
FAST_ENGINE_WINDOW_FIXED_MEMORY_MB = 256.0
FAST_ENGINE_WINDOW_BYTES_PER_POINT = 64
# A point the low-point filter kept out of the surface is labelled noise (7)
# only if it ends up at least this far below the finished ground (metres).
FAST_ENGINE_LOW_NOISE_DEPTH = 0.5


def default_engine_config(engine: str = "classic") -> dict:
    return {
        "_comment": (
            "engine: \"classic\" = the original pipeline. \"fast\" = the "
            "compiled core: many times faster, ground close to but NOT "
            "identical with classic. The fast_engine values are in METRES "
            "whatever the project unit; the first seven carry the names of "
            "TerraScan's Ground routine settings. max_building_size: one "
            "starting point per square of this size - lower it (20 to 30) "
            "for hilly projects where buildings are already in their own "
            "class. steep_iteration_angle: the iteration angle used on "
            "steep ground (iteration_angle is used on flat ground). "
            "ground_band: points this close to the ground surface are "
            "ground too, at most one per 0.3 m square - this is what gives "
            "a dense ground without roughness; 0 = surface points only. "
            "follow_surface: let the ground continue over hill tops, banks "
            "and ridges the plain routine cannot climb. remove_spikes: take "
            "single points that stand out of the surface back out."
        ),
        "version": ENGINE_CONFIG_VERSION,
        "engine": engine,
        "fast_engine": dict(FAST_ENGINE_DEFAULTS),
    }


def apply_engine_config(script_dir: Path) -> str:
    """Apply engine_config.json (if present) to this process.

    Called once when the module is loaded, so every helper process picks up
    the same engine. If the fast engine is asked for but its compiled core
    cannot be loaded on this machine, every process falls back to the
    classic engine in the same way and the reason is kept for the log.
    """
    global ENGINE, ENGINE_FALLBACK_REASON, FAST_ENGINE_SETTINGS
    global ENGINE_CONFIG_UPGRADED
    global TURBO_TARGET_CORE_POINTS, BLOCK_SIZE
    global WINDOW_FIXED_MEMORY_MB, WINDOW_BYTES_PER_POINT
    config_path = script_dir / ENGINE_CONFIG_FILENAME
    engine = "classic"
    values = dict(FAST_ENGINE_DEFAULTS)
    try:
        if config_path.exists():
            data = json.loads(config_path.read_text(encoding="utf-8"))
            engine = str(data.get("engine", "classic")).strip().lower()
            if engine not in ("classic", "fast"):
                raise ValueError(
                    f"engine must be \"classic\" or \"fast\" (got {engine!r})"
                )
            file_values = dict(data.get("fast_engine", {}))
            try:
                file_version = int(data.get("version", 1))
            except (TypeError, ValueError):
                file_version = 1
            if file_version < ENGINE_CONFIG_VERSION:
                file_values = {}
                ENGINE_CONFIG_UPGRADED = True
            for key, value in file_values.items():
                if key not in FAST_ENGINE_KEYS:
                    raise ValueError(f"unknown fast_engine setting {key!r}")
                if key in FAST_ENGINE_SWITCHES:
                    if not isinstance(value, bool):
                        raise ValueError(f"{key} must be true or false")
                    values[key] = value
                else:
                    number = float(value)
                    if not math.isfinite(number) or number < 0:
                        raise ValueError(f"{key} must be a number >= 0")
                    values[key] = number
            if values["iteration_angle"] <= 0 or values["max_building_size"] <= 0:
                raise ValueError(
                    "iteration_angle and max_building_size must be above 0"
                )
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        ENGINE = "classic"
        ENGINE_FALLBACK_REASON = f"could not read {config_path}: {exc}"
        return ENGINE

    settings = {}
    for key, value in values.items():
        if key == "follow_surface":
            settings["follow_rounds"] = 200 if value else 0
            settings["turn_rounds"] = 60 if value else 0
        elif key == "remove_spikes":
            settings["spike_rounds"] = 5 if value else 0
        else:
            settings[FAST_ENGINE_KEYS[key]] = value
    FAST_ENGINE_SETTINGS = settings
    ENGINE = engine
    if engine == "fast":
        try:
            if str(script_dir) not in sys.path:
                sys.path.insert(0, str(script_dir))
            import fast_ground
            fast_ground.load_library()
        except Exception as exc:  # no core for this machine, wrong build...
            ENGINE = "classic"
            ENGINE_FALLBACK_REASON = (
                f"the compiled core could not be loaded "
                f"({type(exc).__name__}: {exc})"
            )
            return ENGINE
        # The compiled core has no need for small windows.
        TURBO_TARGET_CORE_POINTS = FAST_ENGINE_WINDOW_POINTS
        BLOCK_SIZE = max(BLOCK_SIZE, 5000.0)
        WINDOW_FIXED_MEMORY_MB = FAST_ENGINE_WINDOW_FIXED_MEMORY_MB
        WINDOW_BYTES_PER_POINT = FAST_ENGINE_WINDOW_BYTES_PER_POINT
    return ENGINE


def ensure_engine_config(script_dir: Path) -> None:
    """Write the default engine_config.json on first run and say plainly in
    the log which engine this run uses."""
    config_path = script_dir / ENGINE_CONFIG_FILENAME
    if not config_path.exists():
        try:
            config_path.write_text(
                json.dumps(default_engine_config(), indent=2),
                encoding="utf-8",
            )
            log(f"Wrote default engine config: {config_path}")
        except OSError as exc:
            log(f"Could not write default engine config ({exc}).")
    if ENGINE_CONFIG_UPGRADED:
        chosen = "fast" if (ENGINE == "fast" or ENGINE_FALLBACK_REASON) else "classic"
        try:
            config_path.write_text(
                json.dumps(default_engine_config(chosen), indent=2),
                encoding="utf-8",
            )
            log(
                f"{ENGINE_CONFIG_FILENAME} was from an earlier version: "
                "rewritten with the current fast-engine settings "
                f"(engine: {chosen} kept)."
            )
        except OSError as exc:
            log(
                f"{ENGINE_CONFIG_FILENAME} is from an earlier version and "
                f"could not be rewritten ({exc}); the current default "
                "fast-engine settings are used for this run."
            )
    if ENGINE_FALLBACK_REASON:
        log(
            "WARNING: FAST ENGINE NOT USED - "
            f"{ENGINE_FALLBACK_REASON}. Running the classic engine."
        )
    if ENGINE == "fast":
        shown = ", ".join(
            f"{key}={value:g}" for key, value in FAST_ENGINE_SETTINGS.items()
            if not key.endswith("_rounds")
        ) + (
            f", follow_surface={'on' if FAST_ENGINE_SETTINGS.get('follow_rounds') else 'off'}"
            f", remove_spikes={'on' if FAST_ENGINE_SETTINGS.get('spike_rounds') else 'off'}"
        )
        log(
            f"ENGINE: fast (compiled core; {ENGINE_CONFIG_FILENAME}). "
            "Ground is close to, not identical with, the classic engine. "
            f"Settings (metres): {shown}"
        )
    else:
        log(f"ENGINE: classic ({ENGINE_CONFIG_FILENAME}).")


def default_performance_config() -> dict:
    return {
        "_comment": (
            "Speed settings only - neither changes any classification "
            "result. ram_limit_gb: RAM budget in GB for files processed at "
            "the same time. 0 = automatic (55% of installed RAM). On a "
            "machine dedicated to this run a larger value lets more files "
            "run at once; 4 GB of free RAM is always kept spare whatever is "
            "entered. workers: total CPU threads to use. 0 = automatic (80% "
            "of logical CPUs). The same options on the command line "
            "(--ram-limit-gb, --workers) take priority over this file."
        ),
        "ram_limit_gb": 0,
        "workers": 0,
    }


def load_performance_config(script_dir: Path) -> tuple[int, float]:
    """Return (workers, ram_limit_gb) from performance_config.json, creating
    it with automatic defaults on first run like the other config files.
    Anything unreadable or out of range falls back to automatic (0, 0.0)
    with a warning - a bad speed setting must never stop a run."""
    config_path = script_dir / PERFORMANCE_CONFIG_FILENAME
    if not config_path.exists():
        try:
            config_path.write_text(
                json.dumps(default_performance_config(), indent=2),
                encoding="utf-8",
            )
            log(f"Wrote default performance config: {config_path}")
        except OSError as exc:
            log(
                f"Could not write default performance config ({exc}); "
                "using automatic settings."
            )
        return 0, 0.0
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        workers = int(data.get("workers", 0))
        ram_limit_gb = float(data.get("ram_limit_gb", 0))
        if workers < 0 or ram_limit_gb < 0 or not math.isfinite(ram_limit_gb):
            raise ValueError(
                "workers and ram_limit_gb must be zero or greater "
                f"(got {workers}, {ram_limit_gb})"
            )
        return workers, ram_limit_gb
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        log(
            f"WARNING: could not read {config_path} ({exc}); "
            "using automatic speed settings."
        )
        return 0, 0.0


def resolve_input_unit_scale(path: Path, units_setting: str) -> float:
    """Return metres-per-input-unit for this file, or raise SystemExit with
    a clear, actionable message rather than silently guess wrong."""
    if units_setting == "usft":
        return METRES_PER_US_SURVEY_FOOT
    if units_setting == "metres":
        return 1.0

    scale, description = detect_input_unit_scale(path)
    if scale is not None:
        unit_name = "US survey feet" if scale != 1.0 else "metres"
        log(f"  {path.name}: auto-detected {unit_name} ({description})")
        return scale

    raise SystemExit(
        f"Cannot determine the input unit for {path.name}: {description}.\n"
        f"This matters - guessing wrong silently corrupts every coordinate "
        f"by ~3.28x. Set \"units\": \"usft\" or \"units\": \"metres\" in "
        f"{UNITS_CONFIG_FILENAME} (next to the script) for this folder, "
        f"then run again."
    )


# =============================================================================
# =============================================================================
# GROUND CLASSIFICATION ENGINE
#
# CRITICAL UNIT STRATEGY:
#   Input LAS/LAZ XYZ remains on disk exactly as delivered, in whatever unit
#   the file actually uses (feet or metres - see "Input units" in the module
#   docstring above). X/Y/Z are converted to metres ONLY in temporary NumPy
#   arrays before classification runs, using that file's own resolved unit
#   scale. Every metric threshold below is therefore always applied in real
#   metres regardless of the input file's native unit. Only native-coordinate
#   file/halo scheduling values stay in the file's own native unit until
#   that same conversion point.
# =============================================================================

RESET_INPUT_CLASSES = (2, 3, 4, 5, 13)

# Initial model / TerraScan-style settings.
MAX_BUILDING_SIZE = 60.0
MAX_TERRAIN_ANGLE_DEG = 88.0

# Pass 1: initial adaptive TIN lift.
PASS1_ANGLE_DEG = 6.0
PASS1_DISTANCE = 1.40
PASS1_REDUCE_ANGLE_EDGE = 5.0

# Pass 2: reuse current ground and conservatively fill legitimate gaps.
PASS2_ENABLED = True
PASS2_ANGLE_DEG = 4.0
PASS2_DISTANCE = 0.40
PASS2_REDUCE_ANGLE_EDGE = 3.0

# Slope-adaptive iteration angle. TerraScan's own Ground routine documents
# this exact tradeoff but leaves it to the operator to pick one fixed value
# per project ("use a smaller angle value, close to 4.0, in flat terrain and
# a bigger value, close to 10.0, in mountainous terrain"). A single fixed
# value tuned for one project's terrain is exactly why a tool built against
# one dataset performs badly on another with different terrain. This applies
# TerraScan's own documented range automatically per local terrain slope
# (measured from the current triangle, already computed as terrain_slope_deg
# in triangle_metrics) instead of requiring a project-wide manual choice.
# Iteration DISTANCE is deliberately NOT scaled the same way: TerraScan
# documents it as a safety brake against big upward jumps in large/sparse
# triangles specifically to avoid accepting low vegetation as ground, so
# loosening it for sparse areas would fight that purpose rather than serve
# it. Only the angle adapts; distance stays a fixed pass-level constant
# within TerraScan's documented normal range (0.5-1.5 m).
SLOPE_ADAPTIVE_ITERATION_ANGLE = True
# Below this local terrain slope, use the pass's minimum angle (flat-terrain
# behaviour). Above this slope, use the pass's maximum angle (mountainous
# terrain behaviour). Linear ramp in between. These thresholds are generic
# terrain-classification values, not fitted to any one project's data.
SLOPE_ADAPTIVE_FLAT_DEG = 5.0
SLOPE_ADAPTIVE_STEEP_DEG = 35.0
# Pass 1 uses TerraScan's documented flat/mountainous range directly.
PASS1_ANGLE_MIN_DEG = 4.0
PASS1_ANGLE_MAX_DEG = 10.0
# Pass 2 keeps its existing role as the stricter refinement pass: its range
# is scaled down from Pass 1's by the same ratio the original fixed values
# used (4.0/6.0), so it stays proportionally tighter at every slope rather
# than only at the one slope value the old fixed constants implied.
PASS2_ANGLE_MIN_DEG = PASS1_ANGLE_MIN_DEG * (PASS2_ANGLE_DEG / PASS1_ANGLE_DEG)
PASS2_ANGLE_MAX_DEG = PASS1_ANGLE_MAX_DEG * (PASS2_ANGLE_DEG / PASS1_ANGLE_DEG)

# Axelsson (2000), the paper this classifier's docstring already cites,
# describes the ACTUAL mechanism for generalizing across terrain types:
# "Parameters for the TIN densification...are derived from data...
# Statistics from data is collected in the form of discrete histograms of
# surface normal angles and elevation differences...Parameter thresholds
# based on median values are estimated from the histograms...In the
# initialisation phase...statistics are collected from all laser data...
# After initialisation, statistics is collected only from points included
# in the TIN." A fixed angle/distance constant - however carefully chosen,
# and however cleverly adapted by slope - is fundamentally a different (and
# less general) mechanism than this: Axelsson's threshold is never a chosen
# number at all, it is recomputed from the actual noise/relief already
# confirmed in THIS tile's own ground model, every iteration. This is what
# lets one algorithm run unattended across flat farmland, dense forest and
# real mountains without an operator picking a per-project constant.
AXELSSON_DATA_DERIVED_THRESHOLDS = True
# Percentile of the current model's own residual distribution used as next
# iteration's acceptance threshold. Axelsson's paper says "median values";
# a literal median would only ever admit ~50% of genuinely good points per
# iteration, which is overly conservative in practice, so a higher robust
# percentile is used - still derived from the data, not chosen per project.
DATA_DERIVED_PERCENTILE = 90.0
# Floors/ceilings guard against runaway feedback (a noisy early model
# making the threshold too loose, or a very clean model making it too
# tight to admit ordinary LiDAR noise) without reintroducing a fixed
# project-tuned constant as the normal operating value.
DATA_DERIVED_MIN_DISTANCE = 0.15
DATA_DERIVED_MAX_DISTANCE = 3.00
DATA_DERIVED_MIN_ANGLE_DEG = 2.0
DATA_DERIVED_MAX_ANGLE_DEG = 15.0
# How many of the most recently accepted points' residuals to keep for the
# running percentile estimate. Bounded for speed/memory; large enough for a
# stable percentile even on a sparse pass.
DATA_DERIVED_RESIDUAL_BUFFER = 20_000
# Don't switch to data-derived thresholds until enough confirmed-ground
# residuals exist for a stable percentile estimate; before that, iteration 1
# falls back to the pass's fixed settings (the "initialisation phase").
DATA_DERIVED_MIN_SAMPLES = 30
DATA_DERIVED_ESTIMATOR_SEED = 20260821

# Set to zero to disable stopping. TerraScan's option was unchecked in the
# supplied settings. The later adaptive membership stage controls density.
STOP_TRIANGULATION_EDGE = 0.0

# Smoothness and computational controls. The one-metre structural TIN is kept
# deliberately sparser than the final point cloud so its triangles do not
# chase centimetre-scale scatter or low vegetation.
MODEL_SPACING = 1.0
ANGLE_REDUCTION_POWER = 1.0
MAX_TRIANGLE_EDGE_RATIO = 6.0
# Axelsson (2000) and TerraScan's own Ground routine both iterate until no
# more points are added, not for a fixed iteration count. A cap this low was
# found (via the real-data validation suite) to cut Pass 1 off mid-growth on
# some real tiles, leaving the structural model artificially sparse and
# pushing more work than necessary onto the completion stages further down
# the pipeline - which is measurably where most false positives come from.
# Raised well above what any tile in the validation suite needed to reach
# natural convergence ("no new ground model points").
MAX_ITERATIONS_PER_PASS = 200
ALLOW_BELOW_TIN = 0.15
ONE_POINT_PER_TRIANGLE_PASS1 = True

# Final model hygiene. Accepted points that form unsupported upward needles
# relative to a locally fitted ground plane are removed before HAG is built.
TERRAIN_NEIGHBOUR_RADIUS = 10.0
MODEL_SPIKE_NEIGHBOURS = 16
MODEL_SPIKE_RADIUS = TERRAIN_NEIGHBOUR_RADIUS
MODEL_SPIKE_HEIGHT = 0.08
MODEL_SPIKE_SUPPORT_NEIGHBOURS = 24
MODEL_SPIKE_SUPPORT_RADIUS = 4.0
MODEL_SPIKE_SUPPORT_TOLERANCE = 0.06
MODEL_SPIKE_MIN_SUPPORT = 5
MODEL_SPIKE_PRUNE_PASSES = 3
MODEL_LOWER_PLANE_PRUNE_HEIGHT = 0.12
MODEL_LOWER_PLANE_PRUNE_RMSE_FACTOR = 2.0
MODEL_LOWER_PLANE_PRUNE_MAX_RMSE = 0.40

# Gross low-error guard. This is deliberately not a general low-point filter.
# A work minimum is excluded from seed/model construction only when it lies at
# least 1 m below a clean robust local plane and lacks three neighbours at the
# same residual elevation. Continuous valleys, ditches and slopes therefore
# remain eligible. A point is assigned class 7 only after the final TIN confirms
# the same negative residual.
UNSUPPORTED_LOW_GUARD = True
UNSUPPORTED_LOW_MIN_DEPTH = 1.0
UNSUPPORTED_LOW_MAX_PLANE_RMSE = 0.30
UNSUPPORTED_LOW_MIN_SUPPORT = 3
UNSUPPORTED_LOW_CONFIRM_RADIUS = 1.50

# Supported lower-envelope correction. PTD only lifts its initial model; this
# guarded step lets a narrow ditch or depression missed by the coarse seed
# grid pull the final terrain down. Preclassified class 7 points are excluded
# from the ground candidates and are never changed.
LOWER_ENVELOPE_RESCUE = True
LOWER_ENVELOPE_MIN_DEPTH = 0.15
LOWER_ENVELOPE_MAX_DEPTH = 2.0

# Anchored local-plane recovery follows smooth terrain that the strict
# small-triangle angle reduction intentionally stopped short of. A point must
# be close to both the current TIN and a robust plane through nearby minima,
# and close to an existing model point. This is not an unanchored hole fill.
UPPER_SURFACE_RESCUE = True
UPPER_SURFACE_RESCUE_PASSES = 5
UPPER_SURFACE_MIN_HEIGHT = 0.10
UPPER_SURFACE_MAX_HEIGHT = 1.25
UPPER_SURFACE_ANCHOR_RADIUS = 10.0
UPPER_SURFACE_NEIGHBOURS = 20
UPPER_SURFACE_NEIGHBOUR_RADIUS = TERRAIN_NEIGHBOUR_RADIUS
UPPER_SURFACE_PLANE_TOLERANCE = 0.14
UPPER_SURFACE_MAX_RMSE = 0.22
UPPER_SURFACE_MIN_NEIGHBOURS = 6
UPPER_SURFACE_RESIDUAL_SUPPORT_RADIUS = 4.0
UPPER_SURFACE_RESIDUAL_SUPPORT_TOLERANCE = 0.10
UPPER_SURFACE_RESIDUAL_MIN_SUPPORT = 4
UPPER_SURFACE_SUPPORTED_MAX_RMSE = 0.45

# Local lower-surface reconstruction is the fallback for broad convex hills.
# A global seed TIN can cut through a hill, leaving the real surface more than
# 1.4 m above its triangles. Those points cannot be reached by ordinary PTD
# iteration. Reconstructing the robust local plane through modelling minima
# recovers the continuous terrain without using canopy height as the surface.
LOCAL_LOWER_SURFACE_RESCUE = True
LOCAL_LOWER_SURFACE_PLANE_TOLERANCE = 0.18
LOCAL_LOWER_SURFACE_MAX_RMSE = 0.22
LOCAL_LOWER_SURFACE_SUPPORTED_MAX_RMSE = 0.45
LOCAL_LOWER_SURFACE_MIN_SUPPORT = 4

# -----------------------------------------------------------------------------
# Trusted-ground recovery guard.
#
# The original clean metre build used local minima as both candidates and
# evidence during the rescue stages.  That works where the lowest observed
# return is usually terrain, but in dense vegetation or above buildings a
# coherent non-ground lower surface can then support itself and enter class 2.
#
# The strict progressive-TIN result starts the trusted terrain model. Recovery
# then grows progressively from CURRENT accepted terrain; it is not restricted
# to a fixed radius around the original PTD. Candidate minima cannot prove one
# another to be ground within a pass. The final TIN may still bridge a real gap
# with no ground returns; it simply does not manufacture class-2 points there.
# -----------------------------------------------------------------------------
TRUSTED_GROUND_RECOVERY = True

# Use the incoming semantic labels only as a SOFT trust prior for constructing
# the initial PTD backbone. Output behaviour is unchanged: classes 2/3/4/5/13
# are still reset and fully reprocessed. The point is simply that an incoming
# vegetation return should not be allowed to establish the trusted terrain
# before independent terrain evidence exists. Genuine ground that arrived in a
# vegetation class can still join later through the trusted recovery paths.
TRUSTED_BACKBONE_USE_INPUT_PRIOR = True
TRUSTED_BACKBONE_PRIMARY_CLASSES = (0, 2)
TRUSTED_BACKBONE_COVERAGE_GRID = 20.0
TRUSTED_BACKBONE_MIN_COVERAGE = 0.20
TRUSTED_BACKBONE_MIN_POINTS = 100
# Recovery is progressive: every candidate is checked only against terrain
# already accepted BEFORE the current pass. There is deliberately no maximum
# distance back to the original PTD backbone. This lets real ground returns
# propagate through wide wooded areas while preventing a detached vegetation
# or roof surface from mutually self-validating in a single pass.
TRUSTED_SUPPORT_MAX_MODEL_DISTANCE = 2.75
TRUSTED_SUPPORT_NEIGHBOUR_RADIUS = 4.50
TRUSTED_SUPPORT_NEIGHBOURS = 10
TRUSTED_SUPPORT_MIN_NEIGHBOURS = 3
TRUSTED_SUPPORT_MIN_DIRECTION_SECTORS = 2
TRUSTED_SUPPORT_STRONG_NEIGHBOURS = 6
TRUSTED_SUPPORT_FLAT_PLANE_TOLERANCE = 0.12
TRUSTED_SUPPORT_STEEP_PLANE_TOLERANCE = 0.22
TRUSTED_SUPPORT_FULL_SLOPE_DEG = 30.0
TRUSTED_SUPPORT_MAX_PLANE_RMSE = 0.30
TRUSTED_CONNECTION_MIN_SLOPE_DEG = 20.0
TRUSTED_CONNECTION_SLOPE_ALLOWANCE_DEG = 15.0
TRUSTED_CONNECTION_MAX_SLOPE_DEG = 65.0
TRUSTED_CONNECTION_VERTICAL_SLACK = 0.12
TRUSTED_CONNECTED_RESCUE_PASSES = 30

# Sparse-frontier recovery. The strong trusted-support gate above is ideal
# once accepted terrain is locally dense, but it can stall on genuine ground
# beneath woodland where only one or two terrain returns are available near a
# frontier. This second path uses the TIN formed ONLY by already accepted
# terrain as independent evidence. It is deliberately narrow in Z and is used
# only in locally sparse/large triangles, so a detached roof or vegetation
# surface cannot validate itself.
TRUSTED_SPARSE_TIN_RECOVERY = True
TRUSTED_SPARSE_MIN_NEIGHBOURS = 1
TRUSTED_SPARSE_MAX_MODEL_DISTANCE = 3.50
TRUSTED_SPARSE_MIN_TRIANGLE_EDGE = 1.25
TRUSTED_SPARSE_FLAT_UPPER = 0.08
TRUSTED_SPARSE_STEEP_UPPER = 0.18
TRUSTED_SPARSE_FLAT_LOWER = 0.14
TRUSTED_SPARSE_STEEP_LOWER = 0.28
TRUSTED_SPARSE_FULL_SLOPE_DEG = 30.0

# Curvature-aware sparse frontier recovery. This is the third acceptance path
# in v4/v5 and is aimed at datasets such as LGE where genuine ground returns can
# be several metres apart along a curved slope. The fit uses ONLY terrain that
# was already accepted before the current pass, never other candidates. A
# small weighted plane can therefore extrapolate the next terrain step even
# when the trusted TIN chord is a poor vertical predictor. The independent TIN
# is retained as a broad sanity envelope, preventing the plane from walking
# onto detached roofs or vegetation surfaces.
TRUSTED_FRONTIER_PLANE_RECOVERY = True
TRUSTED_FRONTIER_MAX_MODEL_DISTANCE = 5.50
TRUSTED_FRONTIER_NEIGHBOUR_RADIUS = 7.00
TRUSTED_FRONTIER_NEIGHBOURS = 8
TRUSTED_FRONTIER_MIN_NEIGHBOURS = 3
TRUSTED_FRONTIER_FLAT_UPPER = 0.10
TRUSTED_FRONTIER_STEEP_UPPER = 0.22
TRUSTED_FRONTIER_FLAT_LOWER = 0.22
TRUSTED_FRONTIER_STEEP_LOWER = 0.45
TRUSTED_FRONTIER_FULL_SLOPE_DEG = 30.0
TRUSTED_FRONTIER_MAX_PLANE_RMSE = 0.28
TRUSTED_FRONTIER_TIN_UPPER_SANITY = 0.32
TRUSTED_FRONTIER_TIN_LOWER_SANITY = 0.75
TRUSTED_FRONTIER_MIN_TRIANGLE_EDGE = 0.90
TRUSTED_FRONTIER_MIN_XY_SPREAD = 0.30

# Local quadratic fallback for curved terrain. A plane can still under-predict
# a convex road bank or shoulder when sparse ground returns are several metres
# apart. This fit also uses ONLY already accepted terrain. Six or more trusted
# points are required and curvature is bounded so the extrapolation cannot
# freely bend toward a detached vegetation/roof surface.
TRUSTED_QUADRATIC_FRONTIER_RECOVERY = True
TRUSTED_QUADRATIC_MAX_MODEL_DISTANCE = 5.50
TRUSTED_QUADRATIC_NEIGHBOUR_RADIUS = 12.50
TRUSTED_QUADRATIC_NEIGHBOURS = 16
TRUSTED_QUADRATIC_MIN_NEIGHBOURS = 6
TRUSTED_QUADRATIC_FLAT_UPPER = 0.08
TRUSTED_QUADRATIC_STEEP_UPPER = 0.16
TRUSTED_QUADRATIC_FLAT_LOWER = 0.24
TRUSTED_QUADRATIC_STEEP_LOWER = 0.50
TRUSTED_QUADRATIC_FULL_SLOPE_DEG = 30.0
TRUSTED_QUADRATIC_MAX_RMSE = 0.24
TRUSTED_QUADRATIC_TIN_UPPER_SANITY = 1.50
TRUSTED_QUADRATIC_TIN_LOWER_SANITY = 1.50
TRUSTED_QUADRATIC_MIN_TRIANGLE_EDGE = 0.90
TRUSTED_QUADRATIC_MIN_XY_SPREAD = 0.35
TRUSTED_QUADRATIC_MAX_CURVATURE = 0.20
TRUSTED_QUADRATIC_CHUNK_SIZE = 50000

# Breakline-aware directional frontier recovery. Smooth local plane/quadratic
# fits can still miss the exact crest or toe of a bank because trusted points
# from the upper and lower terrain regimes are mixed in one neighbourhood.
# This fifth path fits several one-sided angular wedges using ONLY terrain
# accepted before the current pass. A candidate may match either coherent side
# of the break, but the path activates only where the trusted neighbourhood has
# real vertical relief / slope contrast. It therefore does not globally loosen
# recovery across flat vegetation.
TRUSTED_BREAKLINE_RECOVERY = True
TRUSTED_BREAKLINE_MAX_MODEL_DISTANCE = 5.50
TRUSTED_BREAKLINE_NEIGHBOUR_RADIUS = 7.50
TRUSTED_BREAKLINE_NEIGHBOURS = 12
TRUSTED_BREAKLINE_MIN_NEIGHBOURS = 3
TRUSTED_BREAKLINE_DIRECTION_SECTORS = 8
TRUSTED_BREAKLINE_WEDGE_HALF_WIDTH_DEG = 67.5
TRUSTED_BREAKLINE_FLAT_UPPER = 0.08
TRUSTED_BREAKLINE_STEEP_UPPER = 0.20
TRUSTED_BREAKLINE_FLAT_LOWER = 0.20
TRUSTED_BREAKLINE_STEEP_LOWER = 0.45
TRUSTED_BREAKLINE_FULL_SLOPE_DEG = 30.0
TRUSTED_BREAKLINE_MAX_RMSE = 0.22
TRUSTED_BREAKLINE_TIN_UPPER_SANITY = 0.50
TRUSTED_BREAKLINE_TIN_LOWER_SANITY = 1.00
TRUSTED_BREAKLINE_MIN_TRIANGLE_EDGE = 0.60
TRUSTED_BREAKLINE_MIN_XY_SPREAD = 0.18
TRUSTED_BREAKLINE_MIN_LOCAL_Z_RANGE = 0.20
TRUSTED_BREAKLINE_MIN_SLOPE_CONTRAST_DEG = 6.0
TRUSTED_BREAKLINE_MIN_TIN_SLOPE_DEG = 8.0
TRUSTED_BREAKLINE_CHUNK_SIZE = 50000
# Existing 2/3/4/5/13 labels remain fully reset/reprocessed, but the most
# aggressive breakline-only path uses them as a SOFT confidence prior. A point
# that arrived as higher vegetation must agree more closely with the independent
# trusted TIN before a one-sided breakline extrapolation may promote it. This is
# not a class lock: the strong/sparse/plane/quadratic terrain paths can still
# recover it, and negative TIN residuals are not penalised by this prior.
TRUSTED_BREAKLINE_USE_INPUT_PRIOR = True
TRUSTED_BREAKLINE_PRIOR_CLASS3_MAX_UPPER = 0.28
TRUSTED_BREAKLINE_PRIOR_CLASS4_MAX_UPPER = 0.24
TRUSTED_BREAKLINE_PRIOR_CLASS13_MAX_UPPER = 0.22
TRUSTED_BREAKLINE_PRIOR_CLASS5_MAX_UPPER = 0.18

# The smooth representative TIN is retained, but a validated structural vertex
# on a sharp terrain break must not be replaced by a smoother point from the
# same metre cell. Breaklines are detected from slope contrast between adjacent
# triangles of the already-clean structural terrain model.
PRESERVE_TERRAIN_BREAKLINES = True
BREAKLINE_PRESERVE_MIN_TRIANGLES = 2
BREAKLINE_PRESERVE_MIN_MAX_SLOPE_DEG = 8.0
BREAKLINE_PRESERVE_MIN_SLOPE_CONTRAST_DEG = 10.0

# -----------------------------------------------------------------------------
# v6 terminal terrain-edge recovery
# -----------------------------------------------------------------------------
# These two narrow one-shot stages run AFTER the ordinary progressive recovery
# and model hygiene. They repair isolated residual failures seen in multi-
# dataset QC without loosening the general vegetation-safe recovery rules.
# Crucially, points accepted by these stages are NOT allowed to seed another
# recovery pass. This prevents a shoreline point from walking across water and
# prevents a crest point from walking onto a detached vegetation/roof surface.
TERMINAL_BANK_EDGE_RECOVERY = True
TERMINAL_BANK_MAX_MODEL_DISTANCE = 4.00
TERMINAL_BANK_NEIGHBOUR_RADIUS = 6.00
TERMINAL_BANK_NEIGHBOURS = 12
TERMINAL_BANK_MIN_WEDGE_NEIGHBOURS = 3
TERMINAL_BANK_DIRECTION_SECTORS = 8
TERMINAL_BANK_WEDGE_HALF_WIDTH_DEG = 55.0
TERMINAL_BANK_MAX_WEDGE_RMSE = 0.28
TERMINAL_BANK_MIN_LOCAL_Z_RANGE = 0.30
TERMINAL_BANK_MIN_NEGATIVE_TIN_RESIDUAL = 0.08
TERMINAL_BANK_MAX_NEGATIVE_TIN_RESIDUAL = 2.50
TERMINAL_BANK_PLANE_UPPER = 0.14
TERMINAL_BANK_PLANE_LOWER = 0.42
TERMINAL_BANK_MIN_CONNECTION_SLOPE_DEG = 8.0
TERMINAL_BANK_MAX_CONNECTION_SLOPE_DEG = 82.0
TERMINAL_BANK_MIN_PEER_SUPPORT = 2
TERMINAL_BANK_PEER_RADIUS = 1.60
TERMINAL_BANK_PEER_MAX_SLOPE_DEG = 72.0
TERMINAL_BANK_MIN_XY_SPREAD = 0.12
TERMINAL_BANK_CHUNK_SIZE = 50000

# Rocky ridge / convex crest recovery. A genuine summit can sit materially
# above the Delaunay chord through accepted terrain. It is accepted only when
# TWO substantially opposing one-sided fits, built solely from already
# accepted terrain, independently extrapolate to the same candidate. This is
# much stronger evidence than the single-wedge breakline path and allows a
# wider positive TIN residual without inviting low vegetation onto class 2.
TERMINAL_RIDGE_RECOVERY = True
TERMINAL_RIDGE_MAX_MODEL_DISTANCE = 5.00
TERMINAL_RIDGE_NEIGHBOUR_RADIUS = 9.00
TERMINAL_RIDGE_NEIGHBOURS = 18
TERMINAL_RIDGE_MIN_WEDGE_NEIGHBOURS = 3
TERMINAL_RIDGE_DIRECTION_SECTORS = 8
TERMINAL_RIDGE_WEDGE_HALF_WIDTH_DEG = 50.0
TERMINAL_RIDGE_MIN_OPPOSITION_SECTORS = 3  # 3*45 = 135 degrees
TERMINAL_RIDGE_MAX_WEDGE_RMSE = 0.34
TERMINAL_RIDGE_MIN_LOCAL_Z_RANGE = 0.35
TERMINAL_RIDGE_MIN_POSITIVE_TIN_RESIDUAL = 0.08
TERMINAL_RIDGE_MAX_POSITIVE_TIN_RESIDUAL = 2.25
TERMINAL_RIDGE_FLAT_FIT_TOLERANCE = 0.14
TERMINAL_RIDGE_STEEP_FIT_TOLERANCE = 0.30
TERMINAL_RIDGE_FULL_SLOPE_DEG = 40.0
TERMINAL_RIDGE_MAX_CONNECTION_SLOPE_DEG = 86.0
TERMINAL_RIDGE_MIN_XY_SPREAD = 0.10
TERMINAL_RIDGE_USE_INPUT_PRIOR = True
TERMINAL_RIDGE_PRIOR_CLASS3_TOLERANCE = 0.24
TERMINAL_RIDGE_PRIOR_CLASS4_TOLERANCE = 0.20
TERMINAL_RIDGE_PRIOR_CLASS13_TOLERANCE = 0.18
TERMINAL_RIDGE_PRIOR_CLASS5_TOLERANCE = 0.14
TERMINAL_RIDGE_CHUNK_SIZE = 30000

# Ported from Trusted v8. Sparse, 2.5-D-safe one-sided escarpment repair that
# runs after bank-edge/ridge-crest recovery. Only work-minimum representatives
# that pass a near-vertical/multi-Z column safety test may join the
# STRUCTURAL class-2 model, so a downstream tool that rebuilds its own TIN
# from delivered class-2 points (e.g. TerraScan) never sees a dense
# near-vertical face. This replaces the v6/v7 behaviour of allowing dense
# steep-face membership; it is always on and self-gating (no v6-vs-v8 toggle
# needed) because the column test is what makes it safe on any terrain.
TERMINAL_ESCARPMENT_RECOVERY = True
TERMINAL_ESCARPMENT_MAX_MODEL_DISTANCE = 6.00
TERMINAL_ESCARPMENT_NEIGHBOUR_RADIUS = 8.00
TERMINAL_ESCARPMENT_NEIGHBOURS = 18
TERMINAL_ESCARPMENT_MIN_WEDGE_NEIGHBOURS = 3
TERMINAL_ESCARPMENT_DIRECTION_SECTORS = 8
TERMINAL_ESCARPMENT_WEDGE_HALF_WIDTH_DEG = 50.0
TERMINAL_ESCARPMENT_MAX_WEDGE_RMSE = 0.30
TERMINAL_ESCARPMENT_MIN_XY_SPREAD = 0.12
TERMINAL_ESCARPMENT_MIN_LOCAL_Z_RANGE = 0.45
TERMINAL_ESCARPMENT_MIN_CONNECTION_SLOPE_DEG = 18.0
TERMINAL_ESCARPMENT_MAX_CONNECTION_SLOPE_DEG = 80.0
TERMINAL_ESCARPMENT_MIN_PLANE_SLOPE_DEG = 18.0
TERMINAL_ESCARPMENT_MAX_PLANE_SLOPE_DEG = 80.0
TERMINAL_ESCARPMENT_FLAT_FIT_TOLERANCE = 0.16
TERMINAL_ESCARPMENT_STEEP_FIT_TOLERANCE = 0.30
TERMINAL_ESCARPMENT_FULL_SLOPE_DEG = 50.0
TERMINAL_ESCARPMENT_MIN_ABS_TIN_RESIDUAL = 0.06
TERMINAL_ESCARPMENT_MAX_ABS_TIN_RESIDUAL = 3.00
TERMINAL_ESCARPMENT_PEER_RADIUS = 1.75
TERMINAL_ESCARPMENT_MIN_PEER_SUPPORT = 2
TERMINAL_ESCARPMENT_PEER_MAX_SLOPE_DEG = 80.0
TERMINAL_ESCARPMENT_COLUMN_RADIUS = 0.20
TERMINAL_ESCARPMENT_COLUMN_NEIGHBOURS = 12
TERMINAL_ESCARPMENT_COLUMN_MIN_NEIGHBOURS = 2
TERMINAL_ESCARPMENT_MAX_COLUMN_Z_RANGE = 0.80
TERMINAL_ESCARPMENT_USE_INPUT_PRIOR = True
TERMINAL_ESCARPMENT_PRIOR_CLASS3_TOLERANCE = 0.24
TERMINAL_ESCARPMENT_PRIOR_CLASS4_TOLERANCE = 0.21
TERMINAL_ESCARPMENT_PRIOR_CLASS13_TOLERANCE = 0.19
TERMINAL_ESCARPMENT_PRIOR_CLASS5_TOLERANCE = 0.16
TERMINAL_ESCARPMENT_CHUNK_SIZE = 30000

# Final class-2 density is a local densification of an already accepted model;
# it is not another terrain-recovery mechanism. Requiring proximity to a real
# model vertex prevents a low vegetation return in the middle of a large TIN
# bridge from becoming class 2 merely because its Z happens to lie close to
# the interpolated triangle.
GROUND_MEMBERSHIP_MAX_MODEL_DISTANCE = 2.75
GROUND_MEMBERSHIP_GAP_MAX_MODEL_DISTANCE = 4.00

# Lower-envelope points are allowed a slightly wider negative residual than
# upper-surface recovery, but only when they remain tightly connected to the
# trusted model.  This preserves shallow ditches/depressions without letting
# isolated low vegetation/noise pull the TIN down.
TRUSTED_LOWER_FLAT_TOLERANCE = 0.18
TRUSTED_LOWER_STEEP_TOLERANCE = 0.30
TRUSTED_LOWER_MAX_MODEL_DISTANCE = 2.00

# Representative replacement is deliberately local to an existing model
# vertex. It may improve which source return represents terrain, but it may
# not create a new island deep inside a vegetation/building gap.
REPRESENTATIVE_MAX_MODEL_DISTANCE = 2.75
# Smoothing is NOT a terrain-recovery stage. Any new source return chosen as a
# representative must also lie in a thin perpendicular envelope around the
# already accepted structural TIN. This prevents a locally biased plane at a
# crest/toe from selecting a low vegetation return as the new TIN vertex.
REPRESENTATIVE_TIN_FLAT_UPPER = 0.06
REPRESENTATIVE_TIN_STEEP_UPPER = 0.14
REPRESENTATIVE_TIN_FLAT_LOWER = 0.08
REPRESENTATIVE_TIN_STEEP_LOWER = 0.18
REPRESENTATIVE_TIN_FULL_SLOPE_DEG = 30.0

# Final representative-point smoothing. The progressive/adaptive passes find
# the terrain, then this stage chooses the point closest to a robust local
# terrain plane in adaptively sized cells. Flat ground is moderately dense;
# slopes are denser so cross-sections do not show broken ground.
SMOOTH_REPRESENTATIVE_MODEL = True
REPRESENTATIVE_FINE_SPACING = 0.60
REPRESENTATIVE_FLAT_SPACING = 0.80
REPRESENTATIVE_DENSE_SLOPE_DEG = 10.0
REPRESENTATIVE_PLANE_TOLERANCE = 0.25
REPRESENTATIVE_MAX_RMSE = 0.25
REPRESENTATIVE_SUPPORTED_MAX_RMSE = 0.50
REPRESENTATIVE_MIN_SUPPORT = 4
REPRESENTATIVE_LOWER_TOLERANCE = 0.12
REPRESENTATIVE_FLAT_UPPER_TOLERANCE = 0.08
REPRESENTATIVE_STEEP_UPPER_TOLERANCE = 0.20
REPRESENTATIVE_FULL_SLOPE_DEG = 30.0

# TerraScan retains points accepted while its TIN is being lifted; it does not
# discard them later merely because several fall in the same fixed grid cell.
# Our clean representative TIN remains the interpolation surface, while this
# final membership test restores source returns lying in a very thin envelope
# around that surface. The envelope is perpendicular to the local triangle,
# widens modestly on steep terrain, and tightens as triangles become smaller,
# mirroring the documented "reduce iteration angle" behaviour. This increases
# class-2 density without admitting the whole 0-0.5 m low-vegetation layer.
ADAPTIVE_GROUND_MEMBERSHIP = True
GROUND_MEMBERSHIP_FLAT_UPPER = 0.040
GROUND_MEMBERSHIP_STEEP_UPPER = 0.080
GROUND_MEMBERSHIP_FLAT_LOWER = 0.060
GROUND_MEMBERSHIP_STEEP_LOWER = 0.100
GROUND_MEMBERSHIP_FULL_SLOPE_DEG = 30.0
GROUND_MEMBERSHIP_REDUCE_EDGE = 5.0
GROUND_MEMBERSHIP_SMALL_TRIANGLE_SCALE = 0.72
# Candidate pre-thinning is only a runtime guard; 0.15 m is much finer than
# the requested output density. The actual density is controlled iteratively
# by the current triangle edge: flat terrain stops sooner, while steep terrain
# is allowed a finer mesh. These are longest-triangle-edge targets, not a
# one-point-per-cell spacing.
GROUND_MEMBERSHIP_CANDIDATE_SPACING = 0.15
GROUND_MEMBERSHIP_FLAT_TARGET_EDGE = 0.96
GROUND_MEMBERSHIP_STEEP_TARGET_EDGE = 0.63
GROUND_MEMBERSHIP_MAX_ITERATIONS = 6
GROUND_MEMBERSHIP_MIN_VERTEX_FRACTION = 0.32

# -----------------------------------------------------------------------------
# Density-adaptive FINAL membership
#
# IMPORTANT:
#   The trusted v6 terrain engine above is NOT retuned.
#   These controls affect only which already-observed source returns are kept
#   as final class 2 after the clean v6 terrain/HAG surface has been built.
# -----------------------------------------------------------------------------
AUTO_DENSITY_ADAPTIVE_MEMBERSHIP = True
DENSITY_SAMPLE_MAX_POINTS = 120_000
DENSITY_NEIGHBOUR_ORDER = 8
DENSITY_SPACING_MIN_VALID = 0.05
DENSITY_SPACING_MAX_VALID = 2.00
DENSITY_ESTIMATOR_SEED = 260818

# Measured equivalent acquisition spacing -> final class-2 target mesh.
# Original v6 0.96/0.63 m remains the coarsest behaviour. Dense FPL data can
# request about 0.50/0.33 m without changing any vertical ground tolerance.
DENSITY_FLAT_EDGE_MULTIPLIER = 2.40
DENSITY_MIN_FLAT_TARGET_EDGE = 0.58
DENSITY_MIN_STEEP_TARGET_EDGE = 0.32
DENSITY_CANDIDATE_SPACING_MULTIPLIER = 0.55
DENSITY_MIN_CANDIDATE_SPACING = 0.07
DENSITY_MAX_ADAPTIVE_ITERATIONS = 10

# Two-sided trusted slope/crest FINAL-membership repair.
#
# This is specifically for genuine rising banks / convex crests where the
# smooth representative TIN rounds the terrain and ordinary v6 membership
# leaves a long low-vegetation gap. Candidate points NEVER enter the trusted
# terrain model and NEVER alter the HAG TIN.
TWO_SIDED_BREAK_MEMBERSHIP = True
BREAK_MEMBERSHIP_NEIGHBOUR_RADIUS = 10.0
BREAK_MEMBERSHIP_NEIGHBOURS = 20
BREAK_MEMBERSHIP_MIN_WEDGE_NEIGHBOURS = 3
BREAK_MEMBERSHIP_DIRECTION_SECTORS = 8
BREAK_MEMBERSHIP_WEDGE_HALF_WIDTH_DEG = 52.5
BREAK_MEMBERSHIP_MIN_OPPOSITION_SECTORS = 3
BREAK_MEMBERSHIP_MAX_WEDGE_RMSE = 0.34
BREAK_MEMBERSHIP_MIN_XY_SPREAD = 0.10
BREAK_MEMBERSHIP_MAX_MODEL_DISTANCE = 5.50
BREAK_MEMBERSHIP_MIN_LOCAL_Z_RANGE = 0.22
BREAK_MEMBERSHIP_FLAT_FIT_TOLERANCE = 0.12
BREAK_MEMBERSHIP_STEEP_FIT_TOLERANCE = 0.26
BREAK_MEMBERSHIP_FULL_SLOPE_DEG = 40.0
BREAK_MEMBERSHIP_MAX_TWO_SIDE_PREDICTION_DISAGREEMENT = 0.18
BREAK_MEMBERSHIP_MAX_ABS_STRUCTURAL_TIN_RESIDUAL = 2.25
BREAK_MEMBERSHIP_MAX_CONNECTION_SLOPE_DEG = 86.0
BREAK_MEMBERSHIP_MIN_TERRAIN_SLOPE_DEG = 5.0
BREAK_MEMBERSHIP_MAX_ITERATIONS = 10

# Spike protection applied only to points introduced by the two-sided
# final-membership repair. The original trusted v6 class-2 set is untouched.
BREAK_MEMBERSHIP_SPIKE_NEIGHBOURS = 10
BREAK_MEMBERSHIP_SPIKE_RADIUS = 0.50
BREAK_MEMBERSHIP_SPIKE_SHORT_EDGE = 0.30
BREAK_MEMBERSHIP_SPIKE_MIN_DZ = 0.22
BREAK_MEMBERSHIP_SPIKE_MAX_EDGE_SLOPE_DEG = 80.0

# Final Delaunay TIN geometry audit for NEW two-sided bank/crest members.
#
# Point-pair checks alone do not guarantee a clean downstream class-2 TIN.
# This audit actually triangulates the proposed final membership, compares
# triangles touching new vertices with the frozen trusted structural terrain,
# and removes only new vertices that create repeated anomalously steep triangle
# fans / extreme needles. Existing Trusted-v6 membership is never removed.
BREAK_MEMBERSHIP_TIN_AUDIT = True
BREAK_MEMBERSHIP_TIN_AUDIT_PASSES = 3
BREAK_MEMBERSHIP_TIN_AUDIT_MIN_SLOPE_DEG = 55.0
BREAK_MEMBERSHIP_TIN_AUDIT_SLOPE_ALLOWANCE_DEG = 28.0
BREAK_MEMBERSHIP_TIN_AUDIT_ABSOLUTE_MAX_SLOPE_DEG = 82.0
BREAK_MEMBERSHIP_TIN_AUDIT_MIN_Z_RANGE = 0.16
BREAK_MEMBERSHIP_TIN_AUDIT_MAX_EDGE = 1.75
BREAK_MEMBERSHIP_TIN_AUDIT_MIN_DANGEROUS_TRIANGLES = 2

# Surface-centred representative QC.
#
# The tested TINSafe build recovered the correct terrain shape and density, but
# some slope representatives were the lower return within local vertical
# scatter. Those source points can make a saw-tooth class-2 TIN even though the
# broader bank/crest detection is correct.
#
# This stage NEVER admits a new candidate. It only ranks/rejects candidates that
# have already passed the two-sided trusted bank/crest test.
BREAK_MEMBERSHIP_SURFACE_CENTRED_SELECTION = True

# Candidate must sit close to the mean elevation independently predicted by the
# two opposing trusted terrain fits. Negative (low) residuals are deliberately
# tighter because isolated low representatives create downward TIN needles.
BREAK_MEMBERSHIP_CENTER_FLAT_POSITIVE = 0.11
BREAK_MEMBERSHIP_CENTER_STEEP_POSITIVE = 0.20
BREAK_MEMBERSHIP_CENTER_FLAT_NEGATIVE = 0.07
BREAK_MEMBERSHIP_CENTER_STEEP_NEGATIVE = 0.13
BREAK_MEMBERSHIP_CENTER_FULL_SLOPE_DEG = 40.0

# Peer residual coherence is used only AFTER trusted two-sided eligibility.
# Candidates do not validate each other as ground; peers can only reject an
# isolated low/high representative and influence which already-eligible source
# return is selected.
BREAK_MEMBERSHIP_PEER_QC = True
BREAK_MEMBERSHIP_PEER_RADIUS = 0.85
BREAK_MEMBERSHIP_PEER_NEIGHBOURS = 14
BREAK_MEMBERSHIP_PEER_MIN_NEIGHBOURS = 4
BREAK_MEMBERSHIP_PEER_MAD_FACTOR = 3.0
BREAK_MEMBERSHIP_PEER_MIN_LOW_TOLERANCE = 0.055
BREAK_MEMBERSHIP_PEER_MIN_HIGH_TOLERANCE = 0.090

# Leave-one-out low-spike audit on NEW bank/crest representatives. This is
# intentionally low-side focused: the problem observed in FPL is selection of
# isolated lower returns on a genuine slope. A clean robust neighbouring plane
# must exist before a point can be rejected.
BREAK_MEMBERSHIP_LOW_SPIKE_AUDIT = True
BREAK_MEMBERSHIP_LOW_SPIKE_RADIUS = 1.10
BREAK_MEMBERSHIP_LOW_SPIKE_NEIGHBOURS = 14
BREAK_MEMBERSHIP_LOW_SPIKE_MIN_NEIGHBOURS = 5
BREAK_MEMBERSHIP_LOW_SPIKE_MAX_PLANE_RMSE = 0.12
BREAK_MEMBERSHIP_LOW_SPIKE_MIN_DEPTH = 0.11
BREAK_MEMBERSHIP_LOW_SPIKE_MIN_DIRECTION_SECTORS = 2

# -----------------------------------------------------------------------------
# Global final SmoothTIN representative regularisation.
#
# The trusted v6 model and HAG surface remain unchanged. After ordinary
# membership + two-sided bank/crest recovery, the output class-2 set is
# re-selected from real source returns so that:
#   * flat/open ground is slightly lighter;
#   * slopes/banks are denser;
#   * selected Z values follow the robust local centre of the observed terrain
#     return band rather than the lower envelope;
#   * the ACTUAL final Delaunay TIN remains smooth.
# -----------------------------------------------------------------------------
GLOBAL_SMOOTH_TIN_MEMBERSHIP = True

# Local terrain-centre offset is learned only from points already admitted to
# final ground by the trusted/adaptive/two-sided logic. The value is a robust
# cell median of HAG relative to the frozen smooth v6 TIN.
SMOOTH_TIN_CENTER_CELL = 0.80
SMOOTH_TIN_CENTER_MIN_POINTS = 3
SMOOTH_TIN_CENTER_FLAT_MAX_UP = 0.07
SMOOTH_TIN_CENTER_STEEP_MAX_UP = 0.24
SMOOTH_TIN_CENTER_MAX_DOWN = 0.03
SMOOTH_TIN_CENTER_FULL_SLOPE_DEG = 30.0

# Non-ground-labelled source returns are allowed to compete for the final
# representative only in a VERY thin band around the locally centred terrain.
# This permits replacing a low-biased class-2 point with a better real source
# return without opening a broad low-vegetation acceptance band.
SMOOTH_TIN_POOL_FLAT_LOWER = 0.055
SMOOTH_TIN_POOL_FLAT_UPPER = 0.075
SMOOTH_TIN_POOL_STEEP_LOWER = 0.095
SMOOTH_TIN_POOL_STEEP_UPPER = 0.120
SMOOTH_TIN_POOL_FULL_SLOPE_DEG = 30.0

# Source-point pre-thinning before triangle refinement.
SMOOTH_TIN_CANDIDATE_SPACING = 0.075

# Triangle-density targets. These mirror the desired TerraScan-like pattern:
# slightly lighter flat areas, denser slopes.
SMOOTH_TIN_FLAT_TARGET_EDGE = 0.58
SMOOTH_TIN_STEEP_TARGET_EDGE = 0.32
SMOOTH_TIN_DENSE_SLOPE_DEG = 10.0
SMOOTH_TIN_FULL_SLOPE_DEG = 30.0
SMOOTH_TIN_MAX_ITERATIONS = 10
SMOOTH_TIN_MIN_VERTEX_FRACTION = 0.30

# Score weights. Surface centring dominates selection.
SMOOTH_TIN_PAIR_PENALTY_WEIGHT = 0.10
SMOOTH_TIN_DISTANCE_REWARD = 0.12

# Final robust low/high representative audit. Applies only to membership beyond
# the frozen smooth v6 model.
SMOOTH_TIN_LOCAL_PLANE_AUDIT = True
SMOOTH_TIN_AUDIT_RADIUS = 1.15
SMOOTH_TIN_AUDIT_NEIGHBOURS = 16
SMOOTH_TIN_AUDIT_MIN_NEIGHBOURS = 5
SMOOTH_TIN_AUDIT_MAX_RMSE = 0.11
SMOOTH_TIN_AUDIT_FLAT_RESIDUAL = 0.085
SMOOTH_TIN_AUDIT_STEEP_RESIDUAL = 0.145
SMOOTH_TIN_AUDIT_FULL_SLOPE_DEG = 30.0

# -----------------------------------------------------------------------------
# FPL Fast SmoothTIN / canopy-safe FINAL membership.
# -----------------------------------------------------------------------------
FAST_SMOOTH_TIN = True

# Two-sided recovery should fill genuine gaps, not double-pack an already
# represented slope. Slopes remain denser than flat terrain.
FAST_EXTRA_FLAT_MIN_SEPARATION = 0.30
FAST_EXTRA_STEEP_MIN_SEPARATION = 0.24
FAST_EXTRA_FULL_SLOPE_DEG = 30.0

# Robust local centre of already-accepted terrain returns.
FAST_CENTER_CELL = 0.80
FAST_CENTER_MIN_POINTS = 3
FAST_CENTER_FLAT_MAX_UP = 0.07
FAST_CENTER_STEEP_MAX_UP = 0.22
FAST_CENTER_MAX_DOWN = 0.03
FAST_CENTER_FULL_SLOPE_DEG = 30.0

# Canopy / vertical-complexity cells derived from HAG.
FAST_CANOPY_CELL = 0.80
FAST_CANOPY_ELEVATED_HAG = 0.45
FAST_CANOPY_EXTREME_HAG = 1.20
FAST_CANOPY_MAX_HAG = 12.0
FAST_CANOPY_MIN_ELEVATED_COUNT = 2

# Local-centre residual envelopes.
FAST_OPEN_FLAT_LOWER = 0.09
FAST_OPEN_STEEP_LOWER = 0.15
FAST_OPEN_FLAT_UPPER = 0.10
FAST_OPEN_STEEP_UPPER = 0.16

FAST_CANOPY_FLAT_LOWER = 0.08
FAST_CANOPY_STEEP_LOWER = 0.12
FAST_CANOPY_FLAT_UPPER = 0.045
FAST_CANOPY_STEEP_UPPER = 0.075
FAST_RESIDUAL_FULL_SLOPE_DEG = 30.0

# Independent frozen structural-TIN envelope.
FAST_STRUCT_FLAT_LOWER = 0.11
FAST_STRUCT_STEEP_LOWER = 0.18
FAST_STRUCT_FLAT_UPPER = 0.09
FAST_STRUCT_STEEP_UPPER = 0.15

# Targeted robust plane confirmation, run only on nominated suspects.
FAST_SUSPECT_PLANE_AUDIT = True
FAST_PLANE_MAX_RMSE = 0.12
FAST_PLANE_MIN_NEIGHBOURS = 5
FAST_PLANE_OPEN_FLAT_TOL = 0.10
FAST_PLANE_OPEN_STEEP_TOL = 0.16
FAST_PLANE_CANOPY_FLAT_TOL = 0.055
FAST_PLANE_CANOPY_STEEP_TOL = 0.085
FAST_PLANE_FULL_SLOPE_DEG = 30.0

# -----------------------------------------------------------------------------
# TerraSmooth: direct final-output local-plane regularisation.
#
# This operates on ALL final class-2 members except the protected smooth-v6
# backbone. It is not a ground-recovery stage and cannot create new class-2
# points. Its only job is to remove source returns that make the output TIN
# locally inconsistent / spiky.
# -----------------------------------------------------------------------------
TERRA_SMOOTH_ALL_MEMBER_AUDIT = True
TERRA_SMOOTH_NEIGHBOURS = 14
TERRA_SMOOTH_RADIUS = 1.25
TERRA_SMOOTH_MIN_NEIGHBOURS = 6
TERRA_SMOOTH_CHUNK = 60_000

# Two-stage robust neighbour-plane fit.
TERRA_SMOOTH_TRIM_MIN = 0.055
TERRA_SMOOTH_TRIM_MAD_FACTOR = 2.8
TERRA_SMOOTH_MAX_PLANE_RMSE = 0.115

# Final point residual to the leave-one-out local plane.
# Positive is tighter because elevated vegetation/tree returns are the main
# upward-spike failure. Negative is still controlled to remove downward dents.
TERRA_SMOOTH_OPEN_FLAT_POSITIVE = 0.080
TERRA_SMOOTH_OPEN_STEEP_POSITIVE = 0.125
TERRA_SMOOTH_OPEN_FLAT_NEGATIVE = 0.095
TERRA_SMOOTH_OPEN_STEEP_NEGATIVE = 0.145

TERRA_SMOOTH_CANOPY_FLAT_POSITIVE = 0.045
TERRA_SMOOTH_CANOPY_STEEP_POSITIVE = 0.070
TERRA_SMOOTH_CANOPY_FLAT_NEGATIVE = 0.070
TERRA_SMOOTH_CANOPY_STEEP_NEGATIVE = 0.105

TERRA_SMOOTH_FULL_SLOPE_DEG = 30.0

# Under canopy, use the frozen structural v6 TIN as an additional lockout
# against clusters of vegetation returns supporting each other.
TERRA_SMOOTH_CANOPY_STRUCT_FLAT_POSITIVE = 0.075
TERRA_SMOOTH_CANOPY_STRUCT_STEEP_POSITIVE = 0.135
TERRA_SMOOTH_CANOPY_STRUCT_FLAT_NEGATIVE = 0.105
TERRA_SMOOTH_CANOPY_STRUCT_STEEP_NEGATIVE = 0.175

# Very rough / breakline neighbourhoods are not flattened by the local-plane
# audit. They remain protected by the existing v6 + two-sided break logic.
TERRA_SMOOTH_BREAK_RMSE_SKIP = 0.115

# Automatic large-triangle support. This is applied uniformly to every
# processing block; there are no geographic masks or project-specific areas.
# Work-minimum returns that form a robust local surface may join the ordinary
# progressive membership pool when the clean reference TIN is locally sparse.
# The broader reference-distance allowance repairs genuine gaps while the
# local-plane gate prevents roofs, vegetation and isolated points from being
# used merely because a triangle is large.
SUPPORTED_TRIANGLE_REFINEMENT = True
SUPPORTED_TRIANGLE_MIN_EDGE_FACTOR = 1.75
SUPPORTED_TRIANGLE_MAX_REFERENCE_DISTANCE = 0.25
SUPPORTED_TRIANGLE_PLANE_TOLERANCE = 0.14
SUPPORTED_TRIANGLE_MAX_RMSE = 0.22
SUPPORTED_TRIANGLE_SUPPORTED_MAX_RMSE = 0.45
SUPPORTED_TRIANGLE_MIN_SUPPORT = 4

LOW_VEG_MAX = 0.50
MEDIUM_VEG_MAX = 3.00
HIGH_VEG_MAX = 200.00

# HAG (vegetation height) bands, editable without touching this file: on
# first run this writes hag_config.json beside the script with the values
# above; edit that file per project and it is picked up automatically on
# every subsequent run. Nothing else in the classifier is configured this
# way - HAG bands are the one setting meant to vary per-delivery spec.
HAG_CONFIG_FILENAME = "hag_config.json"


def default_hag_config() -> dict:
    return {
        "_comment": (
            "Vegetation height (HAG) bands in metres. low_veg_max: class 3 "
            "ceiling. medium_veg_max: class 4 ceiling (also class 5 floor). "
            "high_veg_max: class 5 ceiling; HAG above this becomes class 7 "
            "(noise/outlier). Edit and save; picked up on the next run."
        ),
        "low_veg_max": LOW_VEG_MAX,
        "medium_veg_max": MEDIUM_VEG_MAX,
        "high_veg_max": HIGH_VEG_MAX,
    }


def load_hag_config(script_dir: Path) -> tuple[float, float, float]:
    """Load HAG bands from hag_config.json beside the script, creating it
    with current defaults on first run. Falls back to the built-in defaults
    (with a warning) if the file is missing, unreadable, or invalid."""
    config_path = script_dir / HAG_CONFIG_FILENAME
    if not config_path.exists():
        try:
            config_path.write_text(
                json.dumps(default_hag_config(), indent=2), encoding="utf-8"
            )
            log(f"Wrote default HAG config: {config_path}")
        except OSError as exc:
            log(f"Could not write default HAG config ({exc}); using built-in defaults.")
        return LOW_VEG_MAX, MEDIUM_VEG_MAX, HIGH_VEG_MAX

    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        low = float(data["low_veg_max"])
        medium = float(data["medium_veg_max"])
        high = float(data["high_veg_max"])
        if not (0 < low < medium < high):
            raise ValueError(
                f"HAG config must satisfy 0 < low_veg_max < medium_veg_max "
                f"< high_veg_max (got {low}, {medium}, {high})"
            )
        log(
            f"Loaded HAG config: low<={low:g}m, medium<={medium:g}m, "
            f"high<={high:g}m from {config_path}"
        )
        return low, medium, high
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        log(
            f"WARNING: could not read {config_path} ({exc}); "
            "using built-in default HAG bands."
        )
        return LOW_VEG_MAX, MEDIUM_VEG_MAX, HIGH_VEG_MAX


def configure_hag_bands(low: float, medium: float, high: float) -> None:
    """Set the process-local HAG thresholds. Must be re-called at the start
    of every worker process (Windows multiprocessing spawns fresh
    interpreters that do not inherit the parent's mutated globals) -
    mirrors the existing configure_runtime() pattern."""
    global LOW_VEG_MAX, MEDIUM_VEG_MAX, HIGH_VEG_MAX
    LOW_VEG_MAX = low
    MEDIUM_VEG_MAX = medium
    HIGH_VEG_MAX = high

# Large-file block processing. The overlap is deliberately larger than the
# 60 m initial seed grid.
BLOCK_SIZE = 500.0
BLOCK_OVERLAP = 80.0
SINGLE_BLOCK_POINT_LIMIT = 4_000_000

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
TURBO_TARGET_CORE_POINTS = 2_000_000
TURBO_MAX_INTERNAL_WINDOWS = 8
TURBO_MIN_CORE_SIZE = 220.0

# The known FPL log showed ~99% of useful final-membership additions occurred
# within the first three adaptive and first three two-sided iterations.
TURBO_MAX_ADAPTIVE_MEMBERSHIP_ITERATIONS = 3
TURBO_MAX_BREAK_MEMBERSHIP_ITERATIONS = 3

# Memory-safe two-sided break evidence.
#
# In the failed 21.8 M-point FPL run, one buffered window had ~5.6 M unresolved
# candidates. A 20-neighbour query over all of them created 112,263,220-element
# arrays (~857 MiB EACH for float64), with several such arrays alive together.
#
# Process the exact same directional-plane evidence in bounded chunks instead.
BREAK_MEMBERSHIP_EVIDENCE_CHUNK = 50_000

# Turbo2: reject unresolved source points that cannot usefully improve current
# final class-2 density BEFORE expensive 20-neighbour / 8-sector evidence.
TURBO2_DENSITY_NEED_GATE = True
TURBO2_DENSITY_GATE_CHUNK = 250_000

# Keep Windows paths short.
TURBO2_OUTPUT_FOLDER = "AdaptiveGround_Output"
TURBO2_OUTPUT_SUFFIX = "_AdaptiveGround"

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

# Chunking limits peak temporary memory during TIN tests.
TEST_CHUNK_SIZE = 300_000

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
WINDOW_FIXED_MEMORY_MB = 512.0
WINDOW_BYTES_PER_POINT = 256
FILE_FIXED_MEMORY_MB = 384.0
FULL_FILE_BYTES_PER_POINT = 96

# TerraScan-style neighbouring-tile context. For every target LAS/LAZ, points
# from spatially adjacent files are loaded only where they fall within this
# halo around the target bounds. Temporary halo caches mean each neighbour
# file is decompressed once per run rather than once for every adjacent tile.
NEIGHBOUR_HALO = metres_to_usft(10.0)  # 32.808333... USft = 10 m physical.
# Re-derived per run by configure_neighbour_halo() once the project's real
# input unit is known (see detect_input_unit_scale) - the fixed feet value
# above is only the pre-configuration default, kept for any direct caller
# that never resolves a project unit.


def configure_neighbour_halo(metres_per_input_unit: float) -> None:
    """Set NEIGHBOUR_HALO to 10 metres expressed in the project's real input
    unit. Must be called once the input unit is known, before any
    neighbour-tile discovery or halo-buffered classify_file work runs, and
    re-called at the start of any worker process that does not inherit this
    process's mutated globals (Windows multiprocessing spawn)."""
    global NEIGHBOUR_HALO
    NEIGHBOUR_HALO = 10.0 / max(metres_per_input_unit, 1.0e-9)


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
KD_TREE_PARALLEL_MIN_QUERIES = 50_000


@dataclass(frozen=True)
class PassSettings:
    name: str
    angle_deg: float
    distance: float
    reduce_edge: float
    # Slope-adaptive iteration angle range, following TerraScan's own
    # documented guidance ("use a smaller angle value, close to 4.0, in flat
    # terrain and a bigger value, close to 10.0, in mountainous terrain") -
    # applied automatically per local terrain slope instead of requiring an
    # operator to pick one fixed value for the whole project. angle_deg is
    # kept as the non-adaptive fallback (SLOPE_ADAPTIVE_ITERATION_ANGLE=False)
    # and as the reference point Pass 2's relative strictness is derived from.
    angle_min_deg: float = 0.0
    angle_max_deg: float = 0.0


TinSurface = tuple[Delaunay, np.ndarray, np.ndarray, int]
PlaneContext = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


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


PASSES = (
    PassSettings(
        "Pass 1",
        PASS1_ANGLE_DEG,
        PASS1_DISTANCE,
        PASS1_REDUCE_ANGLE_EDGE,
        PASS1_ANGLE_MIN_DEG,
        PASS1_ANGLE_MAX_DEG,
    ),
    PassSettings(
        "Pass 2",
        PASS2_ANGLE_DEG,
        PASS2_DISTANCE,
        PASS2_REDUCE_ANGLE_EDGE,
        PASS2_ANGLE_MIN_DEG,
        PASS2_ANGLE_MAX_DEG,
    ),
)


_LOG_STATE = threading.local()
# Per-window noise-scale state (thread-local: windows may run concurrently
# on separate threads, see ThreadPoolExecutor usage in process_file_windows,
# so this must never be a plain shared global). Set once per classify_block
# call via configure_noise_scale(); read via noise_scale_multiplier().
_NOISE_SCALE_STATE = threading.local()

# The fixed downstream tolerance constants below (GROUND_MEMBERSHIP_*,
# TRUSTED_SUPPORT_*_TOLERANCE, SUPPORTED_TRIANGLE_*) were measured (via the
# real-data validation suite, comparing against actual TerraScan ground
# truth) to have been calibrated for point clouds with roughly 1-2cm of
# genuine local ground surface noise. Applied unscaled to terrain with
# naturally rougher ground (measured directly from TerraScan reference data
# at 2.6-4.6cm on rockier/water-edge terrain - 2-4x noisier), that same
# fixed tolerance sits exactly where real terrain relief and nearby low
# vegetation become hard to tell apart with one fixed number, in both
# directions. NOISE_SCALE_REFERENCE is the noise level (in metres) the
# unscaled constants were evidently tuned for; the measured multiplier is
# clamped so a very noisy or very clean tile still gets tolerances within a
# sane range rather than an unbounded rescale.
NOISE_SCALE_ADAPTIVE = False
NOISE_SCALE_REFERENCE = 0.017
NOISE_SCALE_MIN_MULTIPLIER = 0.5
NOISE_SCALE_MAX_MULTIPLIER = 3.0
NOISE_SCALE_SAMPLE_POINTS = 4_000
NOISE_SCALE_NEIGHBOUR_RADIUS = 1.5
NOISE_SCALE_MIN_NEIGHBOURS = 5

_PRINT_LOCK = threading.Lock()


def log(message: str) -> None:
    """Print atomically, or capture a complete window log in its worker."""
    buffer = getattr(_LOG_STATE, "buffer", None)
    if buffer is not None:
        buffer.append(message)
        return
    prefix = getattr(_LOG_STATE, "prefix", "")
    with _PRINT_LOCK:
        print(f"{prefix}{message}", flush=True)


def configure_runtime(kdtree_workers: int, prefix: str = "") -> None:
    """Set process-local runtime limits without changing classifications."""
    global KD_TREE_WORKERS
    KD_TREE_WORKERS = max(1, int(kdtree_workers))
    _LOG_STATE.prefix = prefix


def estimate_noise_scale_multiplier(
    model_indices: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> float:
    """Estimate this tile's real local ground noise relative to the level
    the fixed downstream tolerances were tuned for, entirely from points
    already confirmed in the structural model (no ground truth available in
    production - this mirrors Axelsson's own "statistics collected only
    from points included in the TIN" philosophy, applied to noise scale
    instead of iteration angle/distance).

    For a sample of model points, fits a local plane to nearby OTHER model
    points and measures the residual spread - genuine terrain roughness (or
    residual low-vegetation contamination) shows up as a larger spread;
    clean, well-behaved ground shows a small one. Returns a clamped
    multiplier, 1.0 meaning "matches the reference level the constants were
    tuned for".
    """
    if not NOISE_SCALE_ADAPTIVE or model_indices.size < NOISE_SCALE_MIN_NEIGHBOURS + 20:
        return 1.0

    rng = np.random.default_rng(DATA_DERIVED_ESTIMATOR_SEED)
    n = min(NOISE_SCALE_SAMPLE_POINTS, model_indices.size)
    sample = model_indices[rng.choice(model_indices.size, size=n, replace=False)]

    model_xy = np.column_stack((x[model_indices], y[model_indices]))
    tree = cKDTree(model_xy)
    sample_xy = np.column_stack((x[sample], y[sample]))
    neighbour_lists = tree.query_ball_point(
        sample_xy, r=NOISE_SCALE_NEIGHBOUR_RADIUS, workers=1
    )

    residual_stds = []
    for idxs in neighbour_lists:
        if len(idxs) < NOISE_SCALE_MIN_NEIGHBOURS:
            continue
        global_idxs = model_indices[np.asarray(idxs, dtype=np.int64)]
        nx, ny, nz = x[global_idxs], y[global_idxs], z[global_idxs]
        design = np.column_stack((nx, ny, np.ones_like(nx)))
        try:
            coef, *_ = np.linalg.lstsq(design, nz, rcond=None)
        except np.linalg.LinAlgError:
            continue
        residual = nz - design @ coef
        residual_stds.append(float(np.std(residual)))

    if len(residual_stds) < 30:
        return 1.0

    measured_scale = float(np.median(residual_stds))
    multiplier = measured_scale / max(NOISE_SCALE_REFERENCE, 1.0e-6)
    return float(np.clip(
        multiplier, NOISE_SCALE_MIN_MULTIPLIER, NOISE_SCALE_MAX_MULTIPLIER
    ))


def configure_noise_scale(multiplier: float) -> None:
    """Set this window/thread's noise-scale multiplier (see
    estimate_noise_scale_multiplier). Thread-local: safe under the
    ThreadPoolExecutor window parallelism in process_file_windows."""
    _NOISE_SCALE_STATE.multiplier = float(multiplier)


def noise_scale_multiplier() -> float:
    """Current thread's noise-scale multiplier, defaulting to 1.0 (matches
    the original fixed-constant behaviour) if never set for this thread."""
    return getattr(_NOISE_SCALE_STATE, "multiplier", 1.0)


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


def tree_query(
    tree: cKDTree,
    points: np.ndarray,
    **kwargs: object,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a cKDTree query, parallelising only worthwhile batches."""
    query_count = 1 if points.ndim == 1 else points.shape[0]
    workers = (
        KD_TREE_WORKERS
        if query_count >= KD_TREE_PARALLEL_MIN_QUERIES
        else 1
    )
    try:
        return tree.query(points, workers=workers, **kwargs)
    except TypeError:
        # scipy before 1.6 did not expose the workers keyword.
        return tree.query(points, **kwargs)


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
    the project's configured HAG bands (see load_hag_config() /
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


def packed_grid_keys(
    x: np.ndarray,
    y: np.ndarray,
    spacing: float,
    origin_x: float,
    origin_y: float,
) -> np.ndarray:
    """Return a stable structured grid key without integer packing overflow."""
    gx = np.floor((x - origin_x) / spacing).astype(np.int64)
    gy = np.floor((y - origin_y) / spacing).astype(np.int64)
    keys = np.empty(x.size, dtype=[("x", "<i8"), ("y", "<i8")])
    keys["x"] = gx
    keys["y"] = gy
    return keys


def lowest_per_grid(
    indices: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    spacing: float,
    origin_x: float,
    origin_y: float,
) -> np.ndarray:
    """Choose the lowest point in every XY grid cell."""
    if indices.size == 0:
        return indices

    keys = packed_grid_keys(
        x[indices], y[indices], spacing, origin_x, origin_y
    )
    order = np.lexsort((z[indices], keys["y"], keys["x"]))
    ordered_indices = indices[order]
    ordered_keys = keys[order]

    first = np.empty(order.size, dtype=bool)
    first[0] = True
    first[1:] = (
        (ordered_keys["x"][1:] != ordered_keys["x"][:-1])
        | (ordered_keys["y"][1:] != ordered_keys["y"][:-1])
    )
    return ordered_indices[first]


def deduplicate_model_points(
    indices: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    origin_x: float,
    origin_y: float,
) -> np.ndarray:
    """Keep the lowest TIN model point in each model-spacing cell."""
    return lowest_per_grid(
        indices,
        x,
        y,
        z,
        MODEL_SPACING,
        origin_x,
        origin_y,
    )


def add_corner_support(
    model_xy: np.ndarray,
    model_z: np.ndarray,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Add four synthetic corner vertices using local terrain extrapolation.

    Axelsson-style initial TINs need to cover every candidate. The corners sit
    just outside the processing bounds and are never written as ground points.
    """
    margin = max(MODEL_SPACING, 0.01)
    corners = np.array(
        [
            [xmin - margin, ymin - margin],
            [xmin - margin, ymax + margin],
            [xmax + margin, ymin - margin],
            [xmax + margin, ymax + margin],
        ],
        dtype=np.float64,
    )
    tree = cKDTree(model_xy)
    k = min(12, model_xy.shape[0])
    _, nearest = tree_query(tree, corners, k=k)
    nearest = np.asarray(nearest, dtype=np.int64)
    if nearest.ndim == 1:
        nearest = nearest[:, None]

    corner_z = np.empty(corners.shape[0], dtype=np.float64)
    for row, corner in enumerate(corners):
        local = nearest[row]
        local_xy = model_xy[local]
        design = np.column_stack(
            (local_xy[:, 0], local_xy[:, 1], np.ones(local.size))
        )
        try:
            coefficients, *_ = np.linalg.lstsq(
                design, model_z[local], rcond=None
            )
            corner_z[row] = (
                coefficients[0] * corner[0]
                + coefficients[1] * corner[1]
                + coefficients[2]
            )
        except np.linalg.LinAlgError:
            corner_z[row] = model_z[local[0]]

    return (
        np.vstack((model_xy, corners)),
        np.concatenate((model_z, corner_z)),
        corners.shape[0],
    )


# ---------------------------------------------------------------------------
# Reuse of identical triangulations (speed only - never changes a result).
#
# Within one classify_block call the same ground model is triangulated many
# times in a row: profiling real tiles showed roughly 40% of all Delaunay
# builds are repeats of a triangulation built moments earlier (the trusted-
# surface loop builds the same TIN three times per iteration, and the
# structural model is rebuilt by five different stages). Qhull is
# deterministic, so triangulating byte-identical points again returns exactly
# the same triangulation; keeping the last few and handing the same object
# back skips that repeated work and nothing else.
#
# Exactness: a stored triangulation is returned only when the new point array
# is element-for-element equal to the one it was built from (np.array_equal
# on the full array, not just a hash match). Callers only ever read
# find_simplex / simplices / transform from a TIN and never modify it.
#
# The cache is per thread (windows of one file may run on parallel threads)
# and exists only while classify_block is running, so nothing is shared
# between windows or files and it is freed as soon as the block finishes.
TIN_CACHE_SIZE = 3
_TIN_CACHE_STATE = threading.local()
QHULL_TIN_OPTIONS = "Qbb Qc Qz Q12"


def delaunay_triangulation(points: np.ndarray) -> Delaunay:
    """Delaunay triangulation of ``points``, reusing an identical recent one."""
    cache = getattr(_TIN_CACHE_STATE, "cache", None)
    if cache is None:
        return Delaunay(points, qhull_options=QHULL_TIN_OPTIONS)

    for position in range(len(cache) - 1, -1, -1):
        cached = cache[position]
        cached_points = cached.points
        if (
            cached_points.shape == points.shape
            and cached_points.dtype == points.dtype
            and np.array_equal(cached_points, points)
        ):
            if position != len(cache) - 1:
                del cache[position]
                cache.append(cached)
            return cached

    tin = Delaunay(points, qhull_options=QHULL_TIN_OPTIONS)
    cache.append(tin)
    if len(cache) > TIN_CACHE_SIZE:
        del cache[0]
    return tin


def build_tin(
    model_indices: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    bounds: tuple[float, float, float, float],
) -> tuple[Delaunay, np.ndarray, np.ndarray, int]:
    """Construct the current Delaunay TIN with synthetic corner support."""
    model_xy = np.column_stack((x[model_indices], y[model_indices]))
    model_z = z[model_indices].astype(np.float64, copy=False)
    model_xy, model_z, corner_count = add_corner_support(
        model_xy, model_z, *bounds
    )

    try:
        tin = delaunay_triangulation(model_xy)
    except QhullError as exc:
        raise InsufficientGroundSupport(
            "Unable to build the ground TIN. The seed geometry may be "
            "collinear or contain insufficient spatial coverage."
        ) from exc
    return tin, model_xy, model_z, corner_count


def build_planar_tin(
    model_indices: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    bounds: tuple[float, float, float, float],
) -> tuple[Delaunay, np.ndarray]:
    """Build the XY-only density TIN used by membership refinement.

    Its vertices are geometrically identical to :func:`build_tin`. Elevation
    and corner-plane fitting are intentionally skipped because membership
    density uses only simplex locations and triangle edge lengths.
    """
    model_xy = np.column_stack((x[model_indices], y[model_indices]))
    xmin, ymin, xmax, ymax = bounds
    margin = max(MODEL_SPACING, 0.01)
    corners = np.array(
        [
            [xmin - margin, ymin - margin],
            [xmin - margin, ymax + margin],
            [xmax + margin, ymin - margin],
            [xmax + margin, ymax + margin],
        ],
        dtype=np.float64,
    )
    model_xy = np.vstack((model_xy, corners))
    try:
        tin = delaunay_triangulation(model_xy)
    except QhullError as exc:
        raise InsufficientGroundSupport(
            "Unable to build the ground-density TIN."
        ) from exc
    return tin, model_xy


def triangle_metrics(
    tin: Delaunay,
    model_xy: np.ndarray,
    model_z: np.ndarray,
    simplex_ids: np.ndarray,
    query_xy: np.ndarray,
    query_z: np.ndarray,
) -> dict[str, np.ndarray]:
    """Calculate plane distance, vertex angle, slope and triangle geometry."""
    vertices = tin.simplices[simplex_ids]
    tri_xy = model_xy[vertices]
    tri_z = model_z[vertices]

    p0 = np.column_stack((tri_xy[:, 0, :], tri_z[:, 0]))
    p1 = np.column_stack((tri_xy[:, 1, :], tri_z[:, 1]))
    p2 = np.column_stack((tri_xy[:, 2, :], tri_z[:, 2]))

    normals = np.cross(p1 - p0, p2 - p0)
    normal_length = np.linalg.norm(normals, axis=1)
    degenerate = normal_length <= 1.0e-12
    normal_length[degenerate] = 1.0
    normals /= normal_length[:, None]

    flip = normals[:, 2] < 0
    normals[flip] *= -1.0

    query_xyz = np.column_stack((query_xy, query_z))
    signed_distance = np.einsum("ij,ij->i", query_xyz - p0, normals)
    projection = query_xyz - signed_distance[:, None] * normals

    vertex_xyz = np.stack((p0, p1, p2), axis=1)
    projected_vertex_distance = np.linalg.norm(
        vertex_xyz - projection[:, None, :], axis=2
    )
    nearest_vertex_distance = np.min(projected_vertex_distance, axis=1)

    angle_deg = np.degrees(
        np.arctan2(
            np.abs(signed_distance),
            np.maximum(nearest_vertex_distance, 1.0e-9),
        )
    )

    edge01 = np.linalg.norm(tri_xy[:, 0] - tri_xy[:, 1], axis=1)
    edge12 = np.linalg.norm(tri_xy[:, 1] - tri_xy[:, 2], axis=1)
    edge20 = np.linalg.norm(tri_xy[:, 2] - tri_xy[:, 0], axis=1)
    edges = np.column_stack((edge01, edge12, edge20))
    max_edge = np.max(edges, axis=1)
    min_edge = np.maximum(np.min(edges, axis=1), 1.0e-9)
    edge_ratio = max_edge / min_edge

    terrain_slope_deg = np.degrees(
        np.arctan2(
            np.linalg.norm(normals[:, :2], axis=1),
            np.maximum(normals[:, 2], 1.0e-12),
        )
    )

    lowest_vertex = np.argmin(tri_z, axis=1)
    rows = np.arange(query_xy.shape[0])
    low_xy = tri_xy[rows, lowest_vertex]
    low_z = tri_z[rows, lowest_vertex]
    horizontal = np.linalg.norm(query_xy - low_xy, axis=1)
    candidate_slope_deg = np.degrees(
        np.arctan2(
            np.maximum(query_z - low_z, 0.0),
            np.maximum(horizontal, 1.0e-9),
        )
    )

    return {
        "signed_distance": signed_distance,
        "distance": np.abs(signed_distance),
        "angle_deg": angle_deg,
        "max_edge": max_edge,
        "edge_ratio": edge_ratio,
        "terrain_slope_deg": terrain_slope_deg,
        "candidate_slope_deg": candidate_slope_deg,
        "degenerate": degenerate,
    }


def adaptive_angle_limit(
    base_angle_deg: float | np.ndarray,
    max_edge: np.ndarray,
    reduce_edge: float,
) -> np.ndarray:
    """
    Reduce the permissible angle linearly toward zero in small triangles.

    TerraScan documents the behaviour but not the exact proprietary curve.
    The power is exposed above so the curve can be made more/less aggressive.

    base_angle_deg may be a scalar (fixed iteration angle) or a per-candidate
    array (slope-adaptive iteration angle - see SLOPE_ADAPTIVE_ITERATION_ANGLE).
    Either way this is layered independently on top: a candidate in a small
    triangle is reduced toward zero regardless of what its slope-based base
    angle was, exactly matching TerraScan's documented "Reduce iteration
    angle when" behaviour.
    """
    base_angle_arr = np.broadcast_to(
        np.asarray(base_angle_deg, dtype=np.float64), max_edge.shape
    )
    if reduce_edge <= 0:
        return base_angle_arr.copy()

    ratio = np.clip(max_edge / reduce_edge, 0.0, 1.0)
    return base_angle_arr * np.power(ratio, ANGLE_REDUCTION_POWER)


def slope_adaptive_base_angle(
    terrain_slope_deg: np.ndarray,
    angle_min_deg: float,
    angle_max_deg: float,
) -> np.ndarray:
    """Iteration angle base value, ramped by local terrain slope.

    Mirrors TerraScan's own documented guidance for the Ground routine's
    Iteration angle setting ("use a smaller angle value, close to 4.0, in
    flat terrain and a bigger value, close to 10.0, in mountainous terrain")
    - but applied automatically per local triangle instead of requiring one
    fixed value to be chosen for an entire project up front. Below
    SLOPE_ADAPTIVE_FLAT_DEG, angle_min_deg applies (flat-terrain behaviour).
    Above SLOPE_ADAPTIVE_STEEP_DEG, angle_max_deg applies (mountainous
    behaviour). Linear ramp between the two.
    """
    span = max(SLOPE_ADAPTIVE_STEEP_DEG - SLOPE_ADAPTIVE_FLAT_DEG, 1.0e-9)
    t = np.clip(
        (terrain_slope_deg - SLOPE_ADAPTIVE_FLAT_DEG) / span, 0.0, 1.0
    )
    return angle_min_deg + t * (angle_max_deg - angle_min_deg)


def accepted_candidates(
    tin: Delaunay,
    model_xy: np.ndarray,
    model_z: np.ndarray,
    candidate_indices: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    settings: PassSettings,
    distance_limit: float | None = None,
    angle_range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Test candidates in memory-bounded chunks against the current TIN.

    Selecting only the best candidate in each triangle before rebuilding the
    TIN is intentionally conservative. It prevents a large initial triangle
    from accepting a whole layer of low vegetation in one jump.

    distance_limit / angle_range override settings.distance /
    settings.angle_min_deg,angle_max_deg when provided - this is how the
    Axelsson-style data-derived threshold (see run_ptd_pass) is applied
    without needing a second copy of this function.

    Returns (accepted_indices, accepted_distance, accepted_angle_deg): the
    per-accepted-point residuals are needed by run_ptd_pass to estimate next
    iteration's data-derived threshold from points now confirmed in the TIN.
    """
    accepted_parts: list[np.ndarray] = []
    simplex_parts: list[np.ndarray] = []
    merit_parts: list[np.ndarray] = []
    residual_distance_parts: list[np.ndarray] = []
    residual_angle_parts: list[np.ndarray] = []

    effective_distance = (
        distance_limit if distance_limit is not None else settings.distance
    )
    if angle_range is not None:
        effective_angle_min, effective_angle_max = angle_range
    else:
        effective_angle_min = settings.angle_min_deg
        effective_angle_max = settings.angle_max_deg

    for start in range(0, candidate_indices.size, TEST_CHUNK_SIZE):
        part = candidate_indices[start : start + TEST_CHUNK_SIZE]
        query_xy = np.column_stack((x[part], y[part]))
        simplex = tin.find_simplex(query_xy)
        valid = simplex >= 0
        if not np.any(valid):
            continue

        valid_part = part[valid]
        valid_xy = query_xy[valid]
        valid_simplex = simplex[valid]
        metrics = triangle_metrics(
            tin,
            model_xy,
            model_z,
            valid_simplex,
            valid_xy,
            z[valid_part],
        )

        if SLOPE_ADAPTIVE_ITERATION_ANGLE and effective_angle_max > 0:
            base_angle = slope_adaptive_base_angle(
                metrics["terrain_slope_deg"],
                effective_angle_min,
                effective_angle_max,
            )
        else:
            base_angle = settings.angle_deg

        angle_limit = adaptive_angle_limit(
            base_angle,
            metrics["max_edge"],
            settings.reduce_edge,
        )

        good = (
            ~metrics["degenerate"]
            & (metrics["signed_distance"] >= -ALLOW_BELOW_TIN)
            & (metrics["distance"] <= effective_distance)
            & (metrics["angle_deg"] <= angle_limit)
            & (metrics["terrain_slope_deg"] <= MAX_TERRAIN_ANGLE_DEG)
            & (metrics["candidate_slope_deg"] <= MAX_TERRAIN_ANGLE_DEG)
            & (metrics["edge_ratio"] <= MAX_TRIANGLE_EDGE_RATIO)
        )

        if STOP_TRIANGULATION_EDGE > 0:
            good &= metrics["max_edge"] >= STOP_TRIANGULATION_EDGE

        if np.any(good):
            accepted_parts.append(valid_part[good])
            simplex_parts.append(valid_simplex[good])
            distance_merit = (
                metrics["distance"][good] / max(effective_distance, 1.0e-9)
            )
            safe_angle_limit = np.maximum(angle_limit[good], 1.0e-9)
            angle_merit = metrics["angle_deg"][good] / safe_angle_limit
            merit_parts.append(distance_merit + angle_merit)
            residual_distance_parts.append(metrics["distance"][good])
            residual_angle_parts.append(metrics["angle_deg"][good])

    empty = np.empty(0, dtype=np.int64)
    empty_f = np.empty(0, dtype=np.float64)
    if not accepted_parts:
        return empty, empty_f, empty_f

    accepted = np.concatenate(accepted_parts)
    residual_distance = np.concatenate(residual_distance_parts)
    residual_angle = np.concatenate(residual_angle_parts)
    if not ONE_POINT_PER_TRIANGLE_PASS1 or settings.name != "Pass 1":
        return accepted, residual_distance, residual_angle

    simplices = np.concatenate(simplex_parts)
    merits = np.concatenate(merit_parts)
    order = np.lexsort((merits, simplices))
    sorted_simplices = simplices[order]
    first = np.empty(order.size, dtype=bool)
    first[0] = True
    first[1:] = sorted_simplices[1:] != sorted_simplices[:-1]
    keep = order[first]
    return accepted[keep], residual_distance[keep], residual_angle[keep]


def run_ptd_pass(
    work_indices: np.ndarray,
    model_indices: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    bounds: tuple[float, float, float, float],
    settings: PassSettings,
) -> np.ndarray:
    """Progressively densify the current ground model.

    When AXELSSON_DATA_DERIVED_THRESHOLDS is enabled, the distance/angle
    acceptance threshold for each iteration is estimated from the residual
    distribution of points already confirmed in the model - Axelsson (2000)
    section 3.2 - rather than held at one fixed value for the whole pass.
    The pass's own fixed settings.distance / angle_min/max_deg are used only
    to seed iteration 1, before any model-specific statistics exist yet
    (Axelsson's "initialisation phase, before the TIN is created").
    """
    model_membership = np.zeros(x.size, dtype=bool)
    model_membership[model_indices] = True
    # Candidates are always a subset of work_indices. Keeping the shrinking
    # list directly avoids scanning every source return on each PTD iteration.
    candidates = work_indices[~model_membership[work_indices]]

    residual_distance_buffer = np.empty(0, dtype=np.float64)
    residual_angle_buffer = np.empty(0, dtype=np.float64)
    rng = np.random.default_rng(DATA_DERIVED_ESTIMATOR_SEED)

    for iteration in range(1, MAX_ITERATIONS_PER_PASS + 1):
        if candidates.size == 0:
            break

        tin, model_xy, model_z, _ = build_tin(
            model_indices, x, y, z, bounds
        )

        if AXELSSON_DATA_DERIVED_THRESHOLDS and residual_distance_buffer.size >= DATA_DERIVED_MIN_SAMPLES:
            current_distance = float(np.clip(
                np.percentile(residual_distance_buffer, DATA_DERIVED_PERCENTILE),
                DATA_DERIVED_MIN_DISTANCE,
                DATA_DERIVED_MAX_DISTANCE,
            ))
            angle_p90 = float(np.clip(
                np.percentile(residual_angle_buffer, DATA_DERIVED_PERCENTILE),
                DATA_DERIVED_MIN_ANGLE_DEG,
                DATA_DERIVED_MAX_ANGLE_DEG,
            ))
            # Preserve the pass's flat/steep spread around the data-derived
            # centre rather than collapsing to one flat number, so genuinely
            # steep sub-areas of THIS tile still get more angle headroom
            # than flat sub-areas of the same tile (slope-adaptive, layered
            # on a data-derived scale instead of a fixed one).
            spread = 0.5 * (settings.angle_max_deg - settings.angle_min_deg)
            current_angle_range = (
                max(DATA_DERIVED_MIN_ANGLE_DEG, angle_p90 - spread),
                min(DATA_DERIVED_MAX_ANGLE_DEG, angle_p90 + spread),
            )
        else:
            current_distance = None
            current_angle_range = None

        accepted, acc_distance, acc_angle = accepted_candidates(
            tin,
            model_xy,
            model_z,
            candidates,
            x,
            y,
            z,
            settings,
            distance_limit=current_distance,
            angle_range=current_angle_range,
        )

        if accepted.size == 0:
            log(
                f"      {settings.name} iteration {iteration}: "
                "no new ground model points"
            )
            break

        if AXELSSON_DATA_DERIVED_THRESHOLDS:
            residual_distance_buffer = np.concatenate(
                (residual_distance_buffer, acc_distance)
            )
            residual_angle_buffer = np.concatenate(
                (residual_angle_buffer, acc_angle)
            )
            if residual_distance_buffer.size > DATA_DERIVED_RESIDUAL_BUFFER:
                keep = rng.choice(
                    residual_distance_buffer.size,
                    size=DATA_DERIVED_RESIDUAL_BUFFER,
                    replace=False,
                )
                residual_distance_buffer = residual_distance_buffer[keep]
                residual_angle_buffer = residual_angle_buffer[keep]

        model_membership[accepted] = True
        model_indices = np.sort(
            np.concatenate((model_indices, accepted))
        )
        candidates = candidates[~model_membership[candidates]]
        threshold_note = (
            f"; data-derived dist<={current_distance:.2f}m angle~{current_angle_range[0]:.1f}-{current_angle_range[1]:.1f}deg"
            if current_distance is not None
            else ""
        )
        log(
            f"      {settings.name} iteration {iteration}: "
            f"+{accepted.size:,} model points "
            f"({model_indices.size:,} total){threshold_note}"
        )
    else:
        log(
            f"      {settings.name}: reached "
            f"{MAX_ITERATIONS_PER_PASS} iterations"
        )

    return model_indices


def prune_upward_model_spikes(
    model_indices: np.ndarray,
    support_indices: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> tuple[np.ndarray, int]:
    """
    Remove isolated upward needles from the accepted terrain model.

    A point is removed only when it is high relative to both the local median
    and a fitted neighbour plane. Residual support is measured against all
    cleaned work minima, not just current model vertices, so genuine ridges
    and breaklines remain even when the strict PTD model is sparse.
    """
    total_removed = 0
    support_xy = np.column_stack((x[support_indices], y[support_indices]))
    support_tree = cKDTree(support_xy)

    for _ in range(MODEL_SPIKE_PRUNE_PASSES):
        if model_indices.size <= MODEL_SPIKE_NEIGHBOURS:
            break

        model_xy = np.column_stack((x[model_indices], y[model_indices]))
        tree = cKDTree(model_xy)
        k = min(MODEL_SPIKE_NEIGHBOURS + 1, model_indices.size)
        distances, neighbours = tree_query(
            tree,
            model_xy,
            k=k,
            distance_upper_bound=MODEL_SPIKE_RADIUS,
        )
        valid = (
            np.isfinite(distances)
            & (neighbours < model_indices.size)
        )
        rows = np.arange(model_indices.size)[:, None]
        valid &= neighbours != rows

        neighbour_z = np.full(neighbours.shape, np.nan, dtype=np.float64)
        neighbour_z[valid] = z[model_indices[neighbours[valid]]]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            local_median = np.nanmedian(neighbour_z, axis=1)

        point_z = z[model_indices]
        nominated = np.flatnonzero(
            np.isfinite(local_median)
            & (point_z > local_median + MODEL_SPIKE_HEIGHT)
        )
        if nominated.size == 0:
            break

        remove = np.zeros(model_indices.size, dtype=bool)
        for position in nominated:
            local_positions = neighbours[position][valid[position]]
            if local_positions.size < 5:
                continue

            local_indices = model_indices[local_positions]
            dx = x[local_indices] - x[model_indices[position]]
            dy = y[local_indices] - y[model_indices[position]]
            design = np.column_stack(
                (dx, dy, np.ones(local_indices.size))
            )
            try:
                coefficients, *_ = np.linalg.lstsq(
                    design, z[local_indices], rcond=None
                )
                residuals = z[local_indices] - design @ coefficients
                residual_median = np.median(residuals)
                mad = np.median(np.abs(residuals - residual_median))
                tolerance = max(0.20, 3.0 * 1.4826 * mad)
                inliers = np.abs(residuals - residual_median) <= tolerance
                if np.count_nonzero(inliers) >= 5:
                    coefficients, *_ = np.linalg.lstsq(
                        design[inliers],
                        z[local_indices][inliers],
                        rcond=None,
                    )
            except np.linalg.LinAlgError:
                continue

            predicted = float(coefficients[2])
            point_residual = point_z[position] - predicted
            if point_residual <= MODEL_SPIKE_HEIGHT:
                continue

            support_k = min(
                MODEL_SPIKE_SUPPORT_NEIGHBOURS + 1,
                support_indices.size,
            )
            support_distances, support_positions = tree_query(
                support_tree,
                model_xy[position],
                k=support_k,
                distance_upper_bound=MODEL_SPIKE_SUPPORT_RADIUS,
            )
            support_distances = np.atleast_1d(support_distances)
            support_positions = np.atleast_1d(support_positions)
            support_valid = (
                np.isfinite(support_distances)
                & (support_positions < support_indices.size)
            )
            nearby = support_indices[support_positions[support_valid]]
            nearby = nearby[nearby != model_indices[position]]
            if nearby.size:
                nearby_dx = x[nearby] - x[model_indices[position]]
                nearby_dy = y[nearby] - y[model_indices[position]]
                nearby_plane = (
                    coefficients[0] * nearby_dx
                    + coefficients[1] * nearby_dy
                    + coefficients[2]
                )
                nearby_residual = z[nearby] - nearby_plane
                residual_support = np.count_nonzero(
                    np.abs(nearby_residual - point_residual)
                    <= MODEL_SPIKE_SUPPORT_TOLERANCE
                )
                if residual_support >= MODEL_SPIKE_MIN_SUPPORT:
                    continue

            remove[position] = True

        removed = int(np.count_nonzero(remove))
        if removed == 0 or model_indices.size - removed < 3:
            break
        model_indices = model_indices[~remove]
        total_removed += removed

    return model_indices, total_removed


def local_plane_support(
    query_indices: np.ndarray,
    support_indices: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    support_tree: cKDTree | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return robust local-plane prediction, RMSE and neighbour count.

    The batched normal-equation solve keeps this practical for large blocks.
    A second solve trims returns far from the first plane, so vegetation does
    not dominate the local terrain estimate.
    """
    predicted = np.full(query_indices.size, np.nan, dtype=np.float64)
    rmse = np.full(query_indices.size, np.inf, dtype=np.float64)
    counts = np.zeros(query_indices.size, dtype=np.int32)
    residual_support = np.zeros(query_indices.size, dtype=np.int32)
    if query_indices.size == 0 or support_indices.size < 3:
        return predicted, rmse, counts, residual_support

    if support_tree is None:
        support_xy = np.column_stack(
            (x[support_indices], y[support_indices])
        )
        support_tree = cKDTree(support_xy)
    k = min(UPPER_SURFACE_NEIGHBOURS + 1, support_indices.size)

    plane_chunk = max(10_000, TEST_CHUNK_SIZE // 3)
    for start in range(0, query_indices.size, plane_chunk):
        stop = min(start + plane_chunk, query_indices.size)
        part = query_indices[start:stop]
        query_xy = np.column_stack((x[part], y[part]))
        distances, neighbours = tree_query(
            support_tree,
            query_xy,
            k=k,
            distance_upper_bound=UPPER_SURFACE_NEIGHBOUR_RADIUS,
        )
        distances = np.atleast_2d(distances)
        neighbours = np.atleast_2d(neighbours)
        valid = np.isfinite(distances) & (neighbours < support_indices.size)

        neighbour_global = np.full(neighbours.shape, -1, dtype=np.int64)
        neighbour_global[valid] = support_indices[neighbours[valid]]
        valid &= neighbour_global != part[:, None]

        dx = np.zeros(neighbours.shape, dtype=np.float64)
        dy = np.zeros(neighbours.shape, dtype=np.float64)
        nz = np.zeros(neighbours.shape, dtype=np.float64)
        point_x = x[part, None]
        point_y = y[part, None]
        dx[valid] = (
            x[neighbour_global[valid]]
            - np.broadcast_to(point_x, neighbours.shape)[valid]
        )
        dy[valid] = (
            y[neighbour_global[valid]]
            - np.broadcast_to(point_y, neighbours.shape)[valid]
        )
        nz[valid] = z[neighbour_global[valid]]

        design = np.empty(
            (dx.shape[0], dx.shape[1], 3), dtype=np.float64
        )
        design[:, :, 0] = dx
        design[:, :, 1] = dy
        design[:, :, 2] = 1.0

        def solve_plane(
            weights: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
            ata = np.einsum(
                "nki,nk,nkj->nij", design, weights, design
            )
            atz = np.einsum("nki,nk,nk->ni", design, weights, nz)
            # Small regularisation handles collinear scan-pattern neighbours.
            ata[:, 0, 0] += 1.0e-8
            ata[:, 1, 1] += 1.0e-8
            ata[:, 2, 2] += 1.0e-8
            coefficients = np.linalg.solve(
                ata, atz[..., None]
            )[..., 0]
            fitted = np.einsum("nki,ni->nk", design, coefficients)
            return coefficients, fitted

        weights = valid.astype(np.float64)
        neighbour_count = np.sum(valid, axis=1)
        coefficients, fitted = solve_plane(weights)
        residual = np.where(valid, nz - fitted, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            residual_median = np.nanmedian(residual, axis=1)
            mad = np.nanmedian(
                np.abs(residual - residual_median[:, None]), axis=1
            )
        trim = np.maximum(0.20, 3.0 * 1.4826 * np.nan_to_num(mad))
        robust = (
            valid
            & (
                np.abs(residual - residual_median[:, None])
                <= trim[:, None]
            )
        )
        robust_count = np.sum(robust, axis=1)
        use_robust = robust_count >= UPPER_SURFACE_MIN_NEIGHBOURS
        final_weights = np.where(use_robust[:, None], robust, valid).astype(
            np.float64
        )
        coefficients, fitted = solve_plane(final_weights)
        final_count = np.sum(final_weights, axis=1)
        squared = np.where(
            final_weights > 0,
            np.square(nz - fitted),
            0.0,
        )
        local_rmse = np.sqrt(
            np.sum(squared, axis=1) / np.maximum(final_count, 1)
        )
        point_residual = z[part] - coefficients[:, 2]
        neighbour_residual = np.where(
            valid, nz - fitted, np.nan
        )
        supported = np.sum(
            valid
            & (
                distances
                <= UPPER_SURFACE_RESIDUAL_SUPPORT_RADIUS
            )
            & (
                np.abs(
                    neighbour_residual - point_residual[:, None]
                )
                <= UPPER_SURFACE_RESIDUAL_SUPPORT_TOLERANCE
            ),
            axis=1,
        )

        good = neighbour_count >= UPPER_SURFACE_MIN_NEIGHBOURS
        predicted[start:stop][good] = coefficients[good, 2]
        rmse[start:stop][good] = local_rmse[good]
        counts[start:stop] = neighbour_count
        residual_support[start:stop] = supported

    return predicted, rmse, counts, residual_support


def remove_unsupported_low_work_points(
    work_indices: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, PlaneContext | None]:
    """
    Exclude only isolated gross lows from initial TIN construction.

    Classification is not changed here. The nominated points are retained for
    a second confirmation against the completed terrain surface; this prevents
    a legitimate continuous depression from being labelled as noise merely
    because the initial TIN was coarse.
    """
    if work_indices.size < UPPER_SURFACE_MIN_NEIGHBOURS + 1:
        return work_indices, np.empty(0, dtype=np.int64), None

    plane_context = local_plane_support(
        work_indices,
        work_indices,
        x,
        y,
        z,
    )
    plane_z, plane_rmse, neighbour_count, residual_support = plane_context
    residual = z[work_indices] - plane_z
    unsupported = (
        np.isfinite(plane_z)
        & (neighbour_count >= UPPER_SURFACE_MIN_NEIGHBOURS)
        & (plane_rmse <= UNSUPPORTED_LOW_MAX_PLANE_RMSE)
        & (residual <= -UNSUPPORTED_LOW_MIN_DEPTH)
        & (residual_support < UNSUPPORTED_LOW_MIN_SUPPORT)
    )
    rejected = work_indices[unsupported]
    if rejected.size == 0 or work_indices.size - rejected.size < 3:
        return work_indices, np.empty(0, dtype=np.int64), plane_context
    # The context is aligned to the pre-removal array and must not be reused
    # after rejected points have been removed from the support set.
    return work_indices[~unsupported], rejected, None


def confirm_unsupported_low_noise(
    source_indices: np.ndarray,
    support_indices: np.ndarray,
    preliminary_low_indices: np.ndarray,
    hag: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    support_tree: cKDTree | None = None,
) -> np.ndarray:
    """Confirm pre-nominated gross lows before assigning class 7."""
    if preliminary_low_indices.size == 0:
        return np.empty(0, dtype=np.int64)

    candidates = source_indices[
        np.isfinite(hag[source_indices])
        & (hag[source_indices] <= -UNSUPPORTED_LOW_MIN_DEPTH)
    ]
    if candidates.size == 0:
        return candidates

    preliminary_tree = cKDTree(
        np.column_stack(
            (x[preliminary_low_indices], y[preliminary_low_indices])
        )
    )
    distance, _ = tree_query(
        preliminary_tree,
        np.column_stack((x[candidates], y[candidates])), k=1
    )
    candidates = candidates[
        distance <= UNSUPPORTED_LOW_CONFIRM_RADIUS
    ]
    if candidates.size == 0:
        return candidates

    plane_z, plane_rmse, neighbour_count, residual_support = (
        local_plane_support(
            candidates,
            support_indices,
            x,
            y,
            z,
            support_tree,
        )
    )
    residual = z[candidates] - plane_z
    confirmed = (
        np.isfinite(plane_z)
        & (neighbour_count >= UPPER_SURFACE_MIN_NEIGHBOURS)
        & (plane_rmse <= UNSUPPORTED_LOW_MAX_PLANE_RMSE)
        & (residual <= -UNSUPPORTED_LOW_MIN_DEPTH)
        & (residual_support < UNSUPPORTED_LOW_MIN_SUPPORT)
    )
    return candidates[confirmed]


def prune_model_above_lower_plane(
    model_indices: np.ndarray,
    work_indices: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    support_tree: cKDTree | None = None,
) -> tuple[np.ndarray, int]:
    """
    Remove accepted vertices that sit above a robust nearby lower surface.

    Progressive TIN growth can occasionally climb a chain of vegetation on a
    hillside. A plane fitted to nearby lower-envelope points exposes that
    failure: the canopy vertex is high above a clean, well-supported plane.
    The RMSE gate avoids flattening genuinely rough or discontinuous terrain.
    """
    plane_z, plane_rmse, neighbour_count, _ = local_plane_support(
        model_indices,
        work_indices,
        x,
        y,
        z,
        support_tree,
    )
    residual = z[model_indices] - plane_z
    adaptive_height = np.maximum(
        MODEL_LOWER_PLANE_PRUNE_HEIGHT,
        MODEL_LOWER_PLANE_PRUNE_RMSE_FACTOR * plane_rmse,
    )
    remove = (
        np.isfinite(plane_z)
        & (neighbour_count >= UPPER_SURFACE_MIN_NEIGHBOURS)
        & (plane_rmse <= MODEL_LOWER_PLANE_PRUNE_MAX_RMSE)
        & (residual > adaptive_height)
    )
    removed = int(np.count_nonzero(remove))
    if removed == 0 or model_indices.size - removed < 3:
        return model_indices, 0
    return model_indices[~remove], removed



def trusted_quadratic_frontier_support(
    query_indices: np.ndarray,
    model_indices: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    bounds: tuple[float, float, float, float],
) -> np.ndarray:
    """Return one-step curved-terrain support from accepted model points only.

    The fit is intentionally local, weighted and bounded. It is never fitted to
    candidate/work-minimum points, so a vegetation or roof patch cannot make
    itself look like terrain. The trusted TIN remains a broad sanity reference.
    """
    result = np.zeros(query_indices.size, dtype=bool)
    if (
        not TRUSTED_QUADRATIC_FRONTIER_RECOVERY
        or query_indices.size == 0
        or model_indices.size < TRUSTED_QUADRATIC_MIN_NEIGHBOURS
    ):
        return result

    model_xy = np.column_stack((x[model_indices], y[model_indices]))
    model_tree = cKDTree(model_xy)
    k = min(TRUSTED_QUADRATIC_NEIGHBOURS, model_indices.size)

    for start in range(0, query_indices.size, TRUSTED_QUADRATIC_CHUNK_SIZE):
        stop = min(start + TRUSTED_QUADRATIC_CHUNK_SIZE, query_indices.size)
        part = query_indices[start:stop]
        qxy = np.column_stack((x[part], y[part]))
        distances, positions = tree_query(
            model_tree,
            qxy,
            k=k,
            distance_upper_bound=TRUSTED_QUADRATIC_NEIGHBOUR_RADIUS,
        )
        distances = np.asarray(distances, dtype=np.float64)
        positions = np.asarray(positions, dtype=np.int64)
        if distances.ndim == 1:
            distances = distances[:, None]
            positions = positions[:, None]
        valid = (
            np.isfinite(distances)
            & (positions >= 0)
            & (positions < model_indices.size)
        )
        counts = np.sum(valid, axis=1)
        nearest = np.min(np.where(valid, distances, np.inf), axis=1)
        initial = (
            (counts >= TRUSTED_QUADRATIC_MIN_NEIGHBOURS)
            & (nearest <= TRUSTED_QUADRATIC_MAX_MODEL_DISTANCE)
        )
        if not np.any(initial):
            continue

        safe = np.clip(positions, 0, max(0, model_indices.size - 1))
        global_idx = model_indices[safe]
        dx = np.zeros(distances.shape, dtype=np.float64)
        dy = np.zeros(distances.shape, dtype=np.float64)
        nz = np.zeros(distances.shape, dtype=np.float64)
        qx = x[part, None]
        qy = y[part, None]
        dx[valid] = x[global_idx[valid]] - np.broadcast_to(qx, distances.shape)[valid]
        dy[valid] = y[global_idx[valid]] - np.broadcast_to(qy, distances.shape)[valid]
        nz[valid] = z[global_idx[valid]]

        weights = np.where(
            valid,
            1.0 / np.square(np.maximum(distances, 0.50)),
            0.0,
        )
        design = np.empty((part.size, distances.shape[1], 6), dtype=np.float64)
        design[:, :, 0] = dx * dx
        design[:, :, 1] = dy * dy
        design[:, :, 2] = dx * dy
        design[:, :, 3] = dx
        design[:, :, 4] = dy
        design[:, :, 5] = 1.0
        ata = np.einsum('nki,nk,nkj->nij', design, weights, design)
        atz = np.einsum('nki,nk,nk->ni', design, weights, nz)
        # Stronger damping on quadratic terms limits unstable curvature while
        # leaving the local gradient/intercept effectively data driven.
        ata[:, 0, 0] += 2.0e-3
        ata[:, 1, 1] += 2.0e-3
        ata[:, 2, 2] += 2.0e-3
        ata[:, 3, 3] += 1.0e-6
        ata[:, 4, 4] += 1.0e-6
        ata[:, 5, 5] += 1.0e-8
        try:
            coeff = np.linalg.solve(ata, atz[..., None])[..., 0]
        except np.linalg.LinAlgError:
            continue

        fitted = np.einsum('nki,ni->nk', design, coeff)
        sq = np.where(valid, np.square(nz - fitted), 0.0)
        rmse = np.sqrt(np.sum(sq, axis=1) / np.maximum(counts, 1))
        predicted = coeff[:, 5]
        residual = z[part] - predicted
        fitted_slope = np.degrees(
            np.arctan(np.linalg.norm(coeff[:, 3:5], axis=1))
        )
        curvature = np.maximum.reduce(
            (
                np.abs(2.0 * coeff[:, 0]),
                np.abs(2.0 * coeff[:, 1]),
                np.abs(coeff[:, 2]),
            )
        )

        # 2-D support spread gate.
        wsum = np.maximum(np.sum(weights, axis=1), 1.0e-12)
        mx = np.sum(weights * dx, axis=1) / wsum
        my = np.sum(weights * dy, axis=1) / wsum
        cdx = dx - mx[:, None]
        cdy = dy - my[:, None]
        cxx = np.sum(weights * cdx * cdx, axis=1) / wsum
        cyy = np.sum(weights * cdy * cdy, axis=1) / wsum
        cxy = np.sum(weights * cdx * cdy, axis=1) / wsum
        trace = cxx + cyy
        disc = np.sqrt(np.maximum((cxx - cyy) ** 2 + 4.0 * cxy * cxy, 0.0))
        xy_spread = np.sqrt(np.maximum(0.5 * (trace - disc), 0.0))

        tin_residual, tin_slope, tin_edge = interpolate_tin_membership_context(
            part, model_indices, x, y, z, bounds
        )
        tin_residual = tin_residual.astype(np.float64, copy=False)
        tin_slope = tin_slope.astype(np.float64, copy=False)
        tin_edge = tin_edge.astype(np.float64, copy=False)
        slope_weight = np.clip(
            np.maximum(tin_slope, np.nan_to_num(fitted_slope))
            / TRUSTED_QUADRATIC_FULL_SLOPE_DEG,
            0.0,
            1.0,
        )
        upper = (
            TRUSTED_QUADRATIC_FLAT_UPPER
            + slope_weight
            * (TRUSTED_QUADRATIC_STEEP_UPPER - TRUSTED_QUADRATIC_FLAT_UPPER)
        )
        lower = (
            TRUSTED_QUADRATIC_FLAT_LOWER
            + slope_weight
            * (TRUSTED_QUADRATIC_STEEP_LOWER - TRUSTED_QUADRATIC_FLAT_LOWER)
        )

        # Direct connection slope to the nearest already accepted model point.
        nearest_pos = np.argmin(np.where(valid, distances, np.inf), axis=1)
        row = np.arange(part.size)
        nearest_global = global_idx[row, nearest_pos]
        horizontal = np.maximum(nearest, 1.0e-6)
        dz_near = np.maximum(
            np.abs(z[part] - z[nearest_global]) - TRUSTED_CONNECTION_VERTICAL_SLACK,
            0.0,
        )
        connection_slope = np.degrees(np.arctan2(dz_near, horizontal))
        allowed = np.clip(
            np.maximum(
                TRUSTED_CONNECTION_MIN_SLOPE_DEG,
                np.maximum(tin_slope, np.nan_to_num(fitted_slope))
                + TRUSTED_CONNECTION_SLOPE_ALLOWANCE_DEG,
            ),
            TRUSTED_CONNECTION_MIN_SLOPE_DEG,
            TRUSTED_CONNECTION_MAX_SLOPE_DEG,
        )

        good = (
            initial
            & np.isfinite(predicted)
            & np.isfinite(rmse)
            & np.isfinite(tin_residual)
            & np.isfinite(tin_edge)
            & (xy_spread >= TRUSTED_QUADRATIC_MIN_XY_SPREAD)
            & (rmse <= TRUSTED_QUADRATIC_MAX_RMSE)
            & (curvature <= TRUSTED_QUADRATIC_MAX_CURVATURE)
            & (residual >= -lower)
            & (residual <= upper)
            & (tin_residual >= -TRUSTED_QUADRATIC_TIN_LOWER_SANITY)
            & (tin_residual <= TRUSTED_QUADRATIC_TIN_UPPER_SANITY)
            & (tin_edge >= TRUSTED_QUADRATIC_MIN_TRIANGLE_EDGE)
            & (connection_slope <= allowed)
        )
        result[start:stop] = good

    return result


def trusted_directional_breakline_support(
    query_indices: np.ndarray,
    model_indices: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    bounds: tuple[float, float, float, float],
    original_classification: np.ndarray | None = None,
) -> np.ndarray:
    """Return one-step support at a sharp terrain crest/toe.

    Only already accepted model points enter the fits. Eight overlapping
    angular wedges are tested around each candidate so points from the upper
    and lower sides of a bank do not have to share one regression surface.
    The gate activates only when the trusted neighbourhood has real relief or
    a material slope contrast, keeping it out of ordinary flat vegetation.
    """
    result = np.zeros(query_indices.size, dtype=bool)
    if (
        not TRUSTED_BREAKLINE_RECOVERY
        or query_indices.size == 0
        or model_indices.size < TRUSTED_BREAKLINE_MIN_NEIGHBOURS
    ):
        return result

    model_xy = np.column_stack((x[model_indices], y[model_indices]))
    model_tree = cKDTree(model_xy)
    k = min(TRUSTED_BREAKLINE_NEIGHBOURS, model_indices.size)

    for start in range(0, query_indices.size, TRUSTED_BREAKLINE_CHUNK_SIZE):
        stop = min(start + TRUSTED_BREAKLINE_CHUNK_SIZE, query_indices.size)
        part = query_indices[start:stop]
        qxy = np.column_stack((x[part], y[part]))
        distances, positions = tree_query(
            model_tree,
            qxy,
            k=k,
            distance_upper_bound=TRUSTED_BREAKLINE_NEIGHBOUR_RADIUS,
        )
        distances = np.asarray(distances, dtype=np.float64)
        positions = np.asarray(positions, dtype=np.int64)
        if distances.ndim == 1:
            distances = distances[:, None]
            positions = positions[:, None]

        valid = (
            np.isfinite(distances)
            & (positions >= 0)
            & (positions < model_indices.size)
        )
        counts = np.sum(valid, axis=1)
        if not np.any(counts >= TRUSTED_BREAKLINE_MIN_NEIGHBOURS):
            continue

        safe = np.clip(positions, 0, max(0, model_indices.size - 1))
        global_idx = model_indices[safe]
        dx = np.zeros(distances.shape, dtype=np.float64)
        dy = np.zeros(distances.shape, dtype=np.float64)
        nz = np.zeros(distances.shape, dtype=np.float64)
        dx[valid] = x[global_idx[valid]] - np.broadcast_to(x[part, None], distances.shape)[valid]
        dy[valid] = y[global_idx[valid]] - np.broadcast_to(y[part, None], distances.shape)[valid]
        nz[valid] = z[global_idx[valid]]

        local_z = np.where(valid, nz, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            z_range = np.nanmax(local_z, axis=1) - np.nanmin(local_z, axis=1)

        tin_residual, tin_slope, tin_edge = interpolate_tin_membership_context(
            part,
            model_indices,
            x,
            y,
            z,
            bounds,
        )
        tin_residual = tin_residual.astype(np.float64, copy=False)
        tin_slope = tin_slope.astype(np.float64, copy=False)
        tin_edge = tin_edge.astype(np.float64, copy=False)

        # Connection slope to each trusted neighbour. The nearest valid path
        # is enough at a real crest/toe, but it must still be terrain-like.
        vertical_step = np.maximum(
            np.abs(nz - z[part, None]) - TRUSTED_CONNECTION_VERTICAL_SLACK,
            0.0,
        )
        connection = np.full(distances.shape, np.nan, dtype=np.float64)
        usable = valid & (distances > 1.0e-6)
        connection[usable] = np.degrees(
            np.arctan2(vertical_step[usable], distances[usable])
        )

        angles = np.degrees(np.arctan2(dy, dx))
        accepted_chunk = np.zeros(part.size, dtype=bool)
        best_slope_contrast = np.zeros(part.size, dtype=np.float64)

        for sector in range(TRUSTED_BREAKLINE_DIRECTION_SECTORS):
            centre = -180.0 + (
                sector + 0.5
            ) * (360.0 / TRUSTED_BREAKLINE_DIRECTION_SECTORS)
            diff = np.abs((angles - centre + 180.0) % 360.0 - 180.0)
            wedge = valid & (diff <= TRUSTED_BREAKLINE_WEDGE_HALF_WIDTH_DEG)
            wcount = np.sum(wedge, axis=1)
            eligible_rows = wcount >= TRUSTED_BREAKLINE_MIN_NEIGHBOURS
            if not np.any(eligible_rows):
                continue

            weights = np.where(
                wedge,
                1.0 / np.square(np.maximum(distances, 0.30)),
                0.0,
            )
            design = np.empty((part.size, distances.shape[1], 3), dtype=np.float64)
            design[:, :, 0] = dx
            design[:, :, 1] = dy
            design[:, :, 2] = 1.0
            ata = np.einsum("nki,nk,nkj->nij", design, weights, design)
            atz = np.einsum("nki,nk,nk->ni", design, weights, nz)
            ata[:, 0, 0] += 1.0e-7
            ata[:, 1, 1] += 1.0e-7
            ata[:, 2, 2] += 1.0e-7
            try:
                coeff = np.linalg.solve(ata, atz[..., None])[..., 0]
            except np.linalg.LinAlgError:
                continue

            fitted = np.einsum("nki,ni->nk", design, coeff)
            sq = np.where(wedge, np.square(nz - fitted), 0.0)
            rmse = np.sqrt(np.sum(sq, axis=1) / np.maximum(wcount, 1))
            predicted = coeff[:, 2]
            residual = z[part] - predicted
            fit_slope = np.degrees(np.arctan(np.linalg.norm(coeff[:, :2], axis=1)))

            wsum = np.maximum(np.sum(weights, axis=1), 1.0e-12)
            mean_dx = np.sum(weights * dx, axis=1) / wsum
            mean_dy = np.sum(weights * dy, axis=1) / wsum
            cdx = dx - mean_dx[:, None]
            cdy = dy - mean_dy[:, None]
            cxx = np.sum(weights * cdx * cdx, axis=1) / wsum
            cyy = np.sum(weights * cdy * cdy, axis=1) / wsum
            cxy = np.sum(weights * cdx * cdy, axis=1) / wsum
            trace = cxx + cyy
            disc = np.sqrt(np.maximum((cxx - cyy) ** 2 + 4.0 * cxy * cxy, 0.0))
            lambda_min = np.maximum(0.5 * (trace - disc), 0.0)
            xy_spread = np.sqrt(lambda_min)

            wedge_distance = np.min(np.where(wedge, distances, np.inf), axis=1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                wedge_connection = np.nanmin(np.where(wedge, connection, np.nan), axis=1)

            slope_for_tolerance = np.maximum(tin_slope, np.nan_to_num(fit_slope))
            sw = np.clip(
                slope_for_tolerance / TRUSTED_BREAKLINE_FULL_SLOPE_DEG,
                0.0,
                1.0,
            )
            upper = (
                TRUSTED_BREAKLINE_FLAT_UPPER
                + sw * (TRUSTED_BREAKLINE_STEEP_UPPER - TRUSTED_BREAKLINE_FLAT_UPPER)
            )
            lower = (
                TRUSTED_BREAKLINE_FLAT_LOWER
                + sw * (TRUSTED_BREAKLINE_STEEP_LOWER - TRUSTED_BREAKLINE_FLAT_LOWER)
            )
            slope_contrast = np.abs(fit_slope - tin_slope)
            best_slope_contrast = np.maximum(
                best_slope_contrast,
                np.nan_to_num(slope_contrast),
            )
            break_context = (
                (z_range >= TRUSTED_BREAKLINE_MIN_LOCAL_Z_RANGE)
                & (
                    (slope_contrast >= TRUSTED_BREAKLINE_MIN_SLOPE_CONTRAST_DEG)
                    | (tin_slope >= TRUSTED_BREAKLINE_MIN_TIN_SLOPE_DEG)
                )
            )
            allowed_connection = np.clip(
                np.maximum(
                    TRUSTED_CONNECTION_MIN_SLOPE_DEG,
                    slope_for_tolerance + TRUSTED_CONNECTION_SLOPE_ALLOWANCE_DEG,
                ),
                TRUSTED_CONNECTION_MIN_SLOPE_DEG,
                TRUSTED_CONNECTION_MAX_SLOPE_DEG,
            )

            prior_ok = np.ones(part.size, dtype=bool)
            if (
                TRUSTED_BREAKLINE_USE_INPUT_PRIOR
                and original_classification is not None
            ):
                prior = original_classification[part]
                prior_upper_cap = np.full(part.size, np.inf, dtype=np.float64)
                prior_upper_cap[prior == 3] = TRUSTED_BREAKLINE_PRIOR_CLASS3_MAX_UPPER
                prior_upper_cap[prior == 4] = TRUSTED_BREAKLINE_PRIOR_CLASS4_MAX_UPPER
                prior_upper_cap[prior == 13] = TRUSTED_BREAKLINE_PRIOR_CLASS13_MAX_UPPER
                prior_upper_cap[prior == 5] = TRUSTED_BREAKLINE_PRIOR_CLASS5_MAX_UPPER
                # Only positive residuals are tightened. A genuine depression
                # or downslope ground return below the trusted TIN is not
                # penalised because it arrived in a vegetation class.
                prior_ok = (tin_residual <= prior_upper_cap) | (tin_residual <= 0.0)

            good = (
                eligible_rows
                & break_context
                & prior_ok
                & np.isfinite(predicted)
                & np.isfinite(rmse)
                & np.isfinite(wedge_distance)
                & np.isfinite(wedge_connection)
                & np.isfinite(tin_residual)
                & np.isfinite(tin_edge)
                & (wedge_distance <= TRUSTED_BREAKLINE_MAX_MODEL_DISTANCE)
                & (xy_spread >= TRUSTED_BREAKLINE_MIN_XY_SPREAD)
                & (rmse <= TRUSTED_BREAKLINE_MAX_RMSE)
                & (residual >= -lower)
                & (residual <= upper)
                & (tin_residual >= -TRUSTED_BREAKLINE_TIN_LOWER_SANITY)
                & (tin_residual <= TRUSTED_BREAKLINE_TIN_UPPER_SANITY)
                & (tin_edge >= TRUSTED_BREAKLINE_MIN_TRIANGLE_EDGE)
                & (wedge_connection <= allowed_connection)
            )
            accepted_chunk |= good

        result[start:stop] = accepted_chunk

    return result


def trusted_surface_support(
    query_indices: np.ndarray,
    model_indices: np.ndarray,
    backbone_indices: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    bounds: tuple[float, float, float, float],
    flat_tolerance: float = TRUSTED_SUPPORT_FLAT_PLANE_TOLERANCE,
    steep_tolerance: float = TRUSTED_SUPPORT_STEEP_PLANE_TOLERANCE,
    max_model_distance: float = TRUSTED_SUPPORT_MAX_MODEL_DISTANCE,
    original_classification: np.ndarray | None = None,
) -> np.ndarray:
    """
    Return candidates supported by terrain accepted before this pass.

    v6 retains the five progressive acceptance paths from v5:

    1. Strong local-model support -- the v2 rule. This requires several
       already accepted terrain neighbours, directional support and a robust
       local plane fitted only to accepted terrain.

    2. Sparse TIN-frontier support -- for genuine ground beneath vegetation
       where the accepted model is locally too sparse to supply 3-6 neighbours.
       A candidate may advance the terrain only when it lies in a large/sparse
       trusted-TIN triangle, is very close to that trusted TIN in perpendicular
       distance, and has a terrain-compatible connection to at least one model
       point.

    3. Curvature-aware frontier-plane support -- for patchy ground on curved
       slopes where the straight TIN chord is a poor vertical predictor. A
       small weighted plane is fitted ONLY to already accepted terrain and may
       extrapolate one controlled step. The trusted TIN remains a broad sanity
       envelope. Candidate points still NEVER support one another within a pass.

    4. Curvature-aware quadratic support -- when a planar frontier fit still
       cannot follow a curved bank/road shoulder, a bounded local quadratic
       fitted only to already accepted terrain predicts one controlled step.

    5. Directional breakline support -- at a crest/toe, upper and lower terrain
       can corrupt a single smooth fit. Several one-sided angular wedges are
       fitted from already accepted terrain and the candidate may agree with
       either coherent side of the break.

    Paths 2-5 let ground propagate through woodland, curved slopes and sharp
    terrain transitions without restoring self-supporting vegetation/roof
    behaviour.
    """
    del backbone_indices
    if query_indices.size == 0:
        return np.empty(0, dtype=bool)
    if model_indices.size < 3:
        return np.zeros(query_indices.size, dtype=bool)

    model_xy = np.column_stack((x[model_indices], y[model_indices]))
    model_tree = cKDTree(model_xy)
    query_xy = np.column_stack((x[query_indices], y[query_indices]))

    # Query a little farther than the strong path because the sparse path is
    # explicitly intended for a terrain frontier with few accepted vertices.
    query_radius = max(
        TRUSTED_SUPPORT_NEIGHBOUR_RADIUS,
        TRUSTED_SPARSE_MAX_MODEL_DISTANCE,
        TRUSTED_FRONTIER_NEIGHBOUR_RADIUS,
    )
    k = min(TRUSTED_SUPPORT_NEIGHBOURS, model_indices.size)
    distances, neighbour_positions = tree_query(
        model_tree,
        query_xy,
        k=k,
        distance_upper_bound=query_radius,
    )
    distances = np.asarray(distances, dtype=np.float64)
    neighbour_positions = np.asarray(neighbour_positions, dtype=np.int64)
    if distances.ndim == 1:
        distances = distances[:, None]
        neighbour_positions = neighbour_positions[:, None]

    valid_neighbour = (
        np.isfinite(distances)
        & (neighbour_positions >= 0)
        & (neighbour_positions < model_indices.size)
    )
    support_count = np.sum(valid_neighbour, axis=1)
    nearest_distance = np.where(
        support_count > 0,
        np.min(np.where(valid_neighbour, distances, np.inf), axis=1),
        np.inf,
    )

    safe_positions = np.clip(
        neighbour_positions, 0, max(0, model_indices.size - 1)
    )
    neighbour_global = model_indices[safe_positions]
    dx = np.zeros(distances.shape, dtype=np.float64)
    dy = np.zeros(distances.shape, dtype=np.float64)
    dz = np.zeros(distances.shape, dtype=np.float64)
    query_x = x[query_indices, None]
    query_y = y[query_indices, None]
    query_z = z[query_indices, None]
    dx[valid_neighbour] = (
        x[neighbour_global[valid_neighbour]]
        - np.broadcast_to(query_x, distances.shape)[valid_neighbour]
    )
    dy[valid_neighbour] = (
        y[neighbour_global[valid_neighbour]]
        - np.broadcast_to(query_y, distances.shape)[valid_neighbour]
    )
    dz[valid_neighbour] = np.abs(
        z[neighbour_global[valid_neighbour]]
        - np.broadcast_to(query_z, distances.shape)[valid_neighbour]
    )

    quadrant = (
        (dx >= 0.0).astype(np.int8)
        + 2 * (dy >= 0.0).astype(np.int8)
    )
    direction_count = np.zeros(query_indices.size, dtype=np.int16)
    for sector in range(4):
        direction_count += np.any(
            valid_neighbour & (quadrant == sector), axis=1
        ).astype(np.int16)
    directional_support = (
        (direction_count >= TRUSTED_SUPPORT_MIN_DIRECTION_SECTORS)
        | (support_count >= TRUSTED_SUPPORT_STRONG_NEIGHBOURS)
    )

    # Strong path: robust plane from accepted model only.
    plane_z, plane_rmse, neighbour_count, _ = local_plane_support(
        query_indices,
        model_indices,
        x,
        y,
        z,
        model_tree,
    )

    # Independent trusted-TIN geometry is also calculated for every candidate.
    # This surface contains only already accepted terrain vertices.
    tin_residual, tin_slope, tin_edge = interpolate_tin_membership_context(
        query_indices,
        model_indices,
        x,
        y,
        z,
        bounds,
    )
    tin_residual = tin_residual.astype(np.float64, copy=False)
    local_slope = tin_slope.astype(np.float64, copy=False)
    tin_edge = tin_edge.astype(np.float64, copy=False)

    slope_weight = np.clip(
        local_slope / TRUSTED_SUPPORT_FULL_SLOPE_DEG,
        0.0,
        1.0,
    )
    plane_tolerance = (
        flat_tolerance
        + slope_weight * (steep_tolerance - flat_tolerance)
    )
    plane_residual = z[query_indices] - plane_z

    # Terrain-compatible vertical connection to the already accepted frontier.
    vertical_step = np.maximum(
        dz - TRUSTED_CONNECTION_VERTICAL_SLACK,
        0.0,
    )
    connection_slope_each = np.full(
        distances.shape, np.nan, dtype=np.float64
    )
    usable_connection = valid_neighbour & (distances > 1.0e-6)
    connection_slope_each[usable_connection] = np.degrees(
        np.arctan2(
            vertical_step[usable_connection],
            distances[usable_connection],
        )
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        connection_slope = np.nanmedian(
            connection_slope_each[:, : min(4, connection_slope_each.shape[1])],
            axis=1,
        )

    allowed_connection_slope = np.clip(
        np.maximum(
            TRUSTED_CONNECTION_MIN_SLOPE_DEG,
            local_slope + TRUSTED_CONNECTION_SLOPE_ALLOWANCE_DEG,
        ),
        TRUSTED_CONNECTION_MIN_SLOPE_DEG,
        TRUSTED_CONNECTION_MAX_SLOPE_DEG,
    )

    strong_support = (
        np.isfinite(plane_z)
        & np.isfinite(nearest_distance)
        & np.isfinite(connection_slope)
        & (support_count >= TRUSTED_SUPPORT_MIN_NEIGHBOURS)
        & directional_support
        & (neighbour_count >= UPPER_SURFACE_MIN_NEIGHBOURS)
        & (plane_rmse <= TRUSTED_SUPPORT_MAX_PLANE_RMSE)
        & (np.abs(plane_residual) <= plane_tolerance)
        & (nearest_distance <= max_model_distance)
        & (connection_slope <= allowed_connection_slope)
    )

    if not TRUSTED_SPARSE_TIN_RECOVERY:
        sparse_support = np.zeros(query_indices.size, dtype=bool)
    else:
        sparse_support = None

    # Sparse path: only the trusted TIN can provide the elevation evidence.
    # Positive tolerance is intentionally tighter than negative tolerance so
    # a low vegetation/roof layer above terrain cannot easily start climbing.
    if sparse_support is None:
        sparse_weight = np.clip(
            local_slope / TRUSTED_SPARSE_FULL_SLOPE_DEG,
            0.0,
            1.0,
        )
        sparse_upper = (
            TRUSTED_SPARSE_FLAT_UPPER
            + sparse_weight
            * (TRUSTED_SPARSE_STEEP_UPPER - TRUSTED_SPARSE_FLAT_UPPER)
        )
        sparse_lower = (
            TRUSTED_SPARSE_FLAT_LOWER
            + sparse_weight
            * (TRUSTED_SPARSE_STEEP_LOWER - TRUSTED_SPARSE_FLAT_LOWER)
        )
        sparse_max_distance = min(
            TRUSTED_SPARSE_MAX_MODEL_DISTANCE,
            max_model_distance + 0.75,
        )
        sparse_support = (
            np.isfinite(tin_residual)
            & np.isfinite(tin_edge)
            & np.isfinite(nearest_distance)
            & np.isfinite(connection_slope)
            & (support_count >= TRUSTED_SPARSE_MIN_NEIGHBOURS)
            & (nearest_distance <= sparse_max_distance)
            & (tin_edge >= TRUSTED_SPARSE_MIN_TRIANGLE_EDGE)
            & (tin_residual >= -sparse_lower)
            & (tin_residual <= sparse_upper)
            & (connection_slope <= allowed_connection_slope)
        )

    # Third path: curvature-aware extrapolation from accepted terrain ONLY.
    # This targets LGE-like sections where real ground returns are several
    # metres apart and the Delaunay chord cuts above/below a curved slope.
    frontier_support = np.zeros(query_indices.size, dtype=bool)
    if TRUSTED_FRONTIER_PLANE_RECOVERY:
        frontier_cols = min(
            TRUSTED_FRONTIER_NEIGHBOURS, distances.shape[1]
        )
        fdist = distances[:, :frontier_cols]
        fvalid = (
            valid_neighbour[:, :frontier_cols]
            & (fdist <= TRUSTED_FRONTIER_NEIGHBOUR_RADIUS)
        )
        fcount = np.sum(fvalid, axis=1)

        # Design coordinates are centred on the candidate, so the fitted
        # intercept is the independently predicted terrain Z at that point.
        fdx = dx[:, :frontier_cols]
        fdy = dy[:, :frontier_cols]
        fpos = neighbour_positions[:, :frontier_cols]
        fsafe = np.clip(fpos, 0, max(0, model_indices.size - 1))
        fglobal = model_indices[fsafe]
        fnz = np.zeros(fdist.shape, dtype=np.float64)
        fnz[fvalid] = z[fglobal[fvalid]]

        # Distance weighting keeps the extrapolation local without allowing a
        # single nearest point to dominate. Candidate points are never in this
        # fit because model_indices is frozen for the whole recovery pass.
        fweights = np.where(
            fvalid, 1.0 / np.square(np.maximum(fdist, 0.35)), 0.0
        )
        design = np.empty((query_indices.size, frontier_cols, 3), dtype=np.float64)
        design[:, :, 0] = fdx
        design[:, :, 1] = fdy
        design[:, :, 2] = 1.0
        ata = np.einsum('nki,nk,nkj->nij', design, fweights, design)
        atz = np.einsum('nki,nk,nk->ni', design, fweights, fnz)
        ata[:, 0, 0] += 1.0e-7
        ata[:, 1, 1] += 1.0e-7
        ata[:, 2, 2] += 1.0e-7
        try:
            coeff = np.linalg.solve(ata, atz[..., None])[..., 0]
        except np.linalg.LinAlgError:
            coeff = np.full((query_indices.size, 3), np.nan, dtype=np.float64)

        fitted = np.einsum('nki,ni->nk', design, coeff)
        sq = np.where(fvalid, np.square(fnz - fitted), 0.0)
        frmse = np.sqrt(np.sum(sq, axis=1) / np.maximum(fcount, 1))
        fpred = coeff[:, 2]
        fresidual = z[query_indices] - fpred
        fslope = np.degrees(
            np.arctan(np.linalg.norm(coeff[:, :2], axis=1))
        )

        # Reject nearly collinear/degenerate support. The smaller eigenvalue
        # of the weighted XY covariance is a compact 2-D spread measure.
        wsum = np.maximum(np.sum(fweights, axis=1), 1.0e-12)
        mean_dx = np.sum(fweights * fdx, axis=1) / wsum
        mean_dy = np.sum(fweights * fdy, axis=1) / wsum
        cdx = fdx - mean_dx[:, None]
        cdy = fdy - mean_dy[:, None]
        cxx = np.sum(fweights * cdx * cdx, axis=1) / wsum
        cyy = np.sum(fweights * cdy * cdy, axis=1) / wsum
        cxy = np.sum(fweights * cdx * cdy, axis=1) / wsum
        trace = cxx + cyy
        disc = np.sqrt(np.maximum((cxx - cyy) ** 2 + 4.0 * cxy * cxy, 0.0))
        lambda_min = np.maximum(0.5 * (trace - disc), 0.0)
        xy_spread = np.sqrt(lambda_min)

        frontier_slope_weight = np.clip(
            np.maximum(local_slope, np.nan_to_num(fslope))
            / TRUSTED_FRONTIER_FULL_SLOPE_DEG,
            0.0,
            1.0,
        )
        frontier_upper = (
            TRUSTED_FRONTIER_FLAT_UPPER
            + frontier_slope_weight
            * (TRUSTED_FRONTIER_STEEP_UPPER - TRUSTED_FRONTIER_FLAT_UPPER)
        )
        frontier_lower = (
            TRUSTED_FRONTIER_FLAT_LOWER
            + frontier_slope_weight
            * (TRUSTED_FRONTIER_STEEP_LOWER - TRUSTED_FRONTIER_FLAT_LOWER)
        )
        frontier_allowed_slope = np.clip(
            np.maximum(
                TRUSTED_CONNECTION_MIN_SLOPE_DEG,
                np.maximum(local_slope, np.nan_to_num(fslope))
                + TRUSTED_CONNECTION_SLOPE_ALLOWANCE_DEG,
            ),
            TRUSTED_CONNECTION_MIN_SLOPE_DEG,
            TRUSTED_CONNECTION_MAX_SLOPE_DEG,
        )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            frontier_connection_slope = np.nanmin(
                connection_slope_each[:, : min(4, connection_slope_each.shape[1])],
                axis=1,
            )

        frontier_support = (
            np.isfinite(fpred)
            & np.isfinite(frmse)
            & np.isfinite(nearest_distance)
            & np.isfinite(tin_residual)
            & np.isfinite(tin_edge)
            & np.isfinite(frontier_connection_slope)
            & (fcount >= TRUSTED_FRONTIER_MIN_NEIGHBOURS)
            & (nearest_distance <= TRUSTED_FRONTIER_MAX_MODEL_DISTANCE)
            & (xy_spread >= TRUSTED_FRONTIER_MIN_XY_SPREAD)
            & (frmse <= TRUSTED_FRONTIER_MAX_PLANE_RMSE)
            & (fresidual >= -frontier_lower)
            & (fresidual <= frontier_upper)
            & (tin_residual >= -TRUSTED_FRONTIER_TIN_LOWER_SANITY)
            & (tin_residual <= TRUSTED_FRONTIER_TIN_UPPER_SANITY)
            & (tin_edge >= TRUSTED_FRONTIER_MIN_TRIANGLE_EDGE)
            & (frontier_connection_slope <= frontier_allowed_slope)
        )

    combined = strong_support | sparse_support | frontier_support
    if TRUSTED_QUADRATIC_FRONTIER_RECOVERY:
        unresolved = ~combined
        if np.any(unresolved):
            quadratic_local = trusted_quadratic_frontier_support(
                query_indices[unresolved],
                model_indices,
                x,
                y,
                z,
                bounds,
            )
            combined[unresolved] |= quadratic_local

    # Final narrow path for true terrain breaks. It is evaluated only for
    # candidates unresolved by the smoother recovery modes.
    if TRUSTED_BREAKLINE_RECOVERY:
        unresolved = ~combined
        if np.any(unresolved):
            breakline_local = trusted_directional_breakline_support(
                query_indices[unresolved],
                model_indices,
                x,
                y,
                z,
                bounds,
                original_classification,
            )
            combined[unresolved] |= breakline_local
    return combined


def _terminal_one_sided_plane_support(
    part: np.ndarray,
    model_indices: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    neighbour_radius: float,
    neighbour_count: int,
    min_wedge_neighbours: int,
    direction_sectors: int,
    wedge_half_width_deg: float,
    max_wedge_rmse: float,
    min_xy_spread: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return directional trusted-plane predictions for terminal QC repairs.

    Columns correspond to angular sectors. Every plane is fitted ONLY to the
    already accepted structural terrain model. Candidate points never enter
    the fit, so the result is independent evidence rather than self-support.
    """
    n = part.size
    predicted = np.full((n, direction_sectors), np.nan, dtype=np.float64)
    rmse_out = np.full_like(predicted, np.inf)
    slope_out = np.full_like(predicted, np.nan)
    nearest_out = np.full_like(predicted, np.inf)
    spread_out = np.zeros_like(predicted)
    counts_out = np.zeros((n, direction_sectors), dtype=np.int16)
    if n == 0 or model_indices.size < min_wedge_neighbours:
        return predicted, rmse_out, slope_out, nearest_out, spread_out, counts_out

    model_xy = np.column_stack((x[model_indices], y[model_indices]))
    tree = cKDTree(model_xy)
    k = min(neighbour_count, model_indices.size)
    qxy = np.column_stack((x[part], y[part]))
    distances, positions = tree_query(
        tree, qxy, k=k, distance_upper_bound=neighbour_radius
    )
    distances = np.asarray(distances, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.int64)
    if distances.ndim == 1:
        distances = distances[:, None]
        positions = positions[:, None]
    valid = (
        np.isfinite(distances)
        & (positions >= 0)
        & (positions < model_indices.size)
    )
    safe = np.clip(positions, 0, max(0, model_indices.size - 1))
    global_idx = model_indices[safe]
    dx = np.zeros(distances.shape, dtype=np.float64)
    dy = np.zeros(distances.shape, dtype=np.float64)
    nz = np.zeros(distances.shape, dtype=np.float64)
    qx = np.broadcast_to(x[part, None], distances.shape)
    qy = np.broadcast_to(y[part, None], distances.shape)
    dx[valid] = x[global_idx[valid]] - qx[valid]
    dy[valid] = y[global_idx[valid]] - qy[valid]
    nz[valid] = z[global_idx[valid]]
    angles = np.degrees(np.arctan2(dy, dx))

    design = np.empty((n, distances.shape[1], 3), dtype=np.float64)
    design[:, :, 0] = dx
    design[:, :, 1] = dy
    design[:, :, 2] = 1.0

    for sector in range(direction_sectors):
        centre = -180.0 + (sector + 0.5) * (360.0 / direction_sectors)
        diff = np.abs((angles - centre + 180.0) % 360.0 - 180.0)
        wedge = valid & (diff <= wedge_half_width_deg)
        count = np.sum(wedge, axis=1)
        counts_out[:, sector] = count.astype(np.int16, copy=False)
        rows = count >= min_wedge_neighbours
        if not np.any(rows):
            continue

        weights = np.where(
            wedge, 1.0 / np.square(np.maximum(distances, 0.30)), 0.0
        )
        ata = np.einsum('nki,nk,nkj->nij', design, weights, design)
        atz = np.einsum('nki,nk,nk->ni', design, weights, nz)
        ata[:, 0, 0] += 1.0e-7
        ata[:, 1, 1] += 1.0e-7
        ata[:, 2, 2] += 1.0e-7
        try:
            coeff = np.linalg.solve(ata, atz[..., None])[..., 0]
        except np.linalg.LinAlgError:
            continue
        fitted = np.einsum('nki,ni->nk', design, coeff)
        sq = np.where(wedge, np.square(nz - fitted), 0.0)
        rmse = np.sqrt(np.sum(sq, axis=1) / np.maximum(count, 1))

        wsum = np.maximum(np.sum(weights, axis=1), 1.0e-12)
        mean_dx = np.sum(weights * dx, axis=1) / wsum
        mean_dy = np.sum(weights * dy, axis=1) / wsum
        cdx = dx - mean_dx[:, None]
        cdy = dy - mean_dy[:, None]
        cxx = np.sum(weights * cdx * cdx, axis=1) / wsum
        cyy = np.sum(weights * cdy * cdy, axis=1) / wsum
        cxy = np.sum(weights * cdx * cdy, axis=1) / wsum
        trace = cxx + cyy
        disc = np.sqrt(np.maximum((cxx - cyy) ** 2 + 4.0 * cxy * cxy, 0.0))
        spread = np.sqrt(np.maximum(0.5 * (trace - disc), 0.0))

        nearest = np.min(np.where(wedge, distances, np.inf), axis=1)
        slope = np.degrees(np.arctan(np.linalg.norm(coeff[:, :2], axis=1)))
        good_fit = (
            rows
            & np.isfinite(coeff[:, 2])
            & np.isfinite(rmse)
            & (rmse <= max_wedge_rmse)
            & (spread >= min_xy_spread)
        )
        predicted[good_fit, sector] = coeff[good_fit, 2]
        rmse_out[good_fit, sector] = rmse[good_fit]
        slope_out[good_fit, sector] = slope[good_fit]
        nearest_out[good_fit, sector] = nearest[good_fit]
        spread_out[good_fit, sector] = spread[good_fit]

    return predicted, rmse_out, slope_out, nearest_out, spread_out, counts_out


def recover_terminal_bank_edges(
    work_indices: np.ndarray,
    model_indices: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    bounds: tuple[float, float, float, float],
) -> np.ndarray:
    """Recover genuine steep bank/shoreline returns below a bridging TIN.

    This is deliberately terminal: accepted points are returned in one batch
    and are never used to discover a second batch. A steep water edge can
    therefore be repaired without allowing the model to walk across a flat
    water surface. Repeated nearby work minima are required as a guard against
    isolated low errors.
    """
    if not TERMINAL_BANK_EDGE_RECOVERY or work_indices.size == 0:
        return np.empty(0, dtype=np.int64)
    unresolved = work_indices[~np.isin(work_indices, model_indices)]
    if unresolved.size == 0 or model_indices.size < TERMINAL_BANK_MIN_WEDGE_NEIGHBOURS:
        return np.empty(0, dtype=np.int64)

    model_tree = cKDTree(np.column_stack((x[model_indices], y[model_indices])))
    model_distance, model_pos = tree_query(
        model_tree, np.column_stack((x[unresolved], y[unresolved])), k=1
    )
    model_distance = np.asarray(model_distance, dtype=np.float64)
    model_pos = np.asarray(model_pos, dtype=np.int64)
    nearest_global = model_indices[np.clip(model_pos, 0, model_indices.size - 1)]
    vertical = np.maximum(
        np.abs(z[unresolved] - z[nearest_global]) - TRUSTED_CONNECTION_VERTICAL_SLACK,
        0.0,
    )
    connection_slope = np.degrees(
        np.arctan2(vertical, np.maximum(model_distance, 1.0e-6))
    )

    tin_residual, _, _ = interpolate_tin_membership_context(
        unresolved, model_indices, x, y, z, bounds
    )
    tin_residual = tin_residual.astype(np.float64, copy=False)

    # Local trusted relief: water/bank repair is only relevant at an actual
    # vertical transition, not on an ordinary flat low surface.
    k = min(TERMINAL_BANK_NEIGHBOURS, model_indices.size)
    distances, positions = tree_query(
        model_tree,
        np.column_stack((x[unresolved], y[unresolved])),
        k=k,
        distance_upper_bound=TERMINAL_BANK_NEIGHBOUR_RADIUS,
    )
    distances = np.asarray(distances, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.int64)
    if distances.ndim == 1:
        distances = distances[:, None]
        positions = positions[:, None]
    valid = np.isfinite(distances) & (positions >= 0) & (positions < model_indices.size)
    safe = np.clip(positions, 0, max(0, model_indices.size - 1))
    nz = np.full(distances.shape, np.nan, dtype=np.float64)
    nz[valid] = z[model_indices[safe[valid]]]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        local_z_range = np.nanmax(nz, axis=1) - np.nanmin(nz, axis=1)

    pred, wrmse, wslope, wnearest, _, _ = _terminal_one_sided_plane_support(
        unresolved,
        model_indices,
        x, y, z,
        TERMINAL_BANK_NEIGHBOUR_RADIUS,
        TERMINAL_BANK_NEIGHBOURS,
        TERMINAL_BANK_MIN_WEDGE_NEIGHBOURS,
        TERMINAL_BANK_DIRECTION_SECTORS,
        TERMINAL_BANK_WEDGE_HALF_WIDTH_DEG,
        TERMINAL_BANK_MAX_WEDGE_RMSE,
        TERMINAL_BANK_MIN_XY_SPREAD,
    )
    residual = z[unresolved, None] - pred
    wedge_connection = np.degrees(
        np.arctan2(
            np.maximum(np.abs(z[unresolved, None] - pred), 0.0),
            np.maximum(wnearest, 1.0e-6),
        )
    )
    wedge_good = (
        np.isfinite(pred)
        & np.isfinite(wrmse)
        & np.isfinite(wnearest)
        & (wnearest <= TERMINAL_BANK_MAX_MODEL_DISTANCE)
        & (residual >= -TERMINAL_BANK_PLANE_LOWER)
        & (residual <= TERMINAL_BANK_PLANE_UPPER)
        & (wedge_connection <= TERMINAL_BANK_MAX_CONNECTION_SLOPE_DEG)
    )
    any_wedge = np.any(wedge_good, axis=1)

    # Repeated local minima support. This does not permit propagation because
    # every candidate is decided against the same frozen pre-terminal model.
    work_tree = cKDTree(np.column_stack((x[work_indices], y[work_indices])))
    pk = min(8, work_indices.size)
    pd, pp = tree_query(
        work_tree,
        np.column_stack((x[unresolved], y[unresolved])),
        k=pk,
        distance_upper_bound=TERMINAL_BANK_PEER_RADIUS,
    )
    pd = np.asarray(pd, dtype=np.float64)
    pp = np.asarray(pp, dtype=np.int64)
    if pd.ndim == 1:
        pd = pd[:, None]
        pp = pp[:, None]
    pvalid = np.isfinite(pd) & (pp >= 0) & (pp < work_indices.size) & (pd > 1.0e-6)
    psafe = np.clip(pp, 0, max(0, work_indices.size - 1))
    peer_global = work_indices[psafe]
    pdz = np.zeros(pd.shape, dtype=np.float64)
    pdz[pvalid] = np.abs(z[peer_global[pvalid]] - np.broadcast_to(z[unresolved, None], pd.shape)[pvalid])
    peer_slope = np.full(pd.shape, np.inf, dtype=np.float64)
    peer_slope[pvalid] = np.degrees(
        np.arctan2(pdz[pvalid], np.maximum(pd[pvalid], 1.0e-6))
    )
    peer_support = np.sum(
        pvalid & (peer_slope <= TERMINAL_BANK_PEER_MAX_SLOPE_DEG), axis=1
    )

    good = (
        np.isfinite(tin_residual)
        & np.isfinite(model_distance)
        & (model_distance <= TERMINAL_BANK_MAX_MODEL_DISTANCE)
        & (tin_residual <= -TERMINAL_BANK_MIN_NEGATIVE_TIN_RESIDUAL)
        & (tin_residual >= -TERMINAL_BANK_MAX_NEGATIVE_TIN_RESIDUAL)
        & (connection_slope >= TERMINAL_BANK_MIN_CONNECTION_SLOPE_DEG)
        & (connection_slope <= TERMINAL_BANK_MAX_CONNECTION_SLOPE_DEG)
        & (local_z_range >= TERMINAL_BANK_MIN_LOCAL_Z_RANGE)
        & (peer_support >= TERMINAL_BANK_MIN_PEER_SUPPORT)
        & any_wedge
    )
    return unresolved[good]


def recover_terminal_ridge_crests(
    work_indices: np.ndarray,
    model_indices: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    bounds: tuple[float, float, float, float],
    original_classification: np.ndarray | None = None,
) -> np.ndarray:
    """Recover isolated convex rocky crests using two-sided trusted support.

    A candidate may sit well above the straight trusted-TIN chord, but TWO
    substantially opposing one-sided planes made only from accepted terrain
    must extrapolate to it. The stage is terminal, so an accepted summit point
    cannot subsequently seed a vegetation climb.
    """
    if not TERMINAL_RIDGE_RECOVERY or work_indices.size == 0:
        return np.empty(0, dtype=np.int64)
    unresolved = work_indices[~np.isin(work_indices, model_indices)]
    if unresolved.size == 0 or model_indices.size < 2 * TERMINAL_RIDGE_MIN_WEDGE_NEIGHBOURS:
        return np.empty(0, dtype=np.int64)

    model_tree = cKDTree(np.column_stack((x[model_indices], y[model_indices])))
    model_distance, model_pos = tree_query(
        model_tree, np.column_stack((x[unresolved], y[unresolved])), k=1
    )
    model_distance = np.asarray(model_distance, dtype=np.float64)
    model_pos = np.asarray(model_pos, dtype=np.int64)
    nearest_global = model_indices[np.clip(model_pos, 0, model_indices.size - 1)]
    vertical = np.maximum(
        np.abs(z[unresolved] - z[nearest_global]) - TRUSTED_CONNECTION_VERTICAL_SLACK,
        0.0,
    )
    connection_slope = np.degrees(
        np.arctan2(vertical, np.maximum(model_distance, 1.0e-6))
    )

    tin_residual, tin_slope, _ = interpolate_tin_membership_context(
        unresolved, model_indices, x, y, z, bounds
    )
    tin_residual = tin_residual.astype(np.float64, copy=False)
    tin_slope = tin_slope.astype(np.float64, copy=False)

    # Trusted neighbourhood relief is mandatory: this path is for a rocky
    # crest/ridge, never a flat patch of low vegetation.
    k = min(TERMINAL_RIDGE_NEIGHBOURS, model_indices.size)
    distances, positions = tree_query(
        model_tree,
        np.column_stack((x[unresolved], y[unresolved])),
        k=k,
        distance_upper_bound=TERMINAL_RIDGE_NEIGHBOUR_RADIUS,
    )
    distances = np.asarray(distances, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.int64)
    if distances.ndim == 1:
        distances = distances[:, None]
        positions = positions[:, None]
    valid = np.isfinite(distances) & (positions >= 0) & (positions < model_indices.size)
    safe = np.clip(positions, 0, max(0, model_indices.size - 1))
    nz = np.full(distances.shape, np.nan, dtype=np.float64)
    nz[valid] = z[model_indices[safe[valid]]]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        local_z_range = np.nanmax(nz, axis=1) - np.nanmin(nz, axis=1)

    pred, wrmse, wslope, wnearest, _, _ = _terminal_one_sided_plane_support(
        unresolved,
        model_indices,
        x, y, z,
        TERMINAL_RIDGE_NEIGHBOUR_RADIUS,
        TERMINAL_RIDGE_NEIGHBOURS,
        TERMINAL_RIDGE_MIN_WEDGE_NEIGHBOURS,
        TERMINAL_RIDGE_DIRECTION_SECTORS,
        TERMINAL_RIDGE_WEDGE_HALF_WIDTH_DEG,
        TERMINAL_RIDGE_MAX_WEDGE_RMSE,
        TERMINAL_RIDGE_MIN_XY_SPREAD,
    )
    residual = z[unresolved, None] - pred
    slope_for_tol = np.maximum(
        np.broadcast_to(tin_slope[:, None], wslope.shape),
        np.nan_to_num(wslope, nan=0.0),
    )
    sw = np.clip(
        slope_for_tol / TERMINAL_RIDGE_FULL_SLOPE_DEG, 0.0, 1.0
    )
    tolerance = (
        TERMINAL_RIDGE_FLAT_FIT_TOLERANCE
        + sw * (
            TERMINAL_RIDGE_STEEP_FIT_TOLERANCE
            - TERMINAL_RIDGE_FLAT_FIT_TOLERANCE
        )
    )

    if TERMINAL_RIDGE_USE_INPUT_PRIOR and original_classification is not None:
        prior = original_classification[unresolved]
        cap = np.full(unresolved.size, np.inf, dtype=np.float64)
        cap[prior == 3] = TERMINAL_RIDGE_PRIOR_CLASS3_TOLERANCE
        cap[prior == 4] = TERMINAL_RIDGE_PRIOR_CLASS4_TOLERANCE
        cap[prior == 13] = TERMINAL_RIDGE_PRIOR_CLASS13_TOLERANCE
        cap[prior == 5] = TERMINAL_RIDGE_PRIOR_CLASS5_TOLERANCE
        tolerance = np.minimum(tolerance, cap[:, None])

    sector_good = (
        np.isfinite(pred)
        & np.isfinite(wrmse)
        & np.isfinite(wnearest)
        & (wnearest <= TERMINAL_RIDGE_MAX_MODEL_DISTANCE)
        & (np.abs(residual) <= tolerance)
    )

    # Require two good wedges whose centres are 135-225 degrees apart. This
    # approximates two-sided ridge support while tolerating irregular rock.
    two_sided = np.zeros(unresolved.size, dtype=bool)
    sectors = TERMINAL_RIDGE_DIRECTION_SECTORS
    for a in range(sectors):
        for b in range(a + 1, sectors):
            delta = min((b - a) % sectors, (a - b) % sectors)
            if delta < TERMINAL_RIDGE_MIN_OPPOSITION_SECTORS:
                continue
            if delta > sectors - TERMINAL_RIDGE_MIN_OPPOSITION_SECTORS:
                continue
            two_sided |= sector_good[:, a] & sector_good[:, b]

    good = (
        np.isfinite(tin_residual)
        & np.isfinite(model_distance)
        & (model_distance <= TERMINAL_RIDGE_MAX_MODEL_DISTANCE)
        & (tin_residual >= TERMINAL_RIDGE_MIN_POSITIVE_TIN_RESIDUAL)
        & (tin_residual <= TERMINAL_RIDGE_MAX_POSITIVE_TIN_RESIDUAL)
        & (connection_slope <= TERMINAL_RIDGE_MAX_CONNECTION_SLOPE_DEG)
        & (local_z_range >= TERMINAL_RIDGE_MIN_LOCAL_Z_RANGE)
        & two_sided
    )
    return unresolved[good]


def recover_terminal_escarpment_representatives(
    ground_candidates: np.ndarray,
    work_indices: np.ndarray,
    model_indices: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    bounds: tuple[float, float, float, float],
    original_classification: np.ndarray | None = None,
) -> np.ndarray:
    """Return sparse 2.5-D-safe representatives on one-sided escarpments.

    Unlike the discarded v7 terminal membership, these points are candidates
    for the STRUCTURAL terrain model itself, so they must remain safe for any
    downstream XY Delaunay TIN built from class 2.  Only one-metre work minima
    are considered; locally multi-valued/near-vertical source columns are
    rejected.  All support planes come only from the frozen accepted model.
    """
    if (
        not TERMINAL_ESCARPMENT_RECOVERY
        or work_indices.size == 0
        or model_indices.size < TERMINAL_ESCARPMENT_MIN_WEDGE_NEIGHBOURS
    ):
        return np.empty(0, dtype=np.int64)

    unresolved = work_indices[~np.isin(work_indices, model_indices)]
    if unresolved.size == 0:
        return np.empty(0, dtype=np.int64)

    model_tree = cKDTree(np.column_stack((x[model_indices], y[model_indices])))
    model_distance, model_pos = tree_query(
        model_tree, np.column_stack((x[unresolved], y[unresolved])), k=1
    )
    model_distance = np.asarray(model_distance, dtype=np.float64)
    model_pos = np.asarray(model_pos, dtype=np.int64)
    nearest_global = model_indices[np.clip(model_pos, 0, model_indices.size - 1)]
    vertical = np.abs(z[unresolved] - z[nearest_global])
    connection_slope = np.degrees(
        np.arctan2(vertical, np.maximum(model_distance, 1.0e-6))
    )

    tin_residual, tin_slope, _ = interpolate_tin_membership_context(
        unresolved, model_indices, x, y, z, bounds
    )
    tin_residual = tin_residual.astype(np.float64, copy=False)
    tin_slope = tin_slope.astype(np.float64, copy=False)

    # Local trusted-model relief confirms this is a terrain edge/escarpment,
    # not an isolated near-flat low-vegetation patch.
    k = min(TERMINAL_ESCARPMENT_NEIGHBOURS, model_indices.size)
    distances, positions = tree_query(
        model_tree,
        np.column_stack((x[unresolved], y[unresolved])),
        k=k,
        distance_upper_bound=TERMINAL_ESCARPMENT_NEIGHBOUR_RADIUS,
    )
    distances = np.asarray(distances, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.int64)
    if distances.ndim == 1:
        distances = distances[:, None]
        positions = positions[:, None]
    valid = np.isfinite(distances) & (positions >= 0) & (positions < model_indices.size)
    safe = np.clip(positions, 0, max(0, model_indices.size - 1))
    nz = np.full(distances.shape, np.nan, dtype=np.float64)
    nz[valid] = z[model_indices[safe[valid]]]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        local_z_range = np.nanmax(nz, axis=1) - np.nanmin(nz, axis=1)

    pred, wrmse, wslope, wnearest, _, _ = _terminal_one_sided_plane_support(
        unresolved,
        model_indices,
        x, y, z,
        TERMINAL_ESCARPMENT_NEIGHBOUR_RADIUS,
        TERMINAL_ESCARPMENT_NEIGHBOURS,
        TERMINAL_ESCARPMENT_MIN_WEDGE_NEIGHBOURS,
        TERMINAL_ESCARPMENT_DIRECTION_SECTORS,
        TERMINAL_ESCARPMENT_WEDGE_HALF_WIDTH_DEG,
        TERMINAL_ESCARPMENT_MAX_WEDGE_RMSE,
        TERMINAL_ESCARPMENT_MIN_XY_SPREAD,
    )
    residual = z[unresolved, None] - pred
    slope_for_tol = np.maximum(
        np.broadcast_to(tin_slope[:, None], wslope.shape),
        np.nan_to_num(wslope, nan=0.0),
    )
    sw = np.clip(
        slope_for_tol / TERMINAL_ESCARPMENT_FULL_SLOPE_DEG, 0.0, 1.0
    )
    tolerance = (
        TERMINAL_ESCARPMENT_FLAT_FIT_TOLERANCE
        + sw * (
            TERMINAL_ESCARPMENT_STEEP_FIT_TOLERANCE
            - TERMINAL_ESCARPMENT_FLAT_FIT_TOLERANCE
        )
    )
    if TERMINAL_ESCARPMENT_USE_INPUT_PRIOR and original_classification is not None:
        prior = original_classification[unresolved]
        cap = np.full(unresolved.size, np.inf, dtype=np.float64)
        cap[prior == 3] = TERMINAL_ESCARPMENT_PRIOR_CLASS3_TOLERANCE
        cap[prior == 4] = TERMINAL_ESCARPMENT_PRIOR_CLASS4_TOLERANCE
        cap[prior == 13] = TERMINAL_ESCARPMENT_PRIOR_CLASS13_TOLERANCE
        cap[prior == 5] = TERMINAL_ESCARPMENT_PRIOR_CLASS5_TOLERANCE
        # Apply the prior only to candidates above the trusted TIN.  A genuine
        # depression/bank below the TIN should not be penalised by an old label.
        tolerance = np.where(
            tin_residual[:, None] > 0.0,
            np.minimum(tolerance, cap[:, None]),
            tolerance,
        )

    wedge_good = (
        np.isfinite(pred)
        & np.isfinite(wrmse)
        & np.isfinite(wnearest)
        & np.isfinite(wslope)
        & (wnearest <= TERMINAL_ESCARPMENT_MAX_MODEL_DISTANCE)
        & (wslope >= TERMINAL_ESCARPMENT_MIN_PLANE_SLOPE_DEG)
        & (wslope <= TERMINAL_ESCARPMENT_MAX_PLANE_SLOPE_DEG)
        & (np.abs(residual) <= tolerance)
    )
    any_wedge = np.any(wedge_good, axis=1)

    # Candidate peer support is only a consistency check; peers never become
    # support planes and this stage remains one-shot/non-propagating.
    work_tree = cKDTree(np.column_stack((x[work_indices], y[work_indices])))
    pk = min(8, work_indices.size)
    pd, pp = tree_query(
        work_tree,
        np.column_stack((x[unresolved], y[unresolved])),
        k=pk,
        distance_upper_bound=TERMINAL_ESCARPMENT_PEER_RADIUS,
    )
    pd = np.asarray(pd, dtype=np.float64)
    pp = np.asarray(pp, dtype=np.int64)
    if pd.ndim == 1:
        pd = pd[:, None]
        pp = pp[:, None]
    pvalid = np.isfinite(pd) & (pp >= 0) & (pp < work_indices.size) & (pd > 1.0e-6)
    psafe = np.clip(pp, 0, max(0, work_indices.size - 1))
    peer_global = work_indices[psafe]
    pdz = np.zeros(pd.shape, dtype=np.float64)
    pdz[pvalid] = np.abs(
        z[peer_global[pvalid]]
        - np.broadcast_to(z[unresolved, None], pd.shape)[pvalid]
    )
    peer_slope = np.full(pd.shape, np.inf, dtype=np.float64)
    peer_slope[pvalid] = np.degrees(
        np.arctan2(pdz[pvalid], np.maximum(pd[pvalid], 1.0e-6))
    )
    peer_support = np.sum(
        pvalid & (peer_slope <= TERMINAL_ESCARPMENT_PEER_MAX_SLOPE_DEG), axis=1
    )

    preliminary = (
        np.isfinite(tin_residual)
        & np.isfinite(model_distance)
        & (model_distance <= TERMINAL_ESCARPMENT_MAX_MODEL_DISTANCE)
        & (np.abs(tin_residual) <= TERMINAL_ESCARPMENT_MAX_ABS_TIN_RESIDUAL)
        & (connection_slope >= TERMINAL_ESCARPMENT_MIN_CONNECTION_SLOPE_DEG)
        & (connection_slope <= TERMINAL_ESCARPMENT_MAX_CONNECTION_SLOPE_DEG)
        & (local_z_range >= TERMINAL_ESCARPMENT_MIN_LOCAL_Z_RANGE)
        & (peer_support >= TERMINAL_ESCARPMENT_MIN_PEER_SUPPORT)
        & any_wedge
    )
    candidates = unresolved[preliminary]
    if candidates.size == 0:
        return candidates

    # 2.5-D safety gate: reject candidate XY columns containing a large local
    # vertical spread.  Such returns may be valid rock-face observations, but
    # they cannot safely coexist in a conventional terrain TIN.
    source_tree = cKDTree(np.column_stack((x[ground_candidates], y[ground_candidates])))
    ck = min(TERMINAL_ESCARPMENT_COLUMN_NEIGHBOURS, ground_candidates.size)
    cd, cp = tree_query(
        source_tree,
        np.column_stack((x[candidates], y[candidates])),
        k=ck,
        distance_upper_bound=TERMINAL_ESCARPMENT_COLUMN_RADIUS,
    )
    cd = np.asarray(cd, dtype=np.float64)
    cp = np.asarray(cp, dtype=np.int64)
    if cd.ndim == 1:
        cd = cd[:, None]
        cp = cp[:, None]
    cvalid = np.isfinite(cd) & (cp >= 0) & (cp < ground_candidates.size)
    csafe = np.clip(cp, 0, max(0, ground_candidates.size - 1))
    cg = ground_candidates[csafe]
    cz = np.full(cd.shape, np.nan, dtype=np.float64)
    cz[cvalid] = z[cg[cvalid]]
    ccount = np.sum(cvalid, axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        column_z_range = np.nanmax(cz, axis=1) - np.nanmin(cz, axis=1)
    column_safe = (
        (ccount < TERMINAL_ESCARPMENT_COLUMN_MIN_NEIGHBOURS)
        | (column_z_range <= TERMINAL_ESCARPMENT_MAX_COLUMN_Z_RANGE)
    )
    return candidates[column_safe]


def recover_terminal_terrain_edges(
    work_indices: np.ndarray,
    model_indices: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    bounds: tuple[float, float, float, float],
    origin_x: float,
    origin_y: float,
    original_classification: np.ndarray | None = None,
) -> tuple[np.ndarray, int, int]:
    """One-shot bank-edge and rocky-crest repair against a frozen model."""
    frozen = model_indices.copy()
    bank = recover_terminal_bank_edges(
        work_indices, frozen, x, y, z, bounds
    )
    ridge = recover_terminal_ridge_crests(
        work_indices, frozen, x, y, z, bounds, original_classification
    )
    if bank.size == 0 and ridge.size == 0:
        return model_indices, 0, 0
    added = np.unique(np.concatenate((bank, ridge)))
    model_indices = np.unique(np.concatenate((model_indices, added)))
    model_indices = deduplicate_model_points(
        model_indices, x, y, z, origin_x, origin_y
    )
    return model_indices, int(bank.size), int(ridge.size)


def recover_local_lower_surface(
    work_indices: np.ndarray,
    model_indices: np.ndarray,
    backbone_indices: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    bounds: tuple[float, float, float, float],
    origin_x: float,
    origin_y: float,
    plane_context: PlaneContext | None = None,
    original_classification: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """
    Recover missed continuous terrain without allowing self-supporting islands.

    This replaces the old unbounded local-minimum rescue.  Recovery may grow
    progressively across a hill/slope, but every new point must fit terrain
    already accepted BEFORE that recovery pass. There is no arbitrary total
    distance back to the original PTD backbone. A roof or vegetation layer
    with no terrain-compatible connection therefore remains a gap in class 2.
    """
    del plane_context  # retained in the signature for minimal caller change
    if not TRUSTED_GROUND_RECOVERY:
        return model_indices, 0

    total_added = 0
    for _ in range(TRUSTED_CONNECTED_RESCUE_PASSES):
        in_model = np.isin(work_indices, model_indices)
        candidates = work_indices[~in_model]
        if candidates.size == 0:
            break

        supported = trusted_surface_support(
            candidates,
            model_indices,
            backbone_indices,
            x,
            y,
            z,
            bounds,
            flat_tolerance=TRUSTED_SUPPORT_FLAT_PLANE_TOLERANCE * noise_scale_multiplier(),
            steep_tolerance=TRUSTED_SUPPORT_STEEP_PLANE_TOLERANCE * noise_scale_multiplier(),
            original_classification=original_classification,
        )
        accepted = candidates[supported]
        if accepted.size == 0:
            break

        previous_model = model_indices.copy()
        model_indices = np.unique(
            np.concatenate((model_indices, accepted))
        )
        model_indices = deduplicate_model_points(
            model_indices,
            x,
            y,
            z,
            origin_x,
            origin_y,
        )
        added = int(
            np.count_nonzero(~np.isin(model_indices, previous_model))
        )
        changed = not np.array_equal(model_indices, previous_model)
        total_added += added
        if not changed:
            break

    return model_indices, total_added


def recover_supported_upper_surface(
    work_indices: np.ndarray,
    model_indices: np.ndarray,
    backbone_indices: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    bounds: tuple[float, float, float, float],
    origin_x: float,
    origin_y: float,
    support_tree: cKDTree | None = None,
    original_classification: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """Grow across modest terrain rises using only trusted-model evidence."""
    del support_tree  # old work-minimum support is intentionally not reused
    if not TRUSTED_GROUND_RECOVERY:
        return model_indices, 0

    total_added = 0
    for _ in range(UPPER_SURFACE_RESCUE_PASSES):
        work_hag = interpolate_hag_at_indices(
            model_indices, work_indices, x, y, z, bounds
        )
        in_model = np.isin(work_indices, model_indices)
        candidates = work_indices[
            ~in_model
            & (work_hag > UPPER_SURFACE_MIN_HEIGHT)
            & (work_hag <= UPPER_SURFACE_MAX_HEIGHT)
        ]
        if candidates.size == 0:
            break

        supported = trusted_surface_support(
            candidates,
            model_indices,
            backbone_indices,
            x,
            y,
            z,
            bounds,
            flat_tolerance=TRUSTED_SUPPORT_FLAT_PLANE_TOLERANCE * noise_scale_multiplier(),
            steep_tolerance=TRUSTED_SUPPORT_STEEP_PLANE_TOLERANCE * noise_scale_multiplier(),
            original_classification=original_classification,
        )
        accepted = candidates[supported]
        if accepted.size == 0:
            break

        previous_model = model_indices.copy()
        model_indices = np.unique(
            np.concatenate((model_indices, accepted))
        )
        model_indices = deduplicate_model_points(
            model_indices, x, y, z, origin_x, origin_y
        )
        added = int(
            np.count_nonzero(~np.isin(model_indices, previous_model))
        )
        changed = not np.array_equal(model_indices, previous_model)
        total_added += added
        if not changed:
            break

    return model_indices, total_added


def interpolate_hag_at_indices(
    model_indices: np.ndarray,
    query_indices: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    bounds: tuple[float, float, float, float],
    surface: TinSurface | None = None,
) -> np.ndarray:
    """Interpolate HAG only at selected indices.

    Recovery stages operate exclusively on the structural work minima. Avoiding
    full-cloud interpolation during each recovery iteration is exact and is a
    major runtime reduction on dense airborne data.
    """
    if query_indices.size == 0:
        return np.empty(0, dtype=np.float32)

    if surface is None:
        surface = build_tin(model_indices, x, y, z, bounds)
    tin, model_xy, model_z, _ = surface
    hag = np.full(query_indices.size, np.nan, dtype=np.float32)

    for start in range(0, query_indices.size, TEST_CHUNK_SIZE):
        stop = min(start + TEST_CHUNK_SIZE, query_indices.size)
        part = query_indices[start:stop]
        query_xy = np.column_stack((x[part], y[part]))
        simplex = tin.find_simplex(query_xy)
        valid = simplex >= 0
        if not np.any(valid):
            continue

        valid_simplex = simplex[valid]
        vertices = tin.simplices[valid_simplex]
        transform = tin.transform[valid_simplex]
        delta = query_xy[valid] - transform[:, 2]
        bary12 = np.einsum("ijk,ik->ij", transform[:, :2], delta)
        bary = np.column_stack((bary12, 1.0 - bary12.sum(axis=1)))
        ground_z = np.einsum("ij,ij->i", bary, model_z[vertices])
        local = np.flatnonzero(valid)
        hag[start + local] = (z[part[valid]] - ground_z).astype(
            np.float32
        )

    missing = ~np.isfinite(hag)
    if np.any(missing):
        actual_xy = np.column_stack((x[model_indices], y[model_indices]))
        tree = cKDTree(actual_xy)
        missing_indices = query_indices[missing]
        _, nearest = tree_query(
            tree,
            np.column_stack((x[missing_indices], y[missing_indices])),
            k=1,
        )
        ground_z = z[model_indices[np.asarray(nearest, dtype=np.int64)]]
        hag[missing] = (z[missing_indices] - ground_z).astype(np.float32)

    return hag


def interpolate_hag(
    model_indices: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    bounds: tuple[float, float, float, float],
    surface: TinSurface | None = None,
    capture_simplex_indices: np.ndarray | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Interpolate HAG for every local point from the final clean TIN."""
    if surface is None:
        surface = build_tin(model_indices, x, y, z, bounds)
    tin, model_xy, model_z, _ = surface
    hag = np.full(x.size, np.nan, dtype=np.float32)
    captured_simplex = (
        np.full(capture_simplex_indices.size, -1, dtype=np.int32)
        if capture_simplex_indices is not None
        else None
    )

    for start in range(0, x.size, TEST_CHUNK_SIZE):
        stop = min(start + TEST_CHUNK_SIZE, x.size)
        query_xy = np.column_stack((x[start:stop], y[start:stop]))
        simplex = tin.find_simplex(query_xy)
        if (
            capture_simplex_indices is not None
            and captured_simplex is not None
        ):
            left = np.searchsorted(
                capture_simplex_indices, start, side="left"
            )
            right = np.searchsorted(
                capture_simplex_indices, stop, side="left"
            )
            if right > left:
                local_capture = (
                    capture_simplex_indices[left:right] - start
                )
                captured_simplex[left:right] = simplex[
                    local_capture
                ].astype(np.int32, copy=False)
        valid = simplex >= 0
        if not np.any(valid):
            continue

        valid_simplex = simplex[valid]
        vertices = tin.simplices[valid_simplex]

        transform = tin.transform[valid_simplex]
        delta = query_xy[valid] - transform[:, 2]
        bary12 = np.einsum("ijk,ik->ij", transform[:, :2], delta)
        bary = np.column_stack((bary12, 1.0 - bary12.sum(axis=1)))
        ground_z = np.einsum("ij,ij->i", bary, model_z[vertices])

        local = np.flatnonzero(valid)
        hag[start + local] = (z[start + local] - ground_z).astype(
            np.float32
        )

    # Corner support should cover the block, but retain a safe nearest-model
    # fallback for any numerical edge cases.
    missing = ~np.isfinite(hag)
    if np.any(missing):
        actual_xy = np.column_stack((x[model_indices], y[model_indices]))
        tree = cKDTree(actual_xy)
        _, nearest = tree_query(
            tree,
            np.column_stack((x[missing], y[missing])), k=1
        )
        ground_z = z[model_indices[np.asarray(nearest, dtype=np.int64)]]
        hag[missing] = (z[missing] - ground_z).astype(np.float32)

    if captured_simplex is not None:
        return hag, captured_simplex
    return hag


def interpolate_tin_slope(
    model_indices: np.ndarray,
    query_indices: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    bounds: tuple[float, float, float, float],
) -> np.ndarray:
    """Return the local TIN slope in degrees at selected source points."""
    slope = np.zeros(query_indices.size, dtype=np.float32)
    if query_indices.size == 0:
        return slope

    tin, _, model_z, _ = build_tin(model_indices, x, y, z, bounds)
    for start in range(0, query_indices.size, TEST_CHUNK_SIZE):
        stop = min(start + TEST_CHUNK_SIZE, query_indices.size)
        part = query_indices[start:stop]
        query_xy = np.column_stack((x[part], y[part]))
        simplex = tin.find_simplex(query_xy)
        valid = simplex >= 0
        if not np.any(valid):
            continue

        simplex_valid = simplex[valid]
        vertices = tin.simplices[simplex_valid]
        transform = tin.transform[simplex_valid, :2, :]
        dz0 = model_z[vertices[:, 0]] - model_z[vertices[:, 2]]
        dz1 = model_z[vertices[:, 1]] - model_z[vertices[:, 2]]
        gradient = (
            dz0[:, None] * transform[:, 0, :]
            + dz1[:, None] * transform[:, 1, :]
        )
        local = np.flatnonzero(valid)
        slope[start + local] = np.degrees(
            np.arctan(np.linalg.norm(gradient, axis=1))
        ).astype(np.float32)

    return slope


def interpolate_tin_membership_context(
    source_indices: np.ndarray,
    model_indices: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    bounds: tuple[float, float, float, float],
    surface: TinSurface | None = None,
    simplex_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return perpendicular residual, slope and longest triangle edge.

    Perpendicular distance matches the progressive-TIN acceptance geometry
    better than vertical HAG on a hillside. Triangle edge length supplies the
    same density cue used by TerraScan's documented angle-reduction option.
    """
    signed_distance = np.full(
        source_indices.size, np.nan, dtype=np.float32
    )
    slope = np.zeros(source_indices.size, dtype=np.float32)
    max_edge = np.full(source_indices.size, np.nan, dtype=np.float32)
    if source_indices.size == 0:
        return signed_distance, slope, max_edge

    if surface is None:
        surface = build_tin(model_indices, x, y, z, bounds)
    tin, model_xy, model_z, _ = surface
    for start in range(0, source_indices.size, TEST_CHUNK_SIZE):
        stop = min(start + TEST_CHUNK_SIZE, source_indices.size)
        part = source_indices[start:stop]
        query_xy = np.column_stack((x[part], y[part]))
        simplex = (
            simplex_ids[start:stop].astype(np.int64, copy=False)
            if simplex_ids is not None
            else tin.find_simplex(query_xy)
        )
        valid = simplex >= 0
        if not np.any(valid):
            continue

        local = np.flatnonzero(valid)
        valid_simplex = simplex[valid]
        vertices = tin.simplices[valid_simplex]
        tri_xy = model_xy[vertices]
        tri_z = model_z[vertices]

        p0 = np.column_stack((tri_xy[:, 0, :], tri_z[:, 0]))
        p1 = np.column_stack((tri_xy[:, 1, :], tri_z[:, 1]))
        p2 = np.column_stack((tri_xy[:, 2, :], tri_z[:, 2]))
        normals = np.cross(p1 - p0, p2 - p0)
        normal_length = np.linalg.norm(normals, axis=1)
        usable = normal_length > 1.0e-12
        normal_length[~usable] = 1.0
        normals /= normal_length[:, None]
        normals[normals[:, 2] < 0] *= -1.0

        query_xyz = np.column_stack((query_xy[valid], z[part[valid]]))
        residual = np.einsum(
            "ij,ij->i", query_xyz - p0, normals
        )
        terrain_slope = np.degrees(
            np.arctan2(
                np.linalg.norm(normals[:, :2], axis=1),
                np.maximum(normals[:, 2], 1.0e-12),
            )
        )
        edge01 = np.linalg.norm(
            tri_xy[:, 0] - tri_xy[:, 1], axis=1
        )
        edge12 = np.linalg.norm(
            tri_xy[:, 1] - tri_xy[:, 2], axis=1
        )
        edge20 = np.linalg.norm(
            tri_xy[:, 2] - tri_xy[:, 0], axis=1
        )
        longest = np.maximum.reduce((edge01, edge12, edge20))

        destination = start + local
        good_destination = destination[usable]
        signed_distance[good_destination] = residual[usable].astype(
            np.float32
        )
        slope[good_destination] = terrain_slope[usable].astype(
            np.float32
        )
        max_edge[good_destination] = longest[usable].astype(np.float32)

    return signed_distance, slope, max_edge



def estimate_acquisition_spacing_density(
    source_indices: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float, int]:
    """Estimate equivalent 2-D acquisition spacing/density in metric arrays.

    The 8th neighbour is used instead of the first because airborne pulses can
    have several returns with nearly identical XY. A deterministic random
    sample keeps runtime bounded; the 2-D sample-density correction converts
    the sampled r_k distance back to a full-cloud equivalent spacing.
    """
    count = int(source_indices.size)
    k_order = max(2, int(DENSITY_NEIGHBOUR_ORDER))
    if count <= k_order + 1:
        return float("nan"), float("nan"), count

    sample_count = min(count, int(DENSITY_SAMPLE_MAX_POINTS))
    if sample_count < count:
        rng = np.random.default_rng(DENSITY_ESTIMATOR_SEED)
        positions = rng.choice(count, size=sample_count, replace=False)
        sample_indices = source_indices[positions]
    else:
        sample_indices = source_indices

    xy = np.column_stack((x[sample_indices], y[sample_indices]))
    finite = np.all(np.isfinite(xy), axis=1)
    xy = xy[finite]
    sample_count = int(xy.shape[0])
    if sample_count <= k_order + 1:
        return float("nan"), float("nan"), sample_count

    tree = cKDTree(xy)
    query_k = min(k_order + 1, sample_count)
    distances, _ = tree_query(tree, xy, k=query_k)
    distances = np.asarray(distances, dtype=np.float64)
    if distances.ndim == 1:
        return float("nan"), float("nan"), sample_count

    rk = distances[:, query_k - 1]
    valid = np.isfinite(rk) & (rk > 1.0e-6)
    if np.count_nonzero(valid) < max(50, sample_count // 100):
        return float("nan"), float("nan"), sample_count

    sample_fraction = min(1.0, sample_count / max(count, 1))
    full_equivalent_rk = rk[valid] * math.sqrt(sample_fraction)
    equivalent_spacing = full_equivalent_rk * math.sqrt(
        math.pi / max(query_k - 1, 1)
    )
    equivalent_spacing = equivalent_spacing[
        (equivalent_spacing >= DENSITY_SPACING_MIN_VALID)
        & (equivalent_spacing <= DENSITY_SPACING_MAX_VALID)
    ]
    if equivalent_spacing.size < 50:
        return float("nan"), float("nan"), sample_count

    spacing = float(np.median(equivalent_spacing))
    density = float(1.0 / max(spacing * spacing, 1.0e-12))
    return spacing, density, sample_count


def adaptive_membership_parameters(
    source_indices: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float, float, int, float, float]:
    """Return density-aware FINAL membership parameters only."""
    base_flat = float(GROUND_MEMBERSHIP_FLAT_TARGET_EDGE)
    base_steep = float(GROUND_MEMBERSHIP_STEEP_TARGET_EDGE)
    base_candidate = float(GROUND_MEMBERSHIP_CANDIDATE_SPACING)
    base_iterations = int(GROUND_MEMBERSHIP_MAX_ITERATIONS)

    if not AUTO_DENSITY_ADAPTIVE_MEMBERSHIP:
        return (
            base_flat,
            base_steep,
            base_candidate,
            base_iterations,
            float("nan"),
            float("nan"),
        )

    spacing, density, _ = estimate_acquisition_spacing_density(
        source_indices, x, y
    )
    if not np.isfinite(spacing):
        return (
            base_flat,
            base_steep,
            base_candidate,
            base_iterations,
            spacing,
            density,
        )

    flat_target = float(np.clip(
        DENSITY_FLAT_EDGE_MULTIPLIER * spacing,
        DENSITY_MIN_FLAT_TARGET_EDGE,
        base_flat,
    ))
    original_ratio = base_steep / max(base_flat, 1.0e-9)
    steep_target = float(np.clip(
        flat_target * original_ratio,
        DENSITY_MIN_STEEP_TARGET_EDGE,
        base_steep,
    ))
    candidate_spacing = float(np.clip(
        DENSITY_CANDIDATE_SPACING_MULTIPLIER * spacing,
        DENSITY_MIN_CANDIDATE_SPACING,
        base_candidate,
    ))

    if flat_target >= base_flat - 1.0e-9:
        iterations = base_iterations
    else:
        refinement = max(base_flat / max(flat_target, 1.0e-9), 1.0)
        extra = int(math.ceil(3.0 * math.log2(refinement)))
        iterations = min(
            int(DENSITY_MAX_ADAPTIVE_ITERATIONS),
            base_iterations + max(1, extra),
        )

    return (
        flat_target,
        steep_target,
        candidate_spacing,
        iterations,
        spacing,
        density,
    )





def smooth_tin_local_center_offsets(
    support_indices: np.ndarray,
    hag: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    slope: np.ndarray,
    origin_x: float,
    origin_y: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build robust per-cell HAG centre offsets from accepted ground only.

    IMPORTANT:
    ``slope`` is compact and aligned one-for-one with ``support_indices``.
    It is NOT a full LAS-length array and therefore must never be indexed by
    global point indices.
    """
    support_indices = np.asarray(support_indices, dtype=np.int64)
    support_slope = np.asarray(slope, dtype=np.float64)

    if support_slope.ndim != 1:
        support_slope = support_slope.reshape(-1)
    if support_slope.size != support_indices.size:
        raise ValueError(
            "smooth_tin_local_center_offsets: slope/support alignment "
            f"mismatch ({support_slope.size:,} slopes for "
            f"{support_indices.size:,} support points)"
        )

    finite = np.isfinite(hag[support_indices])
    support_indices = support_indices[finite]
    support_slope = support_slope[finite]

    if support_indices.size == 0:
        empty_keys = np.empty(
            0, dtype=[("x", "<i8"), ("y", "<i8")]
        )
        return empty_keys, np.empty(0, dtype=np.float64)

    keys = packed_grid_keys(
        x[support_indices],
        y[support_indices],
        SMOOTH_TIN_CENTER_CELL,
        origin_x,
        origin_y,
    )
    order = np.lexsort(
        (
            hag[support_indices],
            keys["y"],
            keys["x"],
        )
    )
    ordered_indices = support_indices[order]
    ordered_slope = support_slope[order]
    ordered_keys = keys[order]

    unique_keys, start, counts = np.unique(
        ordered_keys,
        return_index=True,
        return_counts=True,
    )
    offsets = np.full(unique_keys.size, np.nan, dtype=np.float64)

    for row, (first, count) in enumerate(zip(start, counts)):
        if count < SMOOTH_TIN_CENTER_MIN_POINTS:
            continue

        members = ordered_indices[first : first + count]
        member_slope = ordered_slope[first : first + count]
        values = hag[members].astype(np.float64, copy=False)

        med = float(np.median(values))
        mad = float(np.median(np.abs(values - med)))
        if np.isfinite(mad) and mad > 0.0:
            limit = max(0.035, 3.0 * 1.4826 * mad)
            inliers = np.abs(values - med) <= limit
            if np.count_nonzero(inliers) >= SMOOTH_TIN_CENTER_MIN_POINTS:
                med = float(np.median(values[inliers]))

        finite_slope = member_slope[np.isfinite(member_slope)]
        local_slope = (
            float(np.median(finite_slope))
            if finite_slope.size
            else 0.0
        )
        sw = float(np.clip(
            local_slope / SMOOTH_TIN_CENTER_FULL_SLOPE_DEG,
            0.0,
            1.0,
        ))
        max_up = (
            SMOOTH_TIN_CENTER_FLAT_MAX_UP
            + sw * (
                SMOOTH_TIN_CENTER_STEEP_MAX_UP
                - SMOOTH_TIN_CENTER_FLAT_MAX_UP
            )
        )
        offsets[row] = float(np.clip(
            med,
            -SMOOTH_TIN_CENTER_MAX_DOWN,
            max_up,
        ))

    return unique_keys, offsets


def lookup_smooth_tin_center_offsets(
    query_indices: np.ndarray,
    keys: np.ndarray,
    offsets: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    origin_x: float,
    origin_y: float,
) -> np.ndarray:
    """Return per-point local terrain-centre offset; zero where unsupported."""
    result = np.zeros(query_indices.size, dtype=np.float64)
    if query_indices.size == 0 or keys.size == 0:
        return result

    qkeys = packed_grid_keys(
        x[query_indices],
        y[query_indices],
        SMOOTH_TIN_CENTER_CELL,
        origin_x,
        origin_y,
    )
    positions = np.searchsorted(keys, qkeys)
    safe = np.clip(
        positions, 0, max(0, keys.size - 1)
    )
    matched = (
        (positions < keys.size)
        & (keys[safe] == qkeys)
        & np.isfinite(offsets[safe])
    )
    result[matched] = offsets[safe[matched]]
    return result


def audit_global_smooth_tin_membership(
    base_model_indices: np.ndarray,
    added_indices: np.ndarray,
    structural_model_indices: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    bounds: tuple[float, float, float, float],
) -> np.ndarray:
    """Remove only additional representatives that disrupt a smooth local TIN."""
    if (
        not SMOOTH_TIN_LOCAL_PLANE_AUDIT
        or added_indices.size == 0
    ):
        return np.unique(added_indices)

    membership = np.unique(
        np.concatenate((base_model_indices, added_indices))
    )
    tree = cKDTree(
        np.column_stack((x[membership], y[membership]))
    )
    k = min(
        SMOOTH_TIN_AUDIT_NEIGHBOURS + 1,
        membership.size,
    )

    qxy = np.column_stack((x[added_indices], y[added_indices]))
    distances, positions = tree_query(
        tree,
        qxy,
        k=k,
        distance_upper_bound=SMOOTH_TIN_AUDIT_RADIUS,
    )
    distances = np.asarray(distances, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.int64)
    if distances.ndim == 1:
        distances = distances[:, None]
        positions = positions[:, None]

    safe = np.clip(
        positions, 0, max(0, membership.size - 1)
    )
    neighbour_global = membership[safe]
    own = np.broadcast_to(
        added_indices[:, None], positions.shape
    )
    valid = (
        np.isfinite(distances)
        & (positions >= 0)
        & (positions < membership.size)
        & (neighbour_global != own)
    )

    structural_residual, structural_slope, _ = (
        interpolate_tin_membership_context(
            added_indices,
            structural_model_indices,
            x, y, z,
            bounds,
        )
    )
    del structural_residual

    keep = np.ones(added_indices.size, dtype=bool)

    for row in range(added_indices.size):
        row_valid = valid[row]
        if np.count_nonzero(row_valid) < SMOOTH_TIN_AUDIT_MIN_NEIGHBOURS:
            continue

        neighbours = neighbour_global[row, row_valid]
        dx = x[neighbours] - x[added_indices[row]]
        dy = y[neighbours] - y[added_indices[row]]
        design = np.column_stack(
            (dx, dy, np.ones(neighbours.size))
        )
        try:
            coeff, *_ = np.linalg.lstsq(
                design,
                z[neighbours],
                rcond=None,
            )
        except np.linalg.LinAlgError:
            continue

        fitted = design @ coeff
        residual = z[neighbours] - fitted
        med = float(np.median(residual))
        mad = float(np.median(np.abs(residual - med)))
        trim = max(0.07, 3.0 * 1.4826 * mad)
        inlier = np.abs(residual - med) <= trim
        if np.count_nonzero(inlier) < SMOOTH_TIN_AUDIT_MIN_NEIGHBOURS:
            continue

        try:
            coeff, *_ = np.linalg.lstsq(
                design[inlier],
                z[neighbours][inlier],
                rcond=None,
            )
        except np.linalg.LinAlgError:
            continue

        fitted_inlier = design[inlier] @ coeff
        rmse = float(np.sqrt(np.mean(
            np.square(
                z[neighbours][inlier] - fitted_inlier
            )
        )))
        if rmse > SMOOTH_TIN_AUDIT_MAX_RMSE:
            continue

        slope_value = float(
            structural_slope[row]
            if np.isfinite(structural_slope[row])
            else 0.0
        )
        sw = float(np.clip(
            slope_value / SMOOTH_TIN_AUDIT_FULL_SLOPE_DEG,
            0.0,
            1.0,
        ))
        tolerance = (
            SMOOTH_TIN_AUDIT_FLAT_RESIDUAL
            + sw * (
                SMOOTH_TIN_AUDIT_STEEP_RESIDUAL
                - SMOOTH_TIN_AUDIT_FLAT_RESIDUAL
            )
        )
        point_residual = (
            z[added_indices[row]] - float(coeff[2])
        )
        if abs(point_residual) > tolerance:
            keep[row] = False

    return np.unique(added_indices[keep])


def select_global_smooth_tin_membership(
    ground_candidates: np.ndarray,
    base_model_indices: np.ndarray,
    ordinary_membership: np.ndarray,
    raw_membership: np.ndarray,
    structural_model_indices: np.ndarray,
    hag: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    bounds: tuple[float, float, float, float],
    origin_x: float,
    origin_y: float,
) -> np.ndarray:
    """Fast canopy-safe cleanup of the already-good final membership."""
    stage_start = time.perf_counter()

    raw_membership = np.unique(raw_membership)
    ordinary_membership = np.unique(ordinary_membership)
    base_model_indices = np.unique(base_model_indices)

    if (
        not FAST_SMOOTH_TIN
        or raw_membership.size < 3
        or base_model_indices.size < 3
    ):
        return raw_membership

    _, smooth_slope, _ = interpolate_tin_membership_context(
        raw_membership, base_model_indices,
        x, y, z, bounds,
    )
    structural_residual, structural_slope, _ = (
        interpolate_tin_membership_context(
            raw_membership, structural_model_indices,
            x, y, z, bounds,
        )
    )
    local_slope = np.maximum(
        np.nan_to_num(smooth_slope, nan=0.0),
        np.nan_to_num(structural_slope, nan=0.0),
    )

    # 1) Thin only redundant two-sided additions.
    extra_mask = ~np.isin(raw_membership, ordinary_membership)
    extra_positions = np.flatnonzero(extra_mask)
    keep = np.ones(raw_membership.size, dtype=bool)

    if extra_positions.size and ordinary_membership.size:
        ordinary_tree = cKDTree(
            np.column_stack((x[ordinary_membership], y[ordinary_membership]))
        )
        extra_indices = raw_membership[extra_positions]
        distance, _ = tree_query(
            ordinary_tree,
            np.column_stack((x[extra_indices], y[extra_indices])),
            k=1,
        )
        distance = np.asarray(distance, dtype=np.float64)
        sw = np.clip(
            local_slope[extra_positions] / FAST_EXTRA_FULL_SLOPE_DEG,
            0.0, 1.0,
        )
        min_sep = (
            FAST_EXTRA_FLAT_MIN_SEPARATION
            + sw * (
                FAST_EXTRA_STEEP_MIN_SEPARATION
                - FAST_EXTRA_FLAT_MIN_SEPARATION
            )
        )
        redundant = distance < min_sep
        keep[extra_positions[redundant]] = False
        log(
            "      Fast SmoothTIN density trim: "
            f"{np.count_nonzero(redundant):,} redundant "
            "two-sided slope representatives removed"
        )

    preliminary = raw_membership[keep]
    positions = np.flatnonzero(keep)
    slope = local_slope[positions]
    struct_res = structural_residual[positions]

    # 2) Learn a robust centre from stable already-ground returns.
    stable = (
        np.isfinite(hag[preliminary])
        & (hag[preliminary] > -0.20)
        & (hag[preliminary] < 0.55)
        & np.isfinite(struct_res)
        & (np.abs(struct_res) <= 0.30)
    )
    center_keys, center_values, _ = fast_group_center_offsets(
        preliminary[stable],
        hag,
        slope[stable],
        x, y,
        origin_x, origin_y,
    )
    center = fast_lookup_group_values(
        preliminary,
        center_keys,
        center_values,
        FAST_CENTER_CELL,
        x, y,
        origin_x, origin_y,
        default=np.nan,
    )
    center_supported = np.isfinite(center)
    center = np.nan_to_num(center, nan=0.0)
    centred_residual = (
        hag[preliminary].astype(np.float64, copy=False) - center
    )

    # 3) Mark vertical-complexity / canopy cells.
    canopy_keys = fast_canopy_complexity_keys(
        ground_candidates, hag,
        x, y, origin_x, origin_y,
    )
    canopy = fast_points_in_keys(
        preliminary, canopy_keys,
        x, y, origin_x, origin_y,
    )

    sw = np.clip(
        slope / FAST_RESIDUAL_FULL_SLOPE_DEG, 0.0, 1.0
    )
    open_lower = (
        FAST_OPEN_FLAT_LOWER
        + sw * (FAST_OPEN_STEEP_LOWER - FAST_OPEN_FLAT_LOWER)
    )
    open_upper = (
        FAST_OPEN_FLAT_UPPER
        + sw * (FAST_OPEN_STEEP_UPPER - FAST_OPEN_FLAT_UPPER)
    )
    canopy_lower = (
        FAST_CANOPY_FLAT_LOWER
        + sw * (FAST_CANOPY_STEEP_LOWER - FAST_CANOPY_FLAT_LOWER)
    )
    canopy_upper = (
        FAST_CANOPY_FLAT_UPPER
        + sw * (FAST_CANOPY_STEEP_UPPER - FAST_CANOPY_FLAT_UPPER)
    )
    struct_lower = (
        FAST_STRUCT_FLAT_LOWER
        + sw * (FAST_STRUCT_STEEP_LOWER - FAST_STRUCT_FLAT_LOWER)
    )
    struct_upper = (
        FAST_STRUCT_FLAT_UPPER
        + sw * (FAST_STRUCT_STEEP_UPPER - FAST_STRUCT_FLAT_UPPER)
    )

    centre_bad_open = (
        (centred_residual < -open_lower)
        | (centred_residual > open_upper)
    )
    centre_bad_canopy = (
        (centred_residual < -canopy_lower)
        | (centred_residual > canopy_upper)
    )
    struct_bad = (
        (struct_res < -struct_lower)
        | (struct_res > struct_upper)
    )

    suspect = (
        (
            ~canopy
            & center_supported
            & centre_bad_open
            & struct_bad
        )
        | (
            canopy
            & center_supported
            & centre_bad_canopy
            & struct_bad
        )
        | (
            canopy
            & ~center_supported
            & struct_bad
        )
    )

    # Protected smooth v6 backbone can never be removed here.
    protected = np.isin(preliminary, base_model_indices)
    suspect &= ~protected
    suspect_positions = np.flatnonzero(suspect)

    reject_indices = np.empty(0, dtype=np.int64)
    if suspect_positions.size:
        support_indices = preliminary[~suspect | protected]
        reject_indices = fast_targeted_plane_rejections(
            preliminary[suspect_positions],
            support_indices,
            slope[suspect_positions],
            canopy[suspect_positions],
            x, y, z,
        )

    final_membership = preliminary[
        ~np.isin(preliminary, reject_indices)
    ]
    final_membership = np.unique(final_membership)

    # Direct output-surface audit. Unlike the earlier heuristic-only cleanup,
    # this checks EVERY non-backbone class-2 point against a leave-one-out
    # robust local terrain plane.
    final_canopy = fast_points_in_keys(
        final_membership,
        canopy_keys,
        x, y,
        origin_x, origin_y,
    )
    final_membership, terra_stats = (
        terra_smooth_prune_final_membership(
            final_membership,
            base_model_indices,
            structural_model_indices,
            final_canopy,
            x, y, z,
            bounds,
        )
    )

    elapsed = time.perf_counter() - stage_start
    log(
        "      TerraSmooth final cleanup: "
        f"{final_membership.size:,} class-2 members; "
        f"{terra_stats['audited']:,} non-backbone members audited; "
        f"{terra_stats['open_removed']:,} open-terrain Z outliers removed; "
        f"{terra_stats['canopy_removed']:,} canopy/vertical-complexity "
        f"outliers removed; {elapsed:.1f} s"
    )
    return final_membership





def terra_smooth_local_plane_context(
    query_indices: np.ndarray,
    support_indices: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Robust leave-one-out local plane for final class-2 QC.

    Returns:
      predicted Z at query XY
      neighbour-plane RMSE
      plane slope in degrees
      robust neighbour count

    The query point itself is explicitly excluded. The second solve trims
    neighbour outliers, so a small number of spikes cannot define the plane.
    """
    n = int(query_indices.size)
    predicted = np.full(n, np.nan, dtype=np.float64)
    rmse = np.full(n, np.inf, dtype=np.float64)
    slope = np.full(n, np.nan, dtype=np.float64)
    count = np.zeros(n, dtype=np.int32)

    if n == 0 or support_indices.size < TERRA_SMOOTH_MIN_NEIGHBOURS:
        return predicted, rmse, slope, count

    support_indices = np.asarray(support_indices, dtype=np.int64)
    tree = cKDTree(
        np.column_stack((x[support_indices], y[support_indices]))
    )
    k = min(TERRA_SMOOTH_NEIGHBOURS + 1, support_indices.size)

    def solve_weighted(
        design: np.ndarray,
        nz: np.ndarray,
        weights: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        ata = np.einsum(
            "nki,nk,nkj->nij",
            design, weights, design,
            optimize=True,
        )
        atz = np.einsum(
            "nki,nk,nk->ni",
            design, weights, nz,
            optimize=True,
        )
        ata[:, 0, 0] += 1.0e-8
        ata[:, 1, 1] += 1.0e-8
        ata[:, 2, 2] += 1.0e-8
        coefficients = np.linalg.solve(
            ata, atz[..., None]
        )[..., 0]
        fitted = np.einsum(
            "nki,ni->nk",
            design, coefficients,
            optimize=True,
        )
        return coefficients, fitted

    for start in range(0, n, TERRA_SMOOTH_CHUNK):
        stop = min(start + TERRA_SMOOTH_CHUNK, n)
        part = query_indices[start:stop]
        qxy = np.column_stack((x[part], y[part]))

        distances, positions = tree_query(
            tree,
            qxy,
            k=k,
            distance_upper_bound=TERRA_SMOOTH_RADIUS,
        )
        distances = np.asarray(distances, dtype=np.float64)
        positions = np.asarray(positions, dtype=np.int64)
        if distances.ndim == 1:
            distances = distances[:, None]
            positions = positions[:, None]

        valid = (
            np.isfinite(distances)
            & (positions >= 0)
            & (positions < support_indices.size)
        )
        safe_pos = np.clip(
            positions, 0, max(0, support_indices.size - 1)
        )
        neighbour_global = support_indices[safe_pos]
        valid &= neighbour_global != part[:, None]

        dx = np.zeros(distances.shape, dtype=np.float64)
        dy = np.zeros(distances.shape, dtype=np.float64)
        nz = np.zeros(distances.shape, dtype=np.float64)

        qx = np.broadcast_to(x[part, None], distances.shape)
        qy = np.broadcast_to(y[part, None], distances.shape)
        dx[valid] = x[neighbour_global[valid]] - qx[valid]
        dy[valid] = y[neighbour_global[valid]] - qy[valid]
        nz[valid] = z[neighbour_global[valid]]

        design = np.empty(
            (part.size, distances.shape[1], 3),
            dtype=np.float64,
        )
        design[:, :, 0] = dx
        design[:, :, 1] = dy
        design[:, :, 2] = 1.0

        weights = valid.astype(np.float64)
        initial_count = np.sum(valid, axis=1)
        coefficients, fitted = solve_weighted(
            design, nz, weights
        )

        neighbour_residual = np.where(
            valid, nz - fitted, np.nan
        )
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore", category=RuntimeWarning
            )
            residual_median = np.nanmedian(
                neighbour_residual, axis=1
            )
            mad = np.nanmedian(
                np.abs(
                    neighbour_residual
                    - residual_median[:, None]
                ),
                axis=1,
            )

        trim = np.maximum(
            TERRA_SMOOTH_TRIM_MIN,
            TERRA_SMOOTH_TRIM_MAD_FACTOR
            * 1.4826
            * np.nan_to_num(mad),
        )
        robust = (
            valid
            & (
                np.abs(
                    neighbour_residual
                    - residual_median[:, None]
                )
                <= trim[:, None]
            )
        )
        robust_count = np.sum(robust, axis=1)
        use_robust = (
            robust_count >= TERRA_SMOOTH_MIN_NEIGHBOURS
        )
        final_weights = np.where(
            use_robust[:, None],
            robust,
            valid,
        ).astype(np.float64)

        coefficients, fitted = solve_weighted(
            design, nz, final_weights
        )
        final_count = np.sum(final_weights, axis=1)
        squared = np.where(
            final_weights > 0.0,
            np.square(nz - fitted),
            0.0,
        )
        local_rmse = np.sqrt(
            np.sum(squared, axis=1)
            / np.maximum(final_count, 1)
        )
        local_slope = np.degrees(
            np.arctan(
                np.hypot(
                    coefficients[:, 0],
                    coefficients[:, 1],
                )
            )
        )

        good = (
            (initial_count >= TERRA_SMOOTH_MIN_NEIGHBOURS)
            & (final_count >= TERRA_SMOOTH_MIN_NEIGHBOURS)
            & np.isfinite(coefficients[:, 2])
        )
        predicted[start:stop][good] = coefficients[good, 2]
        rmse[start:stop][good] = local_rmse[good]
        slope[start:stop][good] = local_slope[good]
        count[start:stop] = final_count.astype(np.int32)

    return predicted, rmse, slope, count


def terra_smooth_prune_final_membership(
    final_membership: np.ndarray,
    base_model_indices: np.ndarray,
    structural_model_indices: np.ndarray,
    canopy_mask: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    bounds: tuple[float, float, float, float],
) -> tuple[np.ndarray, dict[str, int]]:
    """Remove locally inconsistent final class-2 vertices.

    The protected smooth-v6 backbone is never removed. Every other final
    class-2 point is checked against a leave-one-out robust local plane.

    Real sharp breaks are protected by the plane-RMSE skip: if the surrounding
    neighbourhood is genuinely non-planar, this pass does not flatten it.
    """
    final_membership = np.unique(final_membership)
    protected = np.isin(
        final_membership, base_model_indices
    )
    audit_positions = np.flatnonzero(~protected)
    if (
        not TERRA_SMOOTH_ALL_MEMBER_AUDIT
        or audit_positions.size == 0
    ):
        return final_membership, {
            "audited": 0,
            "open_removed": 0,
            "canopy_removed": 0,
        }

    audit_indices = final_membership[audit_positions]
    predicted, plane_rmse, plane_slope, count = (
        terra_smooth_local_plane_context(
            audit_indices,
            final_membership,
            x, y, z,
        )
    )
    residual = z[audit_indices] - predicted

    # Structural context is used only as an extra canopy safety lockout.
    struct_residual, struct_slope, _ = (
        interpolate_tin_membership_context(
            audit_indices,
            structural_model_indices,
            x, y, z,
            bounds,
        )
    )

    local_slope = np.maximum(
        np.nan_to_num(plane_slope, nan=0.0),
        np.nan_to_num(struct_slope, nan=0.0),
    )
    sw = np.clip(
        local_slope / TERRA_SMOOTH_FULL_SLOPE_DEG,
        0.0,
        1.0,
    )

    open_positive = (
        TERRA_SMOOTH_OPEN_FLAT_POSITIVE
        + sw * (
            TERRA_SMOOTH_OPEN_STEEP_POSITIVE
            - TERRA_SMOOTH_OPEN_FLAT_POSITIVE
        )
    )
    open_negative = (
        TERRA_SMOOTH_OPEN_FLAT_NEGATIVE
        + sw * (
            TERRA_SMOOTH_OPEN_STEEP_NEGATIVE
            - TERRA_SMOOTH_OPEN_FLAT_NEGATIVE
        )
    )
    canopy_positive = (
        TERRA_SMOOTH_CANOPY_FLAT_POSITIVE
        + sw * (
            TERRA_SMOOTH_CANOPY_STEEP_POSITIVE
            - TERRA_SMOOTH_CANOPY_FLAT_POSITIVE
        )
    )
    canopy_negative = (
        TERRA_SMOOTH_CANOPY_FLAT_NEGATIVE
        + sw * (
            TERRA_SMOOTH_CANOPY_STEEP_NEGATIVE
            - TERRA_SMOOTH_CANOPY_FLAT_NEGATIVE
        )
    )

    structural_positive = (
        TERRA_SMOOTH_CANOPY_STRUCT_FLAT_POSITIVE
        + sw * (
            TERRA_SMOOTH_CANOPY_STRUCT_STEEP_POSITIVE
            - TERRA_SMOOTH_CANOPY_STRUCT_FLAT_POSITIVE
        )
    )
    structural_negative = (
        TERRA_SMOOTH_CANOPY_STRUCT_FLAT_NEGATIVE
        + sw * (
            TERRA_SMOOTH_CANOPY_STRUCT_STEEP_NEGATIVE
            - TERRA_SMOOTH_CANOPY_STRUCT_FLAT_NEGATIVE
        )
    )

    local_good = (
        np.isfinite(predicted)
        & np.isfinite(plane_rmse)
        & (count >= TERRA_SMOOTH_MIN_NEIGHBOURS)
        & (
            plane_rmse
            <= TERRA_SMOOTH_BREAK_RMSE_SKIP
        )
    )

    audit_canopy = canopy_mask[audit_positions]

    open_bad = (
        ~audit_canopy
        & local_good
        & (
            (residual > open_positive)
            | (residual < -open_negative)
        )
    )

    # Under canopy: a local-plane outlier is enough to reject. In addition,
    # an elevated/depressed structural outlier is rejected when the local plane
    # also trends in the same direction, preventing a small vegetation cluster
    # from mutually supporting a spike.
    canopy_plane_bad = (
        audit_canopy
        & local_good
        & (
            (residual > canopy_positive)
            | (residual < -canopy_negative)
        )
    )
    canopy_struct_high = (
        audit_canopy
        & np.isfinite(struct_residual)
        & (struct_residual > structural_positive)
        & (
            (~np.isfinite(residual))
            | (residual > -0.015)
        )
    )
    canopy_struct_low = (
        audit_canopy
        & np.isfinite(struct_residual)
        & (struct_residual < -structural_negative)
        & (
            (~np.isfinite(residual))
            | (residual < 0.015)
        )
    )
    canopy_bad = (
        canopy_plane_bad
        | canopy_struct_high
        | canopy_struct_low
    )

    remove_audit = open_bad | canopy_bad
    remove_indices = audit_indices[remove_audit]

    keep = ~np.isin(final_membership, remove_indices)
    pruned = final_membership[keep]

    return np.unique(pruned), {
        "audited": int(audit_indices.size),
        "open_removed": int(np.count_nonzero(open_bad)),
        "canopy_removed": int(np.count_nonzero(canopy_bad)),
    }



def fast_group_center_offsets(
    support_indices: np.ndarray,
    hag: np.ndarray,
    support_slope: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    origin_x: float,
    origin_y: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised robust per-cell HAG centre from accepted ground."""
    support_indices = np.asarray(support_indices, dtype=np.int64)
    support_slope = np.asarray(support_slope, dtype=np.float64).reshape(-1)
    if support_indices.size != support_slope.size:
        raise ValueError(
            "fast_group_center_offsets: support/slope size mismatch "
            f"({support_indices.size:,} / {support_slope.size:,})"
        )

    finite = np.isfinite(hag[support_indices]) & np.isfinite(support_slope)
    support_indices = support_indices[finite]
    support_slope = support_slope[finite]
    if support_indices.size == 0:
        empty = np.empty(0, dtype=[("x", "<i8"), ("y", "<i8")])
        return empty, np.empty(0), np.empty(0, dtype=np.int32)

    keys = packed_grid_keys(
        x[support_indices], y[support_indices],
        FAST_CENTER_CELL, origin_x, origin_y,
    )
    values = hag[support_indices].astype(np.float64, copy=False)
    order = np.lexsort((values, keys["y"], keys["x"]))
    ordered_keys = keys[order]
    ordered_values = values[order]
    ordered_slope = np.nan_to_num(support_slope[order], nan=0.0)

    unique_keys, starts, counts = np.unique(
        ordered_keys, return_index=True, return_counts=True
    )
    lo = starts + (counts - 1) // 2
    hi = starts + counts // 2
    median = 0.5 * (ordered_values[lo] + ordered_values[hi])

    cell_slope = np.maximum.reduceat(ordered_slope, starts)
    sw = np.clip(
        cell_slope / FAST_CENTER_FULL_SLOPE_DEG, 0.0, 1.0
    )
    max_up = (
        FAST_CENTER_FLAT_MAX_UP
        + sw * (FAST_CENTER_STEEP_MAX_UP - FAST_CENTER_FLAT_MAX_UP)
    )
    median = np.clip(median, -FAST_CENTER_MAX_DOWN, max_up)
    median[counts < FAST_CENTER_MIN_POINTS] = np.nan
    return unique_keys, median, counts.astype(np.int32)


def fast_lookup_group_values(
    query_indices: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    cell_size: float,
    x: np.ndarray,
    y: np.ndarray,
    origin_x: float,
    origin_y: float,
    default: float = np.nan,
) -> np.ndarray:
    result = np.full(query_indices.size, default, dtype=np.float64)
    if query_indices.size == 0 or keys.size == 0:
        return result
    qkeys = packed_grid_keys(
        x[query_indices], y[query_indices],
        cell_size, origin_x, origin_y,
    )
    pos = np.searchsorted(keys, qkeys)
    safe = np.clip(pos, 0, max(0, keys.size - 1))
    matched = (pos < keys.size) & (keys[safe] == qkeys)
    result[matched] = values[safe[matched]]
    return result


def fast_canopy_complexity_keys(
    ground_candidates: np.ndarray,
    hag: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    origin_x: float,
    origin_y: float,
) -> np.ndarray:
    """Packed cells containing strong vertical/canopy evidence."""
    h = hag[ground_candidates]
    elevated_mask = (
        np.isfinite(h)
        & (h >= FAST_CANOPY_ELEVATED_HAG)
        & (h <= FAST_CANOPY_MAX_HAG)
    )
    elevated = ground_candidates[elevated_mask]
    if elevated.size == 0:
        return np.empty(0, dtype=[("x", "<i8"), ("y", "<i8")])

    keys = packed_grid_keys(
        x[elevated], y[elevated],
        FAST_CANOPY_CELL, origin_x, origin_y,
    )
    values = hag[elevated].astype(np.float64, copy=False)
    order = np.lexsort((values, keys["y"], keys["x"]))
    ordered_keys = keys[order]
    ordered_values = values[order]
    unique_keys, starts, counts = np.unique(
        ordered_keys, return_index=True, return_counts=True
    )
    max_hag = np.maximum.reduceat(ordered_values, starts)
    complex_mask = (
        (counts >= FAST_CANOPY_MIN_ELEVATED_COUNT)
        | (max_hag >= FAST_CANOPY_EXTREME_HAG)
    )
    return unique_keys[complex_mask]


def fast_points_in_keys(
    query_indices: np.ndarray,
    complex_keys: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    origin_x: float,
    origin_y: float,
) -> np.ndarray:
    result = np.zeros(query_indices.size, dtype=bool)
    if query_indices.size == 0 or complex_keys.size == 0:
        return result
    qkeys = packed_grid_keys(
        x[query_indices], y[query_indices],
        FAST_CANOPY_CELL, origin_x, origin_y,
    )
    pos = np.searchsorted(complex_keys, qkeys)
    safe = np.clip(pos, 0, max(0, complex_keys.size - 1))
    return (pos < complex_keys.size) & (complex_keys[safe] == qkeys)


def fast_targeted_plane_rejections(
    suspect_indices: np.ndarray,
    support_indices: np.ndarray,
    suspect_slope: np.ndarray,
    suspect_canopy: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    """Confirm only nominated Z outliers against independent robust terrain."""
    if (
        not FAST_SUSPECT_PLANE_AUDIT
        or suspect_indices.size == 0
        or support_indices.size < FAST_PLANE_MIN_NEIGHBOURS
    ):
        return np.empty(0, dtype=np.int64)

    support_tree = cKDTree(
        np.column_stack((x[support_indices], y[support_indices]))
    )
    plane_z, plane_rmse, neighbour_count, _ = local_plane_support(
        suspect_indices, support_indices,
        x, y, z, support_tree,
    )
    residual = z[suspect_indices] - plane_z
    sw = np.clip(
        suspect_slope / FAST_PLANE_FULL_SLOPE_DEG, 0.0, 1.0
    )
    open_tol = (
        FAST_PLANE_OPEN_FLAT_TOL
        + sw * (FAST_PLANE_OPEN_STEEP_TOL - FAST_PLANE_OPEN_FLAT_TOL)
    )
    canopy_tol = (
        FAST_PLANE_CANOPY_FLAT_TOL
        + sw * (FAST_PLANE_CANOPY_STEEP_TOL - FAST_PLANE_CANOPY_FLAT_TOL)
    )
    tolerance = np.where(suspect_canopy, canopy_tol, open_tol)
    reject = (
        np.isfinite(plane_z)
        & (neighbour_count >= FAST_PLANE_MIN_NEIGHBOURS)
        & (plane_rmse <= FAST_PLANE_MAX_RMSE)
        & (np.abs(residual) > tolerance)
    )
    return suspect_indices[reject]



def break_membership_peer_coherence(
    candidate_indices: np.ndarray,
    signed_residual: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return peer-QC keep mask and residual-centering penalty.

    All candidates supplied here have ALREADY passed independent two-sided
    trusted terrain evidence. Peers are therefore used only to reject isolated
    representatives and select the coherent centre of local source scatter.
    """
    keep = np.ones(candidate_indices.size, dtype=bool)
    penalty = np.zeros(candidate_indices.size, dtype=np.float64)
    if (
        not BREAK_MEMBERSHIP_PEER_QC
        or candidate_indices.size < BREAK_MEMBERSHIP_PEER_MIN_NEIGHBOURS + 1
    ):
        return keep, penalty

    xy = np.column_stack((x[candidate_indices], y[candidate_indices]))
    tree = cKDTree(xy)
    k = min(
        BREAK_MEMBERSHIP_PEER_NEIGHBOURS + 1,
        candidate_indices.size,
    )
    distances, positions = tree_query(
        tree,
        xy,
        k=k,
        distance_upper_bound=BREAK_MEMBERSHIP_PEER_RADIUS,
    )
    distances = np.asarray(distances, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.int64)
    if distances.ndim == 1:
        distances = distances[:, None]
        positions = positions[:, None]

    rows = np.arange(candidate_indices.size)[:, None]
    valid = (
        np.isfinite(distances)
        & (positions >= 0)
        & (positions < candidate_indices.size)
        & (positions != rows)
    )
    safe = np.clip(
        positions, 0, max(0, candidate_indices.size - 1)
    )
    peer_residual = np.full(
        positions.shape, np.nan, dtype=np.float64
    )
    peer_residual[valid] = signed_residual[safe[valid]]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median = np.nanmedian(peer_residual, axis=1)
        mad = np.nanmedian(
            np.abs(peer_residual - median[:, None]),
            axis=1,
        )

    count = np.sum(valid, axis=1)
    robust_scale = 1.4826 * np.nan_to_num(mad, nan=0.0)
    low_tol = np.maximum(
        BREAK_MEMBERSHIP_PEER_MIN_LOW_TOLERANCE,
        BREAK_MEMBERSHIP_PEER_MAD_FACTOR * robust_scale,
    )
    high_tol = np.maximum(
        BREAK_MEMBERSHIP_PEER_MIN_HIGH_TOLERANCE,
        BREAK_MEMBERSHIP_PEER_MAD_FACTOR * robust_scale,
    )

    enough = (
        count >= BREAK_MEMBERSHIP_PEER_MIN_NEIGHBOURS
    ) & np.isfinite(median)

    delta = signed_residual - median
    keep[enough] = (
        (delta[enough] >= -low_tol[enough])
        & (delta[enough] <= high_tol[enough])
    )

    scale = np.where(
        delta >= 0.0,
        high_tol,
        low_tol,
    )
    penalty[enough] = (
        np.abs(delta[enough])
        / np.maximum(scale[enough], 1.0e-9)
    )
    return keep, penalty


def audit_added_low_spike_representatives(
    base_membership: np.ndarray,
    added_indices: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    """Remove only NEW isolated low representatives from a clean local plane.

    This is a leave-one-out style check: the candidate itself is excluded from
    its support. It cannot flatten general terrain because rejection requires a
    clean local plane, multi-directional support and a material LOW residual.
    """
    if (
        not BREAK_MEMBERSHIP_LOW_SPIKE_AUDIT
        or added_indices.size == 0
    ):
        return np.unique(added_indices)

    membership = np.unique(
        np.concatenate((base_membership, added_indices))
    )
    tree = cKDTree(
        np.column_stack((x[membership], y[membership]))
    )
    k = min(
        BREAK_MEMBERSHIP_LOW_SPIKE_NEIGHBOURS + 1,
        membership.size,
    )

    query_xy = np.column_stack(
        (x[added_indices], y[added_indices])
    )
    distances, positions = tree_query(
        tree,
        query_xy,
        k=k,
        distance_upper_bound=BREAK_MEMBERSHIP_LOW_SPIKE_RADIUS,
    )
    distances = np.asarray(distances, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.int64)
    if distances.ndim == 1:
        distances = distances[:, None]
        positions = positions[:, None]

    safe = np.clip(
        positions, 0, max(0, membership.size - 1)
    )
    neighbour_global = membership[safe]
    own = np.broadcast_to(
        added_indices[:, None], positions.shape
    )
    valid = (
        np.isfinite(distances)
        & (positions >= 0)
        & (positions < membership.size)
        & (neighbour_global != own)
    )

    keep = np.ones(added_indices.size, dtype=bool)

    for row in range(added_indices.size):
        row_valid = valid[row]
        if np.count_nonzero(row_valid) < BREAK_MEMBERSHIP_LOW_SPIKE_MIN_NEIGHBOURS:
            continue

        neighbours = neighbour_global[row, row_valid]
        dx = x[neighbours] - x[added_indices[row]]
        dy = y[neighbours] - y[added_indices[row]]
        design = np.column_stack(
            (dx, dy, np.ones(neighbours.size))
        )

        try:
            coeff, *_ = np.linalg.lstsq(
                design,
                z[neighbours],
                rcond=None,
            )
        except np.linalg.LinAlgError:
            continue

        fitted = design @ coeff
        residual = z[neighbours] - fitted
        med = float(np.median(residual))
        mad = float(np.median(np.abs(residual - med)))
        trim = max(0.08, 3.0 * 1.4826 * mad)
        inlier = np.abs(residual - med) <= trim
        if np.count_nonzero(inlier) < BREAK_MEMBERSHIP_LOW_SPIKE_MIN_NEIGHBOURS:
            continue

        try:
            coeff, *_ = np.linalg.lstsq(
                design[inlier],
                z[neighbours][inlier],
                rcond=None,
            )
        except np.linalg.LinAlgError:
            continue

        fitted_inlier = design[inlier] @ coeff
        rmse = float(np.sqrt(np.mean(
            np.square(
                z[neighbours][inlier] - fitted_inlier
            )
        )))
        if rmse > BREAK_MEMBERSHIP_LOW_SPIKE_MAX_PLANE_RMSE:
            continue

        # Require support in at least two XY quadrants around the candidate.
        qx = dx[inlier]
        qy = dy[inlier]
        quadrant = (
            (qx >= 0.0).astype(np.int8)
            + 2 * (qy >= 0.0).astype(np.int8)
        )
        direction_count = len(np.unique(quadrant))
        if (
            direction_count
            < BREAK_MEMBERSHIP_LOW_SPIKE_MIN_DIRECTION_SECTORS
        ):
            continue

        predicted = float(coeff[2])
        point_residual = (
            z[added_indices[row]] - predicted
        )
        if (
            point_residual
            <= -BREAK_MEMBERSHIP_LOW_SPIKE_MIN_DEPTH
        ):
            keep[row] = False

    return np.unique(added_indices[keep])



def tin_surface_slope_at_xy(
    surface: TinSurface,
    query_xy: np.ndarray,
) -> np.ndarray:
    """Return triangle slope in degrees from a frozen TIN at arbitrary XY."""
    tin, model_xy, model_z, _ = surface
    result = np.full(query_xy.shape[0], np.nan, dtype=np.float64)
    if query_xy.size == 0:
        return result

    simplex = tin.find_simplex(query_xy)
    valid = simplex >= 0
    if not np.any(valid):
        return result

    vertices = tin.simplices[simplex[valid]]
    tri_xy = model_xy[vertices]
    tri_z = model_z[vertices]

    p0 = np.column_stack((tri_xy[:, 0, :], tri_z[:, 0]))
    p1 = np.column_stack((tri_xy[:, 1, :], tri_z[:, 1]))
    p2 = np.column_stack((tri_xy[:, 2, :], tri_z[:, 2]))
    normals = np.cross(p1 - p0, p2 - p0)
    normal_length = np.linalg.norm(normals, axis=1)
    good = normal_length > 1.0e-12
    if not np.any(good):
        return result

    local = np.full(np.count_nonzero(valid), np.nan, dtype=np.float64)
    normals = normals[good] / normal_length[good, None]
    normals[normals[:, 2] < 0.0] *= -1.0
    local[good] = np.degrees(
        np.arctan2(
            np.linalg.norm(normals[:, :2], axis=1),
            np.maximum(normals[:, 2], 1.0e-12),
        )
    )
    result[valid] = local
    return result


def audit_two_sided_membership_tin_geometry(
    base_membership: np.ndarray,
    added_indices: np.ndarray,
    structural_model_indices: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    bounds: tuple[float, float, float, float],
) -> np.ndarray:
    """Audit the ACTUAL proposed class-2 Delaunay TIN.

    A bad new representative typically produces a fan of triangles much
    steeper than the frozen trusted terrain around it. We therefore triangulate
    base + new members, inspect real triangle geometry, and remove only NEW
    vertices that repeatedly participate in anomalous steep triangles.

    Legitimate steep/break terrain is protected by comparing each triangle to
    the local slope of the frozen structural v6 TIN and by requiring either
    repeated anomalous triangles or an absolute near-vertical triangle.
    """
    if (
        not BREAK_MEMBERSHIP_TIN_AUDIT
        or added_indices.size == 0
        or base_membership.size < 3
        or structural_model_indices.size < 3
    ):
        return np.unique(added_indices)

    base_membership = np.unique(base_membership)
    current_added = np.unique(added_indices)
    structural_surface = build_tin(
        structural_model_indices, x, y, z, bounds
    )

    for audit_pass in range(1, BREAK_MEMBERSHIP_TIN_AUDIT_PASSES + 1):
        if current_added.size == 0:
            break

        membership = np.unique(
            np.concatenate((base_membership, current_added))
        )
        added_mask = np.isin(membership, current_added)
        if not np.any(added_mask):
            break

        final_tin, final_xy, final_z, _ = build_tin(
            membership, x, y, z, bounds
        )
        simplices = final_tin.simplices

        # Synthetic corner vertices sit after real membership vertices.
        real_triangle = np.all(simplices < membership.size, axis=1)
        simplices = simplices[real_triangle]
        if simplices.size == 0:
            break

        tri_xy = final_xy[simplices]
        tri_z = final_z[simplices]

        p0 = np.column_stack((tri_xy[:, 0, :], tri_z[:, 0]))
        p1 = np.column_stack((tri_xy[:, 1, :], tri_z[:, 1]))
        p2 = np.column_stack((tri_xy[:, 2, :], tri_z[:, 2]))

        normals = np.cross(p1 - p0, p2 - p0)
        normal_length = np.linalg.norm(normals, axis=1)
        valid_normal = normal_length > 1.0e-12
        safe_length = np.where(valid_normal, normal_length, 1.0)
        normals = normals / safe_length[:, None]
        normals[normals[:, 2] < 0.0] *= -1.0
        triangle_slope = np.degrees(
            np.arctan2(
                np.linalg.norm(normals[:, :2], axis=1),
                np.maximum(normals[:, 2], 1.0e-12),
            )
        )

        edge01 = np.linalg.norm(
            tri_xy[:, 0] - tri_xy[:, 1], axis=1
        )
        edge12 = np.linalg.norm(
            tri_xy[:, 1] - tri_xy[:, 2], axis=1
        )
        edge20 = np.linalg.norm(
            tri_xy[:, 2] - tri_xy[:, 0], axis=1
        )
        max_edge = np.maximum.reduce((edge01, edge12, edge20))
        z_range = np.max(tri_z, axis=1) - np.min(tri_z, axis=1)
        centroid = np.mean(tri_xy, axis=1)
        trusted_slope = tin_surface_slope_at_xy(
            structural_surface, centroid
        )

        allowed = np.minimum(
            BREAK_MEMBERSHIP_TIN_AUDIT_ABSOLUTE_MAX_SLOPE_DEG,
            np.maximum(
                BREAK_MEMBERSHIP_TIN_AUDIT_MIN_SLOPE_DEG,
                np.nan_to_num(
                    trusted_slope,
                    nan=BREAK_MEMBERSHIP_TIN_AUDIT_MIN_SLOPE_DEG,
                )
                + BREAK_MEMBERSHIP_TIN_AUDIT_SLOPE_ALLOWANCE_DEG,
            ),
        )

        suspicious = (
            valid_normal
            & (max_edge <= BREAK_MEMBERSHIP_TIN_AUDIT_MAX_EDGE)
            & (z_range >= BREAK_MEMBERSHIP_TIN_AUDIT_MIN_Z_RANGE)
            & (triangle_slope > allowed)
        )
        extreme = (
            valid_normal
            & (z_range >= BREAK_MEMBERSHIP_TIN_AUDIT_MIN_Z_RANGE)
            & (
                triangle_slope
                >= BREAK_MEMBERSHIP_TIN_AUDIT_ABSOLUTE_MAX_SLOPE_DEG
            )
        )

        if not np.any(suspicious | extreme):
            break

        suspicious_count = np.zeros(membership.size, dtype=np.int16)
        extreme_count = np.zeros(membership.size, dtype=np.int16)

        for col in range(3):
            pos = simplices[:, col]
            is_new_vertex = added_mask[pos]
            np.add.at(
                suspicious_count,
                pos[is_new_vertex & suspicious],
                1,
            )
            np.add.at(
                extreme_count,
                pos[is_new_vertex & extreme],
                1,
            )

        remove_position = added_mask & (
            (
                suspicious_count
                >= BREAK_MEMBERSHIP_TIN_AUDIT_MIN_DANGEROUS_TRIANGLES
            )
            | (extreme_count >= 1)
        )
        removed_global = membership[remove_position]
        if removed_global.size == 0:
            break

        current_added = current_added[
            ~np.isin(current_added, removed_global)
        ]
        log(
            f"      Final TIN geometry audit pass {audit_pass}: "
            f"removed {removed_global.size:,} new spike-risk vertices"
        )

    return np.unique(current_added)



def audit_two_sided_membership_spikes(
    base_membership: np.ndarray,
    added_indices: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    """Reject only NEW points that form dangerous short/steep TIN edges."""
    if added_indices.size == 0:
        return added_indices

    combined = np.unique(np.concatenate((base_membership, added_indices)))
    tree = cKDTree(np.column_stack((x[combined], y[combined])))
    k = min(BREAK_MEMBERSHIP_SPIKE_NEIGHBOURS + 1, combined.size)
    query_xy = np.column_stack((x[added_indices], y[added_indices]))
    distances, positions = tree_query(
        tree,
        query_xy,
        k=k,
        distance_upper_bound=BREAK_MEMBERSHIP_SPIKE_RADIUS,
    )
    distances = np.asarray(distances, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.int64)
    if distances.ndim == 1:
        distances = distances[:, None]
        positions = positions[:, None]

    valid = (
        np.isfinite(distances)
        & (positions >= 0)
        & (positions < combined.size)
    )
    safe = np.clip(positions, 0, max(0, combined.size - 1))
    neighbour_global = combined[safe]
    own = np.broadcast_to(added_indices[:, None], distances.shape)
    valid &= neighbour_global != own

    dz = np.zeros(distances.shape, dtype=np.float64)
    qz = np.broadcast_to(z[added_indices, None], distances.shape)
    dz[valid] = np.abs(qz[valid] - z[neighbour_global[valid]])
    edge_slope = np.zeros(distances.shape, dtype=np.float64)
    edge_slope[valid] = np.degrees(
        np.arctan2(
            dz[valid],
            np.maximum(distances[valid], 1.0e-6),
        )
    )
    dangerous = (
        valid
        & (distances <= BREAK_MEMBERSHIP_SPIKE_SHORT_EDGE)
        & (dz >= BREAK_MEMBERSHIP_SPIKE_MIN_DZ)
        & (
            edge_slope
            >= BREAK_MEMBERSHIP_SPIKE_MAX_EDGE_SLOPE_DEG
        )
    )
    keep = ~np.any(dangerous, axis=1)
    return added_indices[keep]




def turbo2_density_need_positions(
    unresolved: np.ndarray,
    current_membership: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    bounds: tuple[float, float, float, float],
    steep_target: float,
) -> np.ndarray:
    """Return unresolved positions that can still improve class-2 density.

    Gate 1 is strictly monotonic: a point already too close to current class-2
    membership cannot become farther away when more vertices are added.

    Gate 2 removes points whose containing current class-2 triangle is already
    no larger than the smallest (steep-terrain) target edge. This keeps the
    expensive two-sided slope evidence focused on actual underrepresented TIN
    areas rather than millions of already-dense source returns.
    """
    if (
        not TURBO2_DENSITY_NEED_GATE
        or unresolved.size == 0
        or current_membership.size < 3
    ):
        return np.arange(unresolved.size, dtype=np.int64)

    started = time.perf_counter()

    # Gate 1: existing-member proximity.
    min_vertex_distance = (
        GROUND_MEMBERSHIP_MIN_VERTEX_FRACTION
        * steep_target
    )
    member_tree = cKDTree(
        np.column_stack(
            (x[current_membership], y[current_membership])
        )
    )

    distance_parts: list[np.ndarray] = []
    for start in range(
        0,
        unresolved.size,
        TURBO2_DENSITY_GATE_CHUNK,
    ):
        stop = min(
            start + TURBO2_DENSITY_GATE_CHUNK,
            unresolved.size,
        )
        part = unresolved[start:stop]
        distance, _ = tree_query(
            member_tree,
            np.column_stack((x[part], y[part])),
            k=1,
        )
        distance = np.asarray(distance, dtype=np.float64)
        keep = (
            np.isfinite(distance)
            & (distance >= min_vertex_distance)
        )
        if np.any(keep):
            distance_parts.append(
                np.flatnonzero(keep).astype(np.int64)
                + start
            )

    if not distance_parts:
        log(
            "      TIMING Turbo2 density-need gate: "
            f"{time.perf_counter() - started:.1f} s; "
            f"0 of {unresolved.size:,} unresolved retained"
        )
        return np.empty(0, dtype=np.int64)

    distance_positions = np.concatenate(distance_parts)
    distance_indices = unresolved[distance_positions]

    # Gate 2: current final-membership TIN density.
    density_tin, density_xy = build_planar_tin(
        current_membership,
        x,
        y,
        bounds,
    )
    simplices = density_tin.simplices

    p0 = density_xy[simplices[:, 0]]
    p1 = density_xy[simplices[:, 1]]
    p2 = density_xy[simplices[:, 2]]

    edge01 = np.hypot(
        p0[:, 0] - p1[:, 0],
        p0[:, 1] - p1[:, 1],
    )
    edge12 = np.hypot(
        p1[:, 0] - p2[:, 0],
        p1[:, 1] - p2[:, 1],
    )
    edge20 = np.hypot(
        p2[:, 0] - p0[:, 0],
        p2[:, 1] - p0[:, 1],
    )
    simplex_max_edge = np.maximum.reduce(
        (edge01, edge12, edge20)
    )

    edge_parts: list[np.ndarray] = []
    for start in range(
        0,
        distance_indices.size,
        TURBO2_DENSITY_GATE_CHUNK,
    ):
        stop = min(
            start + TURBO2_DENSITY_GATE_CHUNK,
            distance_indices.size,
        )
        part = distance_indices[start:stop]
        simplex = density_tin.find_simplex(
            np.column_stack((x[part], y[part]))
        )
        inside = simplex >= 0
        keep = np.zeros(part.size, dtype=bool)
        if np.any(inside):
            keep[inside] = (
                simplex_max_edge[simplex[inside]]
                > steep_target
            )
        if np.any(keep):
            edge_parts.append(
                np.flatnonzero(keep).astype(np.int64)
                + start
            )

    if edge_parts:
        edge_local = np.concatenate(edge_parts)
        final_positions = distance_positions[edge_local]
    else:
        final_positions = np.empty(0, dtype=np.int64)

    log(
        "      TIMING Turbo2 density-need gate: "
        f"{time.perf_counter() - started:.1f} s; "
        f"{distance_positions.size:,} survived proximity gate; "
        f"{final_positions.size:,} of {unresolved.size:,} unresolved "
        "still need possible TIN representation"
    )
    return final_positions



def chunked_two_sided_break_evidence(
    part: np.ndarray,
    tin_slope_part: np.ndarray,
    model_indices: np.ndarray,
    model_tree: cKDTree,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Compute exact two-sided break evidence in bounded-memory chunks.

    Returns one record per candidate:
      local_z_range
      two_sided
      best_score
      best_slope
      best_prediction
      best_pair_disagreement

    This is algebraically the same directional trusted-plane test previously
    applied to the entire unresolved population at once. The difference is only
    memory layout: neighbour/design/sector arrays exist for <=50k candidates at
    a time and are discarded before the next chunk.
    """
    part = np.asarray(part, dtype=np.int64)
    tin_slope_part = np.asarray(tin_slope_part, dtype=np.float64)
    if part.size != tin_slope_part.size:
        raise ValueError(
            "chunked_two_sided_break_evidence: part/slope size mismatch "
            f"({part.size:,} / {tin_slope_part.size:,})"
        )

    n = int(part.size)
    local_z_range = np.full(n, np.nan, dtype=np.float64)
    two_sided = np.zeros(n, dtype=bool)
    best_score = np.full(n, np.inf, dtype=np.float64)
    best_slope = np.zeros(n, dtype=np.float64)
    best_prediction = np.full(n, np.nan, dtype=np.float64)
    best_pair_disagreement = np.full(n, np.inf, dtype=np.float64)

    if n == 0:
        return (
            local_z_range,
            two_sided,
            best_score,
            best_slope,
            best_prediction,
            best_pair_disagreement,
        )

    k = min(BREAK_MEMBERSHIP_NEIGHBOURS, model_indices.size)
    sectors = BREAK_MEMBERSHIP_DIRECTION_SECTORS

    for start in range(0, n, BREAK_MEMBERSHIP_EVIDENCE_CHUNK):
        stop = min(start + BREAK_MEMBERSHIP_EVIDENCE_CHUNK, n)
        query_indices = part[start:stop]
        query_tin_slope = tin_slope_part[start:stop]
        qxy = np.column_stack((x[query_indices], y[query_indices]))

        distances, positions = tree_query(
            model_tree,
            qxy,
            k=k,
            distance_upper_bound=BREAK_MEMBERSHIP_NEIGHBOUR_RADIUS,
        )
        distances = np.asarray(distances, dtype=np.float64)
        positions = np.asarray(positions, dtype=np.int64)
        if distances.ndim == 1:
            distances = distances[:, None]
            positions = positions[:, None]

        valid = (
            np.isfinite(distances)
            & (positions >= 0)
            & (positions < model_indices.size)
        )
        safe = np.clip(
            positions, 0, max(0, model_indices.size - 1)
        )
        global_idx = model_indices[safe]

        dx = np.zeros(distances.shape, dtype=np.float64)
        dy = np.zeros(distances.shape, dtype=np.float64)
        nz = np.full(distances.shape, np.nan, dtype=np.float64)

        qx = np.broadcast_to(
            x[query_indices, None], distances.shape
        )
        qy = np.broadcast_to(
            y[query_indices, None], distances.shape
        )
        dx[valid] = x[global_idx[valid]] - qx[valid]
        dy[valid] = y[global_idx[valid]] - qy[valid]
        nz[valid] = z[global_idx[valid]]

        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore", category=RuntimeWarning
            )
            local_z_range[start:stop] = (
                np.nanmax(nz, axis=1)
                - np.nanmin(nz, axis=1)
            )

        angles = np.degrees(np.arctan2(dy, dx))

        design = np.empty(
            (query_indices.size, distances.shape[1], 3),
            dtype=np.float64,
        )
        design[:, :, 0] = dx
        design[:, :, 1] = dy
        design[:, :, 2] = 1.0

        pred = np.full(
            (query_indices.size, sectors),
            np.nan,
            dtype=np.float64,
        )
        wrmse = np.full_like(pred, np.inf)
        wslope = np.full_like(pred, np.nan)
        wnearest = np.full_like(pred, np.inf)

        # Neither of these depends on the sector, so they are computed once
        # per chunk instead of once (or twice) per sector. Same expressions
        # on the same inputs: the values are bit-for-bit what the sector
        # loop computed before.
        inverse_distance_squared = 1.0 / np.square(
            np.maximum(distances, 0.30)
        )
        nz_filled = np.nan_to_num(nz, nan=0.0)

        for sector in range(sectors):
            centre = (
                -180.0
                + (sector + 0.5) * (360.0 / sectors)
            )
            diff = np.abs(
                (angles - centre + 180.0) % 360.0 - 180.0
            )
            wedge = valid & (
                diff <= BREAK_MEMBERSHIP_WEDGE_HALF_WIDTH_DEG
            )
            count = np.sum(wedge, axis=1)
            rows = (
                count
                >= BREAK_MEMBERSHIP_MIN_WEDGE_NEIGHBOURS
            )
            if not np.any(rows):
                continue

            weights = np.where(
                wedge,
                inverse_distance_squared,
                0.0,
            )
            ata = np.einsum(
                "nki,nk,nkj->nij",
                design,
                weights,
                design,
                optimize=True,
            )
            atz = np.einsum(
                "nki,nk,nk->ni",
                design,
                weights,
                nz_filled,
                optimize=True,
            )
            ata[:, 0, 0] += 1.0e-7
            ata[:, 1, 1] += 1.0e-7
            ata[:, 2, 2] += 1.0e-7

            try:
                coeff = np.linalg.solve(
                    ata, atz[..., None]
                )[..., 0]
            except np.linalg.LinAlgError:
                continue

            fitted = np.einsum(
                "nki,ni->nk",
                design,
                coeff,
                optimize=True,
            )
            sq = np.where(
                wedge,
                np.square(
                    nz_filled - fitted
                ),
                0.0,
            )
            rmse = np.sqrt(
                np.sum(sq, axis=1)
                / np.maximum(count, 1)
            )

            wsum = np.maximum(
                np.sum(weights, axis=1), 1.0e-12
            )
            mean_dx = (
                np.sum(weights * dx, axis=1) / wsum
            )
            mean_dy = (
                np.sum(weights * dy, axis=1) / wsum
            )
            cdx = dx - mean_dx[:, None]
            cdy = dy - mean_dy[:, None]
            cxx = (
                np.sum(weights * cdx * cdx, axis=1)
                / wsum
            )
            cyy = (
                np.sum(weights * cdy * cdy, axis=1)
                / wsum
            )
            cxy = (
                np.sum(weights * cdx * cdy, axis=1)
                / wsum
            )
            trace = cxx + cyy
            disc = np.sqrt(
                np.maximum(
                    (cxx - cyy) ** 2
                    + 4.0 * cxy * cxy,
                    0.0,
                )
            )
            spread = np.sqrt(
                np.maximum(
                    0.5 * (trace - disc),
                    0.0,
                )
            )

            nearest = np.min(
                np.where(wedge, distances, np.inf),
                axis=1,
            )
            slope = np.degrees(
                np.arctan(
                    np.linalg.norm(
                        coeff[:, :2], axis=1
                    )
                )
            )
            good_fit = (
                rows
                & np.isfinite(coeff[:, 2])
                & np.isfinite(rmse)
                & (
                    rmse
                    <= BREAK_MEMBERSHIP_MAX_WEDGE_RMSE
                )
                & (
                    spread
                    >= BREAK_MEMBERSHIP_MIN_XY_SPREAD
                )
            )
            pred[good_fit, sector] = coeff[
                good_fit, 2
            ]
            wrmse[good_fit, sector] = rmse[good_fit]
            wslope[good_fit, sector] = slope[good_fit]
            wnearest[good_fit, sector] = nearest[
                good_fit
            ]

        residual = z[query_indices, None] - pred
        slope_for_tolerance = np.maximum(
            np.broadcast_to(
                query_tin_slope[:, None],
                wslope.shape,
            ),
            np.nan_to_num(wslope, nan=0.0),
        )
        sw = np.clip(
            slope_for_tolerance
            / BREAK_MEMBERSHIP_FULL_SLOPE_DEG,
            0.0,
            1.0,
        )
        tolerance = (
            BREAK_MEMBERSHIP_FLAT_FIT_TOLERANCE
            + sw
            * (
                BREAK_MEMBERSHIP_STEEP_FIT_TOLERANCE
                - BREAK_MEMBERSHIP_FLAT_FIT_TOLERANCE
            )
        )

        sector_good = (
            np.isfinite(pred)
            & np.isfinite(wrmse)
            & np.isfinite(wnearest)
            & (
                wnearest
                <= BREAK_MEMBERSHIP_MAX_MODEL_DISTANCE
            )
            & (np.abs(residual) <= tolerance)
        )

        chunk_two_sided = np.zeros(
            query_indices.size, dtype=bool
        )
        chunk_best_score = np.full(
            query_indices.size, np.inf, dtype=np.float64
        )
        chunk_best_slope = np.zeros(
            query_indices.size, dtype=np.float64
        )
        chunk_best_prediction = np.full(
            query_indices.size, np.nan, dtype=np.float64
        )
        chunk_best_disagreement = np.full(
            query_indices.size, np.inf, dtype=np.float64
        )

        for a in range(sectors):
            for b in range(a + 1, sectors):
                delta = min(
                    (b - a) % sectors,
                    (a - b) % sectors,
                )
                if (
                    delta
                    < BREAK_MEMBERSHIP_MIN_OPPOSITION_SECTORS
                ):
                    continue
                if (
                    delta
                    > sectors
                    - BREAK_MEMBERSHIP_MIN_OPPOSITION_SECTORS
                ):
                    continue

                pair_good = (
                    sector_good[:, a]
                    & sector_good[:, b]
                    & (
                        np.abs(
                            pred[:, a] - pred[:, b]
                        )
                        <= BREAK_MEMBERSHIP_MAX_TWO_SIDE_PREDICTION_DISAGREEMENT
                    )
                )
                if not np.any(pair_good):
                    continue

                pair_score = 0.5 * (
                    np.abs(residual[:, a])
                    / np.maximum(
                        tolerance[:, a], 1.0e-9
                    )
                    + np.abs(residual[:, b])
                    / np.maximum(
                        tolerance[:, b], 1.0e-9
                    )
                )
                pair_slope = np.maximum(
                    np.nan_to_num(
                        wslope[:, a], nan=0.0
                    ),
                    np.nan_to_num(
                        wslope[:, b], nan=0.0
                    ),
                )
                pair_prediction = 0.5 * (
                    pred[:, a] + pred[:, b]
                )
                pair_disagreement = np.abs(
                    pred[:, a] - pred[:, b]
                )
                improve = (
                    pair_good
                    & (
                        pair_score
                        < chunk_best_score
                    )
                )
                chunk_best_score[improve] = pair_score[
                    improve
                ]
                chunk_best_slope[improve] = pair_slope[
                    improve
                ]
                chunk_best_prediction[improve] = (
                    pair_prediction[improve]
                )
                chunk_best_disagreement[improve] = (
                    pair_disagreement[improve]
                )
                chunk_two_sided |= pair_good

        two_sided[start:stop] = chunk_two_sided
        best_score[start:stop] = chunk_best_score
        best_slope[start:stop] = chunk_best_slope
        best_prediction[start:stop] = (
            chunk_best_prediction
        )
        best_pair_disagreement[start:stop] = (
            chunk_best_disagreement
        )

    return (
        local_z_range,
        two_sided,
        best_score,
        best_slope,
        best_prediction,
        best_pair_disagreement,
    )



def select_two_sided_break_membership(
    source_indices: np.ndarray,
    current_membership: np.ndarray,
    structural_model_indices: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    bounds: tuple[float, float, float, float],
    origin_x: float,
    origin_y: float,
) -> np.ndarray:
    """Recover final class-2 density across real banks/convex crests.

    Memory-safe Turbo implementation. Classification semantics are unchanged:
    the exact same structural gates, directional trusted-plane evidence,
    two-sided agreement, surface-centred ranking and spike audits are used.
    """
    if (
        not TWO_SIDED_BREAK_MEMBERSHIP
        or source_indices.size == 0
        or current_membership.size < 3
        or structural_model_indices.size < 6
    ):
        return np.unique(current_membership)

    unresolved = source_indices[
        ~np.isin(source_indices, current_membership)
    ]
    if unresolved.size == 0:
        return np.unique(current_membership)

    (
        flat_target,
        steep_target,
        candidate_spacing,
        runtime_iterations,
        spacing,
        density,
    ) = adaptive_membership_parameters(
        source_indices, x, y
    )

    original_unresolved_count = unresolved.size

    density_need_positions = turbo2_density_need_positions(
        unresolved,
        current_membership,
        x,
        y,
        bounds,
        steep_target,
    )
    if density_need_positions.size == 0:
        return np.unique(current_membership)

    unresolved = unresolved[density_need_positions]

    model_tree = cKDTree(
        np.column_stack(
            (
                x[structural_model_indices],
                y[structural_model_indices],
            )
        )
    )

    model_distance, model_position = tree_query(
        model_tree,
        np.column_stack(
            (x[unresolved], y[unresolved])
        ),
        k=1,
    )
    model_distance = np.asarray(
        model_distance, dtype=np.float64
    )
    model_position = np.asarray(
        model_position, dtype=np.int64
    )
    nearest_global = structural_model_indices[
        np.clip(
            model_position,
            0,
            structural_model_indices.size - 1,
        )
    ]
    vertical = np.maximum(
        np.abs(
            z[unresolved] - z[nearest_global]
        )
        - TRUSTED_CONNECTION_VERTICAL_SLACK,
        0.0,
    )
    connection_slope = np.degrees(
        np.arctan2(
            vertical,
            np.maximum(model_distance, 1.0e-6),
        )
    )

    tin_residual, tin_slope, _ = (
        interpolate_tin_membership_context(
            unresolved,
            structural_model_indices,
            x,
            y,
            z,
            bounds,
        )
    )
    tin_residual = tin_residual.astype(
        np.float64, copy=False
    )
    tin_slope = tin_slope.astype(
        np.float64, copy=False
    )

    # These are EXACT gates from the final eligibility expression in the
    # previous implementation. Points that fail here could never become class 2,
    # so there is no reason to allocate 20-neighbour evidence arrays for them.
    cheap_gate = (
        np.isfinite(model_distance)
        & (
            model_distance
            <= BREAK_MEMBERSHIP_MAX_MODEL_DISTANCE
        )
        & np.isfinite(tin_residual)
        & (
            np.abs(tin_residual)
            <= BREAK_MEMBERSHIP_MAX_ABS_STRUCTURAL_TIN_RESIDUAL
        )
        & (
            connection_slope
            <= BREAK_MEMBERSHIP_MAX_CONNECTION_SLOPE_DEG
        )
    )
    evidence_positions = np.flatnonzero(cheap_gate)
    if evidence_positions.size == 0:
        return np.unique(current_membership)

    evidence_indices = unresolved[evidence_positions]
    evidence_tin_slope = tin_slope[evidence_positions]

    evidence_started = time.perf_counter()
    (
        local_z_range,
        two_sided,
        best_score,
        best_slope,
        best_prediction,
        best_pair_disagreement,
    ) = chunked_two_sided_break_evidence(
        evidence_indices,
        evidence_tin_slope,
        structural_model_indices,
        model_tree,
        x,
        y,
        z,
    )
    if PERFORMANCE_STAGE_TIMINGS:
        log(
            "      TIMING chunked two-sided evidence: "
            f"{time.perf_counter() - evidence_started:.1f} s "
            f"for {evidence_indices.size:,} density-needed candidates "
            f"from {original_unresolved_count:,} original unresolved points"
        )

    break_context = (
        (
            local_z_range
            >= BREAK_MEMBERSHIP_MIN_LOCAL_Z_RANGE
        )
        & (
            (
                evidence_tin_slope
                >= BREAK_MEMBERSHIP_MIN_TERRAIN_SLOPE_DEG
            )
            | (
                best_slope
                >= BREAK_MEMBERSHIP_MIN_TERRAIN_SLOPE_DEG
            )
        )
    )
    eligible = two_sided & break_context
    candidate_positions = np.flatnonzero(eligible)
    if candidate_positions.size == 0:
        return np.unique(current_membership)

    candidate_slope_full = np.maximum(
        evidence_tin_slope,
        best_slope,
    )
    centre_sw = np.clip(
        candidate_slope_full
        / BREAK_MEMBERSHIP_CENTER_FULL_SLOPE_DEG,
        0.0,
        1.0,
    )
    centre_positive = (
        BREAK_MEMBERSHIP_CENTER_FLAT_POSITIVE
        + centre_sw
        * (
            BREAK_MEMBERSHIP_CENTER_STEEP_POSITIVE
            - BREAK_MEMBERSHIP_CENTER_FLAT_POSITIVE
        )
    )
    centre_negative = (
        BREAK_MEMBERSHIP_CENTER_FLAT_NEGATIVE
        + centre_sw
        * (
            BREAK_MEMBERSHIP_CENTER_STEEP_NEGATIVE
            - BREAK_MEMBERSHIP_CENTER_FLAT_NEGATIVE
        )
    )
    signed_centre_residual = (
        z[evidence_indices] - best_prediction
    )

    centre_good = (
        np.isfinite(best_prediction)
        & np.isfinite(signed_centre_residual)
        & (
            signed_centre_residual
            >= -centre_negative
        )
        & (
            signed_centre_residual
            <= centre_positive
        )
    )
    candidate_positions = candidate_positions[
        centre_good[candidate_positions]
    ]
    if candidate_positions.size == 0:
        log(
            "      Two-sided break/crest membership: "
            "no surface-centred representatives survived"
        )
        return np.unique(current_membership)

    candidates = evidence_indices[
        candidate_positions
    ]
    candidate_signed_residual = (
        signed_centre_residual[candidate_positions]
    )
    candidate_slope = candidate_slope_full[
        candidate_positions
    ]
    candidate_centre_scale = np.where(
        candidate_signed_residual >= 0.0,
        centre_positive[candidate_positions],
        centre_negative[candidate_positions],
    )
    centre_score = (
        np.abs(candidate_signed_residual)
        / np.maximum(
            candidate_centre_scale, 1.0e-9
        )
    )
    pair_disagreement_score = (
        best_pair_disagreement[candidate_positions]
        / max(
            BREAK_MEMBERSHIP_MAX_TWO_SIDE_PREDICTION_DISAGREEMENT,
            1.0e-9,
        )
    )

    peer_keep, peer_penalty = (
        break_membership_peer_coherence(
            candidates,
            candidate_signed_residual,
            x,
            y,
        )
    )
    if not np.all(peer_keep):
        rejected_peer = int(
            np.count_nonzero(~peer_keep)
        )
        log(
            "      Surface-centred slope representative QC: "
            f"rejected {rejected_peer:,} "
            "isolated peer-residual returns"
        )
        candidates = candidates[peer_keep]
        candidate_positions = candidate_positions[
            peer_keep
        ]
        candidate_signed_residual = (
            candidate_signed_residual[peer_keep]
        )
        candidate_slope = candidate_slope[peer_keep]
        centre_score = centre_score[peer_keep]
        pair_disagreement_score = (
            pair_disagreement_score[peer_keep]
        )
        peer_penalty = peer_penalty[peer_keep]

    if candidates.size == 0:
        return np.unique(current_membership)

    candidate_score = (
        centre_score
        + 0.30 * pair_disagreement_score
        + 0.35 * peer_penalty
    )

    keys = packed_grid_keys(
        x[candidates],
        y[candidates],
        candidate_spacing,
        origin_x,
        origin_y,
    )
    order = np.lexsort(
        (
            candidate_score,
            keys["y"],
            keys["x"],
        )
    )
    ordered_keys = keys[order]
    first = np.empty(order.size, dtype=bool)
    first[0] = True
    first[1:] = (
        (
            ordered_keys["x"][1:]
            != ordered_keys["x"][:-1]
        )
        | (
            ordered_keys["y"][1:]
            != ordered_keys["y"][:-1]
        )
    )
    kept = order[first]
    candidates = candidates[kept]
    candidate_score = candidate_score[kept]
    candidate_slope = candidate_slope[kept]

    membership = np.unique(current_membership)
    base_membership = membership.copy()
    active = np.ones(
        candidates.size, dtype=bool
    )
    max_iterations = min(
        BREAK_MEMBERSHIP_MAX_ITERATIONS,
        max(
            runtime_iterations,
            GROUND_MEMBERSHIP_MAX_ITERATIONS,
        ),
    )

    turbo_break_iterations = min(
        max_iterations,
        TURBO_MAX_BREAK_MEMBERSHIP_ITERATIONS,
    )
    if turbo_break_iterations < max_iterations:
        log(
            "      Turbo two-sided membership: "
            f"capping {max_iterations} requested iterations at "
            f"{turbo_break_iterations}"
        )

    for iteration in range(
        1, turbo_break_iterations + 1
    ):
        pool_positions = np.flatnonzero(active)
        if pool_positions.size == 0:
            break

        density_tin, density_xy = build_planar_tin(
            membership, x, y, bounds
        )
        candidate_parts: list[np.ndarray] = []
        simplex_parts: list[np.ndarray] = []
        merit_parts: list[np.ndarray] = []

        for start in range(
            0,
            pool_positions.size,
            TEST_CHUNK_SIZE,
        ):
            positions_local = pool_positions[
                start : start + TEST_CHUNK_SIZE
            ]
            part = candidates[positions_local]
            qxy = np.column_stack(
                (x[part], y[part])
            )
            simplex = density_tin.find_simplex(qxy)
            inside = simplex >= 0
            active[
                positions_local[~inside]
            ] = False
            if not np.any(inside):
                continue

            positions_local = positions_local[
                inside
            ]
            qxy = qxy[inside]
            simplex = simplex[inside]
            vertices = density_tin.simplices[
                simplex
            ]
            tri_xy = density_xy[vertices]

            edge01 = np.linalg.norm(
                tri_xy[:, 0] - tri_xy[:, 1],
                axis=1,
            )
            edge12 = np.linalg.norm(
                tri_xy[:, 1] - tri_xy[:, 2],
                axis=1,
            )
            edge20 = np.linalg.norm(
                tri_xy[:, 2] - tri_xy[:, 0],
                axis=1,
            )
            current_edge = np.maximum.reduce(
                (edge01, edge12, edge20)
            )
            nearest_vertex = np.min(
                np.linalg.norm(
                    tri_xy - qxy[:, None, :],
                    axis=2,
                ),
                axis=1,
            )
            slope_weight = np.clip(
                candidate_slope[positions_local]
                / GROUND_MEMBERSHIP_FULL_SLOPE_DEG,
                0.0,
                1.0,
            )
            target_edge = (
                flat_target
                + slope_weight
                * (steep_target - flat_target)
            )
            too_close = (
                nearest_vertex
                < GROUND_MEMBERSHIP_MIN_VERTEX_FRACTION
                * target_edge
            )
            complete = current_edge <= target_edge
            active[
                positions_local[
                    too_close | complete
                ]
            ] = False
            useful = ~too_close & ~complete
            if not np.any(useful):
                continue

            positions_local = positions_local[
                useful
            ]
            simplex = simplex[useful]
            nearest_vertex = nearest_vertex[
                useful
            ]
            target_edge = target_edge[useful]
            merit = (
                candidate_score[positions_local]
                - 0.18
                * np.clip(
                    nearest_vertex
                    / np.maximum(
                        target_edge, 1.0e-9
                    ),
                    0.0,
                    2.0,
                )
            )
            candidate_parts.append(
                positions_local
            )
            simplex_parts.append(simplex)
            merit_parts.append(merit)

        if not candidate_parts:
            break

        positions_all = np.concatenate(
            candidate_parts
        )
        simplices_all = np.concatenate(
            simplex_parts
        )
        merit_all = np.concatenate(merit_parts)
        order = np.lexsort(
            (merit_all, simplices_all)
        )
        ordered_simplices = simplices_all[
            order
        ]
        first = np.empty(
            order.size, dtype=bool
        )
        first[0] = True
        first[1:] = (
            ordered_simplices[1:]
            != ordered_simplices[:-1]
        )
        chosen_positions = positions_all[
            order[first]
        ]
        chosen = candidates[
            chosen_positions
        ]
        if chosen.size == 0:
            break

        tree = cKDTree(
            np.column_stack(
                (x[membership], y[membership])
            )
        )
        d, p = tree_query(
            tree,
            np.column_stack(
                (x[chosen], y[chosen])
            ),
            k=1,
        )
        d = np.asarray(
            d, dtype=np.float64
        )
        p = np.asarray(
            p, dtype=np.int64
        )
        neighbour = membership[
            np.clip(
                p,
                0,
                membership.size - 1,
            )
        ]
        dz = np.abs(
            z[chosen] - z[neighbour]
        )
        edge_slope = np.degrees(
            np.arctan2(
                dz,
                np.maximum(d, 1.0e-6),
            )
        )
        safe = ~(
            (
                d
                <= BREAK_MEMBERSHIP_SPIKE_SHORT_EDGE
            )
            & (
                dz
                >= BREAK_MEMBERSHIP_SPIKE_MIN_DZ
            )
            & (
                edge_slope
                >= BREAK_MEMBERSHIP_SPIKE_MAX_EDGE_SLOPE_DEG
            )
        )
        rejected = int(
            np.count_nonzero(~safe)
        )
        chosen_positions = chosen_positions[
            safe
        ]
        chosen = chosen[safe]
        active[
            positions_all[order[first]]
        ] = False

        if rejected:
            log(
                f"      Two-sided break membership iteration "
                f"{iteration}: rejected {rejected:,} "
                "spike-risk candidates"
            )
        if chosen.size == 0:
            continue

        membership = np.concatenate(
            (membership, chosen)
        )
        log(
            f"      Two-sided break membership iteration "
            f"{iteration}: +{chosen.size:,} ground returns"
        )

    added = membership[
        ~np.isin(
            membership, base_membership
        )
    ]

    low_audited = (
        audit_added_low_spike_representatives(
            base_membership,
            added,
            x,
            y,
            z,
        )
    )
    pair_audited = (
        audit_two_sided_membership_spikes(
            base_membership,
            low_audited,
            x,
            y,
            z,
        )
    )
    tin_audited = (
        audit_two_sided_membership_tin_geometry(
            base_membership,
            pair_audited,
            structural_model_indices,
            x,
            y,
            z,
            bounds,
        )
    )
    final_membership = np.unique(
        np.concatenate(
            (base_membership, tin_audited)
        )
    )
    removed_low = int(
        added.size - low_audited.size
    )
    removed_pair = int(
        low_audited.size - pair_audited.size
    )
    removed_tin = int(
        pair_audited.size - tin_audited.size
    )

    if np.isfinite(spacing):
        log(
            "      FPL auto-density profile: "
            f"spacing ~{spacing:.3f} m "
            f"(~{metres_to_usft(spacing):.3f} USft), "
            f"density ~{density:.1f} pts/m^2; "
            f"target edges "
            f"{flat_target:.2f}/{steep_target:.2f} m"
        )
    log(
        "      Two-sided break/crest membership: "
        f"+{tin_audited.size:,} retained"
        + (
            f", {removed_low:,} removed as "
            "isolated low representatives"
            if removed_low
            else ""
        )
        + (
            f", {removed_pair:,} removed by pair audit"
            if removed_pair
            else ""
        )
        + (
            f", {removed_tin:,} removed by Delaunay TIN audit"
            if removed_tin
            else ""
        )
    )
    return final_membership


def select_adaptive_ground_membership(
    source_indices: np.ndarray,
    model_indices: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    bounds: tuple[float, float, float, float],
    origin_x: float,
    origin_y: float,
    work_indices: np.ndarray | None = None,
    work_plane_context: PlaneContext | None = None,
    trusted_anchor_indices: np.ndarray | None = None,
    reference_surface: TinSurface | None = None,
    reference_simplex_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Progressively restore a dense, one-surface-thick class-2 set.

    Eligible returns first pass a narrow perpendicular-distance envelope
    around the clean reference TIN. Starting with its representatives, one
    best-fitting return per still-large triangle is then added and the density
    TIN rebuilt. Iteration stops locally once the slope-dependent target edge
    is reached. This reproduces progressive, adaptive density rather than a
    fixed one-point-per-metre quota.
    """
    (
        runtime_flat_target,
        runtime_steep_target,
        runtime_candidate_spacing,
        runtime_iterations,
        measured_spacing,
        measured_density,
    ) = adaptive_membership_parameters(source_indices, x, y)

    if np.isfinite(measured_spacing):
        log(
            "      FPL auto-density profile: "
            f"spacing ~{measured_spacing:.3f} m "
            f"(~{metres_to_usft(measured_spacing):.3f} USft), "
            f"density ~{measured_density:.1f} pts/m^2; "
            f"membership edges "
            f"{runtime_flat_target:.2f}/{runtime_steep_target:.2f} m; "
            f"candidate spacing {runtime_candidate_spacing:.2f} m; "
            f"iterations {runtime_iterations}"
        )

    residual, slope, max_edge = interpolate_tin_membership_context(
        source_indices,
        model_indices,
        x,
        y,
        z,
        bounds,
        reference_surface,
        reference_simplex_ids,
    )
    valid = (
        np.isfinite(residual)
        & np.isfinite(max_edge)
        & (slope <= MAX_TERRAIN_ANGLE_DEG)
    )
    if not np.any(valid):
        return np.empty(0, dtype=np.int64)

    # Membership is intentionally LOCAL to the accepted structural terrain.
    # A point in the middle of a large TIN bridge must not become class 2 just
    # because its elevation happens to lie within a few centimetres of the
    # interpolated triangle. Genuine sparse ground should first enter the
    # structural model via progressive trusted recovery above.
    model_tree = cKDTree(
        np.column_stack((x[model_indices], y[model_indices]))
    )
    source_model_distance, _ = tree_query(
        model_tree,
        np.column_stack((x[source_indices], y[source_indices])),
        k=1,
    )
    source_model_distance = np.asarray(
        source_model_distance, dtype=np.float64
    )
    del trusted_anchor_indices

    slope_weight = np.clip(
        slope / GROUND_MEMBERSHIP_FULL_SLOPE_DEG,
        0.0,
        1.0,
    )
    upper = (
        GROUND_MEMBERSHIP_FLAT_UPPER
        + slope_weight
        * (
            GROUND_MEMBERSHIP_STEEP_UPPER
            - GROUND_MEMBERSHIP_FLAT_UPPER
        )
    )
    lower = (
        GROUND_MEMBERSHIP_FLAT_LOWER
        + slope_weight
        * (
            GROUND_MEMBERSHIP_STEEP_LOWER
            - GROUND_MEMBERSHIP_FLAT_LOWER
        )
    )

    # TerraScan reduces its iteration angle toward zero inside small
    # triangles. We tighten the membership envelope with the same edge cue,
    # but retain a non-zero floor because this final test must reconstruct all
    # points accepted during earlier, larger-triangle iterations.
    edge_weight = np.clip(
        max_edge / GROUND_MEMBERSHIP_REDUCE_EDGE,
        0.0,
        1.0,
    )
    edge_scale = (
        GROUND_MEMBERSHIP_SMALL_TRIANGLE_SCALE
        + (1.0 - GROUND_MEMBERSHIP_SMALL_TRIANGLE_SCALE)
        * edge_weight
    )
    upper *= edge_scale
    lower *= edge_scale

    accepted = (
        valid
        & (residual >= -lower)
        & (residual <= upper)
        & (source_model_distance <= GROUND_MEMBERSHIP_MAX_MODEL_DISTANCE)
    )
    supported_gap = np.zeros(source_indices.size, dtype=bool)

    if (
        SUPPORTED_TRIANGLE_REFINEMENT
        and work_indices is not None
        and work_plane_context is not None
        and work_indices.size
    ):
        # source_indices and work_indices are both unique global indices. Map
        # the work minima into the source-aligned TIN context without a dense
        # global lookup array.
        work_positions = np.searchsorted(source_indices, work_indices)
        safe_positions = np.clip(
            work_positions, 0, source_indices.size - 1
        )
        mapped = source_indices[safe_positions] == work_indices
        plane_z, plane_rmse, neighbour_count, residual_support = (
            work_plane_context
        )
        work_plane_residual = z[work_indices] - plane_z
        position_slope_weight = np.clip(
            slope[safe_positions]
            / GROUND_MEMBERSHIP_FULL_SLOPE_DEG,
            0.0,
            1.0,
        )
        position_target_edge = (
            runtime_flat_target
            + position_slope_weight
            * (
                runtime_steep_target
                - runtime_flat_target
            )
        )
        _nsm = noise_scale_multiplier()
        direct_support = (
            (plane_rmse <= SUPPORTED_TRIANGLE_MAX_RMSE * _nsm)
            & (
                np.abs(work_plane_residual)
                <= SUPPORTED_TRIANGLE_PLANE_TOLERANCE * _nsm
            )
        )
        repeated_support = (
            (
                plane_rmse
                <= SUPPORTED_TRIANGLE_SUPPORTED_MAX_RMSE * _nsm
            )
            & (
                residual_support
                >= SUPPORTED_TRIANGLE_MIN_SUPPORT
            )
            & (
                np.abs(work_plane_residual)
                <= SUPPORTED_TRIANGLE_MAX_REFERENCE_DISTANCE * _nsm
            )
        )
        # Large-triangle support may extend a little farther than ordinary
        # density membership, but still has to be local to the CURRENT final
        # terrain model. It is not allowed to seed a detached ground island.
        work_model_near = (
            source_model_distance[safe_positions]
            <= GROUND_MEMBERSHIP_GAP_MAX_MODEL_DISTANCE
        )

        gap_eligible = (
            mapped
            & work_model_near
            & valid[safe_positions]
            & (
                neighbour_count
                >= UPPER_SURFACE_MIN_NEIGHBOURS
            )
            & (direct_support | repeated_support)
            & (
                np.abs(residual[safe_positions])
                <= SUPPORTED_TRIANGLE_MAX_REFERENCE_DISTANCE * _nsm
            )
            & (
                max_edge[safe_positions]
                > (
                    SUPPORTED_TRIANGLE_MIN_EDGE_FACTOR
                    * position_target_edge
                )
            )
        )
        supported_positions = safe_positions[gap_eligible]
        supported_gap[supported_positions] = True
        accepted[supported_positions] = True
        if supported_positions.size:
            log(
                "      Globally supported large-triangle candidates: "
                f"{supported_positions.size:,}"
            )

    accepted_positions = np.flatnonzero(accepted)
    if accepted_positions.size == 0:
        return model_indices.copy()

    # Remove only sub-15 cm duplication before iterative triangulation. Keep
    # the return nearest the clean surface in each such cell.
    accepted_indices = source_indices[accepted_positions]
    side_tolerance = np.where(
        residual[accepted_positions] >= 0,
        upper[accepted_positions],
        lower[accepted_positions],
    )
    gap_in_pool = supported_gap[accepted_positions]
    side_tolerance[gap_in_pool] = np.maximum(
        side_tolerance[gap_in_pool],
        SUPPORTED_TRIANGLE_MAX_REFERENCE_DISTANCE,
    )
    surface_score = (
        np.abs(residual[accepted_positions])
        / np.maximum(side_tolerance, 1.0e-9)
    )
    fine_keys = packed_grid_keys(
        x[accepted_indices],
        y[accepted_indices],
        runtime_candidate_spacing,
        origin_x,
        origin_y,
    )
    order = np.lexsort(
        (surface_score, fine_keys["y"], fine_keys["x"])
    )
    ordered_keys = fine_keys[order]
    first = np.empty(order.size, dtype=bool)
    first[0] = True
    first[1:] = (
        (ordered_keys["x"][1:] != ordered_keys["x"][:-1])
        | (ordered_keys["y"][1:] != ordered_keys["y"][:-1])
    )
    kept = order[first]
    pool_indices = accepted_indices[kept]
    pool_residual = residual[accepted_positions[kept]]
    pool_slope = slope[accepted_positions[kept]]
    pool_tolerance = side_tolerance[kept]

    active = ~np.isin(pool_indices, model_indices)
    membership = model_indices.copy()

    turbo_runtime_iterations = min(
        runtime_iterations,
        TURBO_MAX_ADAPTIVE_MEMBERSHIP_ITERATIONS,
    )
    if turbo_runtime_iterations < runtime_iterations:
        log(
            "      Turbo adaptive membership: "
            f"capping {runtime_iterations} requested iterations at "
            f"{turbo_runtime_iterations}"
        )

    for iteration in range(1, turbo_runtime_iterations + 1):
        pool_positions = np.flatnonzero(active)
        if pool_positions.size == 0:
            break

        density_tin, density_xy = build_planar_tin(
            membership, x, y, bounds
        )
        candidate_parts: list[np.ndarray] = []
        simplex_parts: list[np.ndarray] = []
        merit_parts: list[np.ndarray] = []

        for start in range(0, pool_positions.size, TEST_CHUNK_SIZE):
            positions = pool_positions[start : start + TEST_CHUNK_SIZE]
            part = pool_indices[positions]
            query_xy = np.column_stack((x[part], y[part]))
            simplex = density_tin.find_simplex(query_xy)
            inside = simplex >= 0
            active[positions[~inside]] = False
            if not np.any(inside):
                continue

            positions = positions[inside]
            query_xy = query_xy[inside]
            simplex = simplex[inside]
            vertices = density_tin.simplices[simplex]
            tri_xy = density_xy[vertices]
            edge01 = np.linalg.norm(
                tri_xy[:, 0] - tri_xy[:, 1], axis=1
            )
            edge12 = np.linalg.norm(
                tri_xy[:, 1] - tri_xy[:, 2], axis=1
            )
            edge20 = np.linalg.norm(
                tri_xy[:, 2] - tri_xy[:, 0], axis=1
            )
            current_edge = np.maximum.reduce((edge01, edge12, edge20))
            nearest_vertex = np.min(
                np.linalg.norm(
                    tri_xy - query_xy[:, None, :], axis=2
                ),
                axis=1,
            )
            slope_weight = np.clip(
                pool_slope[positions]
                / GROUND_MEMBERSHIP_FULL_SLOPE_DEG,
                0.0,
                1.0,
            )
            target_edge = (
                runtime_flat_target
                + slope_weight
                * (
                    runtime_steep_target
                    - runtime_flat_target
                )
            )
            too_close = (
                nearest_vertex
                < (
                    GROUND_MEMBERSHIP_MIN_VERTEX_FRACTION
                    * target_edge
                )
            )
            locally_complete = current_edge <= target_edge
            active[positions[too_close | locally_complete]] = False
            useful = (
                ~locally_complete
                & ~too_close
            )
            if not np.any(useful):
                continue

            positions = positions[useful]
            simplex = simplex[useful]
            nearest_vertex = nearest_vertex[useful]
            target_edge = target_edge[useful]
            # Prefer a point close to the reference surface, but among near-
            # equals favour one away from existing vertices so it improves
            # the triangle rather than merely duplicating a vertex.
            merit = (
                np.abs(pool_residual[positions])
                / np.maximum(pool_tolerance[positions], 1.0e-9)
                - 0.15
                * np.clip(nearest_vertex / target_edge, 0.0, 2.0)
            )
            candidate_parts.append(positions)
            simplex_parts.append(simplex)
            merit_parts.append(merit)

        if not candidate_parts:
            break

        candidate_positions = np.concatenate(candidate_parts)
        candidate_simplices = np.concatenate(simplex_parts)
        candidate_merit = np.concatenate(merit_parts)
        order = np.lexsort((candidate_merit, candidate_simplices))
        ordered_simplices = candidate_simplices[order]
        first = np.empty(order.size, dtype=bool)
        first[0] = True
        first[1:] = ordered_simplices[1:] != ordered_simplices[:-1]
        chosen_positions = candidate_positions[order[first]]
        chosen = pool_indices[chosen_positions]
        if chosen.size == 0:
            break

        active[chosen_positions] = False
        membership = np.concatenate((membership, chosen))
        log(
            f"      Adaptive membership iteration {iteration}: "
            f"+{chosen.size:,} ground returns"
        )

    return np.unique(membership)



def detect_structural_breakline_vertices(
    model_indices: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    bounds: tuple[float, float, float, float],
) -> np.ndarray:
    """Return accepted structural vertices lying on sharp terrain breaks.

    The input model has already passed canopy/lower-plane and spike pruning.
    Breaklines are therefore detected from slope contrast between adjacent TIN
    triangles, not from raw source points. Preserving these vertices keeps the
    final smooth TIN faithful to genuine crests and toes without globally
    making the representative surface rougher.
    """
    if not PRESERVE_TERRAIN_BREAKLINES or model_indices.size < 3:
        return np.empty(0, dtype=np.int64)

    try:
        tin, model_xy, model_z, _ = build_tin(model_indices, x, y, z, bounds)
    except InsufficientGroundSupport:
        return np.empty(0, dtype=np.int64)

    vertices = tin.simplices
    tri_xy = model_xy[vertices]
    tri_z = model_z[vertices]
    p0 = np.column_stack((tri_xy[:, 0, :], tri_z[:, 0]))
    p1 = np.column_stack((tri_xy[:, 1, :], tri_z[:, 1]))
    p2 = np.column_stack((tri_xy[:, 2, :], tri_z[:, 2]))
    normals = np.cross(p1 - p0, p2 - p0)
    length = np.linalg.norm(normals, axis=1)
    usable = length > 1.0e-12
    length[~usable] = 1.0
    normals /= length[:, None]
    normals[normals[:, 2] < 0] *= -1.0
    slopes = np.degrees(
        np.arctan2(
            np.linalg.norm(normals[:, :2], axis=1),
            np.maximum(normals[:, 2], 1.0e-12),
        )
    )
    slopes[~usable] = np.nan

    n = model_indices.size
    min_slope = np.full(n, np.inf, dtype=np.float64)
    max_slope = np.full(n, -np.inf, dtype=np.float64)
    counts = np.zeros(n, dtype=np.int16)
    for col in range(3):
        v = vertices[:, col]
        valid = (v >= 0) & (v < n) & np.isfinite(slopes)
        if not np.any(valid):
            continue
        vv = v[valid]
        ss = slopes[valid]
        np.minimum.at(min_slope, vv, ss)
        np.maximum.at(max_slope, vv, ss)
        np.add.at(counts, vv, 1)

    contrast = max_slope - min_slope
    keep = (
        (counts >= BREAKLINE_PRESERVE_MIN_TRIANGLES)
        & np.isfinite(contrast)
        & (max_slope >= BREAKLINE_PRESERVE_MIN_MAX_SLOPE_DEG)
        & (contrast >= BREAKLINE_PRESERVE_MIN_SLOPE_CONTRAST_DEG)
    )
    return model_indices[keep]


def select_smooth_model_representatives(
    source_indices: np.ndarray,
    work_indices: np.ndarray,
    rough_model_indices: np.ndarray,
    backbone_indices: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    bounds: tuple[float, float, float, float],
    origin_x: float,
    origin_y: float,
    support_tree: cKDTree | None = None,
) -> np.ndarray:
    """
    Choose one locally smooth class-2 representative in each terrain cell.

    The adaptive TIN first determines where terrain exists. A robust local
    plane through the lower-envelope points then supplies the target elevation
    in every cell. Of all eligible source returns in a supported cell, only the
    point closest to that plane is retained. The permitted positive residual
    increases with slope, preventing hillside gaps without thickening flat
    ground.
    """
    if work_indices.size < 3:
        return rough_model_indices

    breakline_indices = detect_structural_breakline_vertices(
        rough_model_indices, x, y, z, bounds
    )
    if breakline_indices.size:
        log(
            f"      Structural terrain break vertices protected: "
            f"{breakline_indices.size:,}"
        )

    representative_work_indices = lowest_per_grid(
        source_indices,
        x,
        y,
        z,
        REPRESENTATIVE_FINE_SPACING,
        origin_x,
        origin_y,
    )
    if representative_work_indices.size < 3:
        return rough_model_indices

    # Representative selection may refine an existing trusted surface but may
    # not use the lower-envelope work minima as independent evidence.
    representative_model_tree = cKDTree(
        np.column_stack((x[rough_model_indices], y[rough_model_indices]))
    )
    (
        plane_z,
        plane_rmse,
        neighbour_count,
        residual_support,
    ) = local_plane_support(
        representative_work_indices,
        rough_model_indices,
        x,
        y,
        z,
        representative_model_tree,
    )
    representative_xy = np.column_stack(
        (x[representative_work_indices], y[representative_work_indices])
    )
    representative_model_distance, _ = tree_query(
        representative_model_tree, representative_xy, k=1
    )
    # v2 deliberately does not require representatives to remain close to
    # the original PTD backbone.  The rough_model_indices already contain only
    # accepted connected terrain, so proximity to that CURRENT model is the
    # correct guard and allows smooth terrain recovery through wide woodland.
    del backbone_indices
    work_residual = z[representative_work_indices] - plane_z
    directly_supported = (
        (plane_rmse <= REPRESENTATIVE_MAX_RMSE)
        & (
            np.abs(work_residual)
            <= REPRESENTATIVE_PLANE_TOLERANCE
        )
    )
    repeatedly_supported = (
        (plane_rmse <= REPRESENTATIVE_SUPPORTED_MAX_RMSE)
        & (
            residual_support
            >= REPRESENTATIVE_MIN_SUPPORT
        )
    )
    supported_cell = (
        np.isfinite(plane_z)
        & (neighbour_count >= UPPER_SURFACE_MIN_NEIGHBOURS)
        & (directly_supported | repeatedly_supported)
        & (
            representative_model_distance
            <= REPRESENTATIVE_MAX_MODEL_DISTANCE
        )
    )

    local_slope = interpolate_tin_slope(
        rough_model_indices,
        representative_work_indices,
        x,
        y,
        z,
        bounds,
    )
    slope_weight = np.clip(
        local_slope / REPRESENTATIVE_FULL_SLOPE_DEG,
        0.0,
        1.0,
    )
    upper_tolerance = (
        REPRESENTATIVE_FLAT_UPPER_TOLERANCE
        + slope_weight
        * (
            REPRESENTATIVE_STEEP_UPPER_TOLERANCE
            - REPRESENTATIVE_FLAT_UPPER_TOLERANCE
        )
    )

    work_keys = packed_grid_keys(
        x[representative_work_indices],
        y[representative_work_indices],
        REPRESENTATIVE_FINE_SPACING,
        origin_x,
        origin_y,
    )
    source_keys = packed_grid_keys(
        x[source_indices],
        y[source_indices],
        REPRESENTATIVE_FINE_SPACING,
        origin_x,
        origin_y,
    )
    cell_position = np.searchsorted(work_keys, source_keys)
    cell_position = np.clip(
        cell_position,
        0,
        representative_work_indices.size - 1,
    )
    same_cell = (
        (work_keys["x"][cell_position] == source_keys["x"])
        & (work_keys["y"][cell_position] == source_keys["y"])
    )

    residual = z[source_indices] - plane_z[cell_position]
    eligible = (
        same_cell
        & supported_cell[cell_position]
        & np.isfinite(residual)
        & (residual >= -REPRESENTATIVE_LOWER_TOLERANCE)
        & (residual <= upper_tolerance[cell_position])
    )
    candidates = source_indices[eligible]
    candidate_cells = cell_position[eligible]
    candidate_plane_score = np.abs(residual[eligible])
    if candidates.size:
        # Independent structural-TIN gate. The local representative plane can
        # be biased across a sharp break; it may rank candidates, but it may
        # not move the terrain surface away from the already validated TIN.
        rep_tin_residual, rep_tin_slope, _ = interpolate_tin_membership_context(
            candidates,
            rough_model_indices,
            x,
            y,
            z,
            bounds,
        )
        rep_tin_residual = rep_tin_residual.astype(np.float64, copy=False)
        rep_tin_slope = rep_tin_slope.astype(np.float64, copy=False)
        rep_sw = np.clip(
            rep_tin_slope / REPRESENTATIVE_TIN_FULL_SLOPE_DEG,
            0.0,
            1.0,
        )
        rep_upper = (
            REPRESENTATIVE_TIN_FLAT_UPPER
            + rep_sw
            * (REPRESENTATIVE_TIN_STEEP_UPPER - REPRESENTATIVE_TIN_FLAT_UPPER)
        )
        rep_lower = (
            REPRESENTATIVE_TIN_FLAT_LOWER
            + rep_sw
            * (REPRESENTATIVE_TIN_STEEP_LOWER - REPRESENTATIVE_TIN_FLAT_LOWER)
        )
        tin_ok = (
            np.isfinite(rep_tin_residual)
            & (rep_tin_residual >= -rep_lower)
            & (rep_tin_residual <= rep_upper)
        )
        candidates = candidates[tin_ok]
        candidate_cells = candidate_cells[tin_ok]
        candidate_plane_score = candidate_plane_score[tin_ok]
        rep_tin_residual = rep_tin_residual[tin_ok]

    if candidates.size:
        # Plane closeness remains the primary smoothing score; a small TIN
        # residual term breaks ties in favour of the accepted structural
        # surface rather than a nearby vegetation layer.
        score = candidate_plane_score + 0.35 * np.abs(rep_tin_residual)
        order = np.lexsort((score, candidate_cells))
        ordered_cells = candidate_cells[order]
        first = np.empty(order.size, dtype=bool)
        first[0] = True
        first[1:] = ordered_cells[1:] != ordered_cells[:-1]
        selected = order[first]
        representatives = candidates[selected]
        representative_score = score[selected]
        representative_slope = local_slope[candidate_cells[selected]]

        # Keep the full 0.60 m density on slopes. On flatter terrain, retain
        # the best plane-fitting point in each 0.80 m cell. This makes density
        # adaptive without ever creating a vertically thick class-2 band.
        dense = representative_slope >= REPRESENTATIVE_DENSE_SLOPE_DEG
        flat_indices = representatives[~dense]
        flat_score = representative_score[~dense]
        if flat_indices.size:
            flat_keys = packed_grid_keys(
                x[flat_indices],
                y[flat_indices],
                REPRESENTATIVE_FLAT_SPACING,
                origin_x,
                origin_y,
            )
            flat_order = np.lexsort(
                (flat_score, flat_keys["y"], flat_keys["x"])
            )
            ordered_flat_keys = flat_keys[flat_order]
            flat_first = np.empty(flat_order.size, dtype=bool)
            flat_first[0] = True
            flat_first[1:] = (
                (ordered_flat_keys["x"][1:]
                 != ordered_flat_keys["x"][:-1])
                | (ordered_flat_keys["y"][1:]
                   != ordered_flat_keys["y"][:-1])
            )
            flat_indices = flat_indices[flat_order[flat_first]]
        representatives = np.concatenate(
            (representatives[dense], flat_indices)
        )
    else:
        representatives = np.empty(0, dtype=np.int64)

    # Preserve a cleaned adaptive-TIN vertex only where local-plane fitting
    # could not provide a replacement (normally edges or very sparse cells).
    if representatives.size:
        representative_keys = packed_grid_keys(
            x[representatives],
            y[representatives],
            MODEL_SPACING,
            origin_x,
            origin_y,
        )
        representative_order = np.lexsort(
            (representative_keys["y"], representative_keys["x"])
        )
        representatives = representatives[representative_order]
        representative_keys = representative_keys[representative_order]
        rough_keys = packed_grid_keys(
            x[rough_model_indices],
            y[rough_model_indices],
            MODEL_SPACING,
            origin_x,
            origin_y,
        )
        insertion = np.searchsorted(representative_keys, rough_keys)
        safe = np.clip(insertion, 0, representatives.size - 1)
        replaced = (
            (representative_keys["x"][safe] == rough_keys["x"])
            & (representative_keys["y"][safe] == rough_keys["y"])
        )
        fallback = rough_model_indices[~replaced]
        representatives = np.unique(
            np.concatenate((representatives, fallback))
        )
    else:
        representatives = rough_model_indices.copy()

    # A validated structural break vertex is more important to TIN geometry
    # than a smoother replacement from the same 1 m model cell. Remove such
    # replacements before the final hygiene pass.
    if breakline_indices.size and representatives.size:
        break_keys = packed_grid_keys(
            x[breakline_indices], y[breakline_indices], MODEL_SPACING,
            origin_x, origin_y
        )
        break_order = np.lexsort((break_keys["y"], break_keys["x"]))
        breakline_indices = breakline_indices[break_order]
        break_keys = break_keys[break_order]
        rep_keys = packed_grid_keys(
            x[representatives], y[representatives], MODEL_SPACING,
            origin_x, origin_y
        )
        pos = np.searchsorted(break_keys, rep_keys)
        safe = np.clip(pos, 0, breakline_indices.size - 1)
        same_break_cell = (
            (break_keys["x"][safe] == rep_keys["x"])
            & (break_keys["y"][safe] == rep_keys["y"])
        )
        representatives = np.unique(
            np.concatenate((representatives[~same_break_cell], breakline_indices))
        )

    representatives, _ = prune_model_above_lower_plane(
        representatives,
        rough_model_indices,
        x,
        y,
        z,
        representative_model_tree,
    )
    # The structural model passed the same lower-plane/spike hygiene BEFORE
    # breakline detection. Reinsert any break vertex that a single smooth local
    # plane would otherwise remove at a legitimate crest/toe.
    if breakline_indices.size:
        representatives = np.unique(
            np.concatenate((representatives, breakline_indices))
        )
    # The structural model has already passed the more expensive iterative
    # spike prune. Representatives are independently constrained to the robust
    # plane, so repeating that point-by-point pass here would add substantial
    # runtime without improving the selected surface.
    return representatives


def classify_block(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    initial_classification: np.ndarray,
    global_origin: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classify one buffered processing block.

    Thin wrapper: switches the identical-triangulation cache on for the
    duration of this block (see delaunay_triangulation) and guarantees it is
    released afterwards, whatever happens inside.
    """
    if ENGINE == "fast":
        return classify_block_fast(x, y, z, initial_classification)
    previous_cache = getattr(_TIN_CACHE_STATE, "cache", None)
    _TIN_CACHE_STATE.cache = []
    try:
        return _classify_block(
            x, y, z, initial_classification, global_origin
        )
    finally:
        _TIN_CACHE_STATE.cache = previous_cache


def classify_block_fast(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    initial_classification: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classify one block with the compiled core (engine "fast").

    Same contract as _classify_block: the five source classes are reset and
    re-labelled 2 / 3 / 4 / 5 / 7, every other class is left alone, and the
    result is (classification, height above ground, ground point indices).
    """
    import fast_ground

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
    surface_count = int(np.count_nonzero(model))
    ground_count = int(np.count_nonzero(ground))
    log(
        f"      Fast engine: {ground_count:,} ground points of "
        f"{source.size:,} candidates ({surface_count:,} on the surface, "
        f"{ground_count - surface_count:,} within "
        f"{100 * float(run.p['surface_band']):g} cm of it); surface "
        f"followed past its edge +{run.followed:,}, corners turned "
        f"+{run.turned:,}, spikes taken out {run.spikes_removed:,}; "
        f"isolated low points kept out: {int(np.count_nonzero(low)):,} "
        f"({int(np.count_nonzero(low_noise)):,} labelled class 7)"
    )
    if run.seed_square < float(FAST_ENGINE_SETTINGS.get("max_building_size", 60.0)):
        log(
            "      Small block: starting squares reduced to "
            f"{run.seed_square:g} m so the surface has at least three "
            "starting points."
        )
    return classification, hag, source[model]


def _classify_block(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    initial_classification: np.ndarray,
    global_origin: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classify one buffered processing block."""
    original_classification = initial_classification.copy()
    classification = initial_classification.copy()
    reset_mask = reset_input_class_mask(classification)
    classification[reset_mask] = 0
    source_mask = classification == 0

    # TerraScan's Ground routine works from its selected source class. Do not
    # impose a last-return restriction: valid hill-surface points can carry
    # any return number, especially after classification by other software.
    ground_candidates = np.flatnonzero(source_mask)
    if ground_candidates.size < 3:
        raise InsufficientGroundSupport(
            "Fewer than three eligible source-class candidates."
        )

    origin_x, origin_y = global_origin
    work_indices = lowest_per_grid(
        ground_candidates,
        x,
        y,
        z,
        MODEL_SPACING,
        origin_x,
        origin_y,
    )
    preliminary_low_indices = np.empty(0, dtype=np.int64)
    work_plane_context: PlaneContext | None = None
    if UNSUPPORTED_LOW_GUARD:
        (
            work_indices,
            preliminary_low_indices,
            work_plane_context,
        ) = (
            remove_unsupported_low_work_points(
                work_indices,
                x,
                y,
                z,
            )
        )
        if preliminary_low_indices.size:
            log(
                f"      Unsupported gross-low seed candidates excluded: "
                f"{preliminary_low_indices.size:,}"
            )

    work_tree = cKDTree(
        np.column_stack((x[work_indices], y[work_indices]))
    )
    if work_plane_context is None:
        work_plane_context = local_plane_support(
            work_indices,
            work_indices,
            x,
            y,
            z,
            work_tree,
        )
    # Build the initial trusted PTD from the higher-confidence part of the
    # incoming pool when those labels provide useful coverage. This DOES NOT
    # preserve or trust the old classification as final truth: all target
    # classes were already reset above and the full work_indices remain
    # available to recovery. It only stops an incoming vegetation surface from
    # seeding the trusted model before independent terrain exists.
    backbone_work_indices = work_indices
    backbone_prior_used = False
    if TRUSTED_BACKBONE_USE_INPUT_PRIOR:
        primary_mask = np.zeros(ground_candidates.size, dtype=bool)
        for code in TRUSTED_BACKBONE_PRIMARY_CLASSES:
            primary_mask |= original_classification[ground_candidates] == code
        primary_candidates = ground_candidates[primary_mask]
        if primary_candidates.size >= TRUSTED_BACKBONE_MIN_POINTS:
            primary_work = lowest_per_grid(
                primary_candidates,
                x,
                y,
                z,
                MODEL_SPACING,
                origin_x,
                origin_y,
            )
            if primary_work.size >= 3:
                all_keys = packed_grid_keys(
                    x[work_indices],
                    y[work_indices],
                    TRUSTED_BACKBONE_COVERAGE_GRID,
                    origin_x,
                    origin_y,
                )
                primary_keys = packed_grid_keys(
                    x[primary_work],
                    y[primary_work],
                    TRUSTED_BACKBONE_COVERAGE_GRID,
                    origin_x,
                    origin_y,
                )
                all_cells = max(1, np.unique(all_keys).size)
                primary_cells = np.unique(primary_keys).size
                coverage = primary_cells / all_cells
                if coverage >= TRUSTED_BACKBONE_MIN_COVERAGE:
                    backbone_work_indices = primary_work
                    backbone_prior_used = True
                    log(
                        f"      Trusted backbone prior: {primary_work.size:,} "
                        f"work minima across {coverage:.0%} of coarse cells"
                    )
                else:
                    log(
                        f"      Input prior coverage only {coverage:.0%}; "
                        "using full reset pool for PTD backbone"
                    )

    seed_spacing = MAX_BUILDING_SIZE
    seed_indices = lowest_per_grid(
        backbone_work_indices,
        x,
        y,
        z,
        seed_spacing,
        origin_x,
        origin_y,
    )
    while seed_indices.size < 4 and seed_spacing > 2.0 * MODEL_SPACING:
        seed_spacing *= 0.5
        seed_indices = lowest_per_grid(
            backbone_work_indices,
            x,
            y,
            z,
            seed_spacing,
            origin_x,
            origin_y,
        )

    seed_indices = deduplicate_model_points(
        seed_indices, x, y, z, origin_x, origin_y
    )

    if seed_indices.size < 3:
        raise InsufficientGroundSupport(
            "Fewer than three initial ground seed points were found."
        )

    bounds = (
        float(np.min(x)),
        float(np.min(y)),
        float(np.max(x)),
        float(np.max(y)),
    )
    model_indices = seed_indices
    log(
        f"      Work minima: {work_indices.size:,}; "
        f"PTD backbone minima: {backbone_work_indices.size:,}; "
        f"initial seeds: {seed_indices.size:,}"
    )

    for settings in PASSES:
        if settings.name == "Pass 2" and not PASS2_ENABLED:
            continue
        model_indices = run_ptd_pass(
            backbone_work_indices,
            model_indices,
            x,
            y,
            z,
            bounds,
            settings,
        )

    # The strict PTD result seeds the trusted terrain model. v2 does NOT keep
    # later recovery inside a fixed distance of this original backbone. Each
    # recovery pass is supported only by terrain accepted before that pass, so
    # genuine ground can grow through wide woodland without allowing a detached
    # vegetation/roof layer to self-support in one step.
    trusted_backbone_indices = model_indices.copy()
    log(
        f"      Initial trusted PTD model: "
        f"{trusted_backbone_indices.size:,} model points"
    )

    if LOWER_ENVELOPE_RESCUE:
        preliminary_hag = interpolate_hag_at_indices(
            model_indices, work_indices, x, y, z, bounds
        )
        lower_candidates = work_indices[
            (preliminary_hag < -LOWER_ENVELOPE_MIN_DEPTH)
            & (preliminary_hag >= -LOWER_ENVELOPE_MAX_DEPTH)
        ]
        if lower_candidates.size:
            lower_supported = trusted_surface_support(
                lower_candidates,
                model_indices,
                trusted_backbone_indices,
                x,
                y,
                z,
                bounds,
                flat_tolerance=TRUSTED_LOWER_FLAT_TOLERANCE * noise_scale_multiplier(),
                steep_tolerance=TRUSTED_LOWER_STEEP_TOLERANCE * noise_scale_multiplier(),
                max_model_distance=TRUSTED_LOWER_MAX_MODEL_DISTANCE,
                original_classification=original_classification,
            )
            rescue = lower_candidates[lower_supported]
        else:
            rescue = np.empty(0, dtype=np.int64)
        if rescue.size:
            model_indices = np.unique(
                np.concatenate((model_indices, rescue))
            )
            model_indices = deduplicate_model_points(
                model_indices, x, y, z, origin_x, origin_y
            )
            log(
                f"      Trusted lower-envelope points added: "
                f"{rescue.size:,}"
            )
        rejected_lower = lower_candidates.size - rescue.size
        if rejected_lower:
            log(
                f"      Unsupported lower-envelope candidates rejected: "
                f"{rejected_lower:,}"
            )

    if UPPER_SURFACE_RESCUE:
        model_indices, upper_count = recover_supported_upper_surface(
            work_indices,
            model_indices,
            trusted_backbone_indices,
            x,
            y,
            z,
            bounds,
            origin_x,
            origin_y,
            work_tree,
            original_classification,
        )
        if upper_count:
            log(
                f"      Trusted anchored terrain points added: "
                f"{upper_count:,}"
            )

    if LOCAL_LOWER_SURFACE_RESCUE:
        model_indices, local_surface_count = recover_local_lower_surface(
            work_indices,
            model_indices,
            trusted_backbone_indices,
            x,
            y,
            z,
            bounds,
            origin_x,
            origin_y,
            work_plane_context,
            original_classification,
        )
        if local_surface_count:
            log(
                f"      Connected terrain recovery added: "
                f"{local_surface_count:,}"
            )

    model_indices, lower_plane_spikes = prune_model_above_lower_plane(
        model_indices,
        work_indices,
        x,
        y,
        z,
        work_tree,
    )
    if lower_plane_spikes:
        log(
            f"      Canopy vertices above lower surface removed: "
            f"{lower_plane_spikes:,}"
        )

    model_indices, spike_count = prune_upward_model_spikes(
        model_indices, work_indices, x, y, z
    )
    if spike_count:
        log(f"      Unsupported TIN spikes removed: {spike_count:,}")

    # v6 terminal QC repair. Both paths evaluate against the same frozen,
    # already-clean terrain model and run only once, so neither can propagate
    # across water or climb through a detached vegetation/roof layer.
    model_indices, bank_edge_count, rocky_crest_count = recover_terminal_terrain_edges(
        work_indices,
        model_indices,
        x, y, z,
        bounds,
        origin_x, origin_y,
        original_classification,
    )
    if bank_edge_count:
        log(
            f"      Terminal steep-bank/water-edge returns added: "
            f"{bank_edge_count:,}"
        )
    if rocky_crest_count:
        log(
            f"      Terminal rocky crest/ridge returns added: "
            f"{rocky_crest_count:,}"
        )

    # One-sided escarpment repair (ported from Trusted v8). Only sparse,
    # 2.5-D-safe work-minimum representatives may join the STRUCTURAL model.
    # Dense near-vertical/multi-Z face returns are intentionally left
    # non-ground so any downstream class-2 TIN (including one rebuilt by
    # other software from the delivered points) remains stable.
    frozen_for_escarpment = model_indices.copy()
    escarpment = recover_terminal_escarpment_representatives(
        ground_candidates,
        work_indices,
        frozen_for_escarpment,
        x, y, z,
        bounds,
        original_classification,
    )
    if escarpment.size:
        model_indices = np.unique(np.concatenate((model_indices, escarpment)))
        model_indices = deduplicate_model_points(
            model_indices, x, y, z, origin_x, origin_y
        )
        log(
            f"      Terminal 2.5-D escarpment representatives added: "
            f"{escarpment.size:,}"
        )

    # Freeze the fully trusted structural terrain before smoothing. It is
    # used only as independent evidence for final break/crest membership.
    structural_model_indices = model_indices.copy()

    # Estimate this tile's real local ground noise from the structural model
    # itself (no ground truth available in production) and use it to scale
    # the downstream completion-stage tolerances, which were found (via the
    # real-data validation suite) to have been calibrated for one specific
    # project's noise level and not to transfer to noisier/cleaner terrain.
    noise_multiplier = estimate_noise_scale_multiplier(
        structural_model_indices, x, y, z
    )
    configure_noise_scale(noise_multiplier)
    if NOISE_SCALE_ADAPTIVE:
        log(
            f"      Noise-scale multiplier: {noise_multiplier:.2f}x "
            f"(1.0x = reference {NOISE_SCALE_REFERENCE*100:.1f}cm local noise)"
        )

    if SMOOTH_REPRESENTATIVE_MODEL:
        rough_model_count = model_indices.size
        model_indices = select_smooth_model_representatives(
            ground_candidates,
            work_indices,
            model_indices,
            trusted_backbone_indices,
            x,
            y,
            z,
            bounds,
            origin_x,
            origin_y,
            work_tree,
        )
        log(
            f"      Adaptive smooth representatives: "
            f"{model_indices.size:,} "
            f"(from {rough_model_count:,} structural vertices)"
        )

    final_surface = build_tin(model_indices, x, y, z, bounds)
    hag, ground_simplex_ids = interpolate_hag(
        model_indices,
        x,
        y,
        z,
        bounds,
        final_surface,
        ground_candidates,
    )

    # Rebuild the requested classes from the final clean terrain model.
    classification[source_mask] = 0
    classification[model_indices] = 2

    finite = source_mask & np.isfinite(hag)
    high_outlier = finite & (hag > HIGH_VEG_MAX)
    classification[high_outlier] = 7

    # Noise is already preclassified as class 7 and excluded from source_mask.
    # Keep the representative model as the clean interpolation TIN, but retain
    # all source returns in the thin adaptive surface envelope as class 2.
    if ADAPTIVE_GROUND_MEMBERSHIP:
        final_model_tree = cKDTree(
            np.column_stack((x[model_indices], y[model_indices]))
        )
        trusted_membership_context = local_plane_support(
            work_indices,
            model_indices,
            x,
            y,
            z,
            final_model_tree,
        )
        completed_ground = select_adaptive_ground_membership(
            ground_candidates,
            model_indices,
            x,
            y,
            z,
            bounds,
            origin_x,
            origin_y,
            work_indices,
            trusted_membership_context,
            trusted_backbone_indices,
            final_surface,
            ground_simplex_ids,
        )
        classification[completed_ground] = 2
        if completed_ground.size:
            log(
                f"      Triangle-adaptive ground membership: "
                f"{completed_ground.size:,}"
            )

        # Final FPL bank/crest repair. This does NOT modify the terrain/HAG TIN.
        break_completed_ground = select_two_sided_break_membership(
            ground_candidates,
            completed_ground,
            structural_model_indices,
            x,
            y,
            z,
            bounds,
            origin_x,
            origin_y,
        )
        classification[break_completed_ground] = 2

        # Final global output-TIN regularisation. The v6 trusted/HAG TIN stays
        # frozen; only real source returns carrying class 2 are re-selected.
        smooth_output_ground = select_global_smooth_tin_membership(
            ground_candidates,
            model_indices,
            completed_ground,
            break_completed_ground,
            structural_model_indices,
            hag,
            x,
            y,
            z,
            bounds,
            origin_x,
            origin_y,
        )
        current_source_ground = (
            source_mask & (classification == 2)
        )
        classification[current_source_ground] = 0
        classification[smooth_output_ground] = 2
        log(
            f"      Final smooth output class-2 membership: "
            f"{smooth_output_ground.size:,}"
        )

    if UNSUPPORTED_LOW_GUARD:
        confirmed_low_noise = confirm_unsupported_low_noise(
            ground_candidates,
            work_indices,
            preliminary_low_indices,
            hag,
            x,
            y,
            z,
            work_tree,
        )
        classification[confirmed_low_noise] = 7
        if confirmed_low_noise.size:
            log(
                f"      Confirmed isolated low errors assigned class 7: "
                f"{confirmed_low_noise.size:,}"
            )

    remaining = finite & (classification == 0)
    assign_hag_classes(classification, remaining, hag)

    source_ground = np.flatnonzero(
        source_mask & (classification == 2)
    )
    narrow_ground_count = int(
        np.count_nonzero(~np.isin(source_ground, model_indices))
    )
    low_blanket_count = int(
        np.count_nonzero(source_mask & (classification == 3))
    )
    log(
        f"      Smooth ground membership: {model_indices.size:,} model, "
        f"{narrow_ground_count:,} additional thin-surface; "
        f"low-vegetation blanket: {low_blanket_count:,}"
    )

    return classification, hag, model_indices


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


def classify_window_job(
    window_number: int,
    window_count: int,
    core: tuple[float, float, float, float],
    buffered_indices: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    classification_in: np.ndarray,
    global_origin: tuple[float, float],
    overall_xmax: float,
    overall_ymax: float,
    allow_sparse_fallback: bool = False,
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

        cxmin, cymin, cxmax, cymax = core
        x_upper = (
            local_x <= cxmax
            if cxmax >= overall_xmax
            else local_x < cxmax
        )
        y_upper = (
            local_y <= cymax
            if cymax >= overall_ymax
            else local_y < cymax
        )
        core_local = (
            (local_x >= cxmin)
            & x_upper
            & (local_y >= cymin)
            & y_upper
        )

        try:
            block_started = time.perf_counter()
            local_class, local_hag, model = classify_block(
                local_x,
                local_y,
                local_z,
                local_input_class,
                global_origin,
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
    global_origin: tuple[float, float],
    requested_workers: int,
    memory_budget: int,
    core_maxima: tuple[float, float] | None = None,
    allow_sparse_fallback: bool = False,
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
                global_origin,
                overall_xmax,
                overall_ymax,
                allow_sparse_fallback,
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
                global_origin,
                overall_xmax,
                overall_ymax,
                allow_sparse_fallback,
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
    project_origin: tuple[float, float] | None = None,
    input_unit_scale: float | None = None,
) -> float:
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
        las.write(output_path)
        return time.perf_counter() - started

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
            f"{NEIGHBOUR_HALO:.3f} native-unit halo (10 m physical)"
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
        f"{len(windows)}; internal model spacing: {MODEL_SPACING:g} m "
        f"({MODEL_SPACING / unit_scale:.6f} native input units)"
    )
    if project_origin is not None:
        global_origin = (
            project_origin[0] * unit_scale,
            project_origin[1] * unit_scale,
        )
    else:
        global_origin = (float(np.min(target_x)), float(np.min(target_y)))
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
    processing_started = time.perf_counter()
    process_file_windows(
        windows,
        x,
        y,
        z,
        classification_in,
        classification_out,
        hag_out,
        global_origin,
        actual_window_workers,
        memory_budget,
        (target_bounds[2] * unit_scale, target_bounds[3] * unit_scale),
        target_point_count <= SPARSE_FALLBACK_MAX_BUFFERED_POINTS,
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
    return elapsed



def discover_inputs(script_dir: Path) -> list[Path]:
    inputs = sorted(script_dir.glob("*.las"))
    inputs.extend(sorted(script_dir.glob("*.laz")))
    output_dir = script_dir / TURBO2_OUTPUT_FOLDER
    return [
        path
        for path in inputs
        if "_AdaptiveGround" not in path.stem
        and path.parent != output_dir
    ]


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
    """Find files intersecting each target's 10 m expanded bounds."""
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


def estimate_file_peak_memory(
    input_path: Path,
    halo_point_count: int = 0,
    unit_scale: float = METRES_PER_US_SURVEY_FOOT,
) -> int:
    """Estimate one file's resident plus largest-window working memory."""
    try:
        with laspy.open(input_path) as reader:
            header = reader.header
            point_count = int(header.point_count)
            record_size = int(header.point_format.size)
            xmin, ymin = map(float, header.mins[:2])
            xmax, ymax = map(float, header.maxs[:2])
    except Exception:
        # Compressed size is not a reliable RAM predictor, so use a generous
        # multiplier when only filesystem metadata is available.
        return int(
            FILE_FIXED_MEMORY_MB * 1024**2
            + max(input_path.stat().st_size * 10, 1024**3)
        )

    # Header bounds are in the file's native input unit; convert span to
    # metres so the memory scheduler mirrors the 500 m processing-window
    # logic exactly, using this file's actual resolved unit (not always
    # feet - see detect_input_unit_scale).
    width = max((xmax - xmin) * unit_scale, 1.0)
    height = max((ymax - ymin) * unit_scale, 1.0)
    nx = max(1, int(math.ceil(width / BLOCK_SIZE)))
    ny = max(1, int(math.ceil(height / BLOCK_SIZE)))
    window_count = nx * ny
    if window_count == 1:
        largest_buffered_count = point_count
    else:
        # The 2x density factor covers uneven acquisition density while the
        # overlap factor accounts for duplicated edge processing.
        overlap_factor = ((BLOCK_SIZE + 2 * BLOCK_OVERLAP) / BLOCK_SIZE) ** 2
        largest_buffered_count = min(
            point_count,
            int(
                math.ceil(
                    point_count
                    / window_count
                    * overlap_factor
                    * 2.0
                )
            ),
        )

    resident = int(
        FILE_FIXED_MEMORY_MB * 1024**2
        + point_count
        * max(FULL_FILE_BYTES_PER_POINT, record_size + 64)
    )
    halo_resident = int(
        max(0, halo_point_count) * HALO_CACHE_BYTES_PER_POINT
    )
    return (
        resident
        + halo_resident
        + estimate_window_memory(largest_buffered_count + halo_point_count)
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
    project_origin: tuple[float, float],
    hag_bands: tuple[float, float, float] = (LOW_VEG_MAX, MEDIUM_VEG_MAX, HIGH_VEG_MAX),
    unit_scale: float = METRES_PER_US_SURVEY_FOOT,
    window_workers: int = 1,
    window_memory_budget: int | None = None,
) -> tuple[str, float]:
    """Top-level process target; paths avoid transferring point arrays.

    window_workers > 1 runs this file's windows on parallel threads, held to
    window_memory_budget by classify_file's own window scheduler - the same
    mechanism a single file run in the main process has always used.
    """
    input_path = Path(input_path_text)
    configure_runtime(cpu_slots, f"[{input_path.stem}] ")
    configure_hag_bands(*hag_bands)
    configure_neighbour_halo(unit_scale)
    configure_nominal_tile_size(unit_scale)
    elapsed = classify_file(
        input_path,
        Path(output_path_text),
        window_workers=max(1, int(window_workers)),
        memory_budget=window_memory_budget,
        kdtree_workers=cpu_slots,
        neighbour_caches=neighbour_caches,
        target_bounds=target_bounds,
        neighbour_bounds=neighbour_bounds,
        project_origin=project_origin,
        input_unit_scale=unit_scale,
    )
    return input_path.name, elapsed


def reusable_existing_output(input_path: Path, output_path: Path) -> bool:
    """Return true for a complete output with the expected cleaned layout.

    Output intentionally omits configured temporary Extra Bytes such as
    ``Distance``, so byte-for-byte point-format size equality with the input is
    no longer a valid resume test.
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
            output_signature = (
                int(output_header.point_count),
                int(output_header.point_format.id),
                tuple(str(name) for name in output_header.point_format.extra_dimension_names),
            )
        return expected_signature == output_signature
    except Exception:
        return False


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adaptive progressive-TIN ground classification."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="LAS/LAZ inputs. If omitted, scan beside this script.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "Total CPU worker budget. Zero uses a guarded automatic value; "
            "one forces fully sequential processing."
        ),
    )
    parser.add_argument(
        "--ram-limit-gb",
        type=float,
        default=0.0,
        help=(
            "RAM budget in GB for files processed at the same time. Zero "
            "(default) uses 55%% of installed RAM. A larger value, on a "
            "machine dedicated to this run, lets more files run at once; "
            "it is always capped to keep 4 GB of currently free RAM spare."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Reprocess valid outputs that already exist. By default, a "
            "matching header-clean output is reused so interrupted batches "
            "resume efficiently."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    script_dir = Path(__file__).resolve().parent
    hag_bands = load_hag_config(script_dir)
    configure_hag_bands(*hag_bands)
    inputs = args.inputs or discover_inputs(script_dir)
    inputs = [path.resolve() for path in inputs]

    if not inputs:
        raise SystemExit(
            "No LAS/LAZ input found. Put files beside the script or drag "
            "them onto Run_AdaptiveGround_Unified.bat."
        )

    if args.workers < 0:
        raise SystemExit("--workers must be zero or greater.")
    if args.ram_limit_gb < 0:
        raise SystemExit("--ram-limit-gb must be zero or greater.")

    tasks: list[tuple[Path, Path]] = []
    reused_count = 0
    for input_path in inputs:
        if not input_path.exists():
            raise SystemExit(f"Input does not exist: {input_path}")
        if input_path.suffix.lower() not in {".las", ".laz"}:
            raise SystemExit(f"Input must be LAS or LAZ: {input_path}")

        output_dir = (
            args.output_dir.resolve()
            if args.output_dir is not None
            else input_path.parent / TURBO2_OUTPUT_FOLDER
        )
        output_path = (
            output_dir
            / (
                f"{input_path.stem}"
                f"{TURBO2_OUTPUT_SUFFIX}"
                f"{input_path.suffix.lower()}"
            )
        )
        if (
            not args.overwrite
            and reusable_existing_output(input_path, output_path)
        ):
            reused_count += 1
            continue
        tasks.append((input_path, output_path))

    units_setting = load_units_config(script_dir)
    unit_scale_by_path: dict[Path, float] = {}
    for input_path, _output_path in tasks:
        unit_scale_by_path[input_path] = resolve_input_unit_scale(
            input_path, units_setting
        )
    # NEIGHBOUR_HALO is a single process-wide value used while discovering
    # which files are spatial neighbours of each other, before any one
    # file's classification begins - it needs one project-wide unit. All
    # files in a run are expected to share the same project/unit (that is
    # what units_config.json's "per folder" scope means); the first
    # resolved task's unit is used for this shared step, and every actual
    # classification still uses its own file's individually resolved unit.
    if tasks:
        configure_neighbour_halo(unit_scale_by_path[tasks[0][0]])
        configure_nominal_tile_size(unit_scale_by_path[tasks[0][0]])

    if reused_count:
        log(
            f"Resume mode: reusing {reused_count:,} complete existing "
            "output(s). Use --overwrite to rebuild them."
        )
    if not tasks:
        log("All requested outputs are already complete.")
        return 0

    unit_labels = sorted({
        "US survey feet" if abs(scale - 1.0) > 1.0e-9 else "metres"
        for scale in unit_scale_by_path.values()
    })
    unit_summary = " and ".join(unit_labels) if unit_labels else "unknown"
    low_veg, medium_veg, high_veg = hag_bands
    log(
        f"Unit mode: this run's input is {unit_summary} (resolved per file "
        "from CRS metadata or units_config.json - see detect_input_unit_scale "
        "/ resolve_input_unit_scale). Classification geometry is normalized "
        "to metres in memory; LAS XYZ is never rewritten. Output vegetation: "
        f"class 3 <={low_veg:g} m, class 4 {low_veg:g}-{medium_veg:g} m, "
        f"class 5 >={medium_veg:g} m (see hag_config.json); no output class 13."
    )

    ensure_stages_config(script_dir)
    ensure_engine_config(script_dir)
    config_workers, config_ram_limit_gb = load_performance_config(script_dir)
    requested_workers = args.workers if args.workers > 0 else config_workers
    requested_ram_limit_gb = (
        args.ram_limit_gb if args.ram_limit_gb > 0 else config_ram_limit_gb
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
    project_origin = (
        min(item.neighbour_bounds[0] for item in candidate_metadata),
        min(item.neighbour_bounds[1] for item in candidate_metadata),
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

        def classify_in_process(
            task_list: list[tuple[Path, Path]],
            failures: list[tuple[Path, BaseException]] | None = None,
        ) -> None:
            """Classify files one at a time in this process, with
            multi-threaded windows. With ``failures`` given, a file that
            raises is recorded there and the batch continues."""
            window_workers = (
                min(MAX_PARALLEL_WINDOWS, cpu_budget)
                if AUTO_PARALLEL
                else 1
            )
            for input_path, output_path in task_list:
                configure_runtime(cpu_budget)
                configure_neighbour_halo(unit_scale_by_path[input_path])
                configure_nominal_tile_size(unit_scale_by_path[input_path])
                try:
                    classify_file(
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
                        project_origin=project_origin,
                        input_unit_scale=unit_scale_by_path[input_path],
                    )
                except Exception as exc:
                    if failures is None:
                        raise
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
                    "the run, ram_limit_gb in performance_config.json "
                    "raises it (4 GB of free RAM is always kept spare)."
                )
            failures: list[tuple[Path, BaseException]] = []
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
                                project_origin,
                                hag_bands,
                                unit_scale_by_path[input_path],
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
                            name, elapsed = future.result()
                            log(
                                f"Finished {name} in "
                                f"{elapsed / 60.0:.1f} minutes."
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
                classify_in_process(crashed_tasks, failures)
            if failures:
                details = "\n".join(
                    f"  {path}: {error}" for path, error in failures
                )
                raise RuntimeError(
                    f"{len(failures)} file(s) failed:\n{details}"
                )
        else:
            classify_in_process(tasks)
    finally:
        if cache_workspace is not None:
            cache_workspace.cleanup()

    return 0


# Stage switches are applied here, after every default above is defined, and
# in every process that loads this file (see apply_stages_config).
DISABLED_STAGES = apply_stages_config(Path(__file__).resolve().parent)
# The engine choice is applied the same way, after the stages.
apply_engine_config(Path(__file__).resolve().parent)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("\nCancelled by user.")
        raise SystemExit(130)
