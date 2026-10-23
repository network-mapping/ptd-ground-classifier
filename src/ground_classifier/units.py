"""Input unit handling: metres vs US survey feet, read from LAS CRS metadata."""
from __future__ import annotations

import struct
from pathlib import Path

from .utils.loggers import log

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

# Every "foot" appears in "international foot" text too (e.g. some legacy
# state-plane definitions), so the negative check has to come first: if the
# text says "US survey foot"/"ftUS"/"USfoot", or plain "foot"/"feet" without
# an explicit "international" qualifier, treat it as US survey feet. This
# matches the only two linear units this classifier supports.
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
        "Guessing wrong silently corrupts every coordinate by ~3.28x. "
        "Pass --units usft or --units metres, or set \"units\" in the "
        "config file's input section, then run again."
    )
