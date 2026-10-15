"""Run configuration: the JSON files in the config folder.

Resolution order for the folder: --config-dir, then $GROUND_CLASSIFIER_CONFIG,
then <repo>/config. A missing file means built-in defaults; an invalid one is an
error, because silently ignoring a setting would make the manifest misleading.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

from ._log import log

CONFIG_ENV_VAR = "GROUND_CLASSIFIER_CONFIG"

UNITS_CONFIG_FILENAME = "units_config.json"
PERFORMANCE_CONFIG_FILENAME = "performance_config.json"
ENGINE_CONFIG_FILENAME = "engine_config.json"
HAG_CONFIG_FILENAME = "hag_config.json"

DEFAULT_LOW_VEG_MAX = 0.50
DEFAULT_MEDIUM_VEG_MAX = 3.00
DEFAULT_HIGH_VEG_MAX = 200.00

# engine_config.json key -> fast_ground setting (None = a switch, see below).
ENGINE_KEYS = {
    "max_building_size": "max_building_size",
    "terrain_angle": "terrain_angle",
    "iteration_angle": "iteration_angle",
    "steep_iteration_angle": "steep_iteration_angle",
    "iteration_distance": "iteration_distance",
    "reduce_iteration_angle_when_edge_shorter_than": "reduce_angle_edge",
    "stop_triangulation_when_edge_shorter_than": "stop_edge",
    "ground_band": "surface_band",
}
ENGINE_SWITCHES = ("follow_surface", "remove_spikes")
ENGINE_DEFAULTS = {
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


@dataclass(frozen=True)
class RunConfig:
    config_dir: Path
    units: str                                  # "auto", "usft" or "metres"
    workers: int                                # 0 = automatic
    ram_limit_gb: float                         # 0 = automatic
    hag_bands: tuple[float, float, float]       # metres
    engine_values: dict                         # engine_config.json, as the user wrote it
    engine_settings: dict                       # the same, as fast_ground settings
    files_read: tuple[str, ...] = field(default=())

    def as_dict(self) -> dict:
        low, medium, high = self.hag_bands
        return {
            "config_dir": str(self.config_dir),
            "files_read": list(self.files_read),
            "units": self.units,
            "workers": self.workers,
            "ram_limit_gb": self.ram_limit_gb,
            "hag_bands_m": {
                "low_veg_max": low,
                "medium_veg_max": medium,
                "high_veg_max": high,
            },
            "engine": self.engine_values,
        }


def resolve_config_dir(explicit: Path | None = None) -> Path:
    if explicit is not None:
        chosen = Path(explicit)
    elif os.environ.get(CONFIG_ENV_VAR):
        chosen = Path(os.environ[CONFIG_ENV_VAR])
    else:
        chosen = Path(__file__).resolve().parents[2] / "config"
    chosen = chosen.expanduser().resolve()
    if not chosen.is_dir():
        raise SystemExit(f"Config folder does not exist: {chosen}")
    return chosen


def _read_json(config_dir: Path, name: str, files_read: list[str]) -> dict:
    path = config_dir / name
    if not path.exists():
        log(f"{name} not found in {config_dir}; using built-in defaults.")
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("top level must be a JSON object")
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Cannot read {path}: {exc}") from exc
    files_read.append(name)
    return data


def _number(data: dict, key: str, default: float, minimum: float = 0.0) -> float:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemExit(f"{key} must be a number (got {value!r})")
    value = float(value)
    if not math.isfinite(value) or value < minimum:
        raise SystemExit(f"{key} must be a number >= {minimum:g} (got {value!r})")
    return value


def load_run_config(config_dir: Path) -> RunConfig:
    files_read: list[str] = []

    units_data = _read_json(config_dir, UNITS_CONFIG_FILENAME, files_read)
    units = str(units_data.get("units", "auto")).strip().lower()
    units = "metres" if units == "meters" else units
    if units not in ("auto", "usft", "metres"):
        raise SystemExit(
            f"{UNITS_CONFIG_FILENAME}: units must be 'auto', 'usft' or "
            f"'metres' (got {units!r})"
        )

    perf = _read_json(config_dir, PERFORMANCE_CONFIG_FILENAME, files_read)
    workers = int(_number(perf, "workers", 0))
    ram_limit_gb = _number(perf, "ram_limit_gb", 0.0)

    hag = _read_json(config_dir, HAG_CONFIG_FILENAME, files_read)
    low = _number(hag, "low_veg_max", DEFAULT_LOW_VEG_MAX)
    medium = _number(hag, "medium_veg_max", DEFAULT_MEDIUM_VEG_MAX)
    high = _number(hag, "high_veg_max", DEFAULT_HIGH_VEG_MAX)
    if not (0 < low < medium < high):
        raise SystemExit(
            f"{HAG_CONFIG_FILENAME}: need 0 < low_veg_max < medium_veg_max < "
            f"high_veg_max (got {low}, {medium}, {high})"
        )

    engine_data = _read_json(config_dir, ENGINE_CONFIG_FILENAME, files_read)
    unknown = [
        key for key in engine_data
        if key not in ENGINE_DEFAULTS and not key.startswith("_")
    ]
    if unknown:
        raise SystemExit(
            f"{ENGINE_CONFIG_FILENAME}: unknown setting(s) {unknown}"
        )
    values = dict(ENGINE_DEFAULTS)
    for key in ENGINE_DEFAULTS:
        if key in ENGINE_SWITCHES:
            value = engine_data.get(key, values[key])
            if not isinstance(value, bool):
                raise SystemExit(f"{key} must be true or false (got {value!r})")
            values[key] = value
        else:
            values[key] = _number(engine_data, key, values[key])
    if values["iteration_angle"] <= 0 or values["max_building_size"] <= 0:
        raise SystemExit("iteration_angle and max_building_size must be above 0")

    settings: dict = {}
    for key, value in values.items():
        if key == "follow_surface":
            settings["follow_rounds"] = 200 if value else 0
            settings["turn_rounds"] = 60 if value else 0
        elif key == "remove_spikes":
            settings["spike_rounds"] = 5 if value else 0
        else:
            settings[ENGINE_KEYS[key]] = value

    return RunConfig(
        config_dir=config_dir,
        units=units,
        workers=workers,
        ram_limit_gb=ram_limit_gb,
        hag_bands=(low, medium, high),
        engine_values=values,
        engine_settings=settings,
        files_read=tuple(files_read),
    )
