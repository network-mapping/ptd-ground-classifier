"""Typed, validated run parameters, composed under one `Config`.

The YAML config file has one section per `*Param` class below, and its keys are the class's fields; the command
line has `PathParam` and `OptionParam`. `config.load_config` only builds them. Defaults live on the fields (the
engine's come from `fast_ground.Settings`) and carry their documentation in `metadata["doc"]`, which is what
`ground-classify init-config` writes into a new file. Normalising and validation happen in `__post_init__`, so a bad
setting fails at start-up. Errors are `SystemExit` messages, as the command line reports them.
"""
from __future__ import annotations

import dataclasses as dc
import math
from pathlib import Path
from typing import ClassVar

from .. import fast_ground
from .helpers import normalize_path

DEFAULT_LOW_VEG_MAX = 0.50
DEFAULT_MEDIUM_VEG_MAX = 3.00
DEFAULT_HIGH_VEG_MAX = 200.00


def _number(key: str, value, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemExit(f"{key} must be a number (got {value!r})")
    value = float(value)
    if not math.isfinite(value) or value < minimum:
        raise SystemExit(f"{key} must be a number >= {minimum:g} (got {value!r})")
    return value


def _field(default, doc: str):
    return dc.field(default=default, metadata={"doc": doc})


# --- from the command line ---------------------------------------------------------------------------------------

@dc.dataclass
class PathParam:
    # Normalised below, except `inputs`: the manifest records them as typed and `cli.collect_inputs` resolves them.
    inputs: tuple[Path, ...]
    output: Path
    config: Path | None             # None = no config file found: built-in defaults
    export_tin: Path | None = None
    export_dtm: Path | None = None

    def __post_init__(self):
        self.inputs = tuple(Path(path) for path in self.inputs)
        for field in dc.fields(self):
            value = getattr(self, field.name)
            if field.name != "inputs" and value is not None:
                setattr(self, field.name, normalize_path(value))
        if self.config is not None and not self.config.is_file():
            raise SystemExit(f"Config file does not exist: {self.config}")


@dc.dataclass
class OptionParam:
    """The run options exactly as given on the command line (0 / None = not given)."""
    units: str | None
    workers: int
    ram_limit_gb: float
    dtm_cell: float
    tin_max_edge: float
    overwrite: bool
    anchor_seeds: bool


# --- from the sections of the config file ---------------------------------------------------------------------------

@dc.dataclass
class InputParam:
    SECTION: ClassVar[str] = "input"
    DOC: ClassVar[str] = (
        "The LAS/LAZ files being classified. The setting applies to every file processed in one run - keep files of "
        "different units in separate runs. The same option on the command line (--units) takes priority over this file."
    )

    units: str = _field(
        "auto",
        "Coordinate unit of the files: auto, usft or metres. 'auto' reads each file's own CRS metadata and only "
        "proceeds when the unit is unambiguous; the tool refuses to guess rather than silently corrupt output. If a "
        "file has no CRS metadata (common - many production LAS files omit it), set usft or metres explicitly.",
    )

    neighbour_halo: float = _field(
        10.0,
        "Metres around each tile that are read from the adjacent tiles, so ground near a tile edge is classified "
        "with its neighbours' points. This changes classifications near tile edges. 0 = no neighbour context.",
    )

    def __post_init__(self):
        units = str(self.units).strip().lower()
        self.units = "metres" if units == "meters" else units
        if self.units not in ("auto", "usft", "metres"):
            raise SystemExit(
                f"{self.SECTION}.units must be 'auto', 'usft' or "
                f"'metres' (got {self.units!r})"
            )
        self.neighbour_halo = _number("neighbour_halo", self.neighbour_halo)


@dc.dataclass
class PerformanceParam:
    SECTION: ClassVar[str] = "performance"
    DOC: ClassVar[str] = (
        "Speed settings only - neither changes any classification result. The same options on the command line "
        "(--workers, --ram-limit-gb) take priority over this file."
    )

    workers: int = _field(
        0, "Total CPU threads to use. 0 = automatic (80% of logical CPUs).")
    ram_limit_gb: float = _field(
        0.0,
        "RAM budget in GB for files processed at the same time. 0 = automatic (55% of installed RAM). On a machine "
        "dedicated to this run a larger value lets more files run at once; 4 GB of free RAM is always kept spare "
        "whatever is entered.",
    )

    def __post_init__(self):
        self.workers = int(_number("workers", self.workers))
        self.ram_limit_gb = _number("ram_limit_gb", self.ram_limit_gb)


@dc.dataclass
class DtmParam:
    SECTION: ClassVar[str] = "dtm"
    DOC: ClassVar[str] = (
        "The DTM rasters written by --export-dtm. Each cell holds the lowest ground point in it; cells without one "
        "are nodata. The same option on the command line (--dtm-cell) takes priority over this file."
    )

    cell_size: float = _field(
        0.0,
        "Cell size in the unit of the input coordinates (feet for a feet project, metres for a metric one). "
        "0 = the default: 1 for metres, 3 for feet.",
    )

    def __post_init__(self):
        self.cell_size = _number("cell_size", self.cell_size)


@dc.dataclass
class HagParam:
    SECTION: ClassVar[str] = "hag"
    DOC: ClassVar[str] = (
        "Vegetation height above ground (HAG) bands, in metres. Above high_veg_max a point becomes class 7 "
        "(noise/outlier)."
    )

    low_veg_max: float = _field(DEFAULT_LOW_VEG_MAX, "Class 3 ceiling.")
    medium_veg_max: float = _field(DEFAULT_MEDIUM_VEG_MAX, "Class 4 ceiling (also class 5 floor).")
    high_veg_max: float = _field(DEFAULT_HIGH_VEG_MAX, "Class 5 ceiling.")

    def __post_init__(self):
        self.low_veg_max = _number("low_veg_max", self.low_veg_max)
        self.medium_veg_max = _number("medium_veg_max", self.medium_veg_max)
        self.high_veg_max = _number("high_veg_max", self.high_veg_max)
        low, medium, high = self.bands
        if not (0 < low < medium < high):
            raise SystemExit(
                f"{self.SECTION}: need 0 < low_veg_max < medium_veg_max < "
                f"high_veg_max (got {low}, {medium}, {high})"
            )

    @property
    def bands(self) -> tuple[float, float, float]:
        return self.low_veg_max, self.medium_veg_max, self.high_veg_max


def _setting(group: str, name: str, doc: str = ""):
    """A number field that is `fast_ground.Settings.<group>.<name>`; its default is the engine's."""
    def default():
        return getattr(getattr(fast_ground.Settings(), group), name)
    return dc.field(default_factory=default, metadata={"setting": (group, name), "doc": doc})


def _switch(*groups: str, doc: str = ""):
    """A true/false field; false sets `rounds` to 0 in these groups of `fast_ground.Settings`."""
    return dc.field(default=True, metadata={"switch": groups, "doc": doc})


@dc.dataclass
class EngineParam:
    SECTION: ClassVar[str] = "engine"
    DOC: ClassVar[str] = (
        "Fast-engine settings, all in METRES and degrees whatever the project unit. The first seven carry the names "
        "of TerraScan's Ground routine settings."
    )

    # Field order is the manifest's.
    max_building_size: float = _setting(
        "seeds", "max_building_size",
        "One starting point per square of this size; lower it (20 to 30) on hilly projects where buildings are "
        "already in their own class.")
    terrain_angle: float = _setting("acceptance", "terrain_angle")
    iteration_angle: float = _setting(
        "acceptance", "iteration_angle", "Used on flat ground (steep_iteration_angle is used on steep ground).")
    steep_iteration_angle: float = _setting("acceptance", "steep_iteration_angle", "Used on steep ground.")
    iteration_distance: float = _setting("acceptance", "iteration_distance")
    reduce_iteration_angle_when_edge_shorter_than: float = _setting("acceptance", "reduce_angle_edge")
    stop_triangulation_when_edge_shorter_than: float = _setting("acceptance", "stop_edge")
    ground_band: float = _setting(
        "ground_band", "width",
        "Points this close to the ground surface are ground too, at most one per 0.3 m square; "
        "0 = surface points only.")
    follow_surface: bool = _switch(
        "follow", "turn",
        doc="Let the ground continue over hill tops, banks and ridges the plain routine cannot climb.")
    remove_spikes: bool = _switch(
        "spikes", doc="Take single points that stand out of the surface back out.")

    def __post_init__(self):
        for field in dc.fields(self):
            value = getattr(self, field.name)
            if "switch" in field.metadata:
                if not isinstance(value, bool):
                    raise SystemExit(f"{field.name} must be true or false (got {value!r})")
            else:
                setattr(self, field.name, _number(field.name, value))
        self.settings()

    def settings(self) -> fast_ground.Settings:
        """The fast_ground Settings these values make."""
        defaults = fast_ground.Settings()
        changes: dict[str, dict] = {}       # group -> {field: value}
        for field in dc.fields(self):
            value = getattr(self, field.name)
            if "setting" in field.metadata:
                group, name = field.metadata["setting"]
                changes.setdefault(group, {})[name] = value
            elif not value:
                for group in field.metadata["switch"]:
                    changes.setdefault(group, {})["rounds"] = 0
        try:
            return fast_ground.Settings(**{
                group: dc.replace(getattr(defaults, group), **fields)
                for group, fields in changes.items()
            })
        except ValueError as exc:
            raise SystemExit(f"{self.SECTION}: {exc}") from exc


@dc.dataclass
class Config:
    path: PathParam                         # from CLI
    options: OptionParam                    # from CLI
    input_config: InputParam                # from the config file
    performance_config: PerformanceParam
    dtm_config: DtmParam
    hag_config: HagParam
    engine_config: EngineParam
    engine_settings: fast_ground.Settings   # engine_config as the engine's settings (and --anchor-seeds)

    # A command line option, when given, takes priority over the config file.
    @property
    def units(self) -> str:
        return self.options.units or self.input_config.units

    @property
    def workers(self) -> int:
        return self.options.workers or self.performance_config.workers

    @property
    def ram_limit_gb(self) -> float:
        return self.options.ram_limit_gb or self.performance_config.ram_limit_gb

    @property
    def dtm_cell(self) -> float:
        return self.options.dtm_cell or self.dtm_config.cell_size

    def as_dict(self) -> dict:
        low, medium, high = self.hag_config.bands
        return {
            "config_file": str(self.path.config) if self.path.config else None,
            "units": self.input_config.units,
            "neighbour_halo_m": self.input_config.neighbour_halo,
            "workers": self.performance_config.workers,
            "ram_limit_gb": self.performance_config.ram_limit_gb,
            "dtm_cell_size": self.dtm_config.cell_size,
            "hag_bands_m": {
                "low_veg_max": low,
                "medium_veg_max": medium,
                "high_veg_max": high,
            },
            "engine": dc.asdict(self.engine_config),
        }
