"""Run configuration: command line arguments plus one YAML config file, as a `params.Config`.

Config file, in order: --config, then $GROUND_CLASSIFIER_CONFIG, then ground_classifier_config.yml in the current
directory, then config/ground_classifier_config.yml in the current directory. With none of them, the built-in
defaults of the param classes are used. Every section and key is optional; an unknown or invalid one is an error,
because silently ignoring a setting would make the manifest misleading.
"""
from __future__ import annotations

import dataclasses as dc
import textwrap
from dataclasses import replace
from pathlib import Path

import yaml

from . import params as p
from .helpers import env_default
from .loggers import log

CONFIG_ENV_VAR = "GROUND_CLASSIFIER_CONFIG"
CONFIG_FILENAME = "ground_classifier_config.yml"

# Config attribute -> the param class of that section of the file.
SECTIONS = {
    "input_config": p.InputParam,
    "performance_config": p.PerformanceParam,
    "dtm_config": p.DtmParam,
    "hag_config": p.HagParam,
    "engine_config": p.EngineParam,
}


def find_config_file(explicit: Path | None = None) -> Path | None:
    """The config file to use, or None for built-in defaults. An explicit path is never second-guessed."""
    if explicit is not None:
        return explicit
    from_env = env_default(CONFIG_ENV_VAR)
    if from_env:
        return Path(from_env)
    for candidate in (Path.cwd() / CONFIG_FILENAME, Path.cwd() / "config" / CONFIG_FILENAME):
        if candidate.is_file():
            return candidate
    return None


def load_config(args) -> p.Config:
    """Build the `Config` from the parsed arguments (`utils.args.get_args`) and the config file they point to."""
    path_param = p.PathParam(
        inputs=args.inputs,
        output=args.output_dir,
        config=find_config_file(args.config),
        export_tin=args.export_tin,
        export_dtm=args.export_dtm,
    )
    option_param = p.OptionParam(
        units=args.units,
        workers=args.workers,
        ram_limit_gb=args.ram_limit_gb,
        dtm_cell=args.dtm_cell,
        tin_max_edge=args.tin_max_edge,
        overwrite=args.overwrite,
        anchor_seeds=args.anchor_seeds,
    )

    if path_param.config is None:
        log(f"No config file found (--config, ${CONFIG_ENV_VAR}, ./{CONFIG_FILENAME}, ./config/{CONFIG_FILENAME}); "
            "using built-in defaults.")
    data = _read_yaml(path_param.config) if path_param.config else {}
    unknown = [name for name in data if name not in {cls.SECTION for cls in SECTIONS.values()}]
    if unknown:
        raise SystemExit(f"{path_param.config.name}: unknown section(s) {unknown}")
    sections = {attribute: _from_section(cls, data.get(cls.SECTION), path_param.config)
                for attribute, cls in SECTIONS.items()}

    engine_settings = sections["engine_config"].settings()
    if option_param.anchor_seeds:
        engine_settings = replace(engine_settings, seeds=replace(engine_settings.seeds, anchor=True))

    return p.Config(
        path=path_param,
        options=option_param,
        engine_settings=engine_settings,
        **sections,
    )


def _read_yaml(path: Path) -> dict:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SystemExit(f"Cannot read {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise SystemExit(f"Cannot read {path}: top level must be a mapping of sections")
    return data


def _from_section(cls, data, path: Path | None):
    """The param dataclass `cls` from its section of the file (None = absent: all defaults)."""
    if data is None:
        data = {}
    name = path.name if path else CONFIG_FILENAME
    if not isinstance(data, dict):
        raise SystemExit(f"{name}: section '{cls.SECTION}' must be a mapping of settings")
    unknown = [key for key in data if key not in {field.name for field in dc.fields(cls)}]
    if unknown:
        raise SystemExit(f"{name}: section '{cls.SECTION}': unknown setting(s) {unknown}")
    return cls(**data)


# --- a new config file ---------------------------------------------------------------------------------------------

def default_config_text() -> str:
    """Every setting at its default, with its documentation as comments."""
    lines = [
        "# ground-classify settings. Every section and key is optional: a missing one uses the value shown here.",
        f"# Found as --config, ${CONFIG_ENV_VAR}, ./{CONFIG_FILENAME} or ./config/{CONFIG_FILENAME}.",
    ]
    for cls in SECTIONS.values():
        defaults = cls()
        lines += ["", *_comment(cls.DOC, 0), f"{cls.SECTION}:"]
        for field in dc.fields(cls):
            if field.metadata.get("doc"):
                lines += _comment(field.metadata["doc"], 2)
            lines.append(f"  {field.name}: {_scalar(getattr(defaults, field.name))}")
    return "\n".join(lines) + "\n"


def write_default_config(path: Path, force: bool = False) -> Path:
    path = Path(path).expanduser().resolve()
    if path.exists() and not force:
        raise SystemExit(f"{path} already exists; use --force to overwrite it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(default_config_text(), encoding="utf-8")
    return path


def _comment(text: str, indent: int) -> list[str]:
    prefix = " " * indent + "# "
    return textwrap.wrap(text, width=100, initial_indent=prefix, subsequent_indent=prefix)


def _scalar(value) -> str:
    return yaml.safe_dump(value, default_flow_style=True, width=10**6).splitlines()[0]
