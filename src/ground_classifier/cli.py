"""Command line: ground-classify <inputs> [options] | compare <A> <B> [ref] | dtm-diff <A> <B> <out> | init-config [FILE]."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import __version__, manifest, ptdcore
from .las import (
    UNIT_LABELS,
    UNIT_OPTION_NAMES,
    UNITS_AUTO,
    LasDataset,
    LasDatasetError,
    LasHeader,
    LasUnitsError,
    expand_inputs,
    unit_for_scale,
)
from .pipeline import (
    OUTPUT_SUFFIX,
    reusable_existing_output,
    run_batch,
)
from .utils.args import get_args
from .utils.config import CONFIG_FILENAME, load_config, write_default_config
from .utils.loggers import install_logging_bridge, log
from .utils.params import Config


def collect_inputs(paths: list[Path]) -> list[Path]:
    try:
        return expand_inputs(paths)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


def _unit_name(scale: float) -> str:
    return UNIT_OPTION_NAMES[unit_for_scale(scale)]


def _input_scale(task_inputs: list[Path], units: str) -> float:
    """Metres per coordinate unit of the files being classified: as stated by --units or the config file, else
    from their CRS. Exits with a message saying how to state it when the CRS cannot tell."""
    try:
        dataset = LasDataset(task_inputs, units=units)
        if units == UNITS_AUTO and not dataset.files:
            raise LasUnitsError("Every input tile is empty, so there is no CRS to read the units from.")
        scale = dataset.metres_per_unit
    except LasUnitsError as exc:
        raise SystemExit(
            f"{exc}\nPass --units metres, usft or foot, or set \"units\" in the config file's input section, "
            "then run again."
        ) from exc
    except (LasDatasetError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc
    if units == UNITS_AUTO:
        log(f"Input units: {UNIT_LABELS[dataset.units]} (from the CRS, EPSG {dataset.epsg})")
    return scale


def classify(config: Config) -> int:
    
    started, clock = manifest.utc_now(), time.perf_counter()
    locations, options = config.path, config.options
    inputs = collect_inputs(list(locations.inputs))
    output_dir = locations.output

    if any(p.parent == output_dir for p in inputs):
        raise SystemExit(f"Output folder {output_dir} holds input files; choose another --output-dir.")
    try:
        ptdcore.load()
    except OSError as exc:
        raise SystemExit(f"Compiled core unavailable: {exc}") from exc

    pairs = [(p, output_dir / f"{p.stem}{OUTPUT_SUFFIX}{p.suffix.lower()}") for p in inputs]
    records = {i: manifest.file_record(i, o) for i, o in pairs}
    tasks: list[tuple[Path, Path]] = []
    for input_path, output_path in pairs:
        if not options.overwrite and reusable_existing_output(input_path, output_path):
            records[input_path].update(
                status="reused", generated_by=LasHeader.from_file(output_path).generating_software
            )
        else:
            tasks.append((input_path, output_path))

    units_setting = config.units
    requested_workers = config.workers
    requested_ram_gb = config.ram_limit_gb
    resources = None
    status, error, exit_code = "completed", None, 0

    log(f"ground-classifier {__version__} (core {ptdcore.CORE_VERSION}); output: {output_dir}")
    try:
        reused = len(pairs) - len(tasks)
        if reused:
            log(f"Reusing {reused:,} complete output(s) from this version. Use --overwrite to rebuild them.")
        if tasks:
            scale = _input_scale([i for i, _ in tasks], units_setting)
            scale_by_path = {i: scale for i, _ in tasks}
            for input_path, scale in scale_by_path.items():
                records[input_path]["unit"] = _unit_name(scale)
            results, failures, resources = run_batch(
                tasks, inputs, scale_by_path, config.run_settings(),
                requested_workers, requested_ram_gb,
            )
            for input_path, output_path in tasks:
                record = records[input_path]
                if input_path in results:
                    result = results[input_path]
                    record.update(
                        status="processed",
                        point_count=result.point_count,
                        class_counts={str(code): n for code, n in result.class_counts.items()},
                        elapsed_s=round(result.elapsed, 1),
                    )
                    record["generated_by"] = f"ground-classifier {__version__}"
            for input_path, exc in failures:
                records[input_path].update(status="failed", error=f"{type(exc).__name__}: {exc}")
            if failures:
                status, exit_code = "completed_with_failures", 1
                log(f"{len(failures)} file(s) failed:")
                for input_path, exc in failures:
                    log(f"  {input_path}: {exc}")
        else:
            log("All requested outputs are already complete.")
    except KeyboardInterrupt:
        status, error = "aborted", "interrupted"
        raise
    except BaseException as exc:
        status, error = "failed", str(exc) if isinstance(exc, SystemExit) else f"{type(exc).__name__}: {exc}"
        raise
    finally:
        path = manifest.write_manifest(
            output_dir,
            manifest.build_manifest(
                started=started, finished=manifest.utc_now(),
                duration_s=time.perf_counter() - clock, status=status, error=error,
                config=config, resources_used=resources,
                files=[records[i] for i, _ in pairs],
            ),
        )
        log(f"Manifest: {path}")
    return exit_code


def init_config(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="ground-classify init-config",
        description="Write a config file with every setting at its default, for editing.",
    )
    parser.add_argument("file", nargs="?", type=Path, default=Path(CONFIG_FILENAME),
                        help=f"Where to write it (default: ./{CONFIG_FILENAME}).")
    parser.add_argument("--force", action="store_true", help="Overwrite the file if it exists.")
    args = parser.parse_args(argv)
    log(f"Wrote {write_default_config(args.file, args.force)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    install_logging_bridge()
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "compare":
        from . import compare
        return compare.main(argv[1:])
    if argv and argv[0] == "dtm-diff":
        from . import dtm_diff
        return dtm_diff.main(argv[1:])
    if argv and argv[0] == "init-config":
        return init_config(argv[1:])
    return classify(load_config(get_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
