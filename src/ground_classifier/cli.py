"""Command line: ground-classify <inputs> [options] | compare <A> <B> [ref] | dtm-diff <A> <B> <out> | init-config [FILE]."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import laspy

from . import __version__, manifest, ptdcore
from .pipeline import (
    OUTPUT_SUFFIX,
    discover_inputs,
    reusable_existing_output,
    run_batch,
)
from .units import resolve_input_unit_scale
from .utils.args import get_args
from .utils.config import CONFIG_FILENAME, load_config, write_default_config
from .utils.loggers import log
from .utils.params import Config


def collect_inputs(paths: list[Path]) -> list[Path]:
    found: list[Path] = []
    for path in paths:
        path = path.resolve()
        if path.is_dir():
            found.extend(p.resolve() for p in discover_inputs(path))
        elif not path.exists():
            raise SystemExit(f"Input does not exist: {path}")
        elif path.suffix.lower() not in {".las", ".laz"}:
            raise SystemExit(f"Input must be LAS or LAZ: {path}")
        else:
            found.append(path)
    unique = sorted(set(found), key=str)
    if not unique:
        raise SystemExit("No LAS/LAZ input found.")
    return unique


def _unit_name(scale: float) -> str:
    return "metres" if abs(scale - 1.0) < 1.0e-9 else "usft"


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
            with laspy.open(output_path) as reader:
                records[input_path].update(status="reused", generated_by=reader.header.generating_software)
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
            scale_by_path = {i: resolve_input_unit_scale(i, units_setting) for i, _ in tasks}
            for input_path, scale in scale_by_path.items():
                records[input_path]["unit"] = _unit_name(scale)
            results, failures, resources = run_batch(
                tasks, inputs, scale_by_path, config.hag_config.bands, config.engine_settings,
                requested_workers, requested_ram_gb,
                locations.export_tin,
                options.tin_max_edge,
                locations.export_dtm,
                config.dtm_cell,
                config.input_config.neighbour_halo,
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
