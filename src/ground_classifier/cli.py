"""Command line: ground-classify <inputs> [options] | ground-classify compare <A> <B> [ref]."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import laspy

from . import __version__, manifest, ptdcore
from ._log import log
from .config import RunConfig, load_run_config, resolve_config_dir
from .pipeline import (
    OUTPUT_SUFFIX,
    discover_inputs,
    reusable_existing_output,
    run_batch,
)
from .units import resolve_input_unit_scale


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ground-classify",
        description=(
            "Ground classification of LAS/LAZ tiles. "
            "'ground-classify compare <A> <B> [reference]' compares two outputs."
        ),
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="LAS/LAZ files or folders of them.")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output"),
        help="Where outputs and manifest.json are written (default: ./output). Must not be an input folder.",
    )
    parser.add_argument("--config-dir", type=Path, default=None,
                        help="Folder with the *_config.json files (default: $GROUND_CLASSIFIER_CONFIG, then <repo>/config).")
    parser.add_argument("--units", choices=("auto", "usft", "metres"), default=None,
                        help="Input coordinate unit; overrides units_config.json.")
    parser.add_argument("--workers", type=int, default=0,
                        help="Total CPU worker budget; 0 = performance_config.json, then automatic.")
    parser.add_argument("--ram-limit-gb", type=float, default=0.0,
                        help="RAM budget in GB for files processed together; 0 = performance_config.json, then automatic.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Reprocess outputs that already exist from this version.")
    parser.add_argument("--export-tin", type=Path, default=None, metavar="DIR",
                        help="Also write each window's ground TIN as a binary PLY mesh (doubles, in the input's own unit) to DIR, "
                             "for example to open in CloudCompare. Outputs that are reused are not reprocessed: add --overwrite.")
    parser.add_argument("--tin-max-edge", type=float, default=0.0, metavar="M",
                        help="With --export-tin: leave out triangles with an edge longer than M metres (0 = keep all).")
    parser.add_argument("--export-dtm", type=Path, default=None, metavar="DIR",
                        help="Also write each window's ground raster (the lowest class 2 point of each cell, nodata where there is none) "
                             "as a GeoTIFF to DIR, in the input's own unit. Add --overwrite for outputs that already exist.")
    parser.add_argument("--dtm-cell", type=float, default=0.0, metavar="SIZE",
                        help="With --export-dtm: cell size in the input's unit; 0 = dtm_config.json, then 1 m or 3 ft.")
    parser.add_argument("--version", action="version",
                        version=f"ground-classifier {__version__} (core {ptdcore.CORE_VERSION})")
    args = parser.parse_args(argv)
    if args.workers < 0 or args.ram_limit_gb < 0:
        parser.error("--workers and --ram-limit-gb must be zero or greater.")
    if args.dtm_cell < 0:
        parser.error("--dtm-cell must be zero or greater.")
    if args.dtm_cell and args.export_dtm is None:
        parser.error("--dtm-cell needs --export-dtm.")
    if args.tin_max_edge < 0:
        parser.error("--tin-max-edge must be zero or greater.")
    if args.tin_max_edge and args.export_tin is None:
        parser.error("--tin-max-edge needs --export-tin.")
    return args


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


def classify(args: argparse.Namespace) -> int:
    started, clock = manifest.utc_now(), time.perf_counter()
    config: RunConfig = load_run_config(resolve_config_dir(args.config_dir))
    inputs = collect_inputs(args.inputs)
    output_dir = args.output_dir.resolve()
    if any(path.parent == output_dir for path in inputs):
        raise SystemExit(f"Output folder {output_dir} holds input files; choose another --output-dir.")
    try:
        ptdcore.load()
    except OSError as exc:
        raise SystemExit(f"Compiled core unavailable: {exc}") from exc

    pairs = [(p, output_dir / f"{p.stem}{OUTPUT_SUFFIX}{p.suffix.lower()}") for p in inputs]
    records = {i: manifest.file_record(i, o) for i, o in pairs}
    tasks: list[tuple[Path, Path]] = []
    for input_path, output_path in pairs:
        if not args.overwrite and reusable_existing_output(input_path, output_path):
            with laspy.open(output_path) as reader:
                records[input_path].update(status="reused", generated_by=reader.header.generating_software)
        else:
            tasks.append((input_path, output_path))

    units_setting = args.units or config.units
    requested_workers = args.workers or config.workers
    requested_ram_gb = args.ram_limit_gb or config.ram_limit_gb
    cli = {
        "inputs": [str(p) for p in args.inputs],
        "output_dir": str(output_dir),
        "overwrite": args.overwrite,
        "units_option": args.units,
        "workers_option": args.workers,
        "ram_limit_gb_option": args.ram_limit_gb,
        "export_tin": str(args.export_tin.resolve()) if args.export_tin else None,
        "tin_max_edge": args.tin_max_edge,
        "export_dtm": str(args.export_dtm.resolve()) if args.export_dtm else None,
        "dtm_cell_option": args.dtm_cell,
    }
    effective: dict = {
        "units": units_setting,
        "requested_workers": requested_workers,
        "requested_ram_limit_gb": requested_ram_gb,
    }
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
                tasks, inputs, scale_by_path, config.hag_bands, config.engine_settings,
                requested_workers, requested_ram_gb,
                args.export_tin.resolve() if args.export_tin else None,
                args.tin_max_edge,
                args.export_dtm.resolve() if args.export_dtm else None,
                args.dtm_cell or config.dtm_cell,
            )
            effective["resources_used"] = resources
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
                cli=cli, config={**config.as_dict(), "effective": effective},
                files=[records[i] for i, _ in pairs],
            ),
        )
        log(f"Manifest: {path}")
    return exit_code


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "compare":
        from . import compare
        return compare.main(argv[1:])
    return classify(parse_arguments(argv))


if __name__ == "__main__":
    raise SystemExit(main())
