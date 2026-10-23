import argparse
from pathlib import Path

from .. import __version__, ptdcore
from .config import CONFIG_ENV_VAR, CONFIG_FILENAME


def get_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Return the validated command line parameters of a classify run."""

    parser = argparse.ArgumentParser(
        prog="ground-classify",
        description=(
            "Ground classification of LAS/LAZ tiles. "
            "'ground-classify compare <A> <B> [reference]' compares two outputs; "
            "'ground-classify dtm-diff <A> <B> <output>' subtracts the DTM rasters of two folders; "
            "'ground-classify init-config [FILE]' writes a config file with every setting at its default."
        ),
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="LAS/LAZ files or folders of them.")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output"),
        help="Where outputs and manifest.json are written (default: ./output). Must not be an input folder.",
    )
    parser.add_argument("--config", type=Path, default=None, metavar="FILE",
                        help=f"YAML config file (default: ${CONFIG_ENV_VAR}, then ./{CONFIG_FILENAME}, then "
                             f"./config/{CONFIG_FILENAME}, then built-in defaults).")
    parser.add_argument("--units", choices=("auto", "usft", "metres"), default=None,
                        help="Input coordinate unit; overrides the config file.")
    parser.add_argument("--workers", type=int, default=0,
                        help="Total CPU worker budget; 0 = config file, then automatic.")
    parser.add_argument("--ram-limit-gb", type=float, default=0.0,
                        help="RAM budget in GB for files processed together; 0 = config file, then automatic.")
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
                        help="With --export-dtm: cell size in the input's unit; 0 = config file, then 1 m or 3 ft.")
    parser.add_argument("--anchor-seeds", action="store_true",
                        help="TEMPORARY, for comparing runs: cut the starting squares from a fixed grid, so that the surface "
                             "does not depend on where a window starts.")
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
