"""Checks of RunSettings and how the pipeline applies it: defaults, pickling, configure_run(), the unit-dependent settings
and the window layout, and that spawned workers receive non-default settings. usage: python tests/test_run_settings.py"""
import contextlib
import dataclasses
import io
import json
import multiprocessing
import pickle
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ground_classifier import cli, dtm, fast_ground, pipeline
from ground_classifier.las import LasHeader
from ground_classifier.tiles import halo_in_input_units, nominal_tile_size
from ground_classifier.utils.args import get_args
from ground_classifier.utils.config import default_config_text, load_config
from ground_classifier.utils.params import (
    DEFAULT_BLOCK_OVERLAP,
    DEFAULT_BLOCK_SIZE,
    DEFAULT_HIGH_VEG_MAX,
    DEFAULT_LOW_VEG_MAX,
    DEFAULT_MEDIUM_VEG_MAX,
    RunSettings,
    WindowsParam,
)

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "fpl_feet"
FEET = 1200 / 3937

fails = 0
def ok(name, cond, note=""):
    global fails
    print(("ok   " if cond else "FAIL ") + name + (" - " + note if note else ""), flush=True)
    fails += 0 if cond else 1


def windows(extent=(0.0, 0.0, 300.0, 300.0), points=1000):
    return [core for _, core in pipeline.window_cores_for_extent(*extent, points)]


def main():
    defaults = RunSettings()
    ok("defaults: the engine's, the usual HAG bands, 10 m of neighbour context, no exports",
       defaults.engine == fast_ground.Settings() and defaults.hag_bands == (DEFAULT_LOW_VEG_MAX, DEFAULT_MEDIUM_VEG_MAX, DEFAULT_HIGH_VEG_MAX)
       and defaults.neighbour_halo_m == 10.0 and defaults.tin_export_dir is None and defaults.dtm_export_dir is None
       and defaults.tin_max_edge == 0.0 and defaults.dtm_cell == 0.0)
    ok("defaults: windows of 5 km with an 80 m overlap", (defaults.block_size, defaults.block_overlap) == (DEFAULT_BLOCK_SIZE, DEFAULT_BLOCK_OVERLAP) == (5000.0, 80.0))
    try:
        defaults.block_size = 1.0
        frozen = False
    except dataclasses.FrozenInstanceError:
        frozen = True
    ok("settings are immutable", frozen)

    custom = RunSettings(hag_bands=(0.2, 1.0, 50.0), neighbour_halo_m=25.0, tin_export_dir=Path("/tmp/tin"), tin_max_edge=5.0,
                         dtm_export_dir=Path("/tmp/dtm"), dtm_cell=6.0, block_size=150.0, block_overlap=40.0)
    again = pickle.loads(pickle.dumps(custom))
    codes = lambda s: (s.feature_codes.mapped_input_feature_codes, s.feature_codes.application_names,
                       [s.feature_codes("application")[n].code for n in s.feature_codes.application_names])
    # FeatureCodes has no __eq__, so compare it by what it holds and everything else as a dataclass
    ok("pickling: the settings cross to a spawned worker unchanged",
       dataclasses.replace(again, feature_codes=custom.feature_codes) == custom and codes(again) == codes(custom))

    # the windows section of the config file
    ok("windows: the section's defaults are RunSettings'", (WindowsParam().block_size, WindowsParam().block_overlap) == (defaults.block_size, defaults.block_overlap))
    tmp = Path(tempfile.mkdtemp(prefix="runsettings_"))
    def load(text, *extra):
        file = tmp / "c.yml"
        file.write_text(text)
        return load_config(get_args([str(FIXTURE), "--output-dir", str(tmp / "o"), "--config", str(file), *map(str, extra)]))
    def error(text):
        try:
            load(text)
        except SystemExit as exc:
            return str(exc)
        return ""
    config = load("windows:\n  block_size: 250\n  block_overlap: 30\ninput:\n  neighbour_halo: 25\n")
    settings = config.run_settings()
    ok("config: block_size and block_overlap come from the windows section", (settings.block_size, settings.block_overlap) == (250.0, 30.0))
    ok("config: run_settings() carries the rest of the Config too",
       settings.neighbour_halo_m == 25.0 and settings.hag_bands == config.hag_config.bands and settings.engine is config.engine_settings
       and settings.feature_codes is config.feature_codes and settings.tin_export_dir is None and settings.dtm_cell == 0.0)
    config = load("input:\n  units: usft\n", "--export-tin", tmp / "t", "--tin-max-edge", "4", "--export-dtm", tmp / "d", "--dtm-cell", "7", "--anchor-seeds")
    settings = config.run_settings()
    ok("config: ... including the command line's exports, cell and --anchor-seeds",
       (settings.tin_export_dir, settings.tin_max_edge, settings.dtm_export_dir, settings.dtm_cell, settings.engine.seeds.anchor) == ((tmp / "t").resolve(), 4.0, (tmp / "d").resolve(), 7.0, True))
    ok("config: no windows section is the defaults", load("").run_settings().block_size == DEFAULT_BLOCK_SIZE)
    ok("config: block_size must be above 0", "block_size must be a number > 0" in error("windows:\n  block_size: 0\n") and "block_size must be a number > 0" in error("windows:\n  block_size: -5\n"))
    ok("config: block_overlap may be 0 but not negative", error("windows:\n  block_overlap: 0\n") == "" and "block_overlap must be a number >= 0" in error("windows:\n  block_overlap: -1\n"))
    ok("config: both must be numbers", "block_size must be a number" in error("windows:\n  block_size: wide\n") and "block_overlap must be a number" in error("windows:\n  block_overlap: true\n"))
    ok("config: an unknown windows key is an error", "section 'windows': unknown setting(s) ['block_sizes']" in error("windows:\n  block_sizes: 5\n"))
    text = default_config_text()
    ok("init-config: the windows section is written, with its documentation",
       "windows:\n" in text and "block_size: 5000.0" in text and "block_overlap: 80.0" in text and "Keep it larger than engine.max_building_size" in text)
    ok("manifest: the windows are recorded", load("windows:\n  block_size: 250\n").as_dict()["windows"] == {"block_size_m": 250.0, "block_overlap_m": 80.0})

    # configure_run applies every field, and the defaults put everything back
    try:
        pipeline.configure_run(custom)
        ok("configure_run: the pipeline now holds exactly the settings it was given, so every field is applied", pipeline.RUN_SETTINGS is custom)
        ok("configure_run: and what the windows and exports read from it is what was set",
           (pipeline.RUN_SETTINGS.block_size, pipeline.RUN_SETTINGS.block_overlap, pipeline.RUN_SETTINGS.tin_export_dir, pipeline.RUN_SETTINGS.dtm_cell)
           == (150.0, 40.0, Path("/tmp/tin"), 6.0))
        ok("tiles: the halo is in the input unit (25 m of feet), and the nominal tile is 1500 ft",
           abs(halo_in_input_units(25.0, FEET) - 25.0 / FEET) < 1e-9 and nominal_tile_size(FEET) == 1500.0, f"{halo_in_input_units(25.0, FEET):.3f}")
        ok("tiles: metres: the halo is 25 and the nominal tile 500", halo_in_input_units(25.0, 1.0) == 25.0 and nominal_tile_size(1.0) == 500.0)
        ok("tiles: a halo of 0 stays 0", halo_in_input_units(0.0, 1.0) == 0.0)

        # the window layout follows the block size set for the run, not a module constant
        pipeline.configure_run(RunSettings())
        ok("windows: a 300 m tile is one window at the default block size", len(windows()) == 1)
        pipeline.configure_run(RunSettings(block_size=150.0))
        ok("windows: ... and four at a block size of 150 m", len(windows()) == 4, str(windows()))
        pipeline.configure_run(RunSettings(block_size=150.0, block_overlap=40.0))
        import numpy as np
        x = np.array([0.0, 100.0, 139.0, 141.0, 300.0]); y = np.zeros(5)
        picked = pipeline.buffered_indices_for_window(x, y, (0.0, -1.0, 100.0, 1.0), None, None)
        ok("windows: the buffer around a core is the block overlap", picked.tolist() == [0, 1, 2], str(picked.tolist()))
    finally:
        pipeline.configure_run(RunSettings())

    # a run through cli.main. On a machine with little free RAM the scheduler runs the files in this process, so
    # this checks the settings reach the pipeline; the spawned-worker path is checked below.
    def run(config_text, out):
        file = tmp / f"{out}.yml"
        file.write_text(config_text)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = cli.main([str(FIXTURE), "--output-dir", str(tmp / out), "--config", str(file), "--workers", "2", "--overwrite",
                             "--export-dtm", str(tmp / f"{out}_dtm"), "--dtm-cell", "6"])
        return code, buffer.getvalue(), json.loads((tmp / out / "manifest.json").read_text())

    windows_in = lambda text: sorted(int(line.rsplit(":", 1)[1]) for line in text.splitlines() if "processing window(s):" in line)
    code, text, manifest = run("input:\n  units: usft\n  neighbour_halo: 25\n", "whole")
    ok("run: completes", code == 0 and manifest["run"]["status"] == "completed")
    ok("run: the configured halo is used (25 m = 82.021 ft)", text.count("82.021 native-unit halo (25 m physical)") >= 2, str(text.count("native-unit halo")))
    ok("run: ... and the DTM export (cell 6)", list((tmp / "whole_dtm").glob("*.tif")) != [] and
       all(dtm.read_dtm_info(p)[1] == 6.0 for p in (tmp / "whole_dtm").glob("*.tif")))
    ok("run: the manifest records the halo", manifest["parameters"]["config"]["neighbour_halo_m"] == 25.0)
    ok("run: at the default block size each tile is one window", windows_in(text) == [1, 1], str(windows_in(text)))
    code, text, manifest = run("input:\n  units: usft\nwindows:\n  block_size: 80\n", "cut")
    ok("run: tiles are cut at the configured block size (158 x 133 m into 4 windows, the 138 x 12 m strip into 2)",
       code == 0 and windows_in(text) == [2, 4], str(windows_in(text)))
    ok("run: ... and the manifest records it", manifest["parameters"]["config"]["windows"] == {"block_size_m": 80.0, "block_overlap_m": 80.0})

    # a spawned worker, as the scheduler's process pool starts one: it must get everything from the RunSettings it is sent
    source = FIXTURE / "219000_1146000.laz"
    def in_worker(settings, name):
        bounds = LasHeader.from_file(source).extent_xy
        with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
            pool.submit(pipeline.classify_file_worker, str(source), str(tmp / f"{name}.laz"), 1, (), bounds, bounds, settings, FEET).result()
        return sorted((tmp / f"{name}_dtm").glob("*.tif"))
    rasters = in_worker(RunSettings(dtm_export_dir=tmp / "spawn_whole_dtm", dtm_cell=6.0), "spawn_whole")
    ok("worker: default settings: one window, so one raster", [p.name for p in rasters] == ["219000_1146000_w01.tif"], str([p.name for p in rasters]))
    rasters = in_worker(RunSettings(dtm_export_dir=tmp / "spawn_cut_dtm", dtm_cell=6.0, block_size=80.0), "spawn_cut")
    ok("worker: block_size 80 cuts the tile into 4 windows, one raster each",
       [p.name for p in rasters] == [f"219000_1146000_w0{i}.tif" for i in range(1, 5)], str([p.name for p in rasters]))
    ok("worker: ... at the cell size in the settings", all(dtm.read_dtm_info(p)[1] == 6.0 for p in rasters))
    shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{fails} failed" if fails else "\nall passed")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
