"""Checks of the `las` package and the unit handling built on it: units and the override, input expansion, whole-tile
read and write, and the command line's use of them. usage: python tests/test_las.py"""
import contextlib
import io
import json
import logging
import shutil
import sys
import tempfile
import types
from pathlib import Path

import laspy
import numpy as np
import pyproj

from ground_classifier import cli, las
from ground_classifier.utils import loggers
from ground_classifier.utils.params import InputParam

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = sorted((ROOT / "tests" / "fixtures" / "fpl_feet").glob("*.laz"))

fails = 0
def ok(name, cond, note=""):
    global fails
    print(("ok   " if cond else "FAIL ") + name + (" - " + note if note else ""), flush=True)
    fails += 0 if cond else 1


def raises(exception, call):
    try:
        call()
    except exception as error:
        return error
    except BaseException as error:
        return None if print("      unexpected", type(error).__name__, error) else None
    return None


def make_las(path, n=20_000, crs=None, seed=1, extra_dimension=False):
    """A small tile of rolling ground; with a CRS if one is given."""
    rng = np.random.default_rng(seed)
    # A CRS without an EPSG code (international feet) can only be stored as WKT, which point format 6 allows.
    header = laspy.LasHeader(point_format=6 if crs is not None else 3, version="1.4")
    header.scales = (0.001, 0.001, 0.001)
    header.offsets = (500_000, 5_000_000, 0)
    if extra_dimension:
        header.add_extra_dim(laspy.ExtraBytesParams(name="Distance", type=np.int32))
    if crs is not None:
        header.add_crs(crs)
    tile = laspy.LasData(header)
    tile.x = rng.uniform(500_000, 500_200, n)
    tile.y = rng.uniform(5_000_000, 5_000_200, n)
    tile.z = 0.05 * (tile.x - 500_000) + rng.normal(0, 0.02, n) + 100
    tile.classification = np.full(n, 2, np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    tile.write(path)
    return path


def capture(logger_name="ground_classifier.las"):
    records = []
    class Collector(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())
    handler = Collector(level=logging.WARNING)
    logger = logging.getLogger(logger_name)
    logger.addHandler(handler)
    return records, lambda: logger.removeHandler(handler)


def main():
    tmp = Path(tempfile.mkdtemp(prefix="las_"))
    metres = pyproj.CRS.from_epsg(32618)
    usft = pyproj.CRS.from_epsg(6438)
    ift = pyproj.CRS.from_proj4("+proj=utm +zone=18 +datum=WGS84 +units=ft +no_defs")

    # 1. the unit options and conversions
    ok("units: the choices", las.UNITS_CHOICES == ("auto", "usft", "metres", "foot"))
    ok("units: names map to the las units", [las.parse_units_option(v) for v in ("metres", "usft", "foot")] == [las.METRES, las.US_FEET, las.IMPERIAL_FEET])
    ok("units: 'auto' is None, aliases and case are accepted", las.parse_units_option("auto") is None and las.parse_units_option("Meters") == las.METRES and las.parse_units_option(" USFT ") == las.US_FEET)
    ok("units: anything else is an error naming the choices", "auto, usft, metres, foot" in str(raises(ValueError, lambda: las.parse_units_option("feet"))))
    ok("units: metres per unit", las.get_conversion(las.METRES) == 1 and las.get_conversion(las.IMPERIAL_FEET) == 0.3048 and las.get_conversion(las.US_FEET) == 1200 / 3937)
    ok("units: the unit of a scale; an unknown scale is taken as US feet, as before",
       [las.unit_for_scale(s) for s in (1.0, 0.3048, 1200 / 3937, 0.5)] == [las.METRES, las.IMPERIAL_FEET, las.US_FEET, las.US_FEET])

    # 2. units from the CRS
    dataset = las.LasDataset(FIXTURES)
    ok("auto: the real tiles are US survey feet, EPSG 6438", dataset.units == las.US_FEET and dataset.epsg == 6438 and dataset.metres_per_unit == 1200 / 3937)
    for crs, unit, scale in ((metres, las.METRES, 1.0), (usft, las.US_FEET, 1200 / 3937), (ift, las.IMPERIAL_FEET, 0.3048)):
        d = las.LasDataset(make_las(tmp / f"crs_{unit}.las", crs=crs))
        ok(f"auto: a {unit} CRS gives {unit}", d.units == unit and d.metres_per_unit == scale and isinstance(d.metres_per_unit, float))
    mismatch = raises(las.LasDatasetError, lambda: las.LasDataset([tmp / "crs_m.las", tmp / "crs_survey_foot.las"]))
    ok("auto: tiles in different CRSs are an error", mismatch is not None and "mismatch" in str(mismatch))

    # 3. the override
    nocrs = [make_las(tmp / "nocrs" / f"t{i}.las", seed=i) for i in (1, 2)]
    error = raises(las.LasUnitsError, lambda: las.LasDataset(nocrs))
    ok("auto: no CRS is a LasUnitsError that says what to do", error is not None and "nocrs" not in str(error) and "t1.las" in str(error) and "state the units" in str(error))
    ok("a LasUnitsError is a LasDatasetError", issubclass(las.LasUnitsError, las.LasDatasetError))
    for option, unit in (("metres", las.METRES), ("usft", las.US_FEET), ("foot", las.IMPERIAL_FEET)):
        d = las.LasDataset(nocrs, units=option)
        ok(f"override: {option} needs no CRS", d.units == unit and d.metres_per_unit == las.get_conversion(unit) and d.epsg is None and d.srs is None)
    d = las.LasDataset([tmp / "crs_m.las", tmp / "crs_survey_foot.las"], units="usft")
    ok("override: the CRS is not compared, the stated unit is used", d.units == las.US_FEET)
    ok("override: a bad value is a LasDatasetError", raises(las.LasDatasetError, lambda: las.LasDataset(nocrs, units="feet")) is not None)
    ok("override: refreshed() keeps it", las.LasDataset(nocrs, units="foot").refreshed().units == las.IMPERIAL_FEET)

    # 4. a CRS that cannot be parsed does not stop header reads or an explicit unit
    bad = make_las(tmp / "badcrs.las")
    tile = laspy.read(bad)
    tile.header.vlrs.append(laspy.VLR(user_id="LASF_Projection", record_id=2112, description="bad wkt", record_data=b"NOT A CRS\x00"))
    tile.write(bad)
    header = las.LasHeader.from_file(bad)
    ok("header: reading it does not parse the CRS", header.point_count == 20_000 and len(header.extent_xy) == 4)
    ok("header: asking for the CRS says it cannot be parsed", "could not be parsed" in str(raises(las.LasHeaderError, lambda: header.crs)))
    ok("override: an unreadable CRS does not matter", las.LasDataset(bad, units="metres").units == las.METRES)
    ok("auto: an unreadable CRS is a LasUnitsError", raises(las.LasUnitsError, lambda: las.LasDataset(bad)) is not None)

    # 5. empty tiles
    empty = tmp / "empty" / "empty.las"
    empty.parent.mkdir()
    laspy.LasData(laspy.LasHeader(point_format=3, version="1.4")).write(empty)
    records, undo = capture()
    d = las.LasDataset([empty, nocrs[0]], units="metres")
    undo()
    ok("empty: listed in sources, left out of files, and said so", len(d.sources) == 2 and len(d.files) == 1 and any("no points" in r for r in records))
    ok("empty: an all-empty dataset has its stated units", las.LasDataset(empty, units="metres").units == las.METRES)
    ok("empty: but no CRS to read them from", raises(las.LasDatasetError, lambda: las.LasDataset(empty).units) is not None)

    # 6. the extent warning is in metres, whatever the unit
    records, undo = capture()
    las.LasDataset(nocrs[0], units="metres", file_extent_warning_threshold=100)
    las.LasDataset(nocrs[0], units="usft", file_extent_warning_threshold=100)
    las.LasDataset(nocrs[0], units="metres", file_extent_warning_threshold=1000)
    undo()
    ok("extent: warns over the threshold in metres (a 200 unit tile is 61 m in feet), not under it",
       len(records) == 1 and "exceeds" in records[0], str(records))
    ok("extent: the threshold must be a positive number", all(raises(las.LasDatasetError, lambda v=v: las.LasDataset(nocrs[0], units="metres", file_extent_warning_threshold=v)) is not None for v in (0, -1, "1 km", True)))

    # 7. inputs
    inputs = tmp / "folder"
    for name in ("b.laz", "a.las", "c.LAZ"):
        shutil.copy(nocrs[0], inputs / name) if inputs.exists() else (inputs.mkdir(), shutil.copy(nocrs[0], inputs / name))
    (inputs / "notes.txt").write_text("x")
    found = las.expand_inputs([inputs, inputs / "a.las"])
    ok("inputs: a folder gives its LAS and LAZ files, sorted, once, as absolute paths", [p.name for p in found] == ["a.las", "b.laz"] and all(p.is_absolute() for p in found), str([p.name for p in found]))
    ok("inputs: a missing path", "Input does not exist" in str(raises(FileNotFoundError, lambda: las.expand_inputs([tmp / "nope.laz"]))))
    ok("inputs: a file that is not LAS", "Input must be LAS or LAZ" in str(raises(ValueError, lambda: las.expand_inputs([inputs / "notes.txt"]))))
    (tmp / "none").mkdir()
    ok("inputs: nothing found", "No LAS/LAZ input found." in str(raises(FileNotFoundError, lambda: las.expand_inputs([tmp / "none"]))))

    # 8. reading and writing a whole tile
    source = FIXTURES[0]
    tile = las.read_tile(source)
    before = list(tile.point_format.extra_dimension_names)
    removed = las.drop_extra_dimensions(tile, ("distance",))
    ok("tile: an extra dimension is dropped, matched ignoring case, by its stored name", before == ["Distance"] and removed == ("Distance",) and list(tile.point_format.extra_dimension_names) == [])
    ok("tile: nothing to drop, nothing dropped", las.drop_extra_dimensions(tile, ("Distance",)) == ())
    out = tmp / "out" / "tile.laz"
    las.write_tile(out, tile, generating_software="test 1")
    written = laspy.read(out)
    mode = out.stat().st_mode & 0o777
    ok("tile: written, with the folder made, the usual file permissions and no temporary file left",
       out.is_file() and mode != 0o600 and [p.name for p in out.parent.iterdir()] == ["tile.laz"], oct(mode))
    original = laspy.read(source)
    same = all(np.array_equal(original[d], written[d]) for d in written.point_format.dimension_names)
    metadata = lambda h: sorted((v.user_id, v.record_id, bytes(v.record_data_bytes())) for v in h.vlrs if v.user_id not in ("laszip encoded", "LASF_Spec"))
    ok("tile: every remaining dimension, the coordinates' encoding, the VLRs and the point format are the source's",
       same and np.array_equal(original.header.scales, written.header.scales) and np.array_equal(original.header.offsets, written.header.offsets)
       and metadata(original.header) == metadata(written.header) and original.header.point_format.id == written.header.point_format.id
       and str(original.header.version) == str(written.header.version))
    ok("tile: the generating software is set", las.LasHeader.from_file(out).generating_software == "test 1")
    tile.classification = np.full(len(tile.points), 7, np.uint8)
    las.write_tile(out, tile)
    ok("tile: a second write replaces the file", int(laspy.read(out).classification.min()) == 7)
    ok("tile: overwrite=False refuses", raises(FileExistsError, lambda: las.write_tile(out, tile, overwrite=False)) is not None)

    class Failing:
        header = types.SimpleNamespace(generating_software="")
        def write(self, path):
            Path(path).write_bytes(b"half a file")
            raise RuntimeError("disk full")
    ok("tile: a failed write leaves the existing file untouched and no temporary file",
       raises(RuntimeError, lambda: las.write_tile(out, Failing())) is not None and int(laspy.read(out).classification.min()) == 7
       and [p.name for p in out.parent.iterdir()] == ["tile.laz"])
    ok("tile: ... and creates no file where there was none", raises(RuntimeError, lambda: las.write_tile(tmp / "out" / "new.laz", Failing())) is not None and not (tmp / "out" / "new.laz").exists())

    batches = tmp / "batches.laz"
    header = las.LasHeader.from_file(source)
    ok("batches: written from a header and record batches", las.write_record_batches(batches, header, las.iter_record_batches(source, batch_size=100_000)) == int(header.point_count))
    ok("batches: refuses to overwrite unless told to", raises(FileExistsError, lambda: las.write_record_batches(batches, header, las.iter_record_batches(source))) is not None
       and las.write_record_batches(batches, header, las.iter_record_batches(source), overwrite=True) == int(header.point_count))

    # 8b. a file reads its own points, a dataset finds its members, and can take in the files beside them
    member = las.LasFile.from_file(FIXTURES[0])
    whole = member.read()
    ok("file: read() is the whole tile", len(whole.points) == member.header.point_count == 209_219 and whole.header.point_format.id == member.header.point_format_id)
    batches = list(member.iter_record_batches(100_000))
    ok("file: iter_record_batches() gives every point in batches of at most the size", sum(len(b) for b in batches) == 209_219 and max(len(b) for b in batches) <= 100_000 and len(batches) == 3, str([len(b) for b in batches]))
    ok("file: a bad batch size is an error", raises(ValueError, lambda: list(member.iter_record_batches(0))) is not None)
    found = las.LasDataset(FIXTURES)
    ok("dataset: file() finds a member by its path, however it is spelt", found.file(FIXTURES[1]).path == FIXTURES[1] and found.file(FIXTURES[1].parent / ".." / "fpl_feet" / FIXTURES[1].name).path == FIXTURES[1])
    ok("dataset: ... and an empty one", las.LasDataset([empty, nocrs[0]], units="metres").file(empty).header.point_count == 0)
    missing = raises(KeyError, lambda: found.file(nocrs[0]))
    ok("dataset: a path that is not a member is a KeyError saying so", missing is not None and "is not in this LAS dataset" in str(missing))
    ok("dataset: iterating the dataset reads through the same files", next(iter(found.iter_record_batches(1_000_000)))[0].path == FIXTURES[0])

    one = las.LasDataset([FIXTURES[0]], siblings=True)
    ok("siblings: one file brings in the other tiles of its folder, sorted, as absolute paths", [lf.path.name for lf in one.sources] == [p.name for p in FIXTURES] and all(lf.path.is_absolute() for lf in one.sources))
    ok("siblings: the units and CRS then cover all of them", one.units == las.US_FEET and one.epsg == 6438 and one.point_count == 1_794_130)
    ok("siblings: without the option only the given file", [lf.path.name for lf in las.LasDataset([FIXTURES[0]]).sources] == [FIXTURES[0].name])
    nearby = tmp / "nearby"
    for name, crs in (("a.las", metres), ("b.las", None), ("c.las", usft)):
        make_las(nearby / name, crs=crs, seed=7)
    error = raises(las.LasUnitsError, lambda: las.LasDataset([nearby / "a.las"], siblings=True))
    ok("siblings: a neighbour with no CRS stops an auto run, naming it", error is not None and "b.las" in str(error))
    mixed = tmp / "mixed"
    make_las(mixed / "a.las", crs=metres, seed=8); make_las(mixed / "c.las", crs=usft, seed=9)
    ok("siblings: ... and a neighbour in other units is a mismatch", "mismatch" in str(raises(las.LasDatasetError, lambda: las.LasDataset([mixed / "a.las"], siblings=True))))
    ok("siblings: with the units stated nothing is compared", [lf.path.name for lf in las.LasDataset([nearby / "a.las"], siblings=True, units="metres").sources] == ["a.las", "b.las", "c.las"])
    ok("siblings: a file given twice is still an error", raises(las.LasDatasetError, lambda: las.LasDataset([nearby / "a.las", nearby / "a.las"], siblings=True, units="metres")) is not None)
    two = las.LasDataset([nearby / "a.las", FIXTURES[0]], siblings=True, units="usft")
    ok("siblings: the folders of all the given files", len(two.sources) == 5, str([lf.path.name for lf in two.sources]))
    ok("siblings: refreshed() keeps the members", [lf.path for lf in one.refreshed().sources] == [lf.path for lf in one.sources])

    # 9. the logging bridge
    shown = []
    loggers._LOG_STATE.buffer = shown
    try:
        loggers.install_logging_bridge()
        loggers.install_logging_bridge()
        las.LasDataset(empty.parent, units="metres")
    finally:
        del loggers._LOG_STATE.buffer
    ok("logging: las warnings reach log(), once, labelled", len(shown) == 1 and shown[0].startswith("Warning: ") and "no points" in shown[0], str(shown))

    # 10. the config file
    ok("config: units accepts foot and the aliases", [InputParam(units=v).units for v in ("auto", "usft", "foot", "metres", "meters", "FOOT")] == ["auto", "usft", "foot", "metres", "metres", "foot"])
    try:
        InputParam(units="feet")
        message = ""
    except SystemExit as error:
        message = str(error)
    ok("config: any other unit is an error naming the choices", "auto', 'metres', 'usft' or 'foot'" in message, message)

    # 11. the command line
    def run(*args):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            try:
                code = cli.main([str(a) for a in args])
            except SystemExit as error:
                code = str(error)
        return code, buffer.getvalue()

    cli_in = tmp / "cli_in"
    make_las(cli_in / "tile.las", n=30_000, seed=5)
    laspy.LasData(laspy.LasHeader(point_format=3, version="1.4")).write(cli_in / "empty.las")
    code, text = run(cli_in, "--output-dir", tmp / "o_none")
    manifest = json.loads((tmp / "o_none" / "manifest.json").read_text())
    ok("cli: no CRS and no --units stops, saying how to state the units, and the manifest records the failure",
       isinstance(code, str) and "Cannot determine the units" in code and '--units metres, usft or foot' in code
       and manifest["run"]["status"] == "failed" and "Cannot determine the units" in manifest["run"]["error"], str(code)[:80])
    for option in ("metres", "usft", "foot"):
        code, text = run(cli_in, "--output-dir", tmp / f"o_{option}", "--units", option, "--workers", "1")
        manifest = json.loads((tmp / f"o_{option}" / "manifest.json").read_text())
        ok(f"cli: --units {option}: runs, records the unit, writes an output for the empty tile too",
           code == 0 and {f["unit"] for f in manifest["files"]} == {option} and (tmp / f"o_{option}" / "empty_AdaptiveGround.las").is_file()
           and {Path(f["input"]).name: f["status"] for f in manifest["files"]} == {"empty.las": "processed", "tile.las": "processed"}, str(code))
    ok("cli: the empty tile's output is empty and the other's has all its points",
       len(laspy.read(tmp / "o_metres" / "empty_AdaptiveGround.las").points) == 0 and len(laspy.read(tmp / "o_metres" / "tile_AdaptiveGround.las").points) == 30_000)
    ok("cli: the run says what it used and warned about the empty tile", "Unit mode: this run's input is international feet" in run(cli_in, "--output-dir", tmp / "o_foot", "--units", "foot", "--workers", "1", "--overwrite")[1])
    code, text = run(cli_in, "--output-dir", tmp / "o_metres", "--units", "metres", "--workers", "1")
    ok("cli: a second run reuses both outputs", code == 0 and "Reusing 2 complete output(s)" in text, text[-200:])
    config = tmp / "foot.yml"
    config.write_text("input:\n  units: foot\n")
    code, text = run(cli_in, "--output-dir", tmp / "o_cfg", "--config", config, "--workers", "1")
    ok("cli: the config file's unit is used", code == 0 and "international feet" in text)
    code, text = run(cli_in, "--output-dir", tmp / "o_cfg2", "--config", config, "--units", "metres", "--workers", "1")
    ok("cli: --units overrides the config file", code == 0 and "Unit mode: this run's input is metres" in text)
    crs_in = tmp / "cli_crs"
    make_las(crs_in / "tile.las", n=30_000, crs=usft, seed=6)
    code, text = run(crs_in, "--output-dir", tmp / "o_crs", "--workers", "1")
    ok("cli: with a CRS and no --units the unit is read from it, and said", code == 0 and "Input units: US survey feet (from the CRS, EPSG 6438)" in text, str(code))
    ok("cli: input errors keep their messages",
       [run(a)[0] for a in (tmp / "nope.laz", inputs / "notes.txt", tmp / "none")] ==
       [f"Input does not exist: {(tmp / 'nope.laz').resolve()}", f"Input must be LAS or LAZ: {(inputs / 'notes.txt').resolve()}", "No LAS/LAZ input found."])

    # one file of a folder: its neighbours come from the dataset, which holds the whole folder
    code, text = run(FIXTURES[0], "--output-dir", tmp / "o_one", "--workers", "1")
    manifest = json.loads((tmp / "o_one" / "manifest.json").read_text())
    ok("cli: one file of a folder is classified, and only that one is written and reported",
       code == 0 and [Path(f["input"]).name for f in manifest["files"]] == [FIXTURES[0].name] and sorted(p.name for p in (tmp / "o_one").glob("*.laz")) == [FIXTURES[0].stem + "_AdaptiveGround.laz"])
    ok("cli: ... with the neighbouring tile of its folder as context", "Neighbour context:" in text and "from 1 adjacent file(s)" in text, str([l for l in text.splitlines() if "Neighbour" in l]))
    ok("cli: ... in the units read from the whole folder's CRS", "Input units: US survey feet (from the CRS, EPSG 6438)" in text)
    # a neighbour with no CRS, beside a tile that has one
    mixed_in = tmp / "mixed_in"
    make_las(mixed_in / "tile.las", n=20_000, crs=usft, seed=11)
    make_las(mixed_in / "stray.las", n=1_000, seed=12)
    code, text = run(mixed_in / "tile.las", "--output-dir", tmp / "o_mixed", "--workers", "1")
    ok("cli: a file beside it with no CRS stops an auto run, naming it", isinstance(code, str) and "stray.las" in code and "--units" in code, str(code)[:100])
    code, text = run(mixed_in / "tile.las", "--output-dir", tmp / "o_mixed2", "--units", "usft", "--workers", "1")
    ok("cli: ... and with the units stated it is context like any other", code == 0, str(code))

    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{fails} failed" if fails else "\nall passed")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
