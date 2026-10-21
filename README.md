# ground-classifier

Ground classification of LAS/LAZ tiles by progressive TIN densification. The benchmark is TerraScan's
Ground routine; see [docs/engine.md](docs/engine.md) for the algorithm and results.

## Layout

| Path | Contents |
|---|---|
| `src/ground_classifier/` | Python package: `cli`, `pipeline` (tiles, windows, scheduler), `fast_ground` (the engine: low points, starting points, the run), `ptdcore` (Python front for the compiled core), `config`, `units`, `manifest`, `compare`, `ply` (TIN meshes) |
| `native/` | `ptdcore.c` (the compiled core) and `build.sh` |
| `config/` | `engine_config.json`, `hag_config.json`, `units_config.json`, `performance_config.json` |
| `tests/` | core self-checks, the real-tile regression test (`fixtures/`), the memory test and the TerraScan validation suite (`validation/`) |
| `scripts/windows/` | `setup.bat`, `run.bat`, `test.bat` |
| `docs/` | `engine.md`, `reference/` (the paper and notes) |
| `data/` | read-only inputs and the TerraScan benchmark (`output/terra`); not tracked |
| `output/` | our results; not tracked |

## Set up

    make venv      # .venv with the package installed (editable)
    make build     # compile the core into build/ (needs clang or gcc)
    make test        # engine self-checks, ends with ALL OK
    make regression  # real-tile end-to-end regression test, ends with ALL OK
    make memory      # memory-use regression test on the four clips, ends with ALL OK
    make validate    # precision / recall / F1 against the TerraScan clips

## Tests

| Command | What it is for |
|---|---|
| `make test` | Engine checks: synthetic terrain, and the four clips against a ground mask recorded on Linux. Core only. |
| `make regression` | The whole tool on a real tile: has the output changed? Use it while refactoring. |
| `make memory` | The engine's memory use on the four clips: has it grown? Use it while refactoring. |
| `make tin` | The TIN export: PLY writer, `GroundRun.tin()`, per-window meshes, command line. |
| `make validate` | Precision, recall and F1 against TerraScan. A benchmark, not a pass/fail test. |

All of them are plain scripts that need the project venv and a built core (the make targets build it); no test framework.

### Regression test

`tests/test_regression.py` runs `python -m ground_classifier` on the fixture in `tests/fixtures/fpl_feet/` and compares the
result with `expected.npz` in the same folder. The fixture is a real FPL tile (`219000_1146000.laz`, copied unchanged from
`data/FPL_feet/input_from_FLAI`) plus only the points of its neighbour `219000_1144500.laz` within 40 ft of the shared edge,
so the neighbour-tile context is exercised without a 36 MB file. It runs in a couple of seconds.

It checks that the run completes; point counts, coordinates and every other dimension are carried through; the
classification of every point is identical to the reference; and the manifest agrees with the output. It also fails if the
configuration (`engine_config.json`, `hag_config.json`, `units_config.json`) or the fixture files differ from those the
reference was made with. On a mismatch it prints how many points differ and which classes they moved between.

The reference records that the output has not changed, not that it is right. Use it for changes that should not alter
results (refactoring, renaming, speed). Do not edit the fixture or the config to make it pass.

To update the reference after a deliberate change of behaviour:

1. Confirm the change is what you intended (`make validate` shows what it does to accuracy).
2. `.venv/bin/python tests/test_regression.py --write-reference --note "what changed and why"`
3. Commit `expected.npz`; the note, versions, platform and hashes are stored inside it.

To make a reference from an earlier version of the code, export it, build its core, and point `PYTHONPATH` at its `src`:

    mkdir /tmp/ref && git archive <commit> | tar -x -C /tmp/ref && (cd /tmp/ref && sh native/build.sh)
    PYTHONPATH=/tmp/ref/src .venv/bin/python tests/test_regression.py --write-reference --note "commit <commit>"

The reference was made from commit `f6e7655` on macOS arm64. Results are expected to match on other platforms but this has
not been tried; if it fails on a new platform, check that before suspecting the code.

### Memory test

`tests/test_memory.py` measures the memory the engine uses and compares it with `tests/fixtures/memory_reference.json`. The
compiled core allocates nothing (all its arrays are numpy arrays made in Python), so `tracemalloc` sees what the engine holds.
On each of the four clips in `tests/validation/clips` it records, in bytes above the input arrays: the peak of
`GroundRun(...).run()`, what is still held when it finishes, the extra peak of each output (`ground()`, `heights()`,
`model()`, `low()`), the peak of each stage, and what is left once the run is released. One more run of the steep clip starts
with a tiny triangle array so that the growth of those arrays is measured too. It takes a few seconds.

A number passes when it is within 2% (plus 16 KiB) of the reference; a number more than 5% below it passes with a note to
update the reference. The point counts and ground counts must match exactly. It also fails if the clips or
`fast_ground.Settings()` (the defaults) differ from those the reference was made with.

It does not see memory allocated outside Python and numpy: scipy's KD-tree in the low-point filter, laspy and lazrs, and
scratch space inside numpy sorts. It guards the engine, not the whole-process peak.

The stage peaks are taken by wrapping the private stage methods of `GroundRun` (`_filter_low_points`, `_choose_seeds`,
`_build_state`, `_grow`, `_prune_spikes`), so those names must stay.

To update the reference after a deliberate change that alters memory use:

1. Confirm the change is intended and `make test` and `make regression` still pass.
2. `.venv/bin/python tests/test_memory.py --write-reference --note "what changed and why"` (it measures twice and refuses to
   write if the two disagree by more than 0.5%).
3. Commit `memory_reference.json`.

A reference can be made from an earlier version of the code the same way as for the regression test (`PYTHONPATH` at that
version's `src`), provided that version has the stage methods named above.

## Run

    .venv/bin/ground-classify data/LGE_feet/input_from_FLAI --units usft --output-dir output/LGE_feet

Inputs are files or folders. Outputs are written as `<name>_AdaptiveGround.las|laz` into `--output-dir` (default `./output`),
never into an input folder. `ground-classify --help` lists the options; settings live in `config/`
(`--config-dir` or `$GROUND_CLASSIFIER_CONFIG` selects another folder).

## Versions and the manifest

The version is `ground_classifier.__version__` (`ground-classify --version`); the compiled core has its own number
(`PTD_VERSION` in `native/ptdcore.c` = `CORE_VERSION` in `ptdcore.py`). Every run writes `manifest.json` into the output
folder, overwriting the previous one: software versions, git commit and dirty flag, library versions, command line, resolved
settings, and per file the unit, status, point count and class counts. Each output file also carries
`ground-classifier <version>` in its LAS header `generating_software`, and existing outputs are reused only when that matches
the current version (`--overwrite` forces a rerun).

## Exporting the TIN

    .venv/bin/ground-classify <inputs> --export-tin tins [--tin-max-edge 5] --overwrite

writes the ground surface each window was classified against as `tins/<tile>_w<NN>.ply`: a binary PLY mesh with double
coordinates, in the input file's own unit, so it lies over the point cloud (CloudCompare opens it directly and proposes its
own shift for large coordinates; PLY has no offset field). Only the surface vertices are used, and a triangle is written by
the window whose core holds its centroid, so the meshes of a tile's windows are one TIN without overlap. `--tin-max-edge M`
leaves out triangles with an edge longer than M metres (guesses across gaps). Outputs that already exist are skipped, so add
`--overwrite` to export for them. In code: `GroundRun.tin()` gives the triangles, `ground_classifier.ply` writes and reads them.
`tests/test_tin_export.py` checks it (`make tin`).

## Compare

    .venv/bin/ground-classify compare output/LGE_feet output/other [data/<project>/output/terra]

## Windows

`scripts\windows\setup.bat`, then `run.bat <inputs> ...`. The core must be built as `build\ptdcore.dll`
(clang commands in `native/build.sh`); the Windows scripts are untested.
