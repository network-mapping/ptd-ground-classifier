# ground-classifier

Ground classification of LAS/LAZ tiles by progressive TIN densification. The benchmark is TerraScan's
Ground routine; see [docs/engine.md](docs/engine.md) for the algorithm and results.

## Layout

| Path | Contents |
|---|---|
| `src/ground_classifier/` | Python package: `cli`, `pipeline` (tiles, windows, scheduler), `fast_ground` (core wrapper), `config`, `units`, `manifest`, `compare` |
| `native/` | `ptdcore.c` (the compiled core) and `build.sh` |
| `config/` | `engine_config.json`, `hag_config.json`, `units_config.json`, `performance_config.json` |
| `tests/` | core self-checks, the real-tile regression test (`fixtures/`) and the TerraScan validation suite (`validation/`) |
| `scripts/windows/` | `setup.bat`, `run.bat`, `test.bat` |
| `docs/` | `engine.md`, `reference/` (the paper and notes) |
| `data/` | read-only inputs and the TerraScan benchmark (`output/terra`); not tracked |
| `output/` | our results; not tracked |

## Set up

    make venv      # .venv with the package installed (editable)
    make build     # compile the core into build/ (needs clang or gcc)
    make test        # engine self-checks, ends with ALL OK
    make regression  # real-tile end-to-end regression test, ends with ALL OK
    make validate    # precision / recall / F1 against the TerraScan clips

## Tests

| Command | What it is for |
|---|---|
| `make test` | Engine checks: synthetic terrain, and the four clips against a ground mask recorded on Linux. Core only. |
| `make regression` | The whole tool on a real tile: has the output changed? Use it while refactoring. |
| `make validate` | Precision, recall and F1 against TerraScan. A benchmark, not a pass/fail test. |

All three are plain scripts that need the project venv and a built core (the make targets build it); no test framework.

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

## Run

    .venv/bin/ground-classify data/LGE_feet/input_from_FLAI --units usft --output-dir output/LGE_feet

Inputs are files or folders. Outputs are written as `<name>_AdaptiveGround.las|laz` into `--output-dir` (default `./output`),
never into an input folder. `ground-classify --help` lists the options; settings live in `config/`
(`--config-dir` or `$GROUND_CLASSIFIER_CONFIG` selects another folder).

## Versions and the manifest

The version is `ground_classifier.__version__` (`ground-classify --version`); the compiled core has its own number
(`PTD_VERSION` in `native/ptdcore.c` = `CORE_VERSION` in `fast_ground.py`). Every run writes `manifest.json` into the output
folder, overwriting the previous one: software versions, git commit and dirty flag, library versions, command line, resolved
settings, and per file the unit, status, point count and class counts. Each output file also carries
`ground-classifier <version>` in its LAS header `generating_software`, and existing outputs are reused only when that matches
the current version (`--overwrite` forces a rerun).

## Compare

    .venv/bin/ground-classify compare output/LGE_feet output/other [data/<project>/output/terra]

## Windows

`scripts\windows\setup.bat`, then `run.bat <inputs> ...`. The core must be built as `build\ptdcore.dll`
(clang commands in `native/build.sh`); the Windows scripts are untested.
