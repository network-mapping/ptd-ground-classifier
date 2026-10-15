# ground-classifier

Ground classification of LAS/LAZ tiles by progressive TIN densification. The benchmark is TerraScan's
Ground routine; see [docs/engine.md](docs/engine.md) for the algorithm and results.

## Layout

| Path | Contents |
|---|---|
| `src/ground_classifier/` | Python package: `cli`, `pipeline` (tiles, windows, scheduler), `fast_ground` (core wrapper), `config`, `units`, `manifest`, `compare` |
| `native/` | `ptdcore.c` (the compiled core) and `build.sh` |
| `config/` | `engine_config.json`, `hag_config.json`, `units_config.json`, `performance_config.json` |
| `tests/` | core self-checks and the TerraScan validation suite (`validation/`) |
| `scripts/windows/` | `setup.bat`, `run.bat`, `test.bat` |
| `docs/` | `engine.md`, `reference/` (the paper and notes) |
| `data/` | read-only inputs and the TerraScan benchmark (`output/terra`); not tracked |
| `output/` | our results; not tracked |

## Set up

    make venv      # .venv with the package installed (editable)
    make build     # compile the core into build/ (needs clang or gcc)
    make test      # engine self-checks, ends with ALL OK
    make validate  # precision / recall / F1 against the TerraScan clips

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
