# utils: command line, configuration and logging

Turns **command line arguments, an environment variable and one YAML config file** into a tree of **typed, validated dataclasses** (`Config`). Application code receives that tree, or fields of it, and never reads `argparse`, `os.environ` or the config file itself.

```
 argv ──► args.py ──► args (Namespace) ─┐
 $GROUND_CLASSIFIER_CONFIG ────────────┼─► config.py: load_config(args) ──► params.py (Config) ──► app
 ground_classifier_config.yml ─────────┘          (no file: the defaults on the params)
```

## Files

| File | Role |
| --- | --- |
| `args.py` | `get_args(argv=None)` defines the classify command line and its cross-option checks. Returns an `argparse.Namespace`. No business logic. |
| `config.py` | `load_config(args) -> Config`: finds the config file, loads each section into its param dataclass (`cls(**data)`), maps the args onto the CLI params. Also `default_config_text()` / `write_default_config()` for `init-config`. It parses and validates nothing itself. |
| `params.py` | The dataclasses: defaults on the fields (with their documentation in `metadata["doc"]`), normalisation and validation in `__post_init__`. |
| `helpers.py` | `normalize_path` (`expanduser().resolve()`) and `env_default` (first non-empty environment variable, else a fallback). |
| `loggers.py` | `log()`: atomic prints, a per-thread capture buffer (`_LOG_STATE.buffer`) and a per-process line prefix (`_LOG_STATE.prefix`). |

## The config file

One YAML file; each top-level key is a section, and its keys are the fields of the section's param class. Every section and key is optional: what is missing takes the default on the field. Unknown sections and keys are errors.

| Section | Param class | Holds |
| --- | --- | --- |
| `input` | `InputParam` | `units`: `auto`, `usft` or `metres` (`meters` is accepted). `neighbour_halo`: metres of neighbouring-tile context read around each tile (default 10; 0 = none). |
| `performance` | `PerformanceParam` | `workers`, `ram_limit_gb`. |
| `dtm` | `DtmParam` | `cell_size`. |
| `hag` | `HagParam` | `low_veg_max`, `medium_veg_max`, `high_veg_max`; `bands` gives the tuple; ordering is checked. |
| `engine` | `EngineParam` | One field per engine setting. Defaults come from `fast_ground.Settings`, not from copies. Each field's metadata says which `Settings` field it sets (`"setting"`) or which groups a switch turns off (`"switch"`); `EngineParam.settings()` builds the `Settings`. |

**Finding the file**: `--config FILE`, then `$GROUND_CLASSIFIER_CONFIG` (an empty variable counts as unset), then `./ground_classifier_config.yml`, then `./config/ground_classifier_config.yml`, all relative to the current directory. With none of them the built-in defaults are used and a message is logged. A path given by flag or variable must exist.

**`ground-classify init-config [FILE]`** writes every setting at its default, with its documentation as comments (`default_config_text()` reads the field metadata), to `./ground_classifier_config.yml` by default. It will not overwrite a file without `--force`. The documentation of a setting is written once, on its field.

## The `Config` tree

| Attribute | Source | Holds |
| --- | --- | --- |
| `path: PathParam` | CLI | `inputs`, `output`, `config` (None = built-in defaults), `export_tin`, `export_dtm`. Every path is normalised in `__post_init__` (None skipped), except `inputs`, which stay as typed because the manifest records them so. A given `config` must be a file. |
| `options: OptionParam` | CLI | `units`, `workers`, `ram_limit_gb`, `dtm_cell`, `tin_max_edge`, `overwrite`, `anchor_seeds`, exactly as given (0 / None = not given). |
| `input_config`, `performance_config`, `dtm_config`, `hag_config`, `engine_config` | config file | The five sections above. |
| `engine_settings` | `engine_config` + CLI | The `fast_ground.Settings` that `engine_config` makes; `--anchor-seeds` is applied here. |

`Config.units`, `.workers`, `.ram_limit_gb` and `.dtm_cell` give the effective value: **a command line option, when given, wins over the config file**. `Config.as_dict()` is the `config` block of the manifest (`config_file`, then the file's values).

Rule of thumb: **where things are and what to do this run come from the command line; tuning values come from the config file.**

## Behaviour to preserve

- **`--output-dir`** defaults to `./output` in the *current working directory*, not the repository.
- An **invalid** value exits with a message naming the key, because silently ignoring a setting would make the manifest misleading. Errors are `SystemExit` messages.
- `get_args` is only for a classify run. `compare`, `dtm-diff` and `init-config` are dispatched on `argv[0]` by the top-level `cli.main` before any parsing.
- Option cross-checks use `parser.error`, so they exit with code 2 and the usage line.

## How it is used: top-level `cli.py`

```python
def main(argv=None):
    ...
    return classify(load_config(get_args(argv)))
```

`main()` is the only place that calls `get_args()` / `load_config()`. `classify(config)` passes sections or fields down (`config.hag_config.bands`, `config.engine_settings`, `locations.export_tin`, ...). Downstream code does not import `utils.config`. Tests can build the dataclasses directly. `manifest.build_manifest` takes the whole `Config`.

## Logging

`log()` is not the standard `logging` module on purpose. `pipeline` captures each window's messages in a buffer and replays them in order, and sets a `[file] ` prefix per file. `ProcessPoolExecutor` workers are spawned, so `pipeline.configure_*()` re-applies state in each worker. Replacing `log()` needs a spawn-safe worker initializer and changes the console output; do that as a separate change.

## Dependency order

`utils` imports `fast_ground` and `ptdcore` only. It must not import `pipeline` or the top-level `cli`:

`utils` → `fast_ground` → `pipeline` → `cli`

(`pipeline` imports `utils.loggers` and `utils.params`; `fast_ground` imports nothing from `utils`.)

## Adding a setting

1. Add a field to the param class of its section in `params.py`, with a default and its documentation (`_field(default, doc)`, or `_setting` / `_switch` for the engine), and validation in `__post_init__`. `init-config`, the manifest dump and unknown-key checking follow from the field.
2. If it should be overridable at runtime, add an argument in `cli.get_args`, a field on `OptionParam`, map it in `config.load_config`, and give `Config` an effective property.
3. If it belongs in the manifest, add it to `Config.as_dict()`.
4. Pass it down from `classify`; do not read the config from inside the pipeline.

Dependencies: `pyyaml`; otherwise the standard library.
