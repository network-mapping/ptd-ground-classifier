# Adaptive Ground Unified

One tool, replacing the three separate scripts (Turbo2/FPL, Trusted v6, Trusted v8).
Handles US survey feet or metres, all point densities, and flat/forested/rocky/
mountainous terrain from a single script and a single `.bat`.

## The fast engine (switched on in this package)

This package contains two ground engines. The original pipeline ("classic")
is still here, unchanged. The second one ("fast") is the core of the method
rewritten as compiled code. It is a different program, so its ground is
**close to the classic engine's, not identical**.

`engine_config.json` in this package is set to `"engine": "fast"`; the log
then starts with `ENGINE: fast`. Set it to `"classic"` to get the original
pipeline back. If the file is missing, the tool writes one set to
`"classic"`.

This is the second version of the fast engine. The first one was fast and
dense but left spikes in the ground model; this one was reworked for a
smoother ground at the same density and speed (see "Checked on real
output tiles" below).

### What it does

1. **Isolated low points** are set aside (the job of TerraScan's "Low
   points" routine): the lowest point of a 1 m square that lies more than
   0.5 m below its neighbours. They are put back if the finished surface
   comes down to them.
2. **Starting points.** The lowest point of every `max_building_size`
   square. The square is made smaller where the ground is steep (down to
   15 m) and for small tiles that would otherwise get fewer than three
   starting points.
3. **Progressive triangulation**, as TerraScan's Ground routine describes
   it: points are added one at a time while they pass the iteration angle
   and `iteration_distance` tests against the triangle they lie in. The
   iteration angle runs from `iteration_angle` on flat ground (slope up to
   10 degrees) to `steep_iteration_angle` on steep ground (35 degrees and
   more). Triangles steeper than `terrain_angle` take no points - this is
   what stops ground points appearing up tree trunks.
4. **Following the surface.** The plain routine cannot climb a hill top or
   an embankment that holds no starting point. Where the ground stops at a
   long triangle, a plane is fitted through the ground points within 2 m,
   and the lowest point that continues that plane is accepted. The plane is
   only used when the ground points fit it within 6 cm, so brush cannot
   steer it.
5. **Turning corners.** Where the ground bends away behind a ridge or a
   nose, the lowest points around the last ground point are tested as a
   plane through it and accepted if they fit within 5 cm. Not used where
   known ground nearby lies below that plane (that is brush over ground).
6. **Spikes taken out.** A surface point that stands above all but one of
   its neighbours by more than 15 degrees and 4 cm is removed from the
   surface, and so is a pit below all of them by more than 45 degrees and
   0.3 m. Up to five rounds.
7. **Dense ground.** Points within `ground_band` (2 cm) of the finished
   surface are ground as well, at most one per 0.3 m square. This is where
   the density comes from: the surface itself is built from fewer, more
   reliable points, and the points lying on it are added afterwards.
8. **Classes.** Ground 2; everything else 3 / 4 / 5 from height above the
   finished surface with your `hag_config.json` bands; above `high_veg_max`
   or a confirmed isolated low point 7. Classes other than 2, 3, 4, 5 and
   13 in the input are left alone, exactly as with the classic engine.

### Settings (`engine_config.json`)

All in **metres** and degrees, whatever the project unit. The first seven
carry the names of TerraScan's Ground routine settings.

| Setting | Default | Meaning |
|---|---|---|
| `max_building_size` | 60 | One starting point per square of this size. Lower it (20 to 30) on hilly projects where buildings are already in their own class. |
| `terrain_angle` | 70 | Steepest triangle that may still take points. |
| `iteration_angle` | 6 | Used on flat ground. Smaller = thinner, cleaner ground; larger = climbs more. |
| `steep_iteration_angle` | 10 | Used on steep ground. |
| `iteration_distance` | 1.4 | Largest jump up from a triangle. |
| `reduce_iteration_angle_when_edge_shorter_than` | 2.0 | Thins the surface where it is already dense. 0 = off. |
| `stop_triangulation_when_edge_shorter_than` | 0 | 0 = off. |
| `ground_band` | 0.02 | Step 7. 0 = surface points only (much thinner ground). |
| `follow_surface` | true | Steps 4 and 5. false = plain TerraScan-style routine only. |
| `remove_spikes` | true | Step 6. |

An `engine_config.json` written by the first version of the fast engine
(no `"version": 2` line) has its values replaced by these defaults on the
next run; the engine choice is kept and the log says so.

Tiles that already have an output are skipped. To classify them again with
this version, delete the old outputs or run with `--overwrite`.

### Checked on real output tiles

Thirteen tiles classified with the first version of the fast engine were
classified again with this one: five from a flat forested corridor and
eight from a hilly project. "Spike" here means a ground point that lies
this far above or below a plane fitted through the ground points around it.

| | First version | This version |
|---|---|---|
| Ground points, all 13 tiles | 3,058,764 | 3,607,197 (+18%, no tile lower) |
| Spikes over 10 cm, per 100,000 ground points | 1,701 | 1,185 |
| Spikes over 20 cm, per 100,000 | 409 | 177 |
| Spikes over 50 cm, per 100,000 | 85 | 9 |
| Spikes over 2 m (count) | 353 | 6 |

The six left over 2 m sit on tile edges, where the fitted plane has points
on one side only; they are not spikes in the model.

On eleven of the tiles, at most 0.6% of the first version's ground lies
more than 0.5 m above the new surface. On the two steepest it is 2.9% and
5.4%: partly brush the first version had climbed, partly real ground on
hidden ridges that this version does not reach. A smaller
`max_building_size` (20 to 30) helps on such tiles.

### How it compares on the four validation clips

Same clips, same scoring as the validation suite, against TerraScan's
ground. "Surface" columns are in centimetres: how far TerraScan's ground
points lie from this tool's ground surface, and how far this tool's ground
points lie from TerraScan's surface (measured where the other ground has a
point within 1 m).

| Clip | Engine | Time | Precision | Recall | F1 | TerraScan pts vs our surface | Our pts vs TerraScan surface |
|---|---|---|---|---|---|---|---|
| water edge (honi) | classic | 26.8 s | 39 | 42 | 40 | 6.9 | 4.0 |
| | fast | 1.2 s | 39 | 60 | 48 | 8.4 | 3.3 |
| rock tops (honi) | classic | 36.7 s | 28 | 47 | 35 | 14.0 | 18.2 |
| | fast | 1.5 s | 36 | 64 | 46 | 13.1 | 13.2 |
| FPL moderate | classic | 25.2 s | 51 | 80 | 62 | 1.5 | 3.0 |
| | fast | 0.9 s | 53 | 85 | 65 | 1.8 | 2.5 |
| FPL steep | classic | 12.7 s | 72 | 29 | 41 | 5.5 | 12.2 |
| | fast | 0.5 s | 66 | 65 | 65 | 7.9 | 12.2 |

Average F1 44.7 -> 55.9, worst clip 34.7 -> 45.6. On the rock clip both
engines put ground on a rock knob where TerraScan has none; that is most of
the last column there.

**Four clips and thirteen tiles are a thin basis.** Dense urban blocks,
bridges and sparse data are untested. Steps 4 and 5 are the part to watch:
they are what reaches hill tops, and they are also where a wrong step would
put ground on something that is not. Use `Compare_Outputs.bat folderA
folderB [TerraScanFolder]` on real tiles.

### Speed

| Test | classic | fast |
|---|---|---|
| The four validation clips, one core | 101 s | 4.1 s |
| One 1500 ft tile, 15.3 million points, whole tool start to finish, one core | 773 s | 21 s |

Peak memory for the 15.3-million-point tile was 2.8 GB. With the fast
engine a tile is processed as one window, and files run side by side
within the RAM budget (`performance_config.json`).

On a 20-thread, 64 GB Windows machine, the first version of the fast engine
took 1.1 to 1.3 minutes per tile of 37 to 49 million points, LAZ reading
and writing included, with up to 16 tiles running together. This version
measured the same speed as the first one here; it has not been timed on
Windows.

### Linux and Windows

The compiled core is one C file, `ptdcore.c`, with no dependencies at all
(no C library, no allocation; numpy owns the memory). Two builds are
included:

- `libptdcore.so` - Linux, 64-bit x86. Built and tested here.
  `build_core.sh` rebuilds it (one gcc or clang command) for any other
  Linux machine.
- `ptdcore.dll` - Windows, 64-bit. Cross-built here with clang. The first
  version's DLL, built the same way, ran correctly on Windows. **This
  version's DLL has not been run on Windows yet.** The same compiler's
  Linux build gives bit-for-bit the same ground as the gcc build on all
  four clips.

Every process runs a short self-test when it loads the core, and checks
that the core and `fast_ground.py` are the same version. If the core is
missing, blocked, out of date or fails the test, the log says `WARNING:
FAST ENGINE NOT USED - <reason>. Running the classic engine.` and the run
continues with the classic engine - much slower, so check the first lines
of the log.

In a pipeline the engine can also be called directly, without the tool:

```python
import fast_ground
ground, height, low = fast_ground.classify_ground(x, y, z)   # metres in, one flag/height per point out
```

`python test_fast_engine.py validation_suite/clips` (on Windows:
`Test_Fast_Engine.bat`) runs the engine's own checks: valid triangulation,
buildings and walls not climbed, vegetation left out, a hill top reached,
small slivers handled, same answer however memory grows, and - on the four
clips - that this machine produces exactly the ground recorded on Linux
when the package was made. **Run it once on Windows before the first real
run**; it takes about a minute and ends with `ALL OK` or names what failed.

## Speed-up in this version - classification output unchanged

This version is faster and produces exactly the same output files as the
previous one. Nothing about how a point is classified was changed: no
threshold, no window layout, no algorithm step. Three things changed.

**1. Files and windows are scheduled to fill the machine.** The previous
build sized the whole batch by a worst-case memory guess for its largest
file. With large tiles that meant one file at a time for the entire batch,
small tiles included; with small tiles it never ran more than six. The
scheduler now:

- gives every file its own memory estimate, taken from the file's real
  window layout and a point-density count made while the neighbour caches
  are built (an upper bound, with a further 15% margin);
- chooses for each file the mix of "files at once" and "windows in parallel
  inside a file" that gets the most done within the RAM budget and the CPU
  budget (80% of logical CPUs). A file too large to share the machine runs
  by itself with its windows in parallel, exactly as the previous build ran
  it, so no file is run in a slower arrangement than before;
- starts files largest-first, and checks the machine's real free RAM before
  starting each extra file, so it holds back if the estimates turn out low
  or another program needs memory.

The log says what was chosen, for example `Parallel plan: 1 to 16 file(s)
at once, up to 4 window(s) in parallel inside a file`, how many large files
will run one at a time, whether RAM is the limit, and ends with `Most files
running at once: N. Lowest free RAM seen during the run: X GB.`

**2. Identical triangulations are built once.** Inside each window the same
ground model was being triangulated several times in a row (about 40% of all
triangulation builds were exact repeats). The last three are now kept and
reused, and only when the points are element-for-element identical.

**3. Two expressions are computed once per chunk** instead of once per
direction sector in the two-sided break stage. Same expressions, same inputs.

### How "unchanged" was checked

Every comparison below is exact: all classification codes, and for full runs
the entire output file byte for byte, against the previous version.

| Test | Result |
|---|---|
| The 4 validation-suite clips (water edge, rocktops, FPL moderate, FPL steep) | identical; suite table (precision/recall/FP/FN) identical |
| 4 window-sized blocks built from those clips, up to 318 m and 7.4 M points | identical |
| 3-tile feet project (1500 ft tiles, neighbours, one tile split into 2 windows) - sequential; parallel files; parallel windows inside a worker process; a large file run by itself beside shared small files; a tight RAM budget | output files byte-identical in every mode |
| 2-tile metric project (500 m tiles, rocky terrain, parallel windows) | output files byte-identical |
| Scheduler on simulated 358- and 400-tile jobs (20 CPUs / 64 GB and 32 CPUs / 128 GB, tile sizes from 2 M to 65 M points) | every file run once; RAM and CPU budgets never exceeded; by-itself files never shared the machine |

### What was measured, and what to expect

- Measured: one full-size window at 73 points/m2 (318 m, 7.4 M points)
  went from 275 s to 231 s on one core. Sparser data gains more, because
  triangulation is a larger share of its time.
- Measured: two windows of one file on two threads took 3.0 minutes where
  one after the other took 6.1.
- What to expect depends on tile size and RAM. Each window needs roughly
  0.5 GB + 256 bytes per point of working memory, and that is what limits
  how much runs at once:
  - Large tiles (about 58-65 M points on a 64 GB machine): one tile at a
    time with its four windows in parallel, the same arrangement as the
    previous build. The gain there is the per-window saving only.
  - Smaller tiles: several run at once, where the previous build ran them
    one at a time whenever the batch also contained a large tile.
- Not tested: Windows and Python 3.9 with your installed package versions
  beyond the first run log, and LAZ input in the test suite. Compare the
  `Class counts:` lines of a few tiles against a previous run.

## Critical fix in the previous version - please read

**Documentation accuracy pass.** The module's own internal docstring (top
of the .py file) and several code comments still described the *old*,
now-fixed behavior - literally stating "input must be US survey feet" and
hardcoding feet-only unit conversions, directly contradicting the actual
auto-detection logic a few sections below it. This is now corrected
throughout: the docstring, the HAG-band documentation (now describes them
as configurable via `hag_config.json` rather than fixed constants), and
one real user-facing startup log message that was hardcoding "US survey
feet" and the default HAG values regardless of what a given run's files
and config actually resolved to. All changes are comments/docstrings/log
text only - re-validated against the full test suite afterward with
identical results, confirming zero effect on classification.

**A batch-processing crash bug is fixed here.** If a window inside an
otherwise large, dense file happened to land on a coverage void or an
irregular tile boundary and end up with too few points to build a TIN
(`InsufficientGroundSupport`), the tool used to abort the *entire batch* -
every file still queued after the one that hit this would never get
processed. Root cause: the graceful "preserve original classification for
this tiny piece instead of crashing" fallback was only being decided at the
*whole-file* level (skip for anything under 10,000 points), not per
*window* - so a single problem window inside a multi-million-point file had
no fallback available at all. Fixed: each window now independently
qualifies for the same fallback if the window itself (not the file) is
sparse. Verified with a direct reproduction of the exact failure (an empty
window inside a large-file call) - confirmed the unpatched code crashes
with the identical error signature, confirmed the fix resolves it, and
re-ran the full validation suite to confirm this changes nothing about
normal classification (results are identical to before this fix).

An earlier version of this package **always assumed US survey feet input**
and unconditionally converted every coordinate by `x0.3048`, regardless of
what a file actually contained. Run on a genuinely metric project, this
would silently shrink every coordinate by ~3.28x and corrupt the output with
no error message. This is now fixed: the tool reads each file's own CRS
metadata to detect feet vs. metres automatically, and **refuses to guess and
stops with a clear error** rather than proceed when it can't tell. See
"Setting the input unit" below - some of your files (confirmed on the honi
data you sent) have no CRS metadata at all and need an explicit setting.

Also fixed: if you unzip this package and run it in place, the tool's own
output folder (`AdaptiveGround_Output`) and the folder you keep the script
in were checking against each other by name and could wrongly refuse to
find your input files if you happened to name your working folder the same
as an earlier build's output folder. Renamed the output folder and fixed
the comparison to check the actual folder path, not just its name, so this
can't happen again regardless of what you name your working folder.

## Run it

Double-click `Run_AdaptiveGround_Unified.bat`, or drag LAS/LAZ files onto it.
Output goes to `AdaptiveGround_Output\` next to the input, unless you pass
`--output-dir`. Same `--workers`, `--ram-limit-gb`, `--overwrite` flags as before.

On Linux: `python3 adaptive_ground_classifier_unified.py <files or folders>
--output-dir <folder>`, with the same flags. The config files are read from
the folder the script is in.

## Setting the input unit

On first run this writes `units_config.json` beside the script, defaulting
to `"units": "auto"` - it reads each file's CRS metadata and uses that when
unambiguous. If a file has no CRS metadata (this is common - several of
your honi files have none), the tool will stop with a clear error naming
the file. When that happens, edit `units_config.json` and set `"units"` to
`"usft"` or `"metres"` for that folder, then run again. This setting applies
to every file processed from one folder in one run - **keep feet and metric
projects in separate folders**, don't mix them in one run.

## Editing vegetation height (HAG) bands

On first run this writes `hag_config.json` beside the script. Open it, edit
`low_veg_max` / `medium_veg_max` / `high_veg_max` (metres), save — picked up
automatically on the next run. Nothing else needs touching for a per-project
HAG spec change.

## Speed settings

On first run this writes `performance_config.json` beside the script.
Neither setting changes any classification result.

- `ram_limit_gb` - RAM budget, in GB, for files processed at the same time.
  `0` (default) means automatic: 55% of installed RAM. If the log says
  `limited by the RAM budget` and the machine is dedicated to the run, a
  larger value lets more files run at once. Whatever is entered, 4 GB of the
  RAM that is free when the run starts is always kept spare. Use the
  `Lowest free RAM seen during the run` line from a finished run to judge
  how much room there is.
- `workers` - total CPU threads to use. `0` (default) means automatic: 80%
  of logical CPUs.

`--ram-limit-gb` and `--workers` on the command line take priority over the
file.

## Optional stages (these change the result)

On first run this writes `stages_config.json` beside the script. Unlike the
speed settings, switching a stage off changes the output, so every stage is
on by default and the log prints `STAGE SWITCHED OFF: ...` whenever one is
not.

- `two_sided_bank_crest_membership` - the final pass that adds extra
  class-2 points on banks and crests. It never changes the terrain model or
  any height above ground; it only decides whether those points are
  labelled ground (2) or low vegetation (3). It is by far the most
  expensive stage: 53% of processing time at 73 points/m2 and 65% at about
  290 points/m2 in the stage review below.

### Stage review: what each stage costs and what it buys

Each stage was switched off in turn on the four validation clips, which
carry TerraScan's classification of the same points. "Surface" is how far
TerraScan's ground points lie from the triangulated surface through this
tool's ground points (RMSE, average of the four clips).

| Switched off | Time | F1 vs TerraScan | Surface vs TerraScan | Terrain model |
|---|---|---|---|---|
| nothing (standard) | 100% | 44.7 | 7.8 cm | - |
| bank/crest membership | 45% | 44.9 | 8.6 cm | unchanged |
| its TIN audit | 87% | 44.7 | 7.8 cm | unchanged |
| its low-spike audit | 83% | 44.7 | 7.8 cm | unchanged |
| pass 2 | 91% | 44.8 | 7.8 cm | changed |
| upper-surface rescue | 97% | 44.8 | 7.4 cm | changed |
| lower-envelope rescue | 96% | 44.5 | 7.9 cm | changed |
| smooth representative model | 90% | 48.2 | 7.7 cm | changed |
| terminal bank/ridge/escarpment recovery | 87% | 45.3 | 11.3 cm | changed |
| local lower-surface rescue | 61% | 35.3 | 135 cm | changed |
| everything except the core densification | 4% | 15.1 | 258 cm | changed |

Reading it: the bank/crest membership stage costs more than half the time
and, on these clips, changes agreement with TerraScan very little (on the
two FPL clips not at all; on the two rocky/water clips the surface is 1-2
cm closer with it on). The local lower-surface rescue and the terminal
recoveries are essential. The core densification alone is far from usable.
The clips are small and at lower density than production tiles, so judge
any stage on your own tiles before switching it off for a delivery.

### Comparing two outputs

`Compare_Outputs.bat folderA folderB [referenceFolder]` compares the same
tiles classified two ways: how many points changed class, the ground point
counts, and how far apart the two ground surfaces are. With a reference
folder (the same tiles classified by TerraScan, for example) it also
reports precision, recall and distance to the reference ground for each.

## What's new since the three separate tools

- **Merged, not duplicated.** This script is Turbo2 (the most complete of the
  three) with v8's escarpment/steep-face safety logic folded in. Steep-face
  handling is fully automatic (geometry-gated column-safety test), not a
  v6-vs-v8 choice you have to make per project.
- **Real multi-core use on a single tile.** The old hard 4-window cap meant
  one large/dense tile could never use more than 4 cores no matter how many
  you had. Window count now scales with tile size/density (self-limited by
  a geometry floor so small/sparse tiles still run as one window, no
  overhead added there).
- **HAG bands are now a config file**, not a code edit.
- **A validated, reusable test suite** (see below) — this is the thing to
  run again if you make further changes, so you don't accidentally fix one
  terrain type while breaking another the way the original tools did.

## What I tested this against, and how

I built a 4-tile validation suite from your real data (`_input`/`_terra`
pairs from honi and FPL), comparing this tool's output point-for-point
against TerraScan's actual classification on the same points:

| Terrain | TerraScan ground truth | Precision | Recall |
|---|---|---|---|
| Water edge (honi) | 2.5% | 38.7% | 42.0% |
| Rocky/forested (honi) | 3.5% | 27.6% | 46.9% |
| FPL, moderate terrain | 2.4% | 51.1% | 80.0% |
| FPL, steep mountain slope | 31.7% | 71.9% | 29.0% |

For comparison, the ORIGINAL FPL script's raw input classification (before
any reclassification) had precision as low as 12-20% with near-100% recall —
i.e. it was calling almost everything "ground," including most vegetation.
Every number above is a real improvement over that starting point. It is
**not** TerraScan-level agreement yet, and I want to be direct about that
rather than overstate it.

## Honest limitations - please read before relying on this in production

**Steep mountain terrain is the weak point.** On the fpl_steep test (a real
cliff-like slope), recall is only 29% - it misses most of TerraScan's ground
there. I traced this all the way down: it is not a bug I could find and fix.
Classic Progressive TIN Densification (the algorithm both this tool and
TerraScan's Ground routine implement) has this exact, published limitation -
multiple peer-reviewed papers exist specifically because researchers hit
this same wall (see References below). I could not close this gap with
threshold tuning of any kind - I tried five different, well-reasoned
approaches and none worked, which is itself consistent with the literature
saying this needs a structurally different algorithm, not better constants.
**Expect to do the same manual cleanup on extreme mountain terrain that you
already do after running TerraScan there.**

**Dense/rocky/vegetated terrain (rocktops, water edges) is improved but not
solved.** Precision roughly doubled vs. the raw input, but there is a real,
measured cause I found and only partially addressed: your honi data's true
ground is 2-4x noisier (measured directly from TerraScan's own ground truth)
than the FPL data this tool was originally tuned against, and several of the
tool's internal tolerances are fixed absolute values that don't adjust for
that. I built a self-calibrating mechanism for this (`NOISE_SCALE_ADAPTIVE`
in the script, currently **off**) and tried it nine different ways before
writing this:

  1. Loosen the primary acceptance envelope proportionally to measured noise
     → worse on all four terrains (looser tolerance in already-ambiguous
     terrain just accepts more of the wrong things).
  2. Invert it - tighten proportionally to measured noise → helped 2 of 4
     terrains, made the worst-performing one (rocktops) meaningfully worse
     (recall 46.9%→38.1%). Fails the worst-case-must-not-regress bar.
  3. Leave distance/RMSE tolerances alone, require more independent
     corroborating support points instead in noisier terrain → negligible
     effect in every direction, on every terrain (change reaches a path
     that isn't the dominant one).

None of these nine variations produced a clean win, which is itself useful
to know: this isn't a "find the right multiplier" problem within the
current architecture. The self-calibrating noise-measurement machinery
(`estimate_noise_scale_multiplier`, `configure_noise_scale`,
`noise_scale_multiplier`, thread-safe, already validated to correctly reach
the code paths tested) is real, working infrastructure - what's missing is
the right thing to *do* with the measurement, not the measurement itself.
The most promising untried direction, per the literature (see References):
Dong et al.'s approach doesn't scale a number continuously - it detects
when local point density crosses a threshold and *switches* to a
qualitatively different, stricter acceptance rule above that point, rather
than adjusting the existing rule's tolerance. That's a bigger change than
anything attempted here and is the right next thing to try.

**Never independently verify a AI's classification against a paying
client's actual accuracy requirements. Spot-check real output before
delivery**, especially on any tile with real relief. Treat this the way you
already treat TerraScan output - as a strong starting point, not a final
answer without QC.

## The validation suite (for whoever continues this work)

`validate_suite.py` + the 4 saved `.npz` clips it reads are the thing that
made every real finding in this project possible - before I built it, I
made two confident claims that turned out to be measurably wrong (an
"obvious" speed optimization that was actually 5x slower; a windowing
change that helped one terrain type while quietly being untested on
others). Every real number in this README came from running the full suite,
not from reasoning about what should happen.

To extend it: add more real `_input`/`_terra` `.las` pairs (uncompressed -
`.laz` point data can't be read without `laspy`/`lazrs`, which aren't
available in every environment), extract a small buffered clip (see the
extraction pattern already in the session, roughly 25-45m evaluation core +
30-80m context buffer), save as `.npz` with `x,y,z,c_input,c_terra,core_mask`,
add one line to the `CLIPS` dict in `validate_suite.py`. A genuinely flat
tile would be a good addition - everything tested so far is deliberately
hard terrain, and there's no current guard against a change that helps hard
terrain while quietly hurting the easy case.

## References

The literature confirming classic PTD's known steep-terrain limitation:
- Axelsson, P. (2000), "DEM Generation from Laser Scanner Data Using
  Adaptive TIN Models" - the algorithm this tool and TerraScan both
  implement.
- Zhang, J. & Lin, X. (2013), "Filtering airborne LiDAR data by embedding
  smoothness-constrained segmentation in progressive TIN densification,"
  ISPRS J. Photogramm. Remote Sens. 81 - documents PTD failing "around the
  break lines and steep terrain."
- Zhao, X. et al. (2016), "Improved progressive TIN densification filtering
  algorithm for airborne LiDAR data in forested areas," ISPRS J.
  Photogramm. Remote Sens. 117 - documents PTD tending to "remove ground
  points on steep hills and flatten the terrain."
- Dong, Y. et al. (2018), "An Improved Progressive TIN Densification
  Filtering Method Considering the Density and Standard Variance of Point
  Clouds," ISPRS Int. J. Geo-Inf. - documents PTD's type-I (false-positive)
  errors worsening on high-density, high-variance point clouds; the most
  promising published lead for the dense-vegetation problem above.
