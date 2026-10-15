# Ground engine

Progressive TIN densification (Axelsson 2000), following the TerraScan-style ground routine. The source paper and notes are in `docs/reference/`.
The triangulation lives in `native/ptdcore.c` (freestanding C, no dependencies);
`src/ground_classifier/fast_ground.py` drives it from Python. The ground is close to
TerraScan's, not identical.

## Algorithm

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
   finished surface with your `config/hag_config.json` bands; above `high_veg_max`
   or a confirmed isolated low point 7. Classes other than 2, 3, 4, 5 and
   13 in the input are left alone.

## Settings (`config/engine_config.json`)

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

## Validation against TerraScan

`make validate` classifies four clips that carry TerraScan's own classification
(`tests/validation/clips`) and scores class 2 against it. Reference result on Apple Silicon (arm64):

| Clip | Precision % | Recall % | F1 |
|---|---|---|---|
| water edge (honi) | 39.4 | 60.4 | 47.7 |
| rock tops (honi) | 35.5 | 63.9 | 45.6 |
| FPL moderate | 52.6 | 85.3 | 65.1 |
| FPL steep | 65.8 | 64.7 | 65.2 |

Average F1 55.9, worst clip 45.6. Four clips are a thin basis: dense urban blocks, bridges and
sparse data are untested. `make test` also checks that the core reproduces, point for point, the ground
recorded on Linux/x86 (`tests/validation/fast_engine_reference.npz`).

Compare two output folders, optionally against TerraScan tiles:

    ground-classify compare output/A output/B data/<project>/output/terra

## Units and classes

The tool reads each file's CRS metadata to decide between US survey feet and metres and stops if it cannot
tell. Use `--units usft|metres` (or `config/units_config.json`) for files without CRS metadata; keep
feet and metric projects in separate runs. Every threshold is in metres; coordinates on disk are never rewritten.

Input classes 2, 3, 4, 5 and 13 are re-labelled: ground 2; the rest 3 / 4 / 5 by height above ground
using `config/hag_config.json`; above `high_veg_max` or a confirmed isolated low point 7. Other classes are kept.
The HAG bands in `config/hag_config.json` are in metres (the shipped values are 2 ft / 10 ft / 2000 ft for feet projects).

## Scheduling

Tiles are processed as one window up to 120 million points; files run side by side within the RAM budget
(`config/performance_config.json`, `--workers`, `--ram-limit-gb`). Each tile's 10 m neighbour context is read
from the other files in the same input folder.
