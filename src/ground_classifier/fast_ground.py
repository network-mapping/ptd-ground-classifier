"""Fast ground engine: Python side of the compiled core (ptdcore.c).

    run = GroundRun(x, y, z, settings).run()

x, y, z are the candidate points of one block, in metres; settings is a `Settings` (below; omitted = all defaults). Then, per point:
    run.ground()   True for points on the ground surface
    run.heights()  height above the finished ground surface (metres; 0 for the points the surface is built from,
                   within `ground_band.width` for the other ground points)
    run.low()      True for isolated low points that were kept out of the surface
    run.model()    True for the points the surface is triangulated from
and per block: run.followed, run.turned, run.spikes_removed (points added or taken out by the later steps),
run.seed_square (side of the starting squares used, in metres) and run.check() (0 when the triangulation is valid).

GroundRun is the API. `isolated_low_points` and `choose_seeds` are its first two stages, kept as plain functions of
arrays (no run state, no compiled core) so that each can be used and tested on its own. Everything that touches the
compiled core goes through `ptdcore` (loading the library, the arrays it works on, their memory); this module decides
what to ask of it.
"""
from dataclasses import dataclass, fields

import numpy as np

from . import ptdcore

# The settings, grouped by the stage that reads them. The values here are THE defaults: nothing else in the program
# repeats them. Each group is frozen; change one with dataclasses.replace, or build a whole Settings:
#     Settings(follow=Follow(rounds=0), turn=Turn(rounds=0))
# A name that is not a field is an error, so a misspelt setting cannot be silently ignored. The names carry
# TerraScan's where TerraScan has the same setting. All lengths are metres, all angles degrees.
# The scale of the integer coordinates is not a setting: see ptdcore.COORDINATE_SCALE.

@dataclass(frozen=True)
class LowPoints:
    """`isolated_low_points`: the filter before the surface is built (and the give-back after it, see `drop`)."""
    cell: float = 1.0                   # m: isolated low points are searched among the lowest point of each such square
    drop: float = 0.5                   # m: below every neighbour by more than this = isolated low point
    radius: float = 3.0
    neighbours: int = 8
    passes: int = 6


@dataclass(frozen=True)
class Seeds:
    """`choose_seeds`: the starting points."""
    max_building_size: float = 60.0     # m: one lowest point per square of this size starts the surface
    square_min: float = 15.0            # m: ... but never below this
    slope_half: float = 10.0            # deg: where the starting points show ground this steep, the starting square is halved ...
    slope_quarter: float = 20.0         # deg: ... and this steep, quartered (large flat roofs are not found on hillsides)

    def __post_init__(self):
        if self.max_building_size <= 0:
            raise ValueError("max_building_size must be above 0")


@dataclass(frozen=True)
class Acceptance:
    """The plain progressive TIN: which points a triangle takes."""
    terrain_angle: float = 70.0         # deg: steepest ground triangle that may still take points
    iteration_angle: float = 6.0        # deg
    steep_iteration_angle: float = 10.0  # deg: the angle used on steep ground (0 = the same angle everywhere)
    flat_slope: float = 10.0            # deg: ground no steeper than this gets iteration_angle ...
    steep_slope: float = 35.0           # deg: ... ground at least this steep gets steep_iteration_angle
    iteration_distance: float = 1.4     # m
    reduce_angle_edge: float = 2.0      # m: 0 = off ("Reduce iteration angle when edge length <")
    reduce_angle_edge_steep: float = 0.5  # m: the same on steep ground (see steep_slope); -1 = the same everywhere
    stop_edge: float = 0.0              # m: 0 = off ("Stop triangulation when edge length <")
    below_surface_free: float = 0.15    # m: a point this far below the surface is taken without the angle test
    below_surface_max: float = 5.0      # m: but never deeper than this (deeper = an error, not ground)

    def __post_init__(self):
        if self.iteration_angle <= 0:
            raise ValueError("iteration_angle must be above 0")


@dataclass(frozen=True)
class Follow:
    """Following the surface past the "chords" the plain routine cannot climb."""
    rounds: int = 200                   # 0 = plain progressive TIN only
    chord_edge: float = 2.0             # m: a triangle with a longer edge is a "chord": the surface may be followed past it
    step_min: float = 0.5               # m: step length, from the known ground vertex to the new point
    step_max: float = 3.0
    up: float = 6.0                     # deg: the surface may bend up by no more than this per step
    down: float = 30.0                  # deg: ... and down by no more than this
    slope: float = 70.0                 # deg: steepest step, and steepest guide
    guide_radius: float = 2.0           # m: the guide is a plane fitted to the ground already found within this radius
    guide_rms: float = 0.06             # m: ... which must fit it this well (rougher ground gives no guide)
    guide_tilt: float = 2.5             # deg: ... and pin its tilt down to within this
    check_tol: float = 0.3              # m: following only where another corner of the triangle lies this far below the extension


@dataclass(frozen=True)
class Turn:
    """Turning a corner onto a new plane (see ptd_hinge in ptdcore.c); also uses Follow.chord_edge."""
    rounds: int = 60                    # 0 = never turn a corner
    min_dist: float = 0.3               # m: the lowest points are looked at from this far from the ground vertex ...
    max_dist: float = 3.0               # m: ... to this far
    tol: float = 0.05                   # m: how flat their underside must be
    slope: float = 60.0                 # deg: steepest plane that may be turned onto
    bins: int = 6                       # how many sector-and-distance bins (of 20) must agree
    reach: float = 1.0                  # m: ... over at least this range of distances
    check_radius: float = 6.0           # m: no turning onto a plane that known ground this close lies ...
    check_tol: float = 0.25             # m: ... this far below


@dataclass(frozen=True)
class Spikes:
    """Taking single points that stand out of the finished surface back out."""
    rounds: int = 5                     # 0 = leave spikes in
    up_angle: float = 15.0              # deg: a vertex standing this steeply above all but one of its neighbours ...
    up_height: float = 0.04             # m: ... and at least this much above them is taken back out
    down_angle: float = 45.0            # deg: a pit must be this steeply below ALL its neighbours ...
    down_height: float = 0.30           # m: ... and this deep
    neighbours: int = 4


@dataclass(frozen=True)
class GroundBand:
    """The last step: the points near the finished surface that are ground too."""
    width: float = 0.02                 # m: points this close to the finished surface are ground too (0 = surface points only)
    edge: float = 5.0                   # m: ... but only in triangles with no edge longer than this
    cell: float = 0.3                   # m: ... and at most one of them per square of this size (the closest to the surface)


@dataclass(frozen=True)
class Settings:
    low_points: LowPoints = LowPoints()
    seeds: Seeds = Seeds()
    acceptance: Acceptance = Acceptance()
    follow: Follow = Follow()
    turn: Turn = Turn()
    spikes: Spikes = Spikes()
    ground_band: GroundBand = GroundBand()

    def flat(self):
        """{"group.field": value} for every setting."""
        return {f"{g.name}.{f.name}": getattr(getattr(self, g.name), f.name)
                for g in fields(self) for f in fields(getattr(self, g.name))}

    def changed(self):
        """The settings that differ from the defaults, as {"group.field": value}."""
        default = Settings().flat()
        return {k: v for k, v in self.flat().items() if v != default[k]}


# Conversions from the settings above to the parameter structs the compiled core (ptdcore) takes: the core wants
# sines, tangents and plain numbers, not degrees and dataclasses.

def _to_ptdcore_acceptance(a):
    """The acceptance test of the plain progressive TIN, from the settings (degrees become sines and tangents)."""
    return ptdcore.AcceptanceParams(
        scale=ptdcore.COORDINATE_SCALE,
        sin_flat=float(np.sin(np.radians(a.iteration_angle))),
        sin_steep=float(np.sin(np.radians(max(a.steep_iteration_angle, a.iteration_angle)))),
        tan_flat=float(np.tan(np.radians(a.flat_slope))),
        tan_steep=float(np.tan(np.radians(max(a.steep_slope, a.flat_slope + 1.0)))),
        up_dist=float(a.iteration_distance),
        down_dist=float(a.below_surface_max),
        below_free=float(a.below_surface_free),
        tan2_terrain=float(np.tan(np.radians(min(a.terrain_angle, 89.9))) ** 2),
        reduce_edge=float(a.reduce_angle_edge),
        reduce_edge_steep=float(a.reduce_angle_edge_steep),
        stop_edge=float(a.stop_edge))


def _to_ptdcore_follow(f):
    return ptdcore.FollowParams(
        chord_edge=float(f.chord_edge), step_min=float(f.step_min), step_max=float(f.step_max),
        tan_up=float(np.tan(np.radians(f.up))), tan_down=float(np.tan(np.radians(f.down))),
        tan_max_slope=float(np.tan(np.radians(f.slope))),
        guide_radius=float(f.guide_radius), guide_rms=float(f.guide_rms),
        tan_guide_tilt=float(np.tan(np.radians(f.guide_tilt))), check_tol=float(f.check_tol))


def _to_ptdcore_turn(t, chord_edge):
    return ptdcore.TurnParams(
        chord_edge=float(chord_edge), rmin=float(t.min_dist), rmax=float(t.max_dist),
        tol=float(t.tol), tan_max_slope=float(np.tan(np.radians(t.slope))), min_bins=int(t.bins),
        min_reach=float(t.reach), check_radius=float(t.check_radius), check_tol=float(t.check_tol))


def _to_ptdcore_spikes(s):
    return ptdcore.SpikeParams(
        tan_up=float(np.tan(np.radians(s.up_angle))), h_up=float(s.up_height),
        tan_down=float(np.tan(np.radians(s.down_angle))), h_down=float(s.down_height),
        min_neighbours=int(s.neighbours))


def isolated_low_points(xi, yi, z, cell, neighbours, radius, drop, passes, min_neighbours=3):
    """Isolated low points - the job TerraScan's separate 'Low points' routine does before Ground.

    Works in two steps:

    1. Candidates: the data is gridded into `cell`-sized squares and the lowest point of each square is taken.
       This step looks at one square at a time, with no reference to neighbouring squares.
    2. Test: each candidate is compared with the other candidates (the lowest points of the surrounding
       squares) within `radius`. It is a 'low point' when it lies more than `drop` below EVERY one of them: real
       ground always has a neighbour near its own level (the downhill one on a slope, the next one along a
       ditch). Only the per-square lowest points take part; other points in the neighbouring squares are not
       compared.

    Repeated, so a small group of low errors is peeled away one layer at a time: a flagged point is set aside
    and the next-lowest point of its square becomes that square's candidate in the following pass.

    Parameters
    ----------
    xi, yi : integer arrays, one value per point
        Horizontal coordinates as the core holds them: integer steps (`ptdcore.metres_to_steps`), counted from the
        block's lower-left corner (non-negative).
    z : float array, one value per point
        Elevations in metres, compared directly with `drop`.
    cell : float
        Side of the squares, in metres, whose lowest point is examined (the `low_points.cell` setting).
    neighbours : int
        How many nearest lowest points are compared with each one (the `low_points.neighbours` setting), at most.
    radius : float
        How far away, in metres, a neighbour may be (the `low_points.radius` setting).
    drop : float
        How far, in metres, a point must lie below every one of its neighbours to be flagged
        (the `low_points.drop` setting).
    passes : int
        Maximum number of peeling rounds (the `low_points.passes` setting). Stops early when a round flags nothing.
    min_neighbours : int, default 3
        The fewest neighbours that must lie inside `radius` for a point to be tested. A point with fewer is never
        flagged: with so few neighbours there is too little to compare its height against. Must be between 0 and
        `neighbours` (no point can have more than `neighbours` neighbours, so a larger value would flag nothing).
        Not a user setting; `isolated_low_points` is called without it, so the default applies.

    Returns
    -------
    low : bool array, one value per point
        True for the points flagged as isolated low points.
    cell_min : int array, one value per occupied square
        Index of the lowest point of each square that was not flagged. A square whose points were all flagged
        is left out.
    cell_key : int array, same length as `cell_min`
        The square of each entry of `cell_min`, as row * width + column (row = y // step, column = x // step).
    width : int
        Number of square columns, needed to split `cell_key` back into row and column.
    step : int
        Side of a square in steps of `xi` / `yi`.

    Raises
    ------
    ValueError
        If `min_neighbours` is negative or greater than `neighbours`."""
    
    if not 0 <= min_neighbours <= neighbours:
        raise ValueError(
            f"min_neighbours ({min_neighbours}) must be between 0 and neighbours ({neighbours}): "
            f"a point cannot have more than {neighbours} neighbours, so no point could ever be flagged")
    from scipy.spatial import cKDTree
    n = z.size
    step = max(1, ptdcore.length_to_steps(cell))
    width = int(xi.max()) // step + 1
    key = (yi // step).astype(np.int64) * width + (xi // step)
    order = np.lexsort((z, key))            # by points by square, with each square ordered by lowest z
    skey = key[order]
    starts = np.flatnonzero(np.concatenate(([True], skey[1:] != skey[:-1]))) # index of start of each square in the ordered array
    ends = np.concatenate((starts[1:], [n]))                  # index of end of each square in the ordered array
    cursor = starts.copy()                   # position of the lowest not-yet-flagged point of each square in the ordered array
    low = np.zeros(n, bool)
    for _ in range(passes):
        live = cursor < ends
        mins = order[cursor[live]]
        if mins.size <= neighbours:
            break
        xy = ptdcore.steps_to_metres(np.column_stack((xi[mins], yi[mins])))
        dist, nb = cKDTree(xy).query(xy, k=neighbours + 1, distance_upper_bound=radius)
        found = np.isfinite(dist[:, 1:])
        nz = np.where(found, z[mins][np.minimum(nb[:, 1:], mins.size - 1)], np.inf)
        flagged = (found.sum(axis=1) >= min_neighbours) & (z[mins] < nz.min(axis=1) - drop)
        if not flagged.any():
            break
        low[mins[flagged]] = True
        cursor[np.flatnonzero(live)[flagged]] += 1
    live = cursor < ends
    return low, order[cursor[live]], skey[cursor[live]], width, step


def choose_seeds(cell_min, cell_key, width, z, max_building_size, cell, slope_half, slope_quarter, square_min):
    """The starting points of the surface: the lowest clean point of every building-size square.

    From the lowest clean point of each `cell`-sized square (see `isolated_low_points`), takes the lowest of every
    `max_building_size`-sized square. The surface starts as a triangulation of these points, so each must lie on
    real ground: a square as big as the biggest building has a point outside it.

    Two adjustments:

    - A block smaller than about two building squares (a sliver of a tile at the edge of the project) would get
      fewer than the three points a surface needs: the square is halved until there are enough.
    - On steep ground the square is halved or quartered: the lowest point of a 60 m square on a hillside sits at its
      downhill edge and says nothing about the rest of it, and there are no 60 m flat roofs on a hillside. The slope
      is judged from the lowest points of neighbouring starting squares.

    Parameters
    ----------
    cell_min : int array, one value per occupied square
        Index into `z` of the lowest clean point of each `cell` square.
    cell_key : int array, same length as `cell_min`
        The square of each entry, as row * width + column.
    width : int
        Number of square columns, to split `cell_key` back into row and column.
    z : float array, one value per point
        Elevations in metres.
    max_building_size : float
        Side of a starting square, in metres (the `seeds.max_building_size` setting), rounded to whole `cell` squares.
    cell : float
        Side of the `cell` squares, in metres (the `low_points.cell` setting).
    slope_half, slope_quarter : float
        Degrees (the `seeds.slope_half` and `seeds.slope_quarter` settings). Where the lowest points of neighbouring
        starting squares rise at least this steeply, the square is halved, or quartered. A `slope_half` of 0 or
        less turns this off.
    square_min : float
        The starting square is never made smaller than this, in metres, on steep ground (the `seed_square_min`
        setting).

    Returns
    -------
    seeds : int32 array
        Indices into `z` of the starting points, in ascending order.
    seed_square : float
        Side of the starting squares used, in metres, after any halving for a small block.
    squares_split : int
        How many starting squares were found steep enough to be made smaller.

    Raises
    ------
    ValueError
        If fewer than three starting points can be found."""
    ratio = max(1, int(round(max_building_size / cell)))
    if cell_min.size < 3:
        raise ValueError("fewer than three starting points")
    cx = cell_key % width; cy = cell_key // width
    zc = z[cell_min]
    while True:
        bx = cx // ratio; by = cy // ratio
        o = np.lexsort((zc, bx, by))
        first = np.concatenate(([True], (bx[o][1:] != bx[o][:-1]) | (by[o][1:] != by[o][:-1])))
        seeds = np.sort(cell_min[o[first]])
        if seeds.size >= 3 or ratio == 1:
            break
        ratio = max(1, ratio // 2)
    t_half = np.tan(np.radians(slope_half)); t_quarter = np.tan(np.radians(slope_quarter))
    squares_split = 0
    if ratio >= 4 and seeds.size >= 3 and t_half > 0.0:
        nbx = int(bx.max()) + 1; nby = int(by.max()) + 1
        coarse = np.full((nby + 2, nbx + 2), np.nan)
        coarse[by[o[first]] + 1, bx[o[first]] + 1] = zc[o[first]]
        centre = coarse[1:-1, 1:-1]
        rise = np.zeros_like(centre)
        for dy in (0, 1, 2):
            for dx in (0, 1, 2):
                d = np.abs(coarse[dy:dy + nby, dx:dx + nbx] - centre) / (np.hypot(dx - 1, dy - 1) or 1.0)
                rise = np.fmax(rise, np.nan_to_num(d))
        slope = rise / (ratio * float(cell))                     # per coarse square
        level = (slope >= t_half).astype(np.int64) + (slope >= t_quarter)
        lv = level[by, bx]                                       # per small-square minimum
        if lv.any():
            smallest = max(1, int(round(square_min / cell)))
            sub = np.minimum(ratio, np.maximum(smallest, ratio >> lv))   # side of the starting square, in small squares
            sx_ = (cx % ratio) // sub; sy_ = (cy % ratio) // sub
            key2 = ((by * nbx + bx) * 4096 + sy_ * 64 + sx_)
            o2 = np.lexsort((zc, key2))
            first2 = np.concatenate(([True], key2[o2][1:] != key2[o2][:-1]))
            seeds = np.sort(cell_min[o2[first2]])
            squares_split = int((level > 0).sum())
    if seeds.size < 3:
        raise ValueError("fewer than three starting points")
    return np.ascontiguousarray(seeds, dtype=np.int32), ratio * float(cell), squares_split


class GroundRun:
    """One run of the core on one block of candidate points: GroundRun(x, y, z, settings).run().

    The constructor only prepares the coordinates. run() does the work in stages: the low-point filter, the starting
    points, the arrays and state of the compiled core, growing the surface, giving back low points the finished
    surface reaches, and taking out spikes. Calling run() again returns the finished run.

    `initial_capacity`, if set before run(), replaces the first size of the triangle arrays (the tests use it to
    force them to be enlarged many times)."""

    def __init__(self, x, y, z, settings=None, lib=None):
        p = settings or Settings()
        self.settings = p
        self._lib = lib or ptdcore.load()
        n = int(x.size)
        self.n = n
        # The core takes horizontal coordinates only as integer steps (ptdcore.COORDINATE_SCALE), counted here from
        # the block's lower-left corner; x and y must be in metres.
        x0, y0 = float(x.min()), float(y.min())
        xi = ptdcore.metres_to_steps(x, x0)
        yi = ptdcore.metres_to_steps(y, y0)
        xmax, ymax = int(xi.max()), int(yi.max())
        margin = ptdcore.length_to_steps(min(max(p.seeds.max_building_size, 30.0), 100.0))
        if max(xmax, ymax) + 2 * margin >= 2 ** 28:
            raise ValueError("block too large for the compiled core at this scale")
        # memory order = spatial order (2 m rows, then x): the candidates of one triangle sit close together in memory
        row = max(1, ptdcore.length_to_steps(2.0))
        order = np.argsort((yi // row) * (xmax + 1) + xi, kind="stable")
        self.order = order
        self._xi = xi[order]; self._yi = yi[order]
        self._zz = np.asarray(z, dtype=np.float64)[order]
        self._extent = (xmax, ymax, margin)         # of the block in integer steps, and the margin of the corner points

        self._core = None                           # made by run()
        self.initial_capacity = None
        self.turns = 0
        self.turned = 0
        self.spikes_removed = 0
        self.rounds = 0
        self.followed = 0
        self.restored = 0

    def run(self):
        if self._core is not None:          # already run
            return self
        cell_min, cell_key, width = self._filter_low_points()
        self._choose_seeds(cell_min, cell_key, width)
        self._build_state()
        self._grow()
        if self.low_sorted.any():
            # A point set aside as an isolated low point that the finished surface nevertheless comes down to
            # (within low_points.drop) was not isolated after all: give it back and let the surface settle again.
            back = self.low_sorted & (self._core.heights() >= -float(self.settings.low_points.drop))
            if back.any():
                self._core.unlock(back)
                self.low_sorted[back] = False
                self.restored = int(back.sum())
                self._core.mark_all_touched()   # unlocking changes what a triangle may take: look at all of them again
                self._core.requeue()
                self._grow()
        self._prune_spikes()
        return self

    def _filter_low_points(self):
        """Find the isolated low points; returns what `choose_seeds` needs."""
        p = self.settings
        low, cell_min, cell_key, width, _ = isolated_low_points(
            self._xi, self._yi, self._zz, p.low_points.cell, int(p.low_points.neighbours),
            p.low_points.radius, p.low_points.drop, int(p.low_points.passes))
        self.low_sorted = low
        return cell_min, cell_key, width

    def _choose_seeds(self, cell_min, cell_key, width):
        s = self.settings.seeds
        self.seeds, self.seed_square, self.seed_squares_split = choose_seeds(
            cell_min, cell_key, width, self._zz, s.max_building_size, self.settings.low_points.cell, s.slope_half,
            s.slope_quarter, s.square_min)

    def _build_state(self):
        """Hand the block to the compiled core: the arrays it works on, and the first allocation of triangle memory."""
        xmax, ymax, margin = self._extent
        X, Y, Z = ptdcore.with_corners(self._xi, self._yi, self._zz, self.seeds, xmax, ymax, margin)
        del self._xi, self._yi, self._zz        # the core holds its own arrays now
        self._core = ptdcore.Core(X, Y, Z, self.seeds, self.low_sorted,
                                  _to_ptdcore_acceptance(self.settings.acceptance),
                                  self.initial_capacity, self._lib)

    def _follow(self):
        """Plain densification, then following the surface past its edge, until nothing more comes of it."""
        core = self._core
        follow = _to_ptdcore_follow(self.settings.follow)
        core.densify()
        for _ in range(int(self.settings.follow.rounds)):
            k = core.frontier(follow)
            if k == 0:
                break
            self.followed += core.insert_staged(k)
            self.rounds += 1
            core.densify()

    def _grow(self):
        core = self._core
        turn = _to_ptdcore_turn(self.settings.turn, self.settings.follow.chord_edge)
        self._follow()
        for _ in range(int(self.settings.turn.rounds)):
            core.mark_all_touched()     # following has just cleared the marks; turning looks at every long triangle again
            k = core.hinge(turn)
            if k == 0:
                break
            done = core.insert_staged(k)
            self.turned += done
            self.turns += 1
            if done == 0:
                break
            self._follow()

    def _prune_spikes(self):
        """Take spikes back out of the finished surface, let the plain test refill, and look again."""
        core = self._core
        spikes = _to_ptdcore_spikes(self.settings.spikes)
        for _ in range(int(self.settings.spikes.rounds)):
            k = core.spikes(spikes)
            if k == 0:
                break
            done = core.remove_staged(k)
            self.spikes_removed += done
            if done == 0:
                break
            core.densify()

    def _unsort(self, values):
        out = np.empty(self.n, values.dtype)
        out[self.order] = values
        return out

    def model(self):
        """The points the surface is triangulated from."""
        return self._unsort(self._core.surface_mask())

    def _members_sorted(self):
        """Points in the thin band about the surface, thinned to one per small square: the closest to the surface."""
        b = self.settings.ground_band
        band = float(b.width)
        if band <= 0.0:
            return np.zeros(self.n, bool)
        k, members = self._core.members(band, band, float(b.edge))
        cell = float(b.cell)
        if k == 0 or cell <= 0.0:
            return members
        idx = np.flatnonzero(members)
        step = max(1, ptdcore.length_to_steps(cell))
        x, y = self._core.x, self._core.y
        key = (y[idx].astype(np.int64) // step) * (int(x.max()) // step + 1) + x[idx] // step
        off = np.abs(self._core.heights()[idx])
        o = np.lexsort((off, key))
        first = np.concatenate(([True], key[o][1:] != key[o][:-1]))
        keep = np.zeros(self.n, bool)
        keep[idx[o[first]]] = True
        return keep

    def ground(self):
        """Surface points plus the points in the thin band about the surface."""
        return self._unsort(self._core.surface_mask() | self._members_sorted())

    def heights(self):
        return self._unsort(self._core.heights())

    def low(self):
        return self._unsort(self.low_sorted)

    def check(self):
        return self._core.check()
