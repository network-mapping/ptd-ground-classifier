"""Fast ground engine: Python side of the compiled core (ptdcore.c).

    run = GroundRun(x, y, z, settings).run()

x, y, z are the candidate points of one block, in metres; settings overrides DEFAULTS below. Then, per point:
    run.ground()   True for points on the ground surface
    run.heights()  height above the finished ground surface (metres; 0 for the points the surface is built from,
                   within `surface_band` for the other ground points)
    run.low()      True for isolated low points that were kept out of the surface
    run.model()    True for the points the surface is triangulated from
and per block: run.followed, run.turned, run.spikes_removed (points added or taken out by the later steps),
run.seed_square (side of the starting squares used, in metres) and run.check() (0 when the triangulation is valid).

GroundRun is the API. `isolated_low_points` and `choose_seeds` are its first two stages, kept as plain functions of
arrays (no run state, no compiled core) so that each can be used and tested on its own. Everything that touches the
compiled core goes through `ptdcore` (loading the library, the arrays it works on, their memory); this module decides
what to ask of it.
"""
import numpy as np

from . import ptdcore

# The settings carry TerraScan's names where TerraScan has the same setting.
DEFAULTS = {
    "max_building_size": 60.0,       # m: one lowest point per square of this size starts the surface
    "terrain_angle": 70.0,           # deg: steepest ground triangle that may still take points
    "iteration_angle": 6.0,          # deg
    "steep_iteration_angle": 10.0,   # deg: the angle used on steep ground (0 = the same angle everywhere)
    "flat_slope": 10.0,              # deg: ground no steeper than this gets iteration_angle ...
    "steep_slope": 35.0,             # deg: ... ground at least this steep gets steep_iteration_angle
    "iteration_distance": 1.4,       # m
    "reduce_angle_edge": 2.0,        # m: 0 = off ("Reduce iteration angle when edge length <")
    "reduce_angle_edge_steep": 0.5,  # m: the same on steep ground (see steep_slope); -1 = the same everywhere
    "seed_square_min": 15.0,         # m: ... but never below this
    "seed_slope_half": 10.0,         # deg: where the starting points show ground this steep, the starting square is halved ...
    "seed_slope_quarter": 20.0,      # deg: ... and this steep, quartered (large flat roofs are not found on hillsides)
    "stop_edge": 0.0,                # m: 0 = off ("Stop triangulation when edge length <")
    "below_surface_free": 0.15,      # m: a point this far below the surface is taken without the angle test
    "below_surface_max": 5.0,        # m: but never deeper than this (deeper = an error, not ground)
    "follow_rounds": 200,            # 0 = plain progressive TIN only
    "follow_chord_edge": 2.0,        # m: a triangle with a longer edge is a "chord": the surface may be followed past it
    "follow_min": 0.5,               # m: step length, from the known ground vertex to the new point
    "follow_max": 3.0,
    "follow_up": 6.0,                # deg: the surface may bend up by no more than this per step
    "follow_down": 30.0,             # deg: ... and down by no more than this
    "follow_slope": 70.0,            # deg: steepest step, and steepest guide
    "follow_guide_radius": 2.0,      # m: the guide is a plane fitted to the ground already found within this radius
    "follow_guide_rms": 0.06,        # m: ... which must fit it this well (rougher ground gives no guide)
    "follow_guide_tilt": 2.5,        # deg: ... and pin its tilt down to within this
    "follow_check_tol": 0.3,         # m: following only where another corner of the triangle lies this far below the extension
    "turn_rounds": 60,               # 0 = never turn a corner (see ptd_hinge in ptdcore.c)
    "turn_min": 0.3,                 # m: the lowest points are looked at from this far from the ground vertex ...
    "turn_max": 3.0,                 # m: ... to this far
    "turn_tol": 0.05,                # m: how flat their underside must be
    "turn_slope": 60.0,              # deg: steepest plane that may be turned onto
    "turn_bins": 6,                  # how many sector-and-distance bins (of 20) must agree
    "turn_reach": 1.0,               # m: ... over at least this range of distances
    "turn_check_radius": 6.0,        # m: no turning onto a plane that known ground this close lies ...
    "turn_check_tol": 0.25,          # m: ... this far below
    "spike_rounds": 5,               # 0 = leave spikes in
    "spike_up_angle": 15.0,          # deg: a vertex standing this steeply above all but one of its neighbours ...
    "spike_up_height": 0.04,         # m: ... and at least this much above them is taken back out
    "spike_down_angle": 45.0,        # deg: a pit must be this steeply below ALL its neighbours ...
    "spike_down_height": 0.30,       # m: ... and this deep
    "spike_neighbours": 4,
    "surface_band": 0.02,            # m: points this close to the finished surface are ground too (0 = surface points only)
    "surface_band_edge": 5.0,        # m: ... but only in triangles with no edge longer than this
    "surface_cell": 0.3,             # m: ... and at most one of them per square of this size (the closest to the surface)
    "low_cell": 1.0,                 # m: isolated low points are searched among the lowest point of each such square
    "low_drop": 0.5,                 # m: below every neighbour by more than this = isolated low point
    "low_radius": 3.0,
    "low_neighbours": 8,
    "low_passes": 6,
    "unit": 0.001,                   # m: horizontal coordinates are held as whole numbers of this
}


def _acceptance_params(p):
    """The acceptance test of the plain progressive TIN, from the settings (degrees become sines and tangents)."""
    return ptdcore.AcceptanceParams(
        unit=float(p["unit"]),
        sin_flat=float(np.sin(np.radians(p["iteration_angle"]))),
        sin_steep=float(np.sin(np.radians(max(p["steep_iteration_angle"], p["iteration_angle"])))),
        tan_flat=float(np.tan(np.radians(p["flat_slope"]))),
        tan_steep=float(np.tan(np.radians(max(p["steep_slope"], p["flat_slope"] + 1.0)))),
        up_dist=float(p["iteration_distance"]),
        down_dist=float(p["below_surface_max"]),
        below_free=float(p["below_surface_free"]),
        tan2_terrain=float(np.tan(np.radians(min(p["terrain_angle"], 89.9))) ** 2),
        reduce_edge=float(p["reduce_angle_edge"]),
        reduce_edge_steep=float(p["reduce_angle_edge_steep"]),
        stop_edge=float(p["stop_edge"]))


def _follow_params(p):
    return ptdcore.FollowParams(
        chord_edge=float(p["follow_chord_edge"]), step_min=float(p["follow_min"]), step_max=float(p["follow_max"]),
        tan_up=float(np.tan(np.radians(p["follow_up"]))), tan_down=float(np.tan(np.radians(p["follow_down"]))),
        tan_max_slope=float(np.tan(np.radians(p["follow_slope"]))),
        guide_radius=float(p["follow_guide_radius"]), guide_rms=float(p["follow_guide_rms"]),
        tan_guide_tilt=float(np.tan(np.radians(p["follow_guide_tilt"]))), check_tol=float(p["follow_check_tol"]))


def _turn_params(p):
    return ptdcore.TurnParams(
        chord_edge=float(p["follow_chord_edge"]), rmin=float(p["turn_min"]), rmax=float(p["turn_max"]),
        tol=float(p["turn_tol"]), tan_max_slope=float(np.tan(np.radians(p["turn_slope"]))), min_bins=int(p["turn_bins"]),
        min_reach=float(p["turn_reach"]), check_radius=float(p["turn_check_radius"]), check_tol=float(p["turn_check_tol"]))


def _spike_params(p):
    return ptdcore.SpikeParams(
        tan_up=float(np.tan(np.radians(p["spike_up_angle"]))), h_up=float(p["spike_up_height"]),
        tan_down=float(np.tan(np.radians(p["spike_down_angle"]))), h_down=float(p["spike_down_height"]),
        min_neighbours=int(p["spike_neighbours"]))


def isolated_low_points(xi, yi, z, unit, cell, neighbours, radius, drop, passes, min_neighbours=3):
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
        Horizontal coordinates as whole multiples of `unit`, measured from the block's lower-left corner
        (non-negative).
    z : float array, one value per point
        Elevations in metres. Not scaled by `unit`: they are compared directly with `drop`.
    unit : float
        Metres per step of `xi` / `yi` (the `unit` setting). It only quantises the horizontal coordinates;
        it is not the project's coordinate unit.
    cell : float
        Side of the squares, in metres, whose lowest point is examined (the `low_cell` setting).
    neighbours : int
        How many nearest lowest points are compared with each one (the `low_neighbours` setting), at most.
    radius : float
        How far away, in metres, a neighbour may be (the `low_radius` setting).
    drop : float
        How far, in metres, a point must lie below every one of its neighbours to be flagged
        (the `low_drop` setting).
    passes : int
        Maximum number of peeling rounds (the `low_passes` setting). Stops early when a round flags nothing.
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
        Side of a square in units of `xi` / `yi`.

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
    step = max(1, int(round(cell / unit)))
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
        xy = np.column_stack((xi[mins], yi[mins])).astype(np.float64) * unit
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
        Side of a starting square, in metres (the `max_building_size` setting), rounded to whole `cell` squares.
    cell : float
        Side of the `cell` squares, in metres (the `low_cell` setting).
    slope_half, slope_quarter : float
        Degrees (the `seed_slope_half` and `seed_slope_quarter` settings). Where the lowest points of neighbouring
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
        p = dict(DEFAULTS)
        if settings:
            p.update(settings)
        self.p = p
        self._lib = lib or ptdcore.load()
        unit = float(p["unit"])
        n = int(x.size)
        self.n = n
        x0, y0 = float(x.min()), float(y.min())
        xi = np.rint((x - x0) / unit).astype(np.int64)
        yi = np.rint((y - y0) / unit).astype(np.int64)
        xmax, ymax = int(xi.max()), int(yi.max())
        margin = int(round(min(max(p["max_building_size"], 30.0), 100.0) / unit))
        if max(xmax, ymax) + 2 * margin >= 2 ** 28:
            raise ValueError("block too large for the compiled core at this unit")
        # memory order = spatial order (2 m rows, then x): the candidates of one triangle sit close together in memory
        row = max(1, int(round(2.0 / unit)))
        order = np.argsort((yi // row) * (xmax + 1) + xi, kind="stable")
        self.order = order
        self._xi = xi[order]; self._yi = yi[order]
        self._zz = np.asarray(z, dtype=np.float64)[order]
        self._extent = (xmax, ymax, margin)         # of the block in whole units, and the margin of the corner points

        self._core = None                           # made by run()
        self.initial_capacity = None
        self._follow_params = _follow_params(p)
        self._turn_params = _turn_params(p)
        self._spike_params = _spike_params(p)
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
            # (within low_drop) was not isolated after all: give it back and let the surface settle again.
            back = self.low_sorted & (self._core.heights() >= -float(self.p["low_drop"]))
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
        p = self.p
        low, cell_min, cell_key, width, _ = isolated_low_points(
            self._xi, self._yi, self._zz, float(p["unit"]), p["low_cell"], int(p["low_neighbours"]), p["low_radius"],
            p["low_drop"], int(p["low_passes"]))
        self.low_sorted = low
        return cell_min, cell_key, width

    def _choose_seeds(self, cell_min, cell_key, width):
        p = self.p
        self.seeds, self.seed_square, self.seed_squares_split = choose_seeds(
            cell_min, cell_key, width, self._zz, p["max_building_size"], p["low_cell"], p["seed_slope_half"],
            p["seed_slope_quarter"], p["seed_square_min"])

    def _build_state(self):
        """Hand the block to the compiled core: the arrays it works on, and the first allocation of triangle memory."""
        xmax, ymax, margin = self._extent
        X, Y, Z = ptdcore.with_corners(self._xi, self._yi, self._zz, self.seeds, xmax, ymax, margin)
        del self._xi, self._yi, self._zz        # the core holds its own arrays now
        self._core = ptdcore.Core(X, Y, Z, self.seeds, self.low_sorted, _acceptance_params(self.p), self.initial_capacity, self._lib)

    def _follow(self):
        """Plain densification, then following the surface past its edge, until nothing more comes of it."""
        core = self._core
        core.densify()
        for _ in range(int(self.p["follow_rounds"])):
            k = core.frontier(self._follow_params)
            if k == 0:
                break
            self.followed += core.insert_staged(k)
            self.rounds += 1
            core.densify()

    def _grow(self):
        core = self._core
        self._follow()
        for _ in range(int(self.p["turn_rounds"])):
            core.mark_all_touched()     # following has just cleared the marks; turning looks at every long triangle again
            k = core.hinge(self._turn_params)
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
        for _ in range(int(self.p["spike_rounds"])):
            k = core.spikes(self._spike_params)
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
        band = float(self.p["surface_band"])
        if band <= 0.0:
            return np.zeros(self.n, bool)
        k, members = self._core.members(band, band, float(self.p["surface_band_edge"]))
        cell = float(self.p["surface_cell"])
        if k == 0 or cell <= 0.0:
            return members
        idx = np.flatnonzero(members)
        step = max(1, int(round(cell / float(self.p["unit"]))))
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
