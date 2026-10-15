"""Fast ground engine: Python side of the compiled core (ptdcore.c).

    ground, height, low = classify_ground(x, y, z)

x, y, z are the candidate points of one block, in metres. Returns, per point:
    ground  True for points on the ground surface
    height  height above the finished ground surface (metres, 0 for ground points)
    low     True for isolated low points that were kept out of the surface
"""
import ctypes
import os
import sys

import numpy as np

_i32p = ctypes.POINTER(ctypes.c_int32)
_f64p = ctypes.POINTER(ctypes.c_double)
_f32p = ctypes.POINTER(ctypes.c_float)
_u8p = ctypes.POINTER(ctypes.c_uint8)

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


class _State(ctypes.Structure):
    _fields_ = [
        ("n", ctypes.c_int64), ("X", _i32p), ("Y", _i32p), ("Z", _f64p), ("next", _i32p), ("ptri", _i32p),
        ("tv", _i32p), ("tn", _i32p), ("thead", _i32p), ("ttail", _i32p), ("tinq", _u8p), ("queue", _i32p), ("stack", _i32p),
        ("mcap", ctypes.c_int64), ("m", ctypes.c_int64), ("qhead", ctypes.c_int64), ("qcount", ctypes.c_int64),
        ("seeds", _i32p), ("nseeds", ctypes.c_int64), ("seed_pos", ctypes.c_int64),
        ("unit", ctypes.c_double), ("sin2_limit", ctypes.c_double), ("up_dist", ctypes.c_double), ("down_dist", ctypes.c_double),
        ("tan2_terrain", ctypes.c_double), ("reduce_edge", ctypes.c_double), ("stop_edge", ctypes.c_double), ("below_free", ctypes.c_double),
        ("n_ground", ctypes.c_int64), ("n_tests", ctypes.c_int64), ("n_flips", ctypes.c_int64), ("initialised", ctypes.c_int32),
        ("locked", _u8p), ("mode", ctypes.c_int32), ("vtouch", _u8p), ("vtri", _i32p),
        ("sin_flat", ctypes.c_double), ("sin_steep", ctypes.c_double), ("tan_flat", ctypes.c_double), ("tan_steep", ctypes.c_double), ("reduce_edge_steep", ctypes.c_double),
    ]


CORE_VERSION = 10   # must equal PTD_VERSION in native/ptdcore.c
CORE_PATH_ENV_VAR = "GROUND_CLASSIFIER_CORE"
_LIB = None


def library_path():
    """The compiled core: $GROUND_CLASSIFIER_CORE, else <repo>/build/ (made by native/build.sh)."""
    override = os.environ.get(CORE_PATH_ENV_VAR)
    if override:
        return override
    if sys.platform == "win32":
        name = "ptdcore.dll"
    elif sys.platform == "darwin":
        name = "libptdcore.dylib"
    else:
        name = "libptdcore.so"
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "..", "build", name))


def load_library(path=None):
    """Load the compiled core. Raises OSError with a plain message if it cannot be used on this machine."""
    global _LIB
    if _LIB is not None and path is None:
        return _LIB
    path = path or library_path()
    if not os.path.exists(path):
        raise OSError(f"compiled core not found: {path} (run `make build`)")
    lib = ctypes.CDLL(path)
    try:
        lib.ptd_version.argtypes = []; lib.ptd_version.restype = ctypes.c_int64
        version = int(lib.ptd_version())
    except AttributeError:
        version = -1
    if version != CORE_VERSION:
        raise OSError(f"compiled core is version {version}, this script needs {CORE_VERSION} (run `make build`)")
    state = ctypes.POINTER(_State)
    lib.ptd_run.argtypes = [state, ctypes.c_int64]; lib.ptd_run.restype = ctypes.c_int
    lib.ptd_heights.argtypes = [state, _f32p]; lib.ptd_heights.restype = None
    lib.ptd_frontier.argtypes = [state, _f64p, _i32p]; lib.ptd_frontier.restype = ctypes.c_int64
    lib.ptd_hinge.argtypes = [state, _f64p, _i32p]; lib.ptd_hinge.restype = ctypes.c_int64
    lib.ptd_insert.argtypes = [state, _i32p, ctypes.c_int64]; lib.ptd_insert.restype = ctypes.c_int64
    lib.ptd_requeue.argtypes = [state]; lib.ptd_requeue.restype = None
    lib.ptd_spikes.argtypes = [state, _f64p, _i32p]; lib.ptd_spikes.restype = ctypes.c_int64
    lib.ptd_remove.argtypes = [state, _i32p, ctypes.c_int64]; lib.ptd_remove.restype = ctypes.c_int64
    lib.ptd_members.argtypes = [state, ctypes.c_double, ctypes.c_double, ctypes.c_double, _u8p]; lib.ptd_members.restype = ctypes.c_int64
    lib.ptd_check.argtypes = [state]; lib.ptd_check.restype = ctypes.c_int64
    lib.ptd_state_size.argtypes = []; lib.ptd_state_size.restype = ctypes.c_int64
    if lib.ptd_state_size() != ctypes.sizeof(_State):
        raise OSError("compiled core does not match this script (run `make build`)")
    _self_test(lib)
    if path == library_path():
        _LIB = lib
    return lib


def _self_test(lib):
    """A two-millisecond run on a known surface: proves the library really works in this process."""
    rng = np.random.default_rng(1)
    x = rng.uniform(0.0, 80.0, 4000); y = rng.uniform(0.0, 80.0, 4000)
    z = 0.05 * x + 0.02 * y + rng.normal(0.0, 0.01, 4000)
    z[:400] += rng.uniform(2.0, 9.0, 400)          # things standing on the surface
    run = GroundRun(x, y, z, {"max_building_size": 20.0}, lib).run()
    ground = run.ground()
    if run.check() != 0 or ground[:400].any() or ground[400:].sum() < 2000:
        raise OSError("compiled core failed its self-test on this machine")


def _ptr(a, t):
    return a.ctypes.data_as(t)


def isolated_low_points(xi, yi, z, unit, cell, neighbours, radius, drop, passes):
    """Isolated low points - the job TerraScan's separate 'Low points' routine does before Ground.

    Looks only at the lowest point of each `cell`-sized square. Such a point is an error when it lies more than
    `drop` below EVERY neighbouring lowest point within `radius`: real ground always has a neighbour near its own
    level (the downhill one on a slope, the next one along a ditch). Repeated, so a small group of low errors is
    peeled away one layer at a time. Returns (low flags, index of the lowest clean point of each square)."""
    from scipy.spatial import cKDTree
    n = z.size
    step = max(1, int(round(cell / unit)))
    width = int(xi.max()) // step + 1
    key = (yi // step).astype(np.int64) * width + (xi // step)
    order = np.lexsort((z, key))            # by square, lowest first
    skey = key[order]
    starts = np.flatnonzero(np.concatenate(([True], skey[1:] != skey[:-1])))
    ends = np.concatenate((starts[1:], [n]))
    cursor = starts.copy()                   # position of the lowest not-yet-flagged point of each square
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
        flagged = (found.sum(axis=1) >= 3) & (z[mins] < nz.min(axis=1) - drop)
        if not flagged.any():
            break
        low[mins[flagged]] = True
        cursor[np.flatnonzero(live)[flagged]] += 1
    live = cursor < ends
    return low, order[cursor[live]], skey[cursor[live]], width, step


class GroundRun:
    """One run of the core on one block of candidate points."""

    def __init__(self, x, y, z, settings=None, lib=None):
        p = dict(DEFAULTS)
        if settings:
            p.update(settings)
        self.p = p
        self.lib = lib or load_library()
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
        xi = xi[order]; yi = yi[order]
        zz = np.asarray(z, dtype=np.float64)[order]

        low, cell_min, cell_key, width, step = isolated_low_points(
            xi, yi, zz, unit, p["low_cell"], int(p["low_neighbours"]), p["low_radius"], p["low_drop"], int(p["low_passes"]))
        self.low_sorted = low
        # seeds: the lowest clean point of every building-size square (taken from the small-square minima)
        # A block smaller than about two building squares (a sliver of a tile at the edge of the project) would get
        # fewer than the three points a surface needs: the square is halved until there are enough.
        ratio = max(1, int(round(p["max_building_size"] / p["low_cell"])))
        if cell_min.size < 3:
            raise ValueError("fewer than three starting points")
        cx = cell_key % width; cy = cell_key // width
        zc = zz[cell_min]
        while True:
            bx = cx // ratio; by = cy // ratio
            o = np.lexsort((zc, bx, by))
            first = np.concatenate(([True], (bx[o][1:] != bx[o][:-1]) | (by[o][1:] != by[o][:-1])))
            seeds = np.sort(cell_min[o[first]])
            if seeds.size >= 3 or ratio == 1:
                break
            ratio = max(1, ratio // 2)
        # On steep ground the square is halved or quartered: the lowest point of a 60 m square on a hillside sits at
        # its downhill edge and says nothing about the rest of it, and there are no 60 m flat roofs on a hillside.
        t_half = np.tan(np.radians(p["seed_slope_half"])); t_quarter = np.tan(np.radians(p["seed_slope_quarter"]))
        self.seed_squares_split = 0
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
            slope = rise / (ratio * float(p["low_cell"]))            # per coarse square
            level = (slope >= t_half).astype(np.int64) + (slope >= t_quarter)
            lv = level[by, bx]                                       # per small-square minimum
            if lv.any():
                smallest = max(1, int(round(p["seed_square_min"] / p["low_cell"])))
                sub = np.minimum(ratio, np.maximum(smallest, ratio >> lv))   # side of the starting square, in small squares
                sx_ = (cx % ratio) // sub; sy_ = (cy % ratio) // sub
                key2 = ((by * nbx + bx) * 4096 + sy_ * 64 + sx_)
                o2 = np.lexsort((zc, key2))
                first2 = np.concatenate(([True], key2[o2][1:] != key2[o2][:-1]))
                seeds = np.sort(cell_min[o2[first2]])
                self.seed_squares_split = int((level > 0).sum())
        if seeds.size < 3:
            raise ValueError("fewer than three starting points")
        self.seed_square = ratio * float(p["low_cell"])
        self.seeds = np.ascontiguousarray(seeds, dtype=np.int32)

        corners = np.array([[-margin, -margin], [xmax + margin, -margin], [xmax + margin, ymax + margin], [-margin, ymax + margin]], dtype=np.int64)
        sx, sy, sz = xi[seeds], yi[seeds], zz[seeds]
        cz = np.array([sz[np.argmin((sx - cx) ** 2 + (sy - cy) ** 2)] for cx, cy in corners])
        self.X = np.ascontiguousarray(np.concatenate((xi, corners[:, 0])), dtype=np.int32)
        self.Y = np.ascontiguousarray(np.concatenate((yi, corners[:, 1])), dtype=np.int32)
        self.Z = np.ascontiguousarray(np.concatenate((zz, cz)), dtype=np.float64)
        del xi, yi, zz
        self.next = np.empty(n, np.int32)
        self.ptri = np.empty(n, np.int32)
        self.locked = np.ascontiguousarray(low, dtype=np.uint8)

        s = self.state = _State()
        s.n = n
        s.X = _ptr(self.X, _i32p); s.Y = _ptr(self.Y, _i32p); s.Z = _ptr(self.Z, _f64p)
        s.next = _ptr(self.next, _i32p); s.ptri = _ptr(self.ptri, _i32p)
        s.seeds = _ptr(self.seeds, _i32p); s.nseeds = self.seeds.size; s.seed_pos = 0
        s.locked = _ptr(self.locked, _u8p)
        self.vtouch = np.ones(n + 4, np.uint8)
        s.vtouch = _ptr(self.vtouch, _u8p)
        self.vtri = np.full(n + 4, -1, np.int32)
        s.vtri = _ptr(self.vtri, _i32p)
        s.unit = unit
        s.sin2_limit = float(np.sin(np.radians(p["iteration_angle"])) ** 2)
        s.sin_flat = float(np.sin(np.radians(p["iteration_angle"])))
        s.sin_steep = float(np.sin(np.radians(max(p["steep_iteration_angle"], p["iteration_angle"]))))
        s.tan_flat = float(np.tan(np.radians(p["flat_slope"]))); s.tan_steep = float(np.tan(np.radians(max(p["steep_slope"], p["flat_slope"] + 1.0))))
        s.up_dist = float(p["iteration_distance"])
        s.down_dist = float(p["below_surface_max"])
        s.below_free = float(p["below_surface_free"])
        s.tan2_terrain = float(np.tan(np.radians(min(p["terrain_angle"], 89.9))) ** 2)
        s.reduce_edge = float(p["reduce_angle_edge"]); s.stop_edge = float(p["stop_edge"])
        s.reduce_edge_steep = float(p["reduce_angle_edge_steep"])
        s.initialised = 0; s.m = 0; s.mode = 0
        self.follow_prm = np.array([
            p["follow_chord_edge"], p["follow_min"], p["follow_max"], np.tan(np.radians(p["follow_up"])),
            np.tan(np.radians(p["follow_down"])), np.tan(np.radians(p["follow_slope"])),
            p["follow_guide_radius"], p["follow_guide_rms"], np.tan(np.radians(p["follow_guide_tilt"])),
            p["follow_check_tol"]], dtype=np.float64)
        self.turn_prm = np.array([
            p["follow_chord_edge"], p["turn_min"], p["turn_max"], p["turn_tol"], np.tan(np.radians(p["turn_slope"])),
            p["turn_bins"], p["turn_reach"], p["turn_check_radius"], p["turn_check_tol"]], dtype=np.float64)
        self.turns = 0
        self.turned = 0
        self.spike_prm = np.array([
            np.tan(np.radians(p["spike_up_angle"])), p["spike_up_height"],
            np.tan(np.radians(p["spike_down_angle"])), p["spike_down_height"], p["spike_neighbours"]], dtype=np.float64)
        self.spikes_removed = 0
        self.rounds = 0
        self.followed = 0
        self.restored = 0
        self.picks = None
        self.mcap = 0
        self._allocate(max(1024, min(2 * n + 8, 4 * self.seeds.size + n // 8 + 1024)))

    def _allocate(self, mcap):
        s = self.state
        old = self.mcap

        def grow(name, per, dtype):
            a = np.empty(per * mcap + (16 if name == "stack" else 0), dtype)
            if old and name != "stack":
                b = getattr(self, name)
                a[: b.size] = b
            setattr(self, name, a)
            return a

        if old:   # straighten the circular queue before it is enlarged
            live = self.queue[(s.qhead + np.arange(s.qcount)) % old].copy()
        grow("tv", 3, np.int32); grow("tn", 3, np.int32)
        grow("thead", 1, np.int32); grow("ttail", 1, np.int32)
        grow("tinq", 1, np.uint8); grow("queue", 1, np.int32); grow("stack", 2, np.int32)
        if old:
            self.queue[: live.size] = live
            s.qhead = 0
        self.mcap = mcap
        s.mcap = mcap
        s.tv = _ptr(self.tv, _i32p); s.tn = _ptr(self.tn, _i32p)
        s.thead = _ptr(self.thead, _i32p); s.ttail = _ptr(self.ttail, _i32p)
        s.tinq = _ptr(self.tinq, _u8p); s.queue = _ptr(self.queue, _i32p); s.stack = _ptr(self.stack, _i32p)

    def _densify(self):
        while self.lib.ptd_run(ctypes.byref(self.state), 0) != 0:
            self._allocate(min(2 * self.n + 8, 2 * self.mcap))

    def _insert_picks(self, k):
        if self.state.m + 2 * k + 8 > self.mcap:
            self._allocate(min(2 * self.n + 8, max(2 * self.mcap, int(self.state.m) + 2 * k + 8)))
        return int(self.lib.ptd_insert(ctypes.byref(self.state), _ptr(self.picks, _i32p), k))

    def _follow(self):
        """Plain densification, then following the surface past its edge, until nothing more comes of it."""
        self._densify()
        for _ in range(int(self.p["follow_rounds"])):
            k = int(self.lib.ptd_frontier(ctypes.byref(self.state), _ptr(self.follow_prm, _f64p), _ptr(self.picks, _i32p)))
            if k == 0:
                break
            self.followed += self._insert_picks(k)
            self.rounds += 1
            self._densify()

    def _grow(self):
        if self.picks is None:
            self.picks = np.empty(self.n, np.int32)
        self._follow()
        for _ in range(int(self.p["turn_rounds"])):
            self.vtouch[:] = 1      # following has just cleared the marks; turning looks at every long triangle again
            k = int(self.lib.ptd_hinge(ctypes.byref(self.state), _ptr(self.turn_prm, _f64p), _ptr(self.picks, _i32p)))
            if k == 0:
                break
            done = self._insert_picks(k)
            self.turned += done
            self.turns += 1
            if done == 0:
                break
            self._follow()

    def run(self):
        self._grow()
        if self.low_sorted.any():
            # A point set aside as an isolated low point that the finished surface nevertheless comes down to
            # (within low_drop) was not isolated after all: give it back and let the surface settle again.
            back = self.low_sorted & (self._heights_sorted() >= -float(self.p["low_drop"]))
            if back.any():
                self.locked[back] = 0
                self.low_sorted[back] = False
                self.restored = int(back.sum())
                self.vtouch[:] = 1      # unlocking changes what a triangle may take: look at all of them again
                self.lib.ptd_requeue(ctypes.byref(self.state))
                self._grow()
        self._prune_spikes()
        return self

    def _prune_spikes(self):
        """Take spikes back out of the finished surface, let the plain test refill, and look again."""
        for _ in range(int(self.p["spike_rounds"])):
            if self.picks is None:
                self.picks = np.empty(self.n, np.int32)
            k = int(self.lib.ptd_spikes(ctypes.byref(self.state), _ptr(self.spike_prm, _f64p), _ptr(self.picks, _i32p)))
            if k == 0:
                break
            done = int(self.lib.ptd_remove(ctypes.byref(self.state), _ptr(self.picks, _i32p), k))
            self.spikes_removed += done
            if done == 0:
                break
            self._densify()

    def _heights_sorted(self):
        h = np.empty(self.n, np.float32)
        self.lib.ptd_heights(ctypes.byref(self.state), _ptr(h, _f32p))
        return h

    def _unsort(self, values):
        out = np.empty(self.n, values.dtype)
        out[self.order] = values
        return out

    def model(self):
        """The points the surface is triangulated from."""
        return self._unsort(self.ptri == -2)

    def _members_sorted(self):
        """Points in the thin band about the surface, thinned to one per small square: the closest to the surface."""
        band = float(self.p["surface_band"])
        out = np.zeros(self.n, np.uint8)
        if band <= 0.0:
            return out.view(bool)
        k = int(self.lib.ptd_members(ctypes.byref(self.state), band, band, float(self.p["surface_band_edge"]), _ptr(out, _u8p)))
        cell = float(self.p["surface_cell"])
        if k == 0 or cell <= 0.0:
            return out.view(bool)
        idx = np.flatnonzero(out)
        step = max(1, int(round(cell / float(self.p["unit"]))))
        key = (self.Y[idx].astype(np.int64) // step) * (int(self.X[: self.n].max()) // step + 1) + self.X[idx] // step
        off = np.abs(self._heights_sorted()[idx])
        o = np.lexsort((off, key))
        first = np.concatenate(([True], key[o][1:] != key[o][:-1]))
        keep = np.zeros(self.n, bool)
        keep[idx[o[first]]] = True
        return keep

    def ground(self):
        """Surface points plus the points in the thin band about the surface."""
        return self._unsort((self.ptri == -2) | self._members_sorted())

    def heights(self):
        return self._unsort(self._heights_sorted())

    def low(self):
        return self._unsort(self.low_sorted)

    def check(self):
        return int(self.lib.ptd_check(ctypes.byref(self.state)))


def classify_ground(x, y, z, settings=None, lib=None):
    run = GroundRun(x, y, z, settings, lib).run()
    return run.ground(), run.heights(), run.low()
