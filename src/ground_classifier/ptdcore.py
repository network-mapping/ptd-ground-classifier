"""Python front for the compiled core (native/ptdcore.c): the only module that touches ctypes.

    lib = load()                                     # find, load and check the library (once per process)
    X, Y, Z = with_corners(xi, yi, z, seeds, xmax, ymax, margin)
    core = Core(X, Y, Z, seeds, locked, AcceptanceParams(...))
    core.densify()                                   # the plain progressive TIN
    k = core.frontier(FollowParams(...))             # candidates for following the surface past its edge
    core.insert_staged(k)
    ground = core.surface_mask()                     # per point: is it a vertex of the surface?

The core is the progressive-TIN engine: it holds a Delaunay triangulation (the surface), inserts candidate points into
it when they pass the acceptance test, and offers the extra steps built on that (following the surface past its
edge, turning a corner, taking spikes back out). It decides nothing about which points are candidates, where the
surface starts, or what the settings mean: that is `fast_ground.GroundRun`. This module only makes the compiled
functions callable with numpy arrays and keeps their memory safe.

Memory. The compiled core allocates nothing (it is freestanding C, no C library): every array it reads or writes is a
numpy array made here and owned by `Core`. The C state structure holds raw pointers into those arrays, so `Core`
keeps a reference to every array for as long as the structure exists, and points the structure at the new array each
time one is replaced (the triangle arrays are replaced, larger, when the surface outgrows them). Nothing here copies
an array that the core needs unless it must; the methods that return arrays say whether they are views or new arrays.
Bytes held, with n the number of points and T the capacity in triangles:

    per point       X 4, Y 4, Z 8, next 4, ptri 4, locked 1, vtouch 1, vtri 4, staged picks 4   = 34 bytes
    per triangle    tv 12, tn 12, thead 4, ttail 4, tinq 1, queue 4, stack 8                    = 45 bytes (+ 64 fixed)

T starts at `Core.default_capacity` and doubles whenever the core reports it is full, never beyond
`Core.max_capacity(n)` = 2n + 8, which is the most a block of n points can use (see there). Growing replaces the
triangle arrays one at a time, so an old and a new copy of one array are alive together, not of all of them. If the
ceiling were ever reached with the core still asking for room, `Core` raises RuntimeError instead of looping. The
caller's own arrays (the sorted integer coordinates, the low-point flags, the seeds) are not counted here.

Units and conventions. Horizontal coordinates are whole numbers of `AcceptanceParams.unit` metres, relative to a
local origin (|value| < 2**30, the corner vertices included), heights are metres. Point k is index k of every
per-point array; the four corner vertices that bound the surface are indices n .. n + 3 of X, Y and Z and are never
candidates. Per-point arrays are in whatever order the caller gave them (`GroundRun` sorts them spatially, which
keeps the candidates of one triangle close together in memory).

A `Core` is not thread-safe, and it is single use: it holds one block of points from construction to the end.
"""
from __future__ import annotations

import ctypes
import os
import sys
from dataclasses import astuple, dataclass
from typing import NamedTuple

import numpy as np

_i32p = ctypes.POINTER(ctypes.c_int32)
_f64p = ctypes.POINTER(ctypes.c_double)
_f32p = ctypes.POINTER(ctypes.c_float)
_u8p = ctypes.POINTER(ctypes.c_uint8)


class _State(ctypes.Structure):
    """The C structure PtdState (native/ptdcore.c): the same fields in the same order."""
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


def library_path() -> str:
    """Where the compiled core is expected to be.

    Returns
    -------
    str
        `$GROUND_CLASSIFIER_CORE` if that is set, otherwise `<repo>/build/` plus the platform's name for it
        (`ptdcore.dll`, `libptdcore.dylib` or `libptdcore.so`), which `native/build.sh` makes.
    """
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


def load(path: str | None = None) -> ctypes.CDLL:
    """Load the compiled core, check it is the one this module was written for, and prove it works.

    The checks are: the file exists; its version number is `CORE_VERSION`; its state structure has the size of
    `_State`; and a small run on a known surface gives a valid triangulation with the objects standing on it left
    out. Loading with the default path is done once per process and the result is kept.

    Parameters
    ----------
    path : str, optional
        The library to load, instead of `library_path()`. A library loaded from another path is not kept.

    Returns
    -------
    ctypes.CDLL
        The library, with the argument and result types of every function set. Pass it to `Core(lib=...)`;
        by default a `Core` calls this function itself.

    Raises
    ------
    OSError
        With a plain message, if the file is missing, is a different version, does not match the structure, or
        fails the self-test on this machine.
    """
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


def _self_test(lib) -> None:
    """A few milliseconds on a known surface: proves the library really works in this process."""
    rng = np.random.default_rng(1)
    n, unit = 4000, 0.001
    x = rng.uniform(0.0, 80.0, n); y = rng.uniform(0.0, 80.0, n)
    z = 0.05 * x + 0.02 * y + rng.normal(0.0, 0.01, n)
    z[:400] += rng.uniform(2.0, 9.0, 400)          # things standing on the surface
    xi = np.rint(x / unit).astype(np.int64); yi = np.rint(y / unit).astype(np.int64)
    quadrant = (x > 40.0).astype(int) + 2 * (y > 40.0)
    seeds = np.array([np.flatnonzero(quadrant == q)[np.argmin(z[quadrant == q])] for q in range(4)], dtype=np.int32)
    X, Y, Z = with_corners(xi, yi, z, seeds, int(xi.max()), int(yi.max()), 20000)
    sin6 = float(np.sin(np.radians(6.0)))
    params = AcceptanceParams(unit=unit, sin_flat=sin6, sin_steep=sin6, tan_flat=0.18, tan_steep=0.7, up_dist=1.4,
                              down_dist=5.0, below_free=0.15, tan2_terrain=float(np.tan(np.radians(70.0)) ** 2),
                              reduce_edge=2.0, reduce_edge_steep=0.5, stop_edge=0.0)
    core = Core(X, Y, Z, seeds, np.zeros(n, np.uint8), params, lib=lib)
    core.densify()
    surface = core.surface_mask()
    if core.check() != 0 or surface[:400].any() or surface[400:].sum() < 2000:
        raise OSError("compiled core failed its self-test on this machine")


def _ptr(a, t):
    return a.ctypes.data_as(t)


_FIELD_TYPES = dict(_State._fields_)


class _TriArray(NamedTuple):
    """One of the arrays whose size follows the triangle capacity. `name` is also its field in `_State`."""
    name: str
    per: int                    # entries per triangle of capacity
    dtype: type
    extra: int = 0              # entries added whatever the capacity
    carried: bool = True        # its contents are copied across when it is replaced by a larger one


_TRIANGLE_ARRAYS = (
    _TriArray("tv", 3, np.int32),
    _TriArray("tn", 3, np.int32),
    _TriArray("thead", 1, np.int32),
    _TriArray("ttail", 1, np.int32),
    _TriArray("tinq", 1, np.uint8),
    _TriArray("queue", 1, np.int32, carried=False),             # circular: `Core._allocate` unwraps it instead
    _TriArray("stack", 2, np.int32, extra=16, carried=False),   # scratch space of one insertion
)


@dataclass(frozen=True)
class AcceptanceParams:
    """The acceptance test of the plain progressive TIN, as the core wants it: angles already as sines and tangents.

    A candidate in a triangle is accepted if it lies within `up_dist` above and `down_dist` below the triangle's
    plane and, seen from its nearest corner, under an angle no larger than the iteration angle. The angle is
    `sin_flat` on flat ground and grows to `sin_steep` on steep ground; in small triangles it is cut back.

    Attributes
    ----------
    unit : float
        Metres per whole number of the integer coordinates.
    sin_flat : float
        Sine of the iteration angle on ground no steeper than `tan_flat`.
    sin_steep : float
        Sine of the iteration angle on ground at least as steep as `tan_steep`, ramping linearly in between
        (`sin_steep <= sin_flat` switches the ramp off).
    tan_flat, tan_steep : float
        Tangents of the two slopes between which the angle ramps.
    up_dist : float
        Furthest, in metres, a candidate may lie above the triangle's plane (the iteration distance).
    down_dist : float
        Furthest it may lie below it. Deeper candidates are never accepted.
    below_free : float
        A candidate deeper than this below the plane (metres) shows the surface is too high there: it is accepted
        on depth alone, without the angle test. Zero or less switches this off.
    tan2_terrain : float
        Square of the tangent of the terrain angle: triangles steeper than this take no candidates.
    reduce_edge : float
        The angle is cut back in triangles whose longest edge is shorter than this (metres); zero switches it off.
    reduce_edge_steep : float
        The same on steep ground, ramped like the angle; a negative value uses `reduce_edge` everywhere.
    stop_edge : float
        Triangles whose longest edge is shorter than this (metres) are not refined further; zero switches it off.
    mode : int, default 0
        Which of the passing candidates a triangle takes first: 0 the smallest angle, 1 the highest above the
        plane, 2 the farthest from the corners, 3 the largest angle.
    """
    unit: float
    sin_flat: float
    sin_steep: float
    tan_flat: float
    tan_steep: float
    up_dist: float
    down_dist: float
    below_free: float
    tan2_terrain: float
    reduce_edge: float
    reduce_edge_steep: float
    stop_edge: float
    mode: int = 0


@dataclass(frozen=True)
class FollowParams:
    """Settings for `Core.frontier`: following the surface past the edge of what is known.

    Where the known ground stops, the triangle holding the next candidates is a long chord to some distant vertex.
    A plane fitted to the ground already found around the vertex (the guide) is extended instead, and a candidate is
    taken if it rises or drops no more than the allowed angle from that extension. Used only in triangles that
    really are such chords.

    Attributes
    ----------
    chord_edge : float
        A triangle with an edge longer than this (metres) is a chord.
    step_min, step_max : float
        Distance (metres) from the known vertex to the new point, at least and at most.
    tan_up, tan_down : float
        Tangents of the angles by which the surface may bend up and down from the guide, per step.
    tan_max_slope : float
        Tangent of the steepest step, and of the steepest guide.
    guide_radius : float
        The guide is fitted to the surface vertices within this radius (metres) of the known vertex.
    guide_rms : float
        The ground there must fit the guide to within this rms (metres), otherwise there is no guide.
    tan_guide_tilt : float
        The guide's tilt must be known to within the angle with this tangent.
    check_tol : float
        Following is used only where another corner of the triangle lies more than this (metres) below the guide's
        extension.
    """
    chord_edge: float
    step_min: float
    step_max: float
    tan_up: float
    tan_down: float
    tan_max_slope: float
    guide_radius: float
    guide_rms: float
    tan_guide_tilt: float
    check_tol: float

    def to_array(self) -> np.ndarray:
        """The values as the float64 array the core reads, in the order of the fields."""
        return np.array(astuple(self), dtype=np.float64)


@dataclass(frozen=True)
class TurnParams:
    """Settings for `Core.hinge`: turning a corner where the ground changes direction.

    Next to a surface vertex, in a long triangle, the lowest candidate in each small sector-and-distance bin is taken
    and a plane through the vertex is fitted to them. If they lie on one plane, no steeper than allowed, they are
    ground (a bank or rock face has a flat underside to its points; brush does not).

    Attributes
    ----------
    chord_edge : float
        Only triangles with an edge longer than this (metres) are looked at.
    rmin, rmax : float
        The lowest points are taken from this far from the vertex, to this far (metres).
    tol : float
        How flat their underside must be (metres).
    tan_max_slope : float
        Tangent of the steepest plane that may be turned onto.
    min_bins : int
        How many of the 20 bins must agree on the plane.
    min_reach : float
        ... over at least this range of distances (metres).
    check_radius, check_tol : float
        No turning onto a plane that known ground within `check_radius` metres lies more than `check_tol` metres below.
    """
    chord_edge: float
    rmin: float
    rmax: float
    tol: float
    tan_max_slope: float
    min_bins: int
    min_reach: float
    check_radius: float
    check_tol: float

    def to_array(self) -> np.ndarray:
        """The values as the float64 array the core reads, in the order of the fields."""
        return np.array(astuple(self), dtype=np.float64)


@dataclass(frozen=True)
class SpikeParams:
    """Settings for `Core.spikes`: finding vertices that stand out of the finished surface.

    A vertex that stands above all but one of its neighbours, each by more than `tan_up` per metre and `h_up`, is a
    spike (a stem, a stone's edge, a stray return); "all but one" keeps ridge lines and bank edges. A pit must be
    below ALL its neighbours by `tan_down` and `h_down`, because low points are far more often real ground.

    Attributes
    ----------
    tan_up, h_up : float
        Tangent of the angle, and height in metres, by which a spike stands above its neighbours.
    tan_down, h_down : float
        The same for a pit.
    min_neighbours : int
        Vertices with fewer neighbours than this are not judged.
    """
    tan_up: float
    h_up: float
    tan_down: float
    h_down: float
    min_neighbours: int

    def to_array(self) -> np.ndarray:
        """The values as the float64 array the core reads, in the order of the fields."""
        return np.array(astuple(self), dtype=np.float64)


def with_corners(xi, yi, z, seeds, xmax: int, ymax: int, margin: int):
    """Add the four corner vertices that bound the surface, and make the arrays in the form the core wants.

    The corners lie `margin` outside the block on every side, so that the two starting triangles cover every
    candidate. Each takes the height of the nearest seed.

    Parameters
    ----------
    xi, yi : integer arrays, one value per point
        Horizontal coordinates in whole units of the core, from a local origin (non-negative).
    z : float array, one value per point
        Heights in metres.
    seeds : int array
        Indices of the starting points, at least three (used here for the heights of the corners).
    xmax, ymax : int
        Largest `xi` and `yi`.
    margin : int
        Distance of the corners from the block, in whole units.

    Returns
    -------
    X, Y : int32 arrays of length n + 4
    Z : float64 array of length n + 4
        The points, then the corners (counter-clockwise from the lower left). New arrays; the inputs are not
        kept, so the caller may drop them straight away to save memory.
    """
    corners = np.array([[-margin, -margin], [xmax + margin, -margin], [xmax + margin, ymax + margin], [-margin, ymax + margin]], dtype=np.int64)
    sx, sy, sz = xi[seeds], yi[seeds], z[seeds]
    cz = np.array([sz[np.argmin((sx - cx) ** 2 + (sy - cy) ** 2)] for cx, cy in corners])
    X = np.ascontiguousarray(np.concatenate((xi, corners[:, 0])), dtype=np.int32)
    Y = np.ascontiguousarray(np.concatenate((yi, corners[:, 1])), dtype=np.int32)
    Z = np.ascontiguousarray(np.concatenate((z, cz)), dtype=np.float64)
    return X, Y, Z


class Core:
    """One block of points held by the compiled core: the surface, and the steps that build and refine it.

    The surface starts as two triangles covering the block. Every point is a candidate and sits in the list of the
    triangle that contains it. `densify` inserts candidates into the surface until none passes the acceptance test;
    the steps `frontier`, `hinge` and `spikes` find further candidates to insert (or vertices to remove) by other
    rules, and are followed by `insert_staged` or `remove_staged`. `Core` takes care of memory: the triangle arrays
    are enlarged, and the circular work queue straightened, whenever the core reports it is full.

    The picks of `frontier`, `hinge` and `spikes` are held inside the object, in one int32 array of n entries that
    is allocated once, until the next call of `insert_staged` or `remove_staged`; those two take the number the
    finding step returned.

    Parameters
    ----------
    X, Y : int32 arrays of length n + 4
        Horizontal coordinates in whole units of `params.unit`; the last four entries are the corner vertices
        (see `with_corners`). Held, not copied.
    Z : float64 array of length n + 4
        Heights in metres, corners last. Held, not copied.
    seeds : int32 array
        Indices (below n) of the starting points, inserted into the surface before anything is tested. Held, not copied.
    locked : array of n bool or uint8
        True for the candidates that may never be accepted (isolated low points). They still get a height. A
        uint8 copy is made.
    params : AcceptanceParams
        The acceptance test.
    initial_capacity : int, optional
        The first size of the triangle arrays, in triangles. By default `default_capacity(n, len(seeds))`. The tests
        pass a small value to make the arrays grow many times.
    lib : ctypes.CDLL, optional
        The library, from `load`. By default `load()`.

    Notes
    -----
    A candidate leaves its triangle's list when it is accepted (it becomes a vertex of the surface) or when another
    point at exactly the same position is a vertex (it can never be inserted). `surface_mask` shows the vertices.
    """

    def __init__(self, X, Y, Z, seeds, locked, params: AcceptanceParams, initial_capacity: int | None = None, lib=None):
        self._lib = lib or load()
        n = int(X.size) - 4
        self._n = n
        self._X, self._Y, self._Z = X, Y, Z
        self._seeds = seeds
        self._next = np.empty(n, np.int32)
        self._ptri = np.empty(n, np.int32)
        self._locked = np.ascontiguousarray(locked, dtype=np.uint8)
        self._mcap = 0

        s = self._state = _State()
        s.n = n
        s.X = _ptr(self._X, _i32p); s.Y = _ptr(self._Y, _i32p); s.Z = _ptr(self._Z, _f64p)
        s.next = _ptr(self._next, _i32p); s.ptri = _ptr(self._ptri, _i32p)
        s.seeds = _ptr(self._seeds, _i32p); s.nseeds = seeds.size; s.seed_pos = 0
        s.locked = _ptr(self._locked, _u8p)
        self._vtouch = np.ones(n + 4, np.uint8)
        s.vtouch = _ptr(self._vtouch, _u8p)
        self._vtri = np.full(n + 4, -1, np.int32)
        s.vtri = _ptr(self._vtri, _i32p)
        s.unit = params.unit
        s.sin2_limit = params.sin_flat ** 2         # not read by the core; kept so the structure is complete
        s.sin_flat = params.sin_flat
        s.sin_steep = params.sin_steep
        s.tan_flat = params.tan_flat
        s.tan_steep = params.tan_steep
        s.up_dist = params.up_dist
        s.down_dist = params.down_dist
        s.below_free = params.below_free
        s.tan2_terrain = params.tan2_terrain
        s.reduce_edge = params.reduce_edge
        s.stop_edge = params.stop_edge
        s.reduce_edge_steep = params.reduce_edge_steep
        s.initialised = 0; s.m = 0; s.mode = params.mode
        self._allocate(initial_capacity or self.default_capacity(n, seeds.size))
        self._picks = np.empty(n, np.int32)

    @staticmethod
    def max_capacity(n: int) -> int:
        """The most triangles (slots in the triangle arrays) a block of n points can ever use: 2n + 8.

        The surface starts as 2 triangles and each insertion adds exactly 2 (a split; edge flips only rearrange
        triangles). A point is inserted at most once: a point at the same position as a vertex, and a locked point,
        is never inserted, and a vertex taken out as a spike is locked. So at most 2 + 2n slots are ever used. The other 6 are
        slack for the core's check `m + 2 <= capacity`, made before each insertion. Taking a vertex out does not give
        its slots back, but it takes no new ones either.

        The triangle arrays are never grown past this. The first allocation of a block under about 500 points
        (`default_capacity` has a floor of 1024) can be larger; it never needs to grow.
        """
        return 2 * n + 8

    @staticmethod
    def default_capacity(n: int, n_seeds: int) -> int:
        """The first size of the triangle arrays, in triangles, for n points and the given number of seeds.

        An estimate of a seed count plus n / 8 triangles (about 6% of the points becoming vertices, 2 triangles each),
        at least 1024 and at most `max_capacity(n)`. The arrays are enlarged when it proves too small.
        """
        return max(1024, min(Core.max_capacity(n), 4 * n_seeds + n // 8 + 1024))

    @property
    def n(self) -> int:
        """Number of points (the corner vertices not counted)."""
        return self._n

    @property
    def capacity(self) -> int:
        """Size of the triangle arrays now, in triangles."""
        return self._mcap

    @property
    def triangles(self) -> int:
        """Triangles in use."""
        return int(self._state.m)

    @property
    def n_ground(self) -> int:
        """Points inserted into the surface so far (the seeds included)."""
        return int(self._state.n_ground)

    @property
    def x(self) -> np.ndarray:
        """Integer x of the n points: a view of the held array, not a copy. Do not write to it."""
        return self._X[: self._n]

    @property
    def y(self) -> np.ndarray:
        """Integer y of the n points: a view of the held array, not a copy. Do not write to it."""
        return self._Y[: self._n]

    def _resized(self, spec: _TriArray, mcap: int) -> np.ndarray:
        """A new array for `spec` sized for `mcap` triangles, with the old contents at its front if `spec` carries them.

        Called while the old array is still the attribute and `self._mcap` is still the old capacity (0 for the first
        allocation, when there is nothing to carry).
        """
        new = np.empty(spec.per * mcap + spec.extra, spec.dtype)
        if self._mcap and spec.carried:
            old = getattr(self, "_" + spec.name)
            new[: old.size] = old
        return new

    def _allocate(self, mcap: int) -> None:
        """Make the triangle arrays hold `mcap` triangles, carrying over what is in them, and re-point the structure.

        The arrays are replaced one at a time, so an old and a new copy of only one of them are alive together.
        """
        s = self._state
        old = self._mcap
        if old:   # the queue starts at qhead and may wrap round the end: unwrap it to the front of the larger one
            live = self._queue[(s.qhead + np.arange(s.qcount)) % old]
        for spec in _TRIANGLE_ARRAYS:
            setattr(self, "_" + spec.name, self._resized(spec, mcap))
        if old:
            self._queue[: live.size] = live
            s.qhead = 0
        self._mcap = mcap
        s.mcap = mcap
        for spec in _TRIANGLE_ARRAYS:
            setattr(s, spec.name, _ptr(getattr(self, "_" + spec.name), _FIELD_TYPES[spec.name]))

    def _enlarge(self, wanted: int) -> bool:
        """Grow the triangle arrays to `wanted` triangles, but not past `max_capacity`; False if that is no growth."""
        target = min(wanted, self.max_capacity(self._n))
        if target <= self._mcap:
            return False
        self._allocate(target)
        return True

    def _ceiling_error(self) -> RuntimeError:
        return RuntimeError(
            f"the compiled core needs more than {self._mcap} triangles for {self._n} points, but a block of {self._n} "
            f"points cannot use more than max_capacity = {self.max_capacity(self._n)}: the assumptions behind it do not hold")

    def densify(self) -> None:
        """Insert candidates until none passes the acceptance test: the plain progressive TIN.

        The seeds go in first. The triangle arrays are doubled each time the core reports they are full.

        Raises
        ------
        RuntimeError
            If the core reports the arrays full and they cannot grow, because they are at `max_capacity`. This
            cannot happen while the assumptions behind `max_capacity` hold; it is a guard against looping forever.
        """
        while self._lib.ptd_run(ctypes.byref(self._state), 0) != 0:
            if not self._enlarge(2 * self._mcap):
                raise self._ceiling_error()

    def frontier(self, params: FollowParams) -> int:
        """Find candidates by following the surface past the edge of what is known; stage them.

        At most one candidate (the lowest) per triangle corner. Clears the "triangle changed" marks that the scan
        uses, so call `mark_all_touched` before a scan that must look at every triangle again.

        Parameters
        ----------
        params : FollowParams

        Returns
        -------
        int
            How many candidates are staged for `insert_staged`; zero when following has run out.
        """
        prm = params.to_array()
        return int(self._lib.ptd_frontier(ctypes.byref(self._state), _ptr(prm, _f64p), _ptr(self._picks, _i32p)))

    def hinge(self, params: TurnParams) -> int:
        """Find candidates by turning a corner (a plane fitted to the lowest points beside a vertex); stage them.

        Looks at the triangles marked as changed (see `mark_all_touched`).

        Parameters
        ----------
        params : TurnParams

        Returns
        -------
        int
            How many candidates are staged for `insert_staged`; zero when no corner could be turned.
        """
        prm = params.to_array()
        return int(self._lib.ptd_hinge(ctypes.byref(self._state), _ptr(prm, _f64p), _ptr(self._picks, _i32p)))

    def spikes(self, params: SpikeParams) -> int:
        """Find vertices of the surface that stand out as spikes or pits; stage them for removal.

        Parameters
        ----------
        params : SpikeParams

        Returns
        -------
        int
            How many vertices are staged for `remove_staged`.
        """
        prm = params.to_array()
        return int(self._lib.ptd_spikes(ctypes.byref(self._state), _ptr(prm, _f64p), _ptr(self._picks, _i32p)))

    def insert_staged(self, k: int) -> int:
        """Insert the candidates staged by `frontier` or `hinge`, enlarging the triangle arrays first if need be.

        Parameters
        ----------
        k : int
            The number the finding step returned.

        Returns
        -------
        int
            How many were inserted; fewer than `k` when some were no longer in a triangle's list (already taken).

        Raises
        ------
        RuntimeError
            If the arrays cannot hold `k` more insertions and cannot grow, because they are at `max_capacity`.
            This cannot happen while the assumptions behind `max_capacity` hold.
        """
        needed = int(self._state.m) + 2 * k         # slots in use after k insertions
        if needed + 8 > self._mcap:
            grew = self._enlarge(max(2 * self._mcap, needed + 8))
            if not grew and needed > self._mcap:    # at the ceiling, with only the slack left: still enough
                raise self._ceiling_error()
        return int(self._lib.ptd_insert(ctypes.byref(self._state), _ptr(self._picks, _i32p), k))

    def remove_staged(self, k: int) -> int:
        """Take the vertices staged by `spikes` out of the surface; their points become candidates again.

        Removed vertices are locked, so they are not taken back. Follow with `densify` to let the plain test refill.

        Parameters
        ----------
        k : int
            The number `spikes` returned.

        Returns
        -------
        int
            How many were removed.
        """
        return int(self._lib.ptd_remove(ctypes.byref(self._state), _ptr(self._picks, _i32p), k))

    def requeue(self) -> None:
        """Put every triangle that holds candidates back on the work queue.

        Needed after `unlock`, because unlocking changes what a triangle may take.
        """
        self._lib.ptd_requeue(ctypes.byref(self._state))

    def mark_all_touched(self) -> None:
        """Mark every triangle as changed since the last `frontier` scan, so the next scan looks at all of them."""
        self._vtouch[:] = 1

    def unlock(self, mask) -> None:
        """Allow the points where `mask` (n bools, in the order of the arrays) is true to be accepted after all. In place."""
        self._locked[mask] = 0

    def heights(self) -> np.ndarray:
        """Height of every point above the surface as it stands now (zero for vertices).

        Returns
        -------
        float32 array of n
            A new array. For a point exactly over a vertex, the difference of their heights.
        """
        h = np.empty(self._n, np.float32)
        self._lib.ptd_heights(ctypes.byref(self._state), _ptr(h, _f32p))
        return h

    def surface_mask(self) -> np.ndarray:
        """Which points are vertices of the surface. A new bool array of n."""
        return self._ptri == -2

    def members(self, up: float, down: float, max_edge: float) -> tuple[int, np.ndarray]:
        """The candidates in a thin band about the surface, the surface vertices excepted.

        A candidate belongs if it is no more than `up` metres above and `down` below the triangle it is in, and
        the triangle's edges are all shorter than `max_edge` (a long triangle is a guess across a gap). Locked
        candidates never belong.

        Returns
        -------
        count : int
            How many belong.
        mask : bool array of n
            A new array, true for those that do.
        """
        out = np.zeros(self._n, np.uint8)
        k = int(self._lib.ptd_members(ctypes.byref(self._state), up, down, max_edge, _ptr(out, _u8p)))
        return k, out.view(np.bool_)

    def check(self) -> int:
        """Self-check of the triangulation: zero if every triangle is counter-clockwise, the neighbour links are
        mutual, every listed candidate lies inside its triangle, and the Delaunay condition holds; otherwise the
        number of faults found."""
        return int(self._lib.ptd_check(ctypes.byref(self._state)))
