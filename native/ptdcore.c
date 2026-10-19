/*
 * ptdcore.c - compiled core for progressive-TIN ground classification.
 *
 * The scheme is the one TerraScan's guide describes for its Ground routine
 * (and Axelsson 2000): start from low seed points, hold a triangulated
 * surface through the ground found so far, and keep adding points that pass
 * an iteration-angle and an iteration-distance test against the triangle
 * they lie in. Every accepted point becomes a vertex of the surface.
 *
 * What makes it fast is the bookkeeping, not the arithmetic:
 *   - the triangulation is a true incremental Delaunay triangulation: a new
 *     ground point is inserted into the existing surface (split + edge
 *     flips). Nothing is ever rebuilt from scratch;
 *   - every triangle keeps the list of candidate points that lie inside it,
 *     so finding "the points this triangle should test" costs nothing, and
 *     only triangles that actually changed are looked at again.
 *
 * Geometry predicates (which side of a line, inside a circumcircle) are
 * evaluated exactly, on integer coordinates, so the triangulation cannot be
 * corrupted by rounding. LAS coordinates are integers on disk; the caller
 * passes them as integers relative to a local origin (|value| < 2^30).
 *
 * The file is freestanding C: no C library, no memory allocation. Every
 * array is owned by the caller (numpy). That is what lets the same source be
 * built as a Linux shared object and as a Windows DLL with no runtime
 * dependencies.
 */
#include <stdint.h>

typedef int32_t i32;
typedef int64_t i64;
typedef __int128 i128;
typedef uint8_t u8;

#if defined(_WIN32)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

#if defined(_WIN32)
/* the Microsoft linker expects this symbol in any module that uses floating point */
int _fltused = 0;
#endif

#define NEXT3(k) ((k) == 2 ? 0 : (k) + 1)
#define PREV3(k) ((k) == 0 ? 2 : (k)-1)

/* ptri[] codes for a candidate that is no longer in any triangle's list */
#define PT_VERTEX (-2)      /* accepted: it is now a surface vertex */
/* <= -10: same XY as surface vertex (-10 - code); can never be inserted */

typedef struct PtdState {
    /* candidate points 0..n-1, then the four corner vertices n..n+3 */
    i64 n;
    const i32 *X;       /* n + 4 */
    const i32 *Y;       /* n + 4 */
    const double *Z;    /* n + 4, metres */
    i32 *next;          /* n: next candidate in the same triangle, -1 end */
    i32 *ptri;          /* n: triangle holding the candidate, or a PT_ code */
    /* triangulation, capacity mcap triangles */
    i32 *tv;            /* 3 * mcap: vertices, counter-clockwise */
    i32 *tn;            /* 3 * mcap: neighbour across the edge opposite vertex k */
    i32 *thead;         /* mcap: first candidate inside the triangle, -1 none */
    i32 *ttail;         /* mcap */
    u8 *tinq;           /* mcap: triangle is waiting in the queue */
    i32 *queue;         /* mcap: circular queue of triangles to (re)examine */
    i32 *stack;         /* 2 * mcap: scratch for edge legalisation */
    i64 mcap;
    i64 m;              /* triangles in use */
    i64 qhead;
    i64 qcount;
    /* seed points, inserted unconditionally before anything is tested */
    const i32 *seeds;
    i64 nseeds;
    i64 seed_pos;
    /* acceptance rule */
    double scale;       /* metres per integer coordinate step (like a LAS scale) */
    double sin2_limit;  /* sin^2(iteration angle) */
    double up_dist;     /* iteration distance above the triangle plane, m */
    double down_dist;   /* and below it */
    double tan2_terrain;/* tan^2(terrain angle): steeper triangles add nothing */
    double reduce_edge; /* shrink the angle when the longest edge is shorter, m */
    double stop_edge;   /* stop refining triangles whose longest edge is shorter, m */
    double below_free;  /* a candidate deeper than this below the plane (m) shows
                           the surface is too high there: it is accepted on depth
                           alone, without the angle test. <= 0 switches this off */
    /* counters */
    i64 n_ground;
    i64 n_tests;
    i64 n_flips;
    i32 initialised;
    u8 *locked;         /* n, or null: bit 1 = never accept this candidate (it still gets a height);
                           bit 2 on a surface vertex = other points share its XY */
    i32 mode;           /* which passing candidate a triangle takes first: 0 smallest angle,
                           1 highest above the plane, 2 farthest from the vertices, 3 largest angle */
    u8 *vtouch;         /* n + 4, or null: 1 = a triangle at this vertex changed since the last
                           ptd_frontier scan (so only those need to be looked at again) */
    i32 *vtri;          /* n + 4, or null: one triangle that has this vertex */
    /* iteration angle by the slope of the triangle: sin_flat at or below tan_flat, sin_steep at or
       above tan_steep, a straight ramp between (sin_steep <= sin_flat switches it off) */
    double sin_flat, sin_steep, tan_flat, tan_steep;
    double reduce_edge_steep;   /* reduce_edge on steep ground (the same ramp); < 0 = reduce_edge everywhere */
} PtdState;

static double fsqrt(double x)   /* no C library here */
{
    if (!(x > 0.0))
        return 0.0;
    double r = x > 1.0 ? x : 1.0;
    for (int it = 0; it < 40; it++) {
        const double nr = 0.5 * (r + x / r);
        if (nr == r)
            break;
        r = nr;
    }
    return r;
}

/* ---- exact predicates on integer coordinates ---------------------------- */

static inline i64 orient(const PtdState *s, i32 a, i32 b, i32 c)
{
    i64 ax = s->X[a], ay = s->Y[a];
    return ((i64)s->X[b] - ax) * ((i64)s->Y[c] - ay) -
           ((i64)s->Y[b] - ay) * ((i64)s->X[c] - ax);
}

/* > 0 when d lies strictly inside the circumcircle of ccw triangle a, b, c */
static inline int incircle_pos(const PtdState *s, i32 a, i32 b, i32 c, i32 d)
{
    i64 dx = s->X[d], dy = s->Y[d];
    i64 adx = s->X[a] - dx, ady = s->Y[a] - dy;
    i64 bdx = s->X[b] - dx, bdy = s->Y[b] - dy;
    i64 cdx = s->X[c] - dx, cdy = s->Y[c] - dy;
    i64 ad2 = adx * adx + ady * ady;
    i64 bd2 = bdx * bdx + bdy * bdy;
    i64 cd2 = cdx * cdx + cdy * cdy;
    i128 det = (i128)adx * ((i128)bdy * cd2 - (i128)cdy * bd2) -
               (i128)ady * ((i128)bdx * cd2 - (i128)cdx * bd2) +
               (i128)ad2 * ((i128)bdx * cdy - (i128)cdx * bdy);
    return det > 0;
}

/* ---- per-triangle candidate lists and the work queue -------------------- */

static inline void bucket_append(PtdState *s, i32 t, i32 p)
{
    s->next[p] = -1;
    if (s->thead[t] < 0)
        s->thead[t] = p;
    else
        s->next[s->ttail[t]] = p;
    s->ttail[t] = p;
    s->ptri[p] = t;
}

static inline void enqueue(PtdState *s, i32 t)
{
    if (!s->tinq[t]) {
        i64 pos = s->qhead + s->qcount;
        if (pos >= s->mcap)
            pos -= s->mcap;
        s->queue[pos] = t;
        s->qcount++;
        s->tinq[t] = 1;
    }
}

static inline void relink(PtdState *s, i32 t, i32 from, i32 to)
{
    if (t < 0)
        return;
    i32 *n = s->tn + 3 * (i64)t;
    if (n[0] == from)
        n[0] = to;
    else if (n[1] == from)
        n[1] = to;
    else if (n[2] == from)
        n[2] = to;
}

static inline void set_tri(PtdState *s, i32 t, i32 a, i32 b, i32 c, i32 na,
                           i32 nb, i32 nc)
{
    i32 *v = s->tv + 3 * (i64)t, *n = s->tn + 3 * (i64)t;
    v[0] = a; v[1] = b; v[2] = c;
    n[0] = na; n[1] = nb; n[2] = nc;
    if (s->vtouch) {
        s->vtouch[a] = 1; s->vtouch[b] = 1; s->vtouch[c] = 1;
    }
    if (s->vtri) {
        s->vtri[a] = t; s->vtri[b] = t; s->vtri[c] = t;
    }
}

/* ---- insertion: split the containing triangle, then restore Delaunay ---- */

static void insert_point(PtdState *s, i32 t, i32 p)
{
    const i32 a = s->tv[3 * (i64)t], b = s->tv[3 * (i64)t + 1],
              c = s->tv[3 * (i64)t + 2];
    const i32 na = s->tn[3 * (i64)t], nb = s->tn[3 * (i64)t + 1],
              nc = s->tn[3 * (i64)t + 2];
    const i32 t1 = (i32)s->m, t2 = (i32)s->m + 1;
    s->m += 2;

    i32 q = s->thead[t];
    s->thead[t] = s->ttail[t] = -1;
    s->thead[t1] = s->ttail[t1] = -1;
    s->thead[t2] = s->ttail[t2] = -1;
    s->tinq[t1] = s->tinq[t2] = 0;

    set_tri(s, t, p, b, c, na, t1, t2);
    set_tri(s, t1, a, p, c, t, nb, t2);
    set_tri(s, t2, a, b, p, t, t1, nc);
    relink(s, nb, t, t1);
    relink(s, nc, t, t2);
    s->ptri[p] = PT_VERTEX;

    /* Hand the old triangle's candidates to the three new ones. A candidate
     * on a shared edge may go to either side. When p lies exactly on an edge
     * of the old triangle one of the three is flat (it is flipped away just
     * below); it must not be given candidates that a proper triangle holds,
     * so the proper ones are tried first. */
    const int flat0 = orient(s, p, b, c) == 0;   /* t  = (p, b, c) */
    const int flat1 = orient(s, a, p, c) == 0;   /* t1 = (a, p, c) */
    const i32 px = s->X[p], py = s->Y[p];
    while (q >= 0) {
        const i32 nx = s->next[q];
        if (s->X[q] == px && s->Y[q] == py) {
            s->ptri[q] = -10 - p; /* same XY as the new vertex */
            if (s->locked)
                s->locked[p] |= 2;
        } else {
            const i64 oa = orient(s, p, a, q), ob = orient(s, p, b, q),
                      oc = orient(s, p, c, q);
            const int in0 = ob >= 0 && oc <= 0;
            const int in1 = oc >= 0 && oa <= 0;
            const int in2 = oa >= 0 && ob <= 0;
            i32 dst;
            if (in0 && !flat0)
                dst = t;
            else if (in1 && !flat1)
                dst = t1;
            else if (in2)
                dst = t2;
            else if (in0)
                dst = t;
            else
                dst = t1;
            bucket_append(s, dst, q);
        }
        q = nx;
    }
    enqueue(s, t);
    enqueue(s, t1);
    enqueue(s, t2);

    /* Lawson legalisation of the edges opposite p */
    i32 *st = s->stack;
    i64 sp = 0;
    st[sp++] = t;  st[sp++] = 0;
    st[sp++] = t1; st[sp++] = 1;
    st[sp++] = t2; st[sp++] = 2;
    while (sp > 0) {
        const i32 i = st[--sp];
        const i32 tt = st[--sp];
        i32 *v = s->tv + 3 * (i64)tt, *n = s->tn + 3 * (i64)tt;
        const i32 u = n[i];
        if (u < 0)
            continue;
        i32 *uv = s->tv + 3 * (i64)u, *un = s->tn + 3 * (i64)u;
        const i32 j = (un[0] == tt) ? 0 : (un[1] == tt) ? 1 : 2;
        const i32 x = v[NEXT3(i)], y = v[PREV3(i)], d = uv[j];
        /* p exactly on the edge x-y leaves a flat triangle: always flip it */
        const int flat = orient(s, p, x, y) == 0;
        if (!flat && !incircle_pos(s, p, x, y, d))
            continue;
        const i32 ux = un[NEXT3(j)]; /* across x-d */
        const i32 uy = un[PREV3(j)]; /* across d-y */
        const i32 tx = n[NEXT3(i)];  /* across y-p */
        const i32 ty = n[PREV3(i)];  /* across p-x */
        i32 qa = s->thead[tt], qb = s->thead[u];
        s->thead[tt] = s->ttail[tt] = -1;
        s->thead[u] = s->ttail[u] = -1;
        set_tri(s, tt, p, x, d, ux, u, ty);
        set_tri(s, u, p, d, y, uy, tx, tt);
        relink(s, ux, u, tt);
        relink(s, tx, tt, u);
        for (int pass = 0; pass < 2; pass++) {
            i32 r = pass ? qb : qa;
            while (r >= 0) {
                const i32 nx = s->next[r];
                bucket_append(s, orient(s, p, d, r) <= 0 ? tt : u, r);
                r = nx;
            }
        }
        enqueue(s, tt);
        enqueue(s, u);
        s->n_flips++;
        st[sp++] = tt; st[sp++] = 0;
        st[sp++] = u;  st[sp++] = 0;
    }
}

/* ---- the acceptance test ------------------------------------------------ */

/* Best acceptable candidate inside triangle t (-1 if none); *prev_out is the
 * candidate before it in the list, needed to unlink it. "Best" is the one
 * seen under the smallest angle from its closest vertex. */
static i32 pick_candidate(PtdState *s, i32 t, i32 *prev_out)
{
    i32 q = s->thead[t];
    if (q < 0)
        return -1;
    const i32 *v = s->tv + 3 * (i64)t;
    const double u = s->scale;
    const double ax = s->X[v[0]] * u, ay = s->Y[v[0]] * u, az = s->Z[v[0]];
    const double bx = s->X[v[1]] * u, by = s->Y[v[1]] * u, bz = s->Z[v[1]];
    const double cx = s->X[v[2]] * u, cy = s->Y[v[2]] * u, cz = s->Z[v[2]];
    const double e1x = bx - ax, e1y = by - ay, e1z = bz - az;
    const double e2x = cx - ax, e2y = cy - ay, e2z = cz - az;
    const double nx = e1y * e2z - e1z * e2y;
    const double ny = e1z * e2x - e1x * e2z;
    const double nz = e1x * e2y - e1y * e2x;
    if (!(nz > 0.0))
        return -1;
    const double nh2 = nx * nx + ny * ny;
    if (nh2 > s->tan2_terrain * nz * nz)
        return -1;
    const double n2 = nh2 + nz * nz;
    const double l3x = cx - bx, l3y = cy - by;
    double edge2 = e1x * e1x + e1y * e1y;
    const double edge2b = e2x * e2x + e2y * e2y, edge2c = l3x * l3x + l3y * l3y;
    if (edge2b > edge2) edge2 = edge2b;
    if (edge2c > edge2) edge2 = edge2c;
    if (s->stop_edge > 0.0 && edge2 < s->stop_edge * s->stop_edge)
        return -1;
    double lim = s->sin_flat, reduce = s->reduce_edge;
    if (s->tan_steep > s->tan_flat) {
        /* steeper ground is rougher and less evenly sampled: it is given a wider angle,
         * and the angle is cut back less in small triangles */
        double f = (fsqrt(nh2) / nz - s->tan_flat) / (s->tan_steep - s->tan_flat);
        if (f < 0.0) f = 0.0;
        if (f > 1.0) f = 1.0;
        if (s->sin_steep > s->sin_flat)
            lim += f * (s->sin_steep - s->sin_flat);
        if (s->reduce_edge_steep >= 0.0 && s->reduce_edge_steep < reduce)
            reduce += f * (s->reduce_edge_steep - reduce);
    }
    double lim2 = lim * lim;
    if (reduce > 0.0 && edge2 < reduce * reduce)
        lim2 *= edge2 / (reduce * reduce);
    const double up2 = s->up_dist * s->up_dist * n2;
    const double down2 = s->down_dist * s->down_dist * n2;
    const double lim2n2 = lim2 * n2;

    const double free2 = s->below_free > 0.0 ? s->below_free * s->below_free * n2 : -1.0;

    i32 best = -1, best_prev = -1, prev = -1;
    i32 deep = -1, deep_prev = -1;
    double best_ratio = 0.0, deep_dn = 0.0;
    i64 tests = 0;
    while (q >= 0) {
        const double qx = s->X[q] * u, qy = s->Y[q] * u, qz = s->Z[q];
        const double dax = qx - ax, day = qy - ay, daz = qz - az;
        const double dn = dax * nx + day * ny + daz * nz;
        const double dn2 = dn * dn;
        tests++;
        if (s->locked && (s->locked[q] & 1)) {
            /* excluded beforehand (an isolated low point) */
        } else if (dn < 0.0 && free2 >= 0.0 && dn2 > free2 && dn2 <= down2) {
            /* clearly below the surface: the deepest such candidate goes first */
            if (deep < 0 || dn < deep_dn) {
                deep = q;
                deep_prev = prev;
                deep_dn = dn;
            }
        } else if (dn2 <= (dn >= 0.0 ? up2 : down2)) {
            const double dbx = qx - bx, dby = qy - by, dbz = qz - bz;
            const double dcx = qx - cx, dcy = qy - cy, dcz = qz - cz;
            double r2 = dax * dax + day * day + daz * daz;
            const double r2b = dbx * dbx + dby * dby + dbz * dbz;
            const double r2c = dcx * dcx + dcy * dcy + dcz * dcz;
            if (r2b < r2) r2 = r2b;
            if (r2c < r2) r2 = r2c;
            if (r2 > 0.0 && dn2 <= lim2n2 * r2) {
                const double ratio = s->mode == 0 ? dn2 / r2 : s->mode == 1 ? -dn : s->mode == 2 ? -r2 : -dn2 / r2;
                if (best < 0 || ratio < best_ratio) {
                    best = q;
                    best_prev = prev;
                    best_ratio = ratio;
                }
            }
        }
        prev = q;
        q = s->next[q];
    }
    s->n_tests += tests;
    if (deep >= 0) {
        *prev_out = deep_prev;
        return deep;
    }
    *prev_out = best_prev;
    return best;
}

static void unlink_candidate(PtdState *s, i32 t, i32 p, i32 prev)
{
    const i32 nx = s->next[p];
    if (prev < 0)
        s->thead[t] = nx;
    else
        s->next[prev] = nx;
    if (s->ttail[t] == p)
        s->ttail[t] = prev;
}

/* ---- driver -------------------------------------------------------------- */

static void initialise(PtdState *s)
{
    const i32 c0 = (i32)s->n, c1 = c0 + 1, c2 = c0 + 2, c3 = c0 + 3;
    /* corners are given counter-clockwise: two triangles cover everything */
    set_tri(s, 0, c0, c1, c2, -1, 1, -1);
    set_tri(s, 1, c0, c2, c3, -1, -1, 0);
    s->thead[0] = s->ttail[0] = -1;
    s->thead[1] = s->ttail[1] = -1;
    s->tinq[0] = s->tinq[1] = 0;
    s->m = 2;
    s->qhead = 0;
    s->qcount = 0;
    for (i64 p = 0; p < s->n; p++)
        bucket_append(s, orient(s, c0, c2, (i32)p) > 0 ? 1 : 0, (i32)p);
    s->seed_pos = 0;
    s->n_ground = s->n_tests = s->n_flips = 0;
    s->initialised = 1;
}

/* Returns 0 when finished, 1 when the triangle arrays are full (enlarge them
 * and call again), 2 when max_inserts points have been added this call. */
EXPORT int ptd_run(PtdState *s, i64 max_inserts)
{
    if (!s->initialised)
        initialise(s);
    i64 budget = max_inserts;
    while (s->seed_pos < s->nseeds) {
        if (s->m + 2 > s->mcap)
            return 1;
        const i32 p = s->seeds[s->seed_pos++];
        const i32 t = s->ptri[p];
        if (t < 0)
            continue;
        i32 prev = -1, q = s->thead[t];
        while (q >= 0 && q != p) {
            prev = q;
            q = s->next[q];
        }
        if (q != p)
            continue;
        unlink_candidate(s, t, p, prev);
        insert_point(s, t, p);
        s->n_ground++;
    }
    while (s->qcount > 0) {
        if (s->m + 2 > s->mcap)
            return 1;
        const i32 t = s->queue[s->qhead];
        if (++s->qhead >= s->mcap)
            s->qhead = 0;
        s->qcount--;
        s->tinq[t] = 0;
        i32 prev;
        const i32 p = pick_candidate(s, t, &prev);
        if (p < 0)
            continue;
        unlink_candidate(s, t, p, prev);
        insert_point(s, t, p);
        s->n_ground++;
        if (max_inserts > 0 && --budget == 0)
            return 2;
    }
    return 0;
}

/* ---- small helpers ------------------------------------------------------- */

#define RING_CAP 48

/* The vertices around surface vertex V, counter-clockwise. Optionally also
 * the triangles (tri[i] has V, ring[i], ring[i+1]) and the triangle outside
 * each ring edge ring[i] -> ring[i+1]. Returns how many, or -1 when V is on
 * the outer boundary or has more than cap neighbours. */
static int ring_of(const PtdState *s, i32 V, i32 *ring, i32 *tri, i32 *out, int cap)
{
    const i32 t0 = s->vtri[V];
    if (t0 < 0)
        return -1;
    i32 cur = t0;
    int k = 0;
    do {
        const i32 *cv = s->tv + 3 * (i64)cur;
        const int j = cv[0] == V ? 0 : cv[1] == V ? 1 : cv[2] == V ? 2 : -1;
        if (j < 0 || k >= cap)
            return -1;
        ring[k] = cv[NEXT3(j)];
        if (tri) tri[k] = cur;
        if (out) out[k] = s->tn[3 * (i64)cur + j];
        k++;
        cur = s->tn[3 * (i64)cur + NEXT3(j)];
        if (cur < 0)
            return -1;
    } while (cur != t0);
    return k;
}

#define GUIDE_CAP 96
#define GUIDE_HASH 256

/* Least-squares plane through the ground already found around V: V itself
 * and the surface vertices within radius rg of it, gathered ring by ring
 * (up to four rings, GUIDE_CAP vertices). g = {a, b, c}:
 * z = c + a (x - xV) + b (y - yV). Returns 1, or 0 when the ground there is
 * too little, too rough (rms above rms_max) or too much in one line for its
 * tilt to be known to within tan_tilt - a few points a hand apart cannot say
 * which way the surface goes, their own measurement noise tilts them every
 * which way. */
/* Kept out of line and under 4 KB of stack: a larger frame would need the C
 * runtime's stack probe on Windows, and this library has no C runtime. */
__attribute__((noinline))
static int guide_plane(const PtdState *s, i32 V, double rg, double rms_max, double tan_tilt, double *g)
{
    const double u = s->scale;
    const i32 n = (i32)s->n;
    i32 list[GUIDE_CAP], ring[RING_CAP];
    u8 hash[GUIDE_HASH];            /* position in list[], 255 = empty */
    double lx[GUIDE_CAP], ly[GUIDE_CAP], lz[GUIDE_CAP];
    const double rg2 = rg * rg;
    const i64 vx = s->X[V], vy = s->Y[V];
    for (int i = 0; i < GUIDE_HASH; i++) hash[i] = 255;
    int m = 0;
    list[0] = V; lx[0] = 0.0; ly[0] = 0.0; lz[0] = s->Z[V]; m = 1;
    hash[(uint32_t)V * 2654435761u >> 24] = 0;
    int level_start = 0;
    for (int depth = 0; depth < 4 && m < GUIDE_CAP; depth++) {
        const int level_end = m;
        for (int i = level_start; i < level_end && m < GUIDE_CAP; i++) {
            const int k = ring_of(s, list[i], ring, 0, 0, RING_CAP);
            for (int j = 0; j < k && m < GUIDE_CAP; j++) {
                const i32 w = ring[j];
                if (w >= n) continue;
                const double dx = (s->X[w] - vx) * u, dy = (s->Y[w] - vy) * u;
                if (dx * dx + dy * dy > rg2) continue;
                uint32_t h = (uint32_t)w * 2654435761u >> 24;
                int seen = 0;
                while (hash[h] != 255) {
                    if (list[hash[h]] == w) { seen = 1; break; }
                    h = (h + 1) & (GUIDE_HASH - 1);
                }
                if (seen) continue;
                hash[h] = (u8)m;
                list[m] = w; lx[m] = dx; ly[m] = dy; lz[m] = s->Z[w]; m++;
            }
        }
        if (level_end == m)
            break;
        level_start = level_end;
    }
    if (m < 6)
        return 0;
    double sx = 0, sy = 0, sz = 0;
    for (int i = 0; i < m; i++) { sx += lx[i]; sy += ly[i]; sz += lz[i]; }
    const double mx = sx / m, my = sy / m, mz = sz / m;
    double cxx = 0, cxy = 0, cyy = 0, cxz = 0, cyz = 0;
    for (int i = 0; i < m; i++) {
        const double dx = lx[i] - mx, dy = ly[i] - my, dz = lz[i] - mz;
        cxx += dx * dx; cxy += dx * dy; cyy += dy * dy; cxz += dx * dz; cyz += dy * dz;
    }
    const double tr = cxx + cyy, det = cxx * cyy - cxy * cxy;
    if (!(det > 0.0))
        return 0;
    const double a = (cxz * cyy - cyz * cxy) / det, b = (cyz * cxx - cxz * cxy) / det;
    const double c = mz - a * mx - b * my;
    double ss = 0;
    for (int i = 0; i < m; i++) {
        const double r = lz[i] - (c + a * lx[i] + b * ly[i]);
        ss += r * r;
    }
    if (ss > rms_max * rms_max * m)
        return 0;
    /* how well the tilt is pinned down: height scatter over the spread of the support in its narrowest direction */
    const double disc = tr * tr - 4.0 * det;
    const double lmin = 0.5 * (tr - fsqrt(disc > 0.0 ? disc : 0.0));
    double rms = fsqrt(ss / m);
    if (rms < 0.02) rms = 0.02;
    if (!(lmin > 0.0) || rms > tan_tilt * fsqrt(lmin))
        return 0;
    g[0] = a; g[1] = b; g[2] = c;
    return 1;
}

/* ---- following the surface past the edge of what is known ----------------
 *
 * The plain test asks "does the point fit the triangle it lies in?". That
 * triangle is a poor guide where the known ground stops: it is then a long
 * chord to some far-away vertex, passing under a hill or an embankment, and
 * nothing on the hill fits it. There the ground already found next to the
 * point is the better guide: a plane fitted through it (guide_plane) is
 * extended, and the candidate is accepted if it rises no more than tan_up
 * above that extension per metre from V (and drops no more than tan_down).
 *
 * It is only used where the triangle really is such a chord: where one of
 * its other corners lies more than check_tol below the extension. Where the
 * triangle agrees with the ground around V, it is ground with brush or
 * grass over it, and the plain test has the last word.
 *
 * One candidate - the lowest - per triangle corner per round; the plain test
 * then fills in around it before the next round.
 *
 * prm = { chord_edge, dmin, dmax, tan_up, tan_down, tan_max_slope,
 *         guide_radius, guide_rms, tan(guide_tilt), check_tol }
 * Writes the chosen candidates to picks[] and returns how many. */
EXPORT i64 ptd_frontier(PtdState *s, const double *prm, i32 *picks)
{
    const double u = s->scale;
    const double local2 = prm[0] * prm[0], dmin2 = prm[1] * prm[1],
                 dmax2 = prm[2] * prm[2], tan_up = prm[3], tan_down = prm[4],
                 tan_max = prm[5], guide_r = prm[6], guide_rms = prm[7],
                 guide_tilt = prm[8], check_tol = prm[9];
    const i32 n = (i32)s->n;
    i64 count = 0;
    if (!s->vtri)
        return 0;
    for (i64 t = 0; t < s->m; t++) {
        if (s->thead[t] < 0)
            continue;
        const i32 *v = s->tv + 3 * t;
        /* Nothing changed at any of its corners since the last scan: nothing to do. */
        if (s->vtouch && !s->vtouch[v[0]] && !s->vtouch[v[1]] && !s->vtouch[v[2]])
            continue;
        double px[3], py[3], pz[3];
        for (int k = 0; k < 3; k++) {
            px[k] = s->X[v[k]] * u; py[k] = s->Y[v[k]] * u; pz[k] = s->Z[v[k]];
        }
        double e2 = 0.0;
        for (int k = 0; k < 3; k++) {
            const int j = NEXT3(k);
            const double d2 = (px[j] - px[k]) * (px[j] - px[k]) + (py[j] - py[k]) * (py[j] - py[k]);
            if (d2 > e2) e2 = d2;
        }
        if (e2 <= local2)
            continue; /* a small triangle is its own best guide */
        const double e1x = px[1] - px[0], e1y = py[1] - py[0], e1z = pz[1] - pz[0];
        const double e2x = px[2] - px[0], e2y = py[2] - py[0], e2z = pz[2] - pz[0];
        const double nx = e1y * e2z - e1z * e2y, ny = e1z * e2x - e1x * e2z,
                     nz = e1x * e2y - e1y * e2x;
        if (!(nz > 0.0))
            continue;
        i32 best[3] = {-1, -1, -1};
        double best_tan[3] = {0.0, 0.0, 0.0};
        int state[3] = {0, 0, 0};          /* 0 not looked at yet, 1 has a guide, -1 has none */
        double guide[3][3];
        for (i32 q = s->thead[t]; q >= 0; q = s->next[q]) {
            if (s->locked && (s->locked[q] & 1))
                continue;
            const double qx = s->X[q] * u, qy = s->Y[q] * u, qz = s->Z[q];
            if ((qx - px[0]) * nx + (qy - py[0]) * ny + (qz - pz[0]) * nz <= 0.0)
                continue; /* below the triangle: not this rule's business */
            int k = 0;
            double d2 = (qx - px[0]) * (qx - px[0]) + (qy - py[0]) * (qy - py[0]);
            for (int j = 1; j < 3; j++) {
                const double c2 = (qx - px[j]) * (qx - px[j]) + (qy - py[j]) * (qy - py[j]);
                if (c2 < d2) { d2 = c2; k = j; }
            }
            const i32 V = v[k];
            if (V >= n || d2 < dmin2 || d2 > dmax2 || state[k] < 0)
                continue;
            if (state[k] == 0) {
                state[k] = -1;
                double *g = guide[k];
                if (!guide_plane(s, V, guide_r, guide_rms, guide_tilt, g))
                    continue;
                if (g[0] * g[0] + g[1] * g[1] > tan_max * tan_max)
                    continue; /* a guide steeper than the steepest allowed step is a wall, not ground */
                /* Following is for where this triangle is a chord: where one of its other
                 * corners lies clearly below the way the ground around V is heading. Where
                 * they agree with it, the triangle is as good a guide as the plane and the
                 * plain test has already said no. */
                int chord = 0;
                for (int j = 0; j < 3 && !chord; j++) {
                    if (j == k) continue;
                    const double wx = px[j] - px[k], wy = py[j] - py[k];
                    if (pz[j] < g[2] + g[0] * wx + g[1] * wy - (check_tol + tan_up * fsqrt(wx * wx + wy * wy)))
                        chord = 1;
                }
                if (!chord)
                    continue;
                state[k] = 1;
            }
            const double *g = guide[k];
            const double d = fsqrt(d2);
            const double step = (qz - pz[k]) / d;
            if (step > tan_max || step < -tan_max)
                continue;
            const double td = (qz - (g[2] + g[0] * (qx - px[k]) + g[1] * (qy - py[k]))) / d;
            if (td > tan_up || td < -tan_down)
                continue;
            if (best[k] < 0 || td < best_tan[k]) {
                best[k] = q;
                best_tan[k] = td;
            }
        }
        for (int k = 0; k < 3; k++)
            if (best[k] >= 0)
                picks[count++] = best[k];
    }
    if (s->vtouch)
        for (i64 p = 0; p < s->n + 4; p++)
            s->vtouch[p] = 0;
    return count;
}

/* ---- turning a corner: the lowest points form a new plane -----------------
 *
 * Following only continues ground straight on. At the foot of a bank, the
 * bottom of a gully or the start of a ridge the ground turns, and neither the
 * triangle nor the extended ground says where to. What does say it is the
 * points themselves: next to a ground vertex V, in a long triangle, the
 * LOWEST candidate in each small sector-and-distance bin is taken, and a
 * plane through V is fitted to those. If they lie on one plane (to within
 * tol), no steeper than tan_max, they are ground: a bank or a rock face has a
 * flat underside to its points, brush and branches do not. Bins whose lowest
 * point is not ground (a bush where no pulse got through) stand above the
 * plane and are dropped from the fit, highest first, as long as enough bins
 * are left.
 *
 * prm = { chord_edge, rmin, rmax, tol, tan_max_slope, min_bins, min_reach, check_radius, check_tol }
 * Writes the accepted candidates to picks[] and returns how many. */
#define HINGE_RADIAL 5
#define HINGE_ANGULAR 4
#define HINGE_BINS (HINGE_RADIAL * HINGE_ANGULAR)
EXPORT i64 ptd_hinge(PtdState *s, const double *prm, i32 *picks)
{
    const double u = s->scale;
    const double chord2 = prm[0] * prm[0], rmin = prm[1], rmax = prm[2], tol = prm[3],
                 tan_max = prm[4], min_reach = prm[6], check_r = prm[7], check_tol = prm[8];
    const int min_bins = (int)prm[5];
    const i32 n = (i32)s->n;
    i64 count = 0;
    for (i64 t = 0; t < s->m; t++) {
        if (s->thead[t] < 0)
            continue;
        const i32 *v = s->tv + 3 * t;
        if (s->vtouch && !s->vtouch[v[0]] && !s->vtouch[v[1]] && !s->vtouch[v[2]])
            continue;
        double px[3], py[3], pz[3];
        for (int k = 0; k < 3; k++) {
            px[k] = s->X[v[k]] * u; py[k] = s->Y[v[k]] * u; pz[k] = s->Z[v[k]];
        }
        double e2 = 0.0;
        for (int k = 0; k < 3; k++) {
            const int j = NEXT3(k);
            const double d2 = (px[j] - px[k]) * (px[j] - px[k]) + (py[j] - py[k]) * (py[j] - py[k]);
            if (d2 > e2) e2 = d2;
        }
        if (e2 <= chord2)
            continue;
        const double e1x = px[1] - px[0], e1y = py[1] - py[0], e1z = pz[1] - pz[0];
        const double e2x = px[2] - px[0], e2y = py[2] - py[0], e2z = pz[2] - pz[0];
        const double nx = e1y * e2z - e1z * e2y, ny = e1z * e2x - e1x * e2z,
                     nz = e1x * e2y - e1y * e2x;
        if (!(nz > 0.0))
            continue;
        i32 bin_q[3][HINGE_BINS];
        double bin_s[3][HINGE_BINS];
        int any[3] = {0, 0, 0};
        for (int k = 0; k < 3; k++)
            for (int b = 0; b < HINGE_BINS; b++) bin_q[k][b] = -1;
        for (i32 q = s->thead[t]; q >= 0; q = s->next[q]) {
            if (s->locked && (s->locked[q] & 1))
                continue;
            const double qx = s->X[q] * u, qy = s->Y[q] * u, qz = s->Z[q];
            if ((qx - px[0]) * nx + (qy - py[0]) * ny + (qz - pz[0]) * nz <= 0.0)
                continue;
            int k = 0;
            double d2 = (qx - px[0]) * (qx - px[0]) + (qy - py[0]) * (qy - py[0]);
            for (int j = 1; j < 3; j++) {
                const double c2 = (qx - px[j]) * (qx - px[j]) + (qy - py[j]) * (qy - py[j]);
                if (c2 < d2) { d2 = c2; k = j; }
            }
            if (v[k] >= n || d2 < rmin * rmin || d2 >= rmax * rmax)
                continue;
            const double d = fsqrt(d2);
            /* which way from V: q - V = alpha (a - V) + beta (b - V), both >= 0 inside the triangle */
            const int ia = NEXT3(k), ib = PREV3(k);
            const double ax = px[ia] - px[k], ay = py[ia] - py[k], bx = px[ib] - px[k], by = py[ib] - py[k];
            const double den = ax * by - ay * bx;
            if (!(den > 0.0))
                continue;
            const double dx = qx - px[k], dy = qy - py[k];
            /* scaled so that equal alpha and beta mean the middle of the corner whatever the edge lengths */
            const double la = fsqrt(ax * ax + ay * ay), lb = fsqrt(bx * bx + by * by);
            double alpha = (dx * by - dy * bx) / den * la, beta = (ax * dy - ay * dx) / den * lb;
            if (alpha < 0.0) alpha = 0.0;
            if (beta < 0.0) beta = 0.0;
            const double sum = alpha + beta;
            int ja = sum > 0.0 ? (int)(HINGE_ANGULAR * beta / sum) : 0;
            if (ja >= HINGE_ANGULAR) ja = HINGE_ANGULAR - 1;
            int jr = (int)(HINGE_RADIAL * (d - rmin) / (rmax - rmin));
            if (jr >= HINGE_RADIAL) jr = HINGE_RADIAL - 1;
            const int b = jr * HINGE_ANGULAR + ja;
            const double sl = (qz - pz[k]) / d;
            if (bin_q[k][b] < 0 || sl < bin_s[k][b]) {
                bin_q[k][b] = q;
                bin_s[k][b] = sl;
                any[k] = 1;
            }
        }
        for (int k = 0; k < 3; k++) {
            if (!any[k])
                continue;
            i32 lq[HINGE_BINS];
            double lx[HINGE_BINS], ly[HINGE_BINS], lz[HINGE_BINS];
            int m = 0;
            for (int b = 0; b < HINGE_BINS; b++) {
                const i32 q = bin_q[k][b];
                if (q < 0) continue;
                lq[m] = q; lx[m] = s->X[q] * u - px[k]; ly[m] = s->Y[q] * u - py[k]; lz[m] = s->Z[q] - pz[k]; m++;
            }
            double a = 0.0, b = 0.0;
            int good = 0;
            while (m >= min_bins) {
                double sxx = 0, sxy = 0, syy = 0, sxz = 0, syz = 0;
                for (int i = 0; i < m; i++) {
                    sxx += lx[i] * lx[i]; sxy += lx[i] * ly[i]; syy += ly[i] * ly[i];
                    sxz += lx[i] * lz[i]; syz += ly[i] * lz[i];
                }
                const double det = sxx * syy - sxy * sxy;
                if (!(det > 0.02 * sxx * syy))
                    break;                  /* all in one direction from V: the tilt across it is unknown */
                a = (sxz * syy - syz * sxy) / det; b = (syz * sxx - sxz * sxy) / det;
                double ss = 0, worst = -1e300, deepest = 0.0;
                int iw = -1;
                for (int i = 0; i < m; i++) {
                    const double r = lz[i] - (a * lx[i] + b * ly[i]);
                    ss += r * r;
                    if (r > worst) { worst = r; iw = i; }
                    if (r < deepest) deepest = r;
                }
                if (ss <= tol * tol * m && worst <= 2.0 * tol && deepest >= -2.0 * tol) {
                    good = 1;
                    break;
                }
                if (iw < 0 || worst <= 0.0)
                    break;
                m--;                        /* drop the bin that stands highest above the plane, fit again */
                lq[iw] = lq[m]; lx[iw] = lx[m]; ly[iw] = ly[m]; lz[iw] = lz[m];
            }
            if (!good || a * a + b * b > tan_max * tan_max)
                continue;
            double dmin = 1e300, dmax = 0.0;
            for (int i = 0; i < m; i++) {
                const double d2 = lx[i] * lx[i] + ly[i] * ly[i];
                if (d2 < dmin) dmin = d2;
                if (d2 > dmax) dmax = d2;
            }
            if (fsqrt(dmax) - fsqrt(dmin) < min_reach)
                continue;                   /* a plane seen over a hand's breadth is no plane */
            /* Known ground close by that lies clearly below the new plane says the plane is the
             * underside of something standing over that ground - brush with ground seen through it. */
            int over = 0;
            for (int j = 0; j < 3 && !over; j++) {
                if (j == k || v[j] >= n) continue;
                const double wx = px[j] - px[k], wy = py[j] - py[k];
                if (wx * wx + wy * wy > check_r * check_r) continue;
                if (pz[j] - pz[k] < a * wx + b * wy - check_tol)
                    over = 1;
            }
            if (over)
                continue;
            for (int i = 0; i < m; i++)
                picks[count++] = lq[i];
        }
    }
    if (s->vtouch)
        for (i64 p = 0; p < s->n + 4; p++)
            s->vtouch[p] = 0;
    return count;
}

/* Insert candidates chosen by ptd_frontier. The caller guarantees room for
 * 2 * count more triangles. Returns how many were inserted. */
EXPORT i64 ptd_insert(PtdState *s, const i32 *picks, i64 count)
{
    i64 done = 0;
    for (i64 i = 0; i < count; i++) {
        const i32 p = picks[i];
        const i32 t = s->ptri[p];
        if (t < 0 || s->m + 2 > s->mcap)
            continue;
        i32 prev = -1, q = s->thead[t];
        while (q >= 0 && q != p) {
            prev = q;
            q = s->next[q];
        }
        if (q != p)
            continue;
        unlink_candidate(s, t, p, prev);
        insert_point(s, t, p);
        s->n_ground++;
        done++;
    }
    return done;
}

/* Height of every candidate above the finished surface (0 for vertices). */
EXPORT void ptd_heights(const PtdState *s, float *out)
{
    const double u = s->scale;
    for (i64 p = 0; p < s->n; p++) {
        const i32 t = s->ptri[p];
        if (t == PT_VERTEX) {
            out[p] = 0.0f;
        } else if (t <= -10) {
            out[p] = (float)(s->Z[p] - s->Z[-10 - t]);
        } else {
            const i32 *v = s->tv + 3 * (i64)t;
            const double ax = s->X[v[0]] * u, ay = s->Y[v[0]] * u, az = s->Z[v[0]];
            const double e1x = s->X[v[1]] * u - ax, e1y = s->Y[v[1]] * u - ay,
                         e1z = s->Z[v[1]] - az;
            const double e2x = s->X[v[2]] * u - ax, e2y = s->Y[v[2]] * u - ay,
                         e2z = s->Z[v[2]] - az;
            const double nx = e1y * e2z - e1z * e2y;
            const double ny = e1z * e2x - e1x * e2z;
            const double nz = e1x * e2y - e1y * e2x;
            const double dx = s->X[p] * u - ax, dy = s->Y[p] * u - ay;
            const double zs = nz != 0.0 ? az - (nx * dx + ny * dy) / nz : az;
            out[p] = (float)(s->Z[p] - zs);
        }
    }
}

/* ---- taking spikes back out ------------------------------------------------
 *
 * A surface vertex that stands above all but one of its neighbours, each by
 * more than tan_up per metre and by more than h_up, is a spike: a stem, a
 * stone's edge, a stray return. "All but one" keeps ridge lines, wall tops
 * and bank edges, which always have a neighbour along them at their own
 * height on each side. A pit is the mirror image, but must be below ALL its
 * neighbours: low points are far more often real ground than high ones.
 *
 * prm = { tan_up, h_up, tan_down, h_down, min_neighbours }
 * Writes the spikes to list[] and returns how many. */
EXPORT i64 ptd_spikes(PtdState *s, const double *prm, i32 *list)
{
    const double u = s->scale, tan_up = prm[0], h_up = prm[1], tan_down = prm[2], h_down = prm[3];
    const int need = (int)prm[4];
    const i32 n = (i32)s->n;
    i64 count = 0;
    i32 ring[RING_CAP];
    if (!s->vtri)
        return 0;
    for (i32 p = 0; p < n; p++) {
        if (s->ptri[p] != PT_VERTEX || (s->locked && (s->locked[p] & 2)))
            continue;
        const int k = ring_of(s, p, ring, 0, 0, RING_CAP);
        if (k < 3)
            continue;
        const double zp = s->Z[p];
        double lo1 = 1e300, lo2 = 1e300;   /* two smallest slopes down to a neighbour */
        double hi1 = -1e300;               /* largest slope down to a neighbour */
        double top1 = -1e300, top2 = -1e300, bottom = 1e300;   /* neighbour heights */
        int m = 0;
        for (int i = 0; i < k; i++) {
            const i32 w = ring[i];
            if (w >= n) continue;
            const double dx = (s->X[w] - s->X[p]) * u, dy = (s->Y[w] - s->Y[p]) * u;
            const double d = fsqrt(dx * dx + dy * dy);
            if (!(d > 0.0)) continue;
            const double zw = s->Z[w], sl = (zp - zw) / d;
            if (sl < lo1) { lo2 = lo1; lo1 = sl; } else if (sl < lo2) lo2 = sl;
            if (sl > hi1) hi1 = sl;
            if (zw > top1) { top2 = top1; top1 = zw; } else if (zw > top2) top2 = zw;
            if (zw < bottom) bottom = zw;
            m++;
        }
        if (m < need)
            continue;
        if (tan_up > 0.0 && lo2 > tan_up && zp - top2 > h_up)
            list[count++] = p;
        else if (tan_down > 0.0 && hi1 < -tan_down && bottom - zp > h_down)
            list[count++] = p;
    }
    return count;
}

/* Take vertex p out of the surface: the triangles around it are replaced by
 * the Delaunay triangulation of the hole (ears whose circumcircle holds no
 * other corner of the hole), their candidates are handed to the new
 * triangles, and p becomes a candidate again - a locked one. Nothing is
 * touched unless the whole hole can be filled. */
__attribute__((noinline))
static int remove_vertex(PtdState *s, i32 p)
{
    i32 P[RING_CAP], T[RING_CAP], F[RING_CAP];
    const int k = ring_of(s, p, P, T, F, RING_CAP);
    if (k < 3)
        return 0;
    /* F[i]: what lies outside edge P[i] -> P[i+1]: a triangle (>= 0), nothing (-1),
     * or new triangle j (-2 - j). */
    i32 nv[RING_CAP][3], nn[RING_CAP][3];
    int m = k, made = 0;
    while (m > 3) {
        int ear = -1;
        for (int i = 0; i < m && ear < 0; i++) {
            const i32 a = P[(i + m - 1) % m], b = P[i], c = P[(i + 1) % m];
            if (orient(s, a, b, c) <= 0)
                continue;
            int ok = 1;
            for (int j = 0; j < m && ok; j++) {
                const i32 q = P[j];
                if (q != a && q != b && q != c && incircle_pos(s, a, b, c, q))
                    ok = 0;
            }
            if (ok)
                ear = i;
        }
        if (ear < 0)
            return 0;
        const int i = ear, im = (i + m - 1) % m;
        nv[made][0] = P[im]; nv[made][1] = P[i]; nv[made][2] = P[(i + 1) % m];
        nn[made][0] = F[i];      /* opposite a: edge b-c */
        nn[made][1] = -1;        /* opposite b: edge c-a, the new diagonal - filled in by its other side */
        nn[made][2] = F[im];     /* opposite c: edge a-b */
        if (F[i] <= -2) nn[-2 - F[i]][1] = -2 - made;
        if (F[im] <= -2) nn[-2 - F[im]][1] = -2 - made;
        F[im] = -2 - made;       /* edge a -> c now has the new triangle outside it */
        for (int j = i; j < m - 1; j++) { P[j] = P[j + 1]; F[j] = F[j + 1]; }
        made++;
        m--;
    }
    if (orient(s, P[0], P[1], P[2]) <= 0)
        return 0;
    nv[made][0] = P[0]; nv[made][1] = P[1]; nv[made][2] = P[2];
    nn[made][0] = F[1]; nn[made][1] = F[2]; nn[made][2] = F[0];
    for (int e = 0; e < 3; e++)
        if (nn[made][e] <= -2) nn[-2 - nn[made][e]][1] = -2 - made;
    made++;

    /* ---- commit ---- */
    i32 head = -1;
    for (int i = 0; i < k; i++) {
        const i32 t = T[i];
        if (s->thead[t] >= 0) {
            s->next[s->ttail[t]] = head;
            head = s->thead[t];
        }
        s->thead[t] = s->ttail[t] = -1;
    }
    for (int j = 0; j < made; j++) {
        i32 nb[3];
        for (int e = 0; e < 3; e++)
            nb[e] = nn[j][e] <= -2 ? T[-2 - nn[j][e]] : nn[j][e];
        set_tri(s, T[j], nv[j][0], nv[j][1], nv[j][2], nb[0], nb[1], nb[2]);
    }
    for (int j = 0; j < made; j++)
        for (int e = 0; e < 3; e++) {
            const i32 x = nn[j][e];
            if (x < 0)
                continue;       /* nothing outside, or one of the new triangles */
            const i32 va = nv[j][NEXT3(e)], vb = nv[j][PREV3(e)];
            const i32 *xv = s->tv + 3 * (i64)x;
            i32 *xn = s->tn + 3 * (i64)x;
            for (int q = 0; q < 3; q++)
                if (xv[q] != va && xv[q] != vb) { xn[q] = T[j]; break; }
        }
    for (int i = made; i < k; i++) {         /* the two triangles left over */
        i32 *dv = s->tv + 3 * (i64)T[i], *dn = s->tn + 3 * (i64)T[i];
        dv[0] = dv[1] = dv[2] = -1;
        dn[0] = dn[1] = dn[2] = -1;
    }
    if (s->locked)
        s->locked[p] |= 1;
    s->next[p] = head;
    head = p;
    while (head >= 0) {
        const i32 q = head;
        head = s->next[q];
        int dst = 0;
        for (int j = 0; j < made; j++)
            if (orient(s, nv[j][0], nv[j][1], q) >= 0 && orient(s, nv[j][1], nv[j][2], q) >= 0 &&
                orient(s, nv[j][2], nv[j][0], q) >= 0) { dst = j; break; }
        bucket_append(s, T[dst], q);
    }
    for (int j = 0; j < made; j++)
        enqueue(s, T[j]);
    s->n_ground--;
    return 1;
}

/* Remove the listed vertices (see ptd_spikes). Returns how many were removed. */
EXPORT i64 ptd_remove(PtdState *s, const i32 *list, i64 count)
{
    i64 done = 0;
    if (!s->vtri || !s->locked)
        return 0;
    for (i64 i = 0; i < count; i++) {
        const i32 p = list[i];
        if (p < 0 || p >= s->n || s->ptri[p] != PT_VERTEX || (s->locked[p] & 2))
            continue;
        done += remove_vertex(s, p);
    }
    return done;
}

/* Put every triangle that holds candidates back on the work queue (used
 * after candidates have been unlocked). */
EXPORT void ptd_requeue(PtdState *s)
{
    for (i64 t = 0; t < s->m; t++)
        if (s->thead[t] >= 0)
            enqueue(s, (i32)t);
}

/* Bumped whenever the behaviour or the interface changes; ptdcore.py
 * refuses a library whose number is not the one it was written for. */
#define PTD_VERSION 10
EXPORT i64 ptd_version(void)
{
    return PTD_VERSION;
}

/* Lets the Python side confirm it was built against the same state layout. */
EXPORT i64 ptd_state_size(void)
{
    return (i64)sizeof(PtdState);
}

/* Candidates that lie in a thin band about the finished surface: no more
 * than `up` above and `down` below the triangle they are in (metres, measured
 * straight up), in triangles whose edges are all shorter than max_edge - a
 * long triangle is a guess across a gap, not a surface to hang points on.
 * Locked candidates (isolated low points, removed spikes) are never members.
 * out[p] = 1 for a member, 0 otherwise; returns how many. */
EXPORT i64 ptd_members(const PtdState *s, double up, double down, double max_edge, u8 *out)
{
    const double u = s->scale, e2max = max_edge * max_edge;
    const i32 n = (i32)s->n;
    i64 count = 0;
    for (i64 p = 0; p < s->n; p++)
        out[p] = 0;
    for (i64 t = 0; t < s->m; t++) {
        if (s->thead[t] < 0)
            continue;
        const i32 *v = s->tv + 3 * t;
        if (v[0] >= n || v[1] >= n || v[2] >= n)
            continue;
        const double ax = s->X[v[0]] * u, ay = s->Y[v[0]] * u, az = s->Z[v[0]];
        const double e1x = s->X[v[1]] * u - ax, e1y = s->Y[v[1]] * u - ay, e1z = s->Z[v[1]] - az;
        const double e2x = s->X[v[2]] * u - ax, e2y = s->Y[v[2]] * u - ay, e2z = s->Z[v[2]] - az;
        const double e3x = e2x - e1x, e3y = e2y - e1y;
        if (e1x * e1x + e1y * e1y > e2max || e2x * e2x + e2y * e2y > e2max || e3x * e3x + e3y * e3y > e2max)
            continue;
        const double nx = e1y * e2z - e1z * e2y, ny = e1z * e2x - e1x * e2z, nz = e1x * e2y - e1y * e2x;
        if (!(nz > 0.0))
            continue;
        for (i32 q = s->thead[t]; q >= 0; q = s->next[q]) {
            if (s->locked && (s->locked[q] & 1))
                continue;
            const double h = s->Z[q] - (az - (nx * (s->X[q] * u - ax) + ny * (s->Y[q] * u - ay)) / nz);
            if (h <= up && h >= -down) {
                out[q] = 1;
                count++;
            }
        }
    }
    return count;
}

/* Self-check used by the tests: 0 if every triangle is counter-clockwise,
 * neighbour links are mutual, every listed candidate lies inside its
 * triangle, and no neighbouring vertex lies inside a circumcircle. */
EXPORT i64 ptd_check(const PtdState *s)
{
    i64 bad = 0;
    for (i64 t = 0; t < s->m; t++) {
        const i32 *v = s->tv + 3 * t, *n = s->tn + 3 * t;
        if (v[0] < 0)
            continue;   /* left over by a removed vertex */
        if (orient(s, v[0], v[1], v[2]) <= 0)
            bad++;
        for (int k = 0; k < 3; k++) {
            const i32 u = n[k];
            if (u < 0)
                continue;
            const i32 *un = s->tn + 3 * (i64)u, *uv = s->tv + 3 * (i64)u;
            const int j = (un[0] == t) ? 0 : (un[1] == t) ? 1 : (un[2] == t) ? 2 : -1;
            if (j < 0) {
                bad++;
                continue;
            }
            if (uv[NEXT3(j)] != v[PREV3(k)] || uv[PREV3(j)] != v[NEXT3(k)])
                bad++;
            if (incircle_pos(s, v[0], v[1], v[2], uv[j]))
                bad++;
        }
        for (i32 q = s->thead[t]; q >= 0; q = s->next[q]) {
            if (s->ptri[q] != t)
                bad++;
            if (orient(s, v[0], v[1], q) < 0 || orient(s, v[1], v[2], q) < 0 ||
                orient(s, v[2], v[0], q) < 0)
                bad++;
        }
    }
    return bad;
}
