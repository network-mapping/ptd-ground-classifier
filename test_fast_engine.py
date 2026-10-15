"""Checks of the compiled core. usage: test_engine.py [clips_dir]"""
import sys, os, numpy as np
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
import fast_ground as fg
lib = fg.load_library()
other = os.path.join(here, "libptdcore_clang.so"); lib2 = fg.load_library(other) if os.path.exists(other) else None
fails = 0
def ok(name, cond, note=""):
    global fails
    print(("ok   " if cond else "FAIL ") + name + (" - " + note if note else ""), flush=True)
    fails += 0 if cond else 1
def go(x, y, z, settings=None, library=lib, small=False):
    r = fg.GroundRun(x, y, z, settings, library)
    if small:
        r.mcap = 0; r._allocate(1024)      # force the triangle arrays to be enlarged many times mid-run
    return r.run()
rng = np.random.default_rng(5)
# 1. random terrain with objects: triangulation valid, objects rejected
n = 200_000; x = rng.uniform(0, 400, n); y = rng.uniform(0, 400, n); z = 3 * np.sin(x / 40) + 2 * np.cos(y / 55) + rng.normal(0, 0.02, n)
obj = rng.random(n) < 0.3; z[obj] += rng.uniform(1.5, 20, obj.sum())
r = go(x, y, z); g = r.ground()
ok("random terrain: triangulation valid", r.check() == 0)
ok("random terrain: objects rejected, ground found", g[obj].mean() < 0.002 and g[~obj].mean() > 0.5, f"objects taken {100*g[obj].mean():.3f}%, ground taken {100*g[~obj].mean():.1f}%")
h = r.heights(); mod = r.model(); band = float(r.p["surface_band"])
ok("heights: zero on surface points, within the band on the other ground points, objects above", np.all(h[mod] == 0) and np.all(np.abs(h[g]) <= band + 1e-6) and np.median(h[obj]) > 1.0, f"{int(mod.sum()):,} surface points + {int(g.sum() - mod.sum()):,} within {100*band:g} cm of the surface")
# 2. exactly collinear / regular grid / duplicate XY
gx, gy = np.meshgrid(np.arange(0, 120, 0.5), np.arange(0, 120, 0.5)); gx = gx.ravel(); gy = gy.ravel(); gz = 0.01 * gx + rng.normal(0, 0.005, gx.size)
gx = np.concatenate((gx, gx[:5000])); gy = np.concatenate((gy, gy[:5000])); gz = np.concatenate((gz, gz[:5000] + 0.3))
r = go(gx, gy, gz); ok("regular grid with duplicate XY: triangulation valid", r.check() == 0, f"{int(r.ground().sum()):,} ground of {gx.size:,}")
# 3. the answer does not depend on how the memory grew
r2 = go(x, y, z, small=True); ok("same result when arrays are enlarged mid-run", np.array_equal(g, r2.ground()) and np.array_equal(h, r2.heights()))
# 4. isolated low points: kept out, labelled; a real pit is kept
z2 = z.copy(); low = np.flatnonzero(~obj)[:300]; z2[low] -= rng.uniform(1.0, 30.0, 300)
r = go(x, y, z2); g2 = r.ground(); ok("isolated low points kept out of the ground", g2[low].sum() == 0 and r.low()[low].mean() > 0.95, f"flagged {100*r.low()[low].mean():.0f}%")
keep = ~np.isin(np.arange(n), low); dh = np.abs(r.heights() - h)[keep]
ok("surface elsewhere unchanged by the low points", np.percentile(dh, 99) < 0.05, f"heights differ by {100*np.percentile(dh, 99):.1f} cm at the 99th percentile, {100*dh.max():.0f} cm at most")
# 5. a hill top the plain routine cannot climb is reached by following the surface
n = 300_000; x = rng.uniform(0, 300, n); y = rng.uniform(0, 300, n); d = np.hypot(x - 150, y - 150)
z = np.where(d < 18, 12.0, np.where(d < 30, 12.0 * (30 - d) / 12.0, 0.0)) + rng.normal(0, 0.02, n)   # flat-topped hill, 45 degree sides
top = d < 15
plain = go(x, y, z, {"follow_rounds": 0, "turn_rounds": 0}).ground(); full = go(x, y, z)
ok("hill top: missed by the plain routine, reached by following and turning", plain[top].mean() < 0.05 and full.ground()[top].mean() > 0.5 and full.check() == 0, f"plain {100*plain[top].mean():.1f}%, with following {100*full.ground()[top].mean():.1f}%")
# 6. a building (vertical walls) is NOT climbed
z = np.where((np.abs(x - 150) < 20) & (np.abs(y - 150) < 20), 6.0, 0.0) + rng.normal(0, 0.02, n); roof = (np.abs(x - 150) < 19.5) & (np.abs(y - 150) < 19.5)
r = go(x, y, z); ok("building roof not taken as ground", r.ground()[roof].sum() == 0, f"{int(r.ground()[roof].sum())} roof points taken")
# 6b. the same building with returns on its walls: still not climbed
wn = 20_000; side = rng.integers(0, 4, wn); t = rng.uniform(-20, 20, wn); wx = np.where(side == 0, -20, np.where(side == 1, 20, t)) + 150; wy = np.where(side == 2, -20, np.where(side == 3, 20, t)) + 150
wx = np.where(side < 2, wx, 150 + t); x2 = np.concatenate((x, wx + rng.normal(0, 0.02, wn))); y2 = np.concatenate((y, wy + rng.normal(0, 0.02, wn))); z2 = np.concatenate((z, rng.uniform(0.0, 6.0, wn)))
r = go(x2, y2, z2); g2 = r.ground(); ok("building with wall returns: roof not taken", g2[:n][roof].sum() == 0 and (g2[n:] & (z2[n:] > 0.5)).sum() < 20, f"{int(g2[:n][roof].sum())} roof points, {int((g2[n:] & (z2[n:] > 0.5)).sum())} wall points above 0.5 m taken")
# 6c. low vegetation over gently rolling ground: stays out
x3 = rng.uniform(0, 300, n); y3 = rng.uniform(0, 300, n); base = 2 * np.sin(x3 / 30) + 1.5 * np.cos(y3 / 45); veg = rng.random(n) < 0.6
z3 = base + rng.normal(0, 0.02, n) + np.where(veg, rng.uniform(0.15, 2.5, n), 0.0)
r = go(x3, y3, z3); g3 = r.ground(); ok("low vegetation not taken as ground", g3[veg].mean() < 0.005 and g3[~veg].mean() > 0.5, f"vegetation taken {100*g3[veg].mean():.2f}%, ground taken {100*g3[~veg].mean():.1f}%")
# 6d. a sliver smaller than one building square (edge-of-project tile): still classified
n4 = 120_000; x4 = rng.uniform(0, 30, n4); y4 = rng.uniform(0, 50, n4); veg4 = rng.random(n4) < 0.5
z4 = 0.03 * x4 + rng.normal(0, 0.02, n4) + np.where(veg4, rng.uniform(0.3, 15, n4), 0.0)
r = go(x4, y4, z4); g4 = r.ground(); h4 = r.heights()
ok("small sliver (30 x 50 m): classified with smaller starting squares", r.check() == 0 and g4[veg4].mean() < 0.005 and g4.sum() > 5000 and np.percentile(np.abs(h4[~veg4]), 99) < 0.1 and np.percentile(h4[veg4], 1) > 0.2,
   f"starting square {r.seed_square:g} m, {int(g4.sum()):,} ground points, vegetation taken {100*g4[veg4].mean():.2f}%, bare points within {100*np.percentile(np.abs(h4[~veg4]), 99):.0f} cm of the surface")
x5 = rng.uniform(0, 8, 20_000); y5 = rng.uniform(0, 8, 20_000); z5 = rng.normal(0, 0.02, 20_000)
r = go(x5, y5, z5); h5 = r.heights(); ok("tiny patch (8 x 8 m): classified", r.check() == 0 and r.ground().sum() > 100 and np.percentile(np.abs(h5), 99) < 0.1, f"starting square {r.seed_square:g} m, {int(r.ground().sum()):,} ground points")
# 7. real clips: valid, and identical between the gcc and clang builds
clips = sys.argv[1] if len(sys.argv) > 1 else None
if clips:
    ref_path = os.path.join(os.path.dirname(os.path.abspath(clips)), "fast_engine_reference.npz")
    ref = dict(np.load(ref_path)) if os.path.exists(ref_path) else {}
    made = {}
    for f in sorted(x for x in os.listdir(clips) if x.endswith(".npz")):
        d = np.load(os.path.join(clips, f)); m = np.isin(d["c_input"], (2, 3, 4, 5, 13)); a = go(d["x"][m], d["y"][m], d["z"][m])
        ok(f"{f}: triangulation valid", a.check() == 0, f"{int(a.ground().sum()):,} ground of {int(m.sum()):,}")
        made[f] = np.packbits(a.ground())
        if f in ref:   # the result recorded on Linux when this package was made
            diff = int(np.unpackbits(ref[f])[: int(m.sum())].astype(bool).__xor__(a.ground()).sum())
            ok(f"{f}: same ground as the reference result made on Linux", diff == 0, f"{diff:,} points differ")
        if lib2 is not None:
            b = go(d["x"][m], d["y"][m], d["z"][m], library=lib2)
            ok(f"{f}: gcc and clang builds give the identical result", np.array_equal(a.ground(), b.ground()) and np.array_equal(a.heights(), b.heights()))
    if "--write-reference" in sys.argv:
        np.savez_compressed(ref_path, **made); print("wrote", ref_path)
print("ALL OK" if fails == 0 else f"{fails} FAILED"); sys.exit(1 if fails else 0)
