"""Compare two sets of classified LAS/LAZ outputs point by point.

    ground-classify compare <A> <B> [<reference>]

<A> and <B> are folders (or single files) holding the same tiles classified
two different ways, for example with and without an optional stage. Files
are matched by name. <reference>, if given, is a folder with the same tiles
classified by another tool (for example TerraScan); its class 2 is treated
as the reference ground.

For every tile it reports
  * how many points have a different class, and the commonest changes;
  * the number of ground (class 2) points in each;
  * how far apart the two ground surfaces are: the triangulated surface
    through A's ground points is sampled at B's ground points, and the
    other way round (RMSE, 95th percentile and maximum, in the file's own
    units);
  * against the reference, if given: precision and recall of each ground
    class, and how far the reference ground points lie from each surface.

Nothing is written or changed.
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    import laspy
    import numpy as np
    from scipy.interpolate import LinearNDInterpolator
except ImportError as exc:
    raise SystemExit(
        f"Missing Python package: {getattr(exc, 'name', exc)}. Run this with "
        "the environment you installed the classifier into (make venv)."
    ) from exc

SUFFIX = "_AdaptiveGround"


def tile_key(path: Path) -> str:
    stem = path.stem
    return stem[: -len(SUFFIX)] if stem.endswith(SUFFIX) else stem


def collect(target: Path) -> dict[str, Path]:
    if target.is_file():
        return {tile_key(target): target}
    files = sorted(target.glob("*.las")) + sorted(target.glob("*.laz"))
    return {tile_key(path): path for path in files}


def surface_gap(
    surface_xyz: tuple[np.ndarray, np.ndarray, np.ndarray],
    sample_xyz: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[float, float, float, int] | None:
    """Vertical distance from sample points to the TIN through surface points."""
    sx, sy, sz = surface_xyz
    px, py, pz = sample_xyz
    if sx.size < 4 or px.size == 0:
        return None
    interpolate = LinearNDInterpolator(np.column_stack((sx, sy)), sz)
    gap = pz - interpolate(np.column_stack((px, py)))
    gap = np.abs(gap[np.isfinite(gap)])
    if gap.size == 0:
        return None
    return (
        float(np.sqrt(np.mean(gap**2))),
        float(np.percentile(gap, 95)),
        float(np.max(gap)),
        int(gap.size),
    )


def describe_gap(label: str, gap: tuple[float, float, float, int] | None) -> str:
    if gap is None:
        return f"    {label}: not enough ground points to compare"
    return (
        f"    {label}: RMSE {gap[0]:.4f}, 95% within {gap[1]:.4f}, "
        f"largest {gap[2]:.3f}  (file units, {gap[3]:,} points)"
    )


def main(argv: list[str] | None = None) -> int:
    argv = [str(a) for a in (sys.argv[1:] if argv is None else argv)]
    if len(argv) not in (2, 3):
        print(__doc__)
        return 2
    a_files, b_files = collect(Path(argv[0])), collect(Path(argv[1]))
    reference = collect(Path(argv[2])) if len(argv) == 3 else {}
    common = sorted(set(a_files) & set(b_files))
    if not common:
        raise SystemExit("No tiles with the same name were found in both A and B.")
    print(f"A = {argv[0]}\nB = {argv[1]}")
    if reference:
        print(f"reference = {argv[2]}")
    total_points = total_changed = total_a = total_b = 0
    for key in common:
        a, b = laspy.read(a_files[key]), laspy.read(b_files[key])
        ax, ay, az = (np.asarray(v, dtype=np.float64) for v in (a.x, a.y, a.z))
        bx, by, bz = (np.asarray(v, dtype=np.float64) for v in (b.x, b.y, b.z))
        ca = np.asarray(a.classification, dtype=np.int64)
        cb = np.asarray(b.classification, dtype=np.int64)
        print(f"\n{key}: {ca.size:,} points")
        if ca.size != cb.size or not (
            np.array_equal(ax, bx) and np.array_equal(ay, by) and np.array_equal(az, bz)
        ):
            print("    The two files do not hold the same points in the same order; skipped.")
            continue
        changed = ca != cb
        ga, gb = ca == 2, cb == 2
        total_points += ca.size
        total_changed += int(changed.sum())
        total_a += int(ga.sum())
        total_b += int(gb.sum())
        print(
            f"    class differs on {int(changed.sum()):,} points "
            f"({100.0 * changed.mean():.3f}%)"
        )
        if changed.any():
            pairs, counts = np.unique(
                ca[changed] * 1000 + cb[changed], return_counts=True
            )
            order = np.argsort(-counts)[:5]
            print(
                "    commonest changes (class in A -> class in B): "
                + ", ".join(
                    f"{int(pairs[i]) // 1000}->{int(pairs[i]) % 1000}: {int(counts[i]):,}"
                    for i in order
                )
            )
        print(
            f"    ground points: A {int(ga.sum()):,}, B {int(gb.sum()):,} "
            f"({100.0 * (int(gb.sum()) - int(ga.sum())) / max(1, int(ga.sum())):+.1f}%)"
        )
        print(describe_gap("B's ground points against A's ground surface",
                           surface_gap((ax[ga], ay[ga], az[ga]), (bx[gb], by[gb], bz[gb]))))
        print(describe_gap("A's ground points against B's ground surface",
                           surface_gap((bx[gb], by[gb], bz[gb]), (ax[ga], ay[ga], az[ga]))))
        if key in reference:
            r = laspy.read(reference[key])
            rx, ry, rz = (np.asarray(v, dtype=np.float64) for v in (r.x, r.y, r.z))
            cr = np.asarray(r.classification, dtype=np.int64)
            same_points = cr.size == ca.size and np.array_equal(rx, ax) and np.array_equal(ry, ay)
            gr = cr == 2
            for label, mask, xyz in (("A", ga, (ax, ay, az)), ("B", gb, (bx, by, bz))):
                if same_points:
                    hit = int(np.sum(mask & gr))
                    print(
                        f"    {label} against the reference ground: precision "
                        f"{100.0 * hit / max(1, int(mask.sum())):.1f}%, recall "
                        f"{100.0 * hit / max(1, int(gr.sum())):.1f}%"
                    )
                print(describe_gap(
                    f"reference ground points against {label}'s ground surface",
                    surface_gap((xyz[0][mask], xyz[1][mask], xyz[2][mask]), (rx[gr], ry[gr], rz[gr])),
                ))
    if total_points:
        print(
            f"\nALL TILES: {total_points:,} points; class differs on "
            f"{total_changed:,} ({100.0 * total_changed / total_points:.3f}%); "
            f"ground points A {total_a:,}, B {total_b:,}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
