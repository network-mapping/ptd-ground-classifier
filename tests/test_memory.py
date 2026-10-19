"""Memory regression test: does a change that should not alter memory use alter it?

Purpose: tell whether a refactor has added array copies, kept arrays alive longer, or changed how the triangle arrays
grow. It records that memory use has not changed, not that it is small. The compiled core allocates nothing (every
array is a numpy array made in Python), so the numpy allocations counted by tracemalloc are the engine's memory.

Usage (project venv active, core built; `make memory` builds first):
    python tests/test_memory.py                      run the test; ends with ALL OK or N FAILED, exit code 1 on failure
    python tests/test_memory.py --write-reference --note "..."     record the current numbers as the reference

What is measured, on each of the four real clips in tests/validation/clips (default settings), in bytes above the
input arrays: the peak while GroundRun(x, y, z).run() runs; what is still held when it has finished; the extra peak
of each output (ground, heights, model, low); the peak of each stage (constructor, low-point filter, starting points,
state of the compiled core, growing the surface, spike removal); and what is left once the run is released. One more
run of the steep clip starts with a tiny triangle array (initial_capacity = 1024), so the growth path is measured too.

The stage peaks exist because the peak of the whole run is the largest of its stages: a larger core stage could hide
under a larger low-point stage. They are taken by wrapping the private stage methods of GroundRun for the duration
of the measurement, so those methods must keep their names (the test says so if one is missing).

A number passes when it is no more than 2% (plus 16 KiB) above the reference. A number more than 5% below the reference
passes with a note: update the reference so the improvement is kept. The point counts and the number of ground points
must match exactly, so that the numbers are known to come from the same work.

What it cannot see: tracemalloc counts memory allocated through Python and numpy. Memory allocated inside other
native code (scipy's cKDTree in the low-point filter, laspy and lazrs reading and writing files, scratch space inside
numpy sorts) is not counted. The test covers the engine, not the whole-process peak.

Reference: tests/fixtures/memory_reference.json. It records the code version, the library versions and platform it was
made with (byte counts of numpy arrays should not depend on the platform; a different numpy version only gives a
warning), the hashes of the clips, and a hash of the default settings: a change of either makes the test fail,
because the numbers could not then be compared.

Updating the reference, after a deliberate change that alters memory use:
    1. Check the change is intended and that make test and make regression still pass.
    2. python tests/test_memory.py --write-reference --note "what changed and why"
       (it measures twice and refuses to write if the two measurements disagree by more than 0.5%)
    3. Commit memory_reference.json.

Making a reference from another version of the code (e.g. an earlier commit, before a refactor): the engine is
imported, so point PYTHONPATH at that version's src folder, with its core built:
    mkdir /tmp/ref && git archive <commit> | tar -x -C /tmp/ref && (cd /tmp/ref && sh native/build.sh)
    PYTHONPATH=/tmp/ref/src python tests/test_memory.py --write-reference --note "commit <commit>"
That version must have the stage methods named above, otherwise the stage peaks cannot be taken.
"""
import argparse
import gc
import hashlib
import json
import platform
import subprocess
import sys
import tracemalloc
from pathlib import Path

import numpy as np

from ground_classifier import fast_ground as fg

ROOT = Path(__file__).resolve().parents[1]
CLIPS = ROOT / "tests" / "validation" / "clips"
REFERENCE = ROOT / "tests" / "fixtures" / "memory_reference.json"
INPUT_CLASSES = (2, 3, 4, 5, 13)            # the classes the pipeline hands to the engine, as in test_fast_engine.py
GROWTH_CLIP = "fpl_steep_clip"
GROWTH_CAPACITY = 1024
STAGES = ("_filter_low_points", "_choose_seeds", "_build_state", "_grow", "_prune_spikes")
STAGE_LABEL = {"init": "constructor", "_filter_low_points": "low points", "_choose_seeds": "starting points",
               "_build_state": "core state", "_grow": "grow", "_prune_spikes": "spikes"}
RESULTS = ("ground", "heights", "model", "low")
TOLERANCE = 0.02            # a number may be this much above the reference
IMPROVED = 0.05             # ... and is worth a note when it is this much below
SLACK = 16 * 1024           # bytes: small differences in Python's own bookkeeping
REPEATABLE = 0.005          # --write-reference: two measurements must agree this closely
RELEASED_LIMIT = 64 * 1024  # bytes that may remain after the run is released

fails = 0


def ok(name, cond, note=""):
    global fails
    print(("ok   " if cond else "FAIL ") + name + (" - " + note if note else ""), flush=True)
    fails += 0 if cond else 1


def mib(n):
    return f"{n / 2**20:.1f} MiB"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def defaults_fingerprint():
    return hashlib.sha256(json.dumps(fg.Settings().flat(), sort_keys=True).encode()).hexdigest()


def current():
    return tracemalloc.get_traced_memory()[0]


def peak_since(before):
    return tracemalloc.get_traced_memory()[1] - before


def whole_run(x, y, z, initial_capacity):
    """Peak and retained memory of GroundRun(x, y, z).run(), the peak of each output, and what is left on release."""
    gc.collect()
    base = current()
    tracemalloc.reset_peak()
    run = fg.GroundRun(x, y, z)
    if initial_capacity:
        run.initial_capacity = initial_capacity
    run.run()
    out = {"run_peak": peak_since(base), "held": current() - base}
    for name in RESULTS:
        before = current()
        tracemalloc.reset_peak()
        result = getattr(run, name)()
        out["result_" + name] = peak_since(before)
        if name == "ground":
            out["n_ground"] = int(result.sum())
        del result
    del run
    gc.collect()
    out["released"] = current() - base
    return out


def stage_peaks(x, y, z, initial_capacity):
    """Extra peak of the constructor and of each stage of run(), by wrapping the stage methods."""
    peaks = {}
    originals = {name: getattr(fg.GroundRun, name) for name in STAGES}

    def hook(name, original):
        def wrapper(self, *args, **kwargs):
            before = current()
            tracemalloc.reset_peak()
            try:
                return original(self, *args, **kwargs)
            finally:
                peaks[name] = max(peaks.get(name, 0), peak_since(before))
        return wrapper

    for name, original in originals.items():
        setattr(fg.GroundRun, name, hook(name, original))
    try:
        gc.collect()
        before = current()
        tracemalloc.reset_peak()
        run = fg.GroundRun(x, y, z)
        peaks["init"] = peak_since(before)
        if initial_capacity:
            run.initial_capacity = initial_capacity
        run.run()
    finally:
        for name, original in originals.items():
            setattr(fg.GroundRun, name, original)
    del run
    return peaks


def load_clip(stem):
    d = np.load(CLIPS / f"{stem}.npz")
    m = np.isin(d["c_input"], INPUT_CLASSES)
    return d["x"][m], d["y"][m], d["z"][m]


def measure_case(stem, initial_capacity):
    x, y, z = load_clip(stem)
    out = {"n": int(x.size)}
    out.update(whole_run(x, y, z, initial_capacity))
    out["stage"] = stage_peaks(x, y, z, initial_capacity)
    return out


def cases():
    """Case name -> (clip, initial_capacity)."""
    out = {p.stem: (p.stem, None) for p in sorted(CLIPS.glob("*.npz"))}
    out[GROWTH_CLIP + "+growth"] = (GROWTH_CLIP, GROWTH_CAPACITY)
    return out


def measure_all():
    # lazy imports and the library load are not the engine's memory: do them before counting
    rng = np.random.default_rng(3)
    fg.GroundRun(rng.uniform(0, 80, 3000), rng.uniform(0, 80, 3000), rng.normal(0, 0.02, 3000)).run().ground()
    tracemalloc.start(1)
    try:
        return {name: measure_case(stem, cap) for name, (stem, cap) in cases().items()}
    finally:
        tracemalloc.stop()


def numbers(case):
    """The numbers of one case that are compared with a tolerance, by name."""
    out = {k: v for k, v in case.items() if k not in ("n", "n_ground", "stage", "released")}
    out.update({"stage." + k: v for k, v in case["stage"].items()})
    return out


def git_commit():
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        return "unknown"
    return (rev or "unknown") + (" + uncommitted changes" if dirty else "")


def write_reference(note):
    first = measure_all()
    second = measure_all()
    reference = {}
    for name in first:
        for key in ("n", "n_ground"):
            if first[name][key] != second[name][key]:
                print(f"FAIL {name}: {key} differs between two runs ({first[name][key]} / {second[name][key]})")
                return 1
        merged = {"n": first[name]["n"], "n_ground": first[name]["n_ground"], "released": max(first[name]["released"], second[name]["released"])}
        for key, a in numbers(first[name]).items():
            b = numbers(second[name])[key]
            if abs(a - b) > REPEATABLE * max(a, b) + SLACK:
                print(f"FAIL {name}: {key} is not repeatable ({a:,} / {b:,} bytes); not writing a reference")
                return 1
        stage_keys = first[name]["stage"].keys()
        merged.update({k: max(first[name][k], second[name][k]) for k in first[name] if k not in ("n", "n_ground", "stage", "released")})
        merged["stage"] = {k: max(first[name]["stage"][k], second[name]["stage"][k]) for k in stage_keys}
        reference[name] = merged
    import scipy
    meta = {
        "note": note, "commit": git_commit(),
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "platform": platform.platform(),
        "defaults_sha256": defaults_fingerprint(),
        "clips_sha256": {p.name: sha256(p) for p in sorted(CLIPS.glob("*.npz"))},
    }
    REFERENCE.parent.mkdir(parents=True, exist_ok=True)
    REFERENCE.write_text(json.dumps({"meta": meta, "cases": reference}, indent=1, sort_keys=True) + "\n")
    print("wrote", REFERENCE)
    print(json.dumps(meta, indent=1))
    for name, case in reference.items():
        print(f"{name}: peak {mib(case['run_peak'])}, held {mib(case['held'])}, {case['n']:,} points")
    return 0


def within(value, ref):
    return value <= ref * (1 + TOLERANCE) + SLACK


def check_case(name, got, ref):
    n = got["n"]
    same = got["n"] == ref["n"] and got["n_ground"] == ref["n_ground"]
    ok(f"{name}: same work as the reference ({n:,} points, {got['n_ground']:,} ground)", same,
       "" if same else f"reference has {ref['n']:,} points, {ref['n_ground']:,} ground")
    notes = []

    def line(label, keys, got_values, ref_values):
        bad = [k for k in keys if not within(got_values[k], ref_values[k])]
        low = [k for k in keys if got_values[k] < ref_values[k] * (1 - IMPROVED) - SLACK]
        detail = "; ".join(f"{k} {got_values[k]:,} bytes, reference {ref_values[k]:,}" for k in bad)
        ok(f"{name}: {label}", not bad, detail)
        if low and not bad:
            notes.append(f"{name}: {label} is more than {IMPROVED:.0%} below the reference ({', '.join(low)}): update the reference to keep the improvement")

    line(f"peak during run ({got['run_peak'] / n:.0f} B/point)", ["run_peak"], got, ref)
    line(f"held after run ({got['held'] / n:.0f} B/point)", ["held"], got, ref)
    line("peak of each output array", ["result_" + r for r in RESULTS], got, ref)
    stages = [s for s in STAGE_LABEL if s in ref["stage"]]
    line("peak of each stage (" + ", ".join(f"{STAGE_LABEL[s]} {got['stage'][s] / 2**20:.1f}" for s in stages) + " MiB)",
         stages, got["stage"], ref["stage"])
    ok(f"{name}: memory is returned when the run is released ({got['released']:,} bytes left)", got["released"] <= RELEASED_LIMIT)
    for note in notes:
        print("note " + note)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--write-reference", action="store_true", help="record the current numbers as the reference")
    parser.add_argument("--note", default="", help="with --write-reference: which version of the code made it")
    args = parser.parse_args()
    if args.write_reference and not args.note:
        parser.error("--write-reference needs --note saying which version of the code is being recorded")

    missing = [n for n in STAGES if not hasattr(fg.GroundRun, n)]
    ok("GroundRun has the stage methods the test wraps", not missing, f"missing: {missing}" if missing else "")
    if missing:
        return 1
    if args.write_reference:
        return write_reference(args.note)

    data = json.loads(REFERENCE.read_text())
    meta, ref = data["meta"], data["cases"]
    print(f"reference: {meta['note']}; commit {meta['commit']}; made on {meta['platform']}", flush=True)
    same = meta["defaults_sha256"] == defaults_fingerprint()
    ok("default settings unchanged since the reference was made", same,
       "" if same else "fast_ground.Settings() defaults differ: rewrite the reference if that is intended")
    ok("clips unchanged since the reference was made",
       meta["clips_sha256"] == {p.name: sha256(p) for p in sorted(CLIPS.glob("*.npz"))})
    import scipy
    for lib, now in (("python", platform.python_version()), ("numpy", np.__version__), ("scipy", scipy.__version__)):
        if meta[lib].split(".")[:2] != now.split(".")[:2]:
            print(f"note {lib} {now} differs from the reference's {meta[lib]}: small differences are possible")

    got = measure_all()
    ok("every case of the reference was measured", sorted(got) == sorted(ref),
       "" if sorted(got) == sorted(ref) else f"measured {sorted(got)}, reference {sorted(ref)}")
    for name in sorted(set(got) & set(ref)):
        check_case(name, got[name], ref[name])

    print("ALL OK" if fails == 0 else f"{fails} FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
