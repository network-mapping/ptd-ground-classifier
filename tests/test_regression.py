"""Regression test on a real tile: runs the ground-classify CLI end to end and compares with a recorded result.

Purpose: tell whether a change that should not alter results (refactoring, renaming, speed) has altered them. It
records that the output has not changed, not that it is right; accuracy is measured by validation/validate_suite.py.

Usage (project venv active, core built; `make regression` builds first):
    python tests/test_regression.py                  run the test; ends with ALL OK or N FAILED, exit code 1 on failure
    python tests/test_regression.py --write-reference --note "..."     record the current result as the reference

Fixture (tests/fixtures/fpl_feet, US survey feet):
    219000_1146000.laz  the whole tile, copied unchanged from data/FPL_feet/input_from_FLAI
    219000_1144500.laz  only the points of the neighbouring tile within 40 ft of the shared edge (y >= 1145960),
                        so the neighbour-tile context is exercised without a 36 MB file
    expected.npz        the output classification of both tiles, and a `meta` record of where it came from
                        (note, versions, platform, hashes of the config and of the fixture tiles)

What is checked, per tile: the run completes; same point count; X, Y, Z and every other dimension carried through;
generating_software set; classification identical to the reference (on a mismatch: how many points differ and between
which classes); manifest.json agrees with the output. The input, hag and engine sections of the config file and the fixture
tiles must also be the ones the reference was made with, otherwise a difference could not be told from a code change.

Updating the reference, after a deliberate change of behaviour:
    1. Check the change is intended (make validate shows what it does to accuracy).
    2. python tests/test_regression.py --write-reference --note "what changed and why"
    3. Commit expected.npz.
Never edit the fixture or the config to make the test pass; the config hash check exists to catch that.

Making a reference from another version of the code (e.g. an earlier commit, before a refactor): the CLI runs as a
subprocess, so point PYTHONPATH at that version's src folder, with its core built:
    mkdir /tmp/ref && git archive <commit> | tar -x -C /tmp/ref && (cd /tmp/ref && sh native/build.sh)
    PYTHONPATH=/tmp/ref/src python tests/test_regression.py --write-reference --note "commit <commit>"
The configuration used is always <repo>/config/ground_classifier_config.yml, whichever version of the code runs. Code from
before the single config file (it reads config/*.json and takes --config-dir) cannot be pointed at it.

The current reference was made from commit f6e7655 on macOS arm64.
"""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import laspy
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "fpl_feet"
REFERENCE = FIXTURE / "expected.npz"
CONFIG_FILE = ROOT / "config" / "ground_classifier_config.yml"
CONFIG_SECTIONS = ("input", "hag", "engine")   # the ones that change results
OUTPUT_SUFFIX = "_AdaptiveGround"

fails = 0


def ok(name, cond, note=""):
    global fails
    print(("ok   " if cond else "FAIL ") + name + (" - " + note if note else ""), flush=True)
    fails += 0 if cond else 1


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def config_fingerprint():
    # Only the sections that change results; comments, performance and dtm settings do not.
    sections = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    chosen = {name: sections.get(name) for name in CONFIG_SECTIONS}
    return hashlib.sha256(json.dumps(chosen, sort_keys=True).encode()).hexdigest()


def run_cli(output_dir):
    cmd = [sys.executable, "-m", "ground_classifier", str(FIXTURE), "--output-dir", str(output_dir),
           "--config", str(CONFIG_FILE), "--units", "usft", "--workers", "1", "--overwrite"]
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)


def describe_change(old, new):
    """Counts of points that moved between classes, largest first."""
    changed = old != new
    pairs, counts = np.unique(np.column_stack((old[changed], new[changed])), axis=0, return_counts=True)
    rows = sorted(zip(counts, pairs.tolist()), reverse=True)[:6]
    return "; ".join(f"{a}->{b}: {n:,}" for n, (a, b) in rows)


def write_reference(output_dir, note):
    arrays = {}
    for path in sorted(FIXTURE.glob("*.laz")):
        arrays[f"{path.stem}__classification"] = np.asarray(laspy.read(output_dir / f"{path.stem}{OUTPUT_SUFFIX}.laz").classification)
    versions = subprocess.run([sys.executable, "-m", "ground_classifier", "--version"], cwd=ROOT,
                              capture_output=True, text=True).stdout.strip()
    import scipy
    meta = {
        "note": note,
        "software": versions,
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "laspy": laspy.__version__,
        "platform": platform.platform(),
        "config_sha256": config_fingerprint(),
        "inputs_sha256": {p.name: sha256(p) for p in sorted(FIXTURE.glob("*.laz"))},
    }
    np.savez_compressed(REFERENCE, meta=np.array(json.dumps(meta)), **arrays)
    print("wrote", REFERENCE)
    print(json.dumps(meta, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--write-reference", action="store_true", help="record the current result as the reference")
    parser.add_argument("--note", default="", help="with --write-reference: which version of the code made it")
    args = parser.parse_args()
    if args.write_reference and not args.note:
        parser.error("--write-reference needs --note saying which version of the code is being recorded")

    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp)
        done = run_cli(output_dir)
        ok("ground-classify runs to completion", done.returncode == 0, f"exit code {done.returncode}")
        if done.returncode != 0:
            print(done.stdout[-2000:], done.stderr[-2000:])
            return 1
        if args.write_reference:
            write_reference(output_dir, args.note)
            return 0

        ref = np.load(REFERENCE)
        meta = json.loads(str(ref["meta"]))
        print(f"reference: {meta['software']}; {meta['note']}; made on {meta['platform']}", flush=True)
        ok("configuration unchanged since the reference was made", meta["config_sha256"] == config_fingerprint(),
           "input / hag / engine config differ: rewrite the reference if that is intended")
        ok("fixture tiles unchanged since the reference was made",
           meta["inputs_sha256"] == {p.name: sha256(p) for p in sorted(FIXTURE.glob("*.laz"))})

        manifest = json.loads((output_dir / "manifest.json").read_text())
        ok("manifest: run completed, every file processed", manifest["run"]["status"] == "completed"
           and all(f["status"] == "processed" for f in manifest["files"]))
        recorded = {Path(f["input"]).stem: f for f in manifest["files"]}

        for path in sorted(FIXTURE.glob("*.laz")):
            stem = path.stem
            source = laspy.read(path)
            out = laspy.read(output_dir / f"{stem}{OUTPUT_SUFFIX}.laz")
            ok(f"{stem}: same number of points", len(out.points) == len(source.points), f"{len(out.points):,}")
            ok(f"{stem}: coordinates never rewritten",
               all(np.array_equal(getattr(out, d), getattr(source, d)) for d in ("X", "Y", "Z")))
            kept = [d for d in out.point_format.dimension_names if d != "classification"]
            changed = [d for d in kept if not np.array_equal(out[d], source[d])]
            ok(f"{stem}: every other dimension carried through", not changed, f"changed: {changed}" if changed else "")
            ok(f"{stem}: written by this tool", str(out.header.generating_software).startswith("ground-classifier"),
               str(out.header.generating_software))
            expected = ref[f"{stem}__classification"]
            got = np.asarray(out.classification)
            different = int(np.count_nonzero(expected != got))
            ok(f"{stem}: classification identical to the reference", different == 0,
               f"{different:,} of {got.size:,} points differ ({describe_change(expected, got)})" if different
               else f"{got.size:,} points")
            counts = {str(c): int(n) for c, n in zip(*np.unique(got, return_counts=True))}
            ok(f"{stem}: manifest class counts match the output",
               all(counts.get(code, 0) == n for code, n in recorded[stem]["class_counts"].items()))

    print("ALL OK" if fails == 0 else f"{fails} FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
