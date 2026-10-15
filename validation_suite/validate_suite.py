"""Standing multi-terrain validation suite.

Runs a given classifier module against ALL real test clips at once and
reports precision/recall for each against real TerraScan ground truth, so
a change that helps one terrain type can't quietly regress another without
it showing up here. This is the harness discipline the project needs:
every parameter change gets checked against every terrain type, every time.

Usage:
    python3 validate_suite.py <module_name>

<module_name> must be importable (its .py file's directory needs to be on
sys.path before this runs) and expose the same classify_block(x, y, z,
initial_classification, global_origin) signature as the production script.
"""
import sys
import time
from pathlib import Path
import numpy as np

CLIPS_DIR = Path(__file__).resolve().parent / "clips"
CLIPS = {
    "water_edge (honi)": CLIPS_DIR / "water_clip_buffered.npz",
    "rocktops (honi)": CLIPS_DIR / "rocktops_clip_buffered.npz",
    "fpl_moderate": CLIPS_DIR / "fpl_clip_buffered.npz",
    "fpl_steep": CLIPS_DIR / "fpl_steep_clip.npz",
}


def run_one(mod, path):
    data = np.load(path)
    x, y, z, c_input, c_terra, core_mask = (
        data["x"], data["y"], data["z"],
        data["c_input"], data["c_terra"], data["core_mask"],
    )
    global_origin = (float(x.min()), float(y.min()))
    t0 = time.perf_counter()
    classification, hag, model = mod.classify_block(x, y, z, c_input, global_origin)
    elapsed = time.perf_counter() - t0

    classification = classification[core_mask]
    c_terra_core = c_terra[core_mask]
    n = classification.size

    ours_ground = classification == 2
    terra_ground = c_terra_core == 2
    agree = ours_ground & terra_ground
    fp = ours_ground & ~terra_ground
    fn = ~ours_ground & terra_ground

    precision = agree.sum() / max(1, ours_ground.sum())
    recall = agree.sum() / max(1, terra_ground.sum())
    f1 = 2 * precision * recall / max(1e-9, precision + recall)

    return {
        "n": n,
        "elapsed": elapsed,
        "terra_ground_pct": 100 * terra_ground.sum() / n,
        "our_ground_pct": 100 * ours_ground.sum() / n,
        "precision": 100 * precision,
        "recall": 100 * recall,
        "f1": 100 * f1,
        "fp": int(fp.sum()),
        "fn": int(fn.sum()),
    }


def run_suite(mod, label=""):
    print(f"\n{'='*78}")
    print(f"VALIDATION SUITE{f' - {label}' if label else ''}")
    print(f"{'='*78}")
    print(f"{'terrain':<22}{'n':>9}{'time':>7}{'terra%':>8}{'ours%':>8}"
          f"{'prec%':>8}{'recall%':>8}{'F1':>7}{'FP':>8}{'FN':>8}")
    results = {}
    for name, path in CLIPS.items():
        r = run_one(mod, path)
        results[name] = r
        print(f"{name:<22}{r['n']:>9,}{r['elapsed']:>6.1f}s{r['terra_ground_pct']:>7.1f}%"
              f"{r['our_ground_pct']:>7.1f}%{r['precision']:>7.1f}%{r['recall']:>7.1f}%"
              f"{r['f1']:>7.1f}{r['fp']:>8,}{r['fn']:>8,}")
    avg_f1 = sum(r["f1"] for r in results.values()) / len(results)
    min_f1 = min(r["f1"] for r in results.values())
    print(f"\nAverage F1: {avg_f1:.1f}   Worst-case F1: {min_f1:.1f}  "
          f"(worst case matters most - a fix that only helps the average "
          f"while tanking one terrain type is not a real fix)")
    return results


if __name__ == "__main__":
    import importlib
    # Default: the classifier one directory up (delivered layout is
    # adaptive_ground_classifier_unified.py next to a validation_suite/ folder).
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    mod_name = sys.argv[1] if len(sys.argv) > 1 else "adaptive_ground_classifier_unified"
    mod = importlib.import_module(mod_name)
    if hasattr(mod, "configure_runtime"):
        mod.configure_runtime(kdtree_workers=1)
    run_suite(mod, label=mod_name)
