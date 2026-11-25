"""Checks of the reclassification module: which input classes are re-labelled, the HAG bands, the output codes, the
engine, and that nothing in it reads module state. usage: python tests/test_reclassify.py"""
import contextlib
import dataclasses
import io
import subprocess
import sys

import numpy as np

from ground_classifier import pipeline, reclassify, window_runner
from ground_classifier.utils.params import FeatureCodesParam, RunSettings

fails = 0
def ok(name, cond, note=""):
    global fails
    print(("ok   " if cond else "FAIL ") + name + (" - " + note if note else ""), flush=True)
    fails += 0 if cond else 1


def quiet(call):
    """The result of `call`, and what it logged."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = call()
    return result, buffer.getvalue()


def raised(call):
    try:
        call()
    except Exception as exc:
        return exc
    return None


def custom_codes():
    """Other output codes, and input code 6 re-labelled as noise too."""
    return FeatureCodesParam(
        input_feature_codes={1: "Unclassified", 2: "Ground", 3: "Low", 4: "Medium", 5: "High", 6: "Building", 13: "Candidate"},
        application_feature_codes={
            "unclassified": {"code": 0, "input_feature_codes": [13]},
            "ground": {"code": 20, "input_feature_codes": [2]},
            "low_vegetation": {"code": 30, "input_feature_codes": [3]},
            "medium_vegetation": {"code": 40, "input_feature_codes": [4]},
            "high_vegetation": {"code": 50, "input_feature_codes": [5]},
            "noise": {"code": 70, "input_feature_codes": [6]},
        },
    ).feature_codes()


def scene(seed=3):
    """Flat ground with a 5 m high building on it (all input class 2), and a few points of class 1 well above.
    The noise (3 mm) is well inside the engine's 2 cm ground band, so the flat points are ground."""
    rng = np.random.default_rng(seed)
    n = 20_000
    x = rng.uniform(0, 100, n); y = rng.uniform(0, 100, n); z = rng.normal(0, 0.003, n)
    building = (x > 40) & (x < 46) & (y > 40) & (y < 46)
    z[building] = 5.0 + rng.normal(0, 0.003, building.sum())
    cls = np.full(n, 2, np.uint8)
    other = np.arange(200)                   # input class 1: not re-labelled
    cls[other] = 1; z[other] += 100.0
    return x, y, z, cls, building, other


def main():
    default = RunSettings()
    fc = default.feature_codes

    # 1. the codes
    ok("codes: the default output codes", [reclassify.application_code(fc, n) for n in ("unclassified", "ground", "low_vegetation", "medium_vegetation", "high_vegetation", "noise")] == [0, 2, 3, 4, 5, 7])
    other_fc = custom_codes()
    ok("codes: the configured ones", [reclassify.application_code(other_fc, n) for n in ("ground", "low_vegetation", "noise")] == [20, 30, 70])

    # 2. which points are re-labelled
    cls = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 13], np.uint8)
    mask = reclassify.reset_input_class_mask(cls, fc)
    ok("mask: the default input classes 2, 3, 4, 5 and 13, and no other", cls[mask].tolist() == [2, 3, 4, 5, 13])
    ok("mask: follows the feature codes: class 6 as well, and no other change", cls[reclassify.reset_input_class_mask(cls, other_fc)].tolist() == [2, 3, 4, 5, 6, 13])

    # 3. the vegetation bands
    hag = np.array([-1.0, 0.5, 0.6, 2.9, 3.0, 50.0, 50.1, 2.0])
    remaining = np.array([True] * 7 + [False])
    out = np.zeros(8, np.uint8)
    reclassify.assign_hag_classes(out, remaining, hag, (0.5, 3.0, 50.0), fc)
    ok("bands: low up to and including low_veg_max, medium between, high from medium_veg_max up to and including high_veg_max; above that and outside `remaining` untouched",
       out.tolist() == [3, 3, 4, 4, 5, 5, 0, 0], str(out.tolist()))
    out = np.zeros(8, np.uint8)
    reclassify.assign_hag_classes(out, remaining, hag, (0.2, 1.0, 10.0), other_fc)
    # idx 0 -0.9 -> low (<= 0.2); 0.5, 0.6 -> medium (0.2 < h < 1); 2.9, 3.0 -> high (1 <= h <= 10); 50.0 and 50.1 are above high_veg_max
    ok("bands: other bands and other codes", out.tolist() == [30, 40, 40, 50, 50, 0, 0, 0], str(out.tolist()))

    # 4. a block
    x, y, z, cls, building, other = scene()
    cls_before = cls.copy()
    (classification, hag, ground_model), text = quiet(lambda: reclassify.classify_block(x, y, z, cls, default))
    flat = ~building & (cls == 2)
    ok("block: the flat ground is class 2", (classification[flat] == 2).mean() > 0.99, f"{(classification[flat] == 2).mean():.4f}")
    ok("block: the 5 m building is high vegetation (class 5), its height above ground about 5 m",
       (classification[building] == 5).mean() > 0.95 and abs(float(np.nanmedian(hag[building])) - 5.0) < 0.2, f"{(classification[building] == 5).mean():.3f}, {np.nanmedian(hag[building]):.2f} m")
    ok("block: points of a class that is not re-labelled are left alone, without a height", (classification[other] == 1).all() and np.isnan(hag[other]).all())
    ok("block: the third result is indices of ground points", ground_model.dtype.kind in "iu" and (classification[ground_model] == 2).all() and len(ground_model) > 15_000)
    ok("block: the input classification is not changed", np.array_equal(cls, cls_before))
    ok("block: says what the engine did", "Fast engine:" in text and "ground points of" in text)

    # 5. everything comes from the arguments
    bands = dataclasses.replace(default, hag_bands=(0.5, 3.0, 4.0))
    (c_noise, _, _), _ = quiet(lambda: reclassify.classify_block(x, y, z, cls, bands))
    ok("arguments: a 5 m building above high_veg_max is noise (7)", (c_noise[building] == 7).mean() > 0.95)
    bands = dataclasses.replace(default, hag_bands=(0.5, 10.0, 20.0))
    (c_medium, _, _), _ = quiet(lambda: reclassify.classify_block(x, y, z, cls, bands))
    ok("arguments: ... and below medium_veg_max it is medium vegetation (4)", (c_medium[building] == 4).mean() > 0.95)
    (c_codes, _, _), _ = quiet(lambda: reclassify.classify_block(x, y, z, cls, dataclasses.replace(default, feature_codes=other_fc)))
    ok("arguments: other feature codes: other output codes (ground 20, high vegetation 50)", (c_codes[flat] == 20).mean() > 0.99 and (c_codes[building] == 50).mean() > 0.95)
    try:
        pipeline.configure_run(RunSettings(hag_bands=(9.0, 9.5, 10.0), feature_codes=other_fc, engine=dataclasses.replace(default.engine, ground_band=dataclasses.replace(default.engine.ground_band, width=0.5))))
        (again, _, _), _ = quiet(lambda: reclassify.classify_block(x, y, z, cls, default))
    finally:
        pipeline.configure_run(RunSettings())
    ok("arguments: the pipeline's own settings make no difference (nothing is read from module state)", np.array_equal(again, classification))
    tight = dataclasses.replace(default, engine=dataclasses.replace(default.engine, ground_band=dataclasses.replace(default.engine.ground_band, width=0.0)))
    (c_tight, _, g_tight), _ = quiet(lambda: reclassify.classify_block(x, y, z, cls, tight))
    ok("arguments: the engine settings are used (ground_band 0 keeps only surface points, so fewer ground points)", int((c_tight == 2).sum()) < int((classification == 2).sum()), f"{int((c_tight == 2).sum()):,} < {int((classification == 2).sum()):,}")

    # 6. the TIN
    faces = []
    quiet(lambda: reclassify.classify_block(x, y, z, cls, default, faces))
    ok("tin: appended to tin_out as indices into the block", len(faces) == 1 and faces[0].ndim == 2 and faces[0].shape[1] == 3 and faces[0].max() < len(x))
    short = []
    quiet(lambda: reclassify.classify_block(x, y, z, cls, default, short, 2.0))
    ok("tin: tin_max_edge leaves out long triangles", len(short) == 1 and len(short[0]) < len(faces[0]), f"{len(short[0]):,} < {len(faces[0]):,}")

    # 7. too little to work with
    few = np.array([2, 2, 1, 1, 1], np.uint8)
    pts = np.arange(5, dtype=float)
    err = raised(lambda: reclassify.classify_block(pts, pts * 0, pts * 0, few, default))
    ok("sparse: fewer than three eligible points is InsufficientGroundSupport", isinstance(err, reclassify.InsufficientGroundSupport) and "Fewer than three" in str(err), str(err))
    err = raised(lambda: reclassify.classify_block(pts, pts * 0, pts * 0, np.full(5, 1, np.uint8), default))
    ok("sparse: ... and none at all too", isinstance(err, reclassify.InsufficientGroundSupport))
    ok("sparse: it is a RuntimeError, as the window runner expects", issubclass(reclassify.InsufficientGroundSupport, RuntimeError) and window_runner.InsufficientGroundSupport is reclassify.InsufficientGroundSupport)

    # 8. independent of the pipeline
    code = "import sys; import ground_classifier.reclassify; print('ground_classifier.pipeline' in sys.modules)"
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    ok("module: importing reclassify does not import the pipeline", done.returncode == 0 and done.stdout.strip() == "False", done.stdout.strip() + done.stderr[-200:])

    print(f"\n{fails} failed" if fails else "\nall passed")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
