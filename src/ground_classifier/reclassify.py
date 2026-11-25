"""Reclassification: how a block of points gets its class codes.

Input classes that the config's feature_codes section maps to a feature (by default 2, 3, 4, 5 and 13) are reset
and re-labelled; every other class is kept. HAG = height above the finished ground surface, in metres. The output
codes below are the defaults:
    2  Ground
    3  Low vegetation:    HAG <= low_veg_max
    4  Medium vegetation: low_veg_max < HAG < medium_veg_max
    5  High vegetation:   medium_veg_max <= HAG <= high_veg_max
    7  Noise: isolated low points, and HAG > high_veg_max

Everything here takes what it needs as arguments (the engine settings, HAG bands and feature codes come in a
`RunSettings`), and none of it reads module state, so it behaves the same in the main process, in a spawned worker
and in a test. Coordinates are in metres.
"""
from __future__ import annotations

import numpy as np

from . import fast_ground
from .feature_codes import FeatureCodes
from .utils.loggers import log
from .utils.params import RunSettings

# A point the low-point filter kept out of the surface is labelled noise
# only if it ends up at least this far below the finished ground (metres).
LOW_NOISE_DEPTH = 0.5


class InsufficientGroundSupport(RuntimeError):
    """The available geometry cannot support a progressive ground TIN."""


def application_code(feature_codes: FeatureCodes, name: str) -> int:
    """The class code written for the feature `name` (e.g. "ground")."""
    return feature_codes("application")[name].code


def reset_input_class_mask(classification: np.ndarray, feature_codes: FeatureCodes) -> np.ndarray:
    """Return the mask of points in a re-labelled input class, without np.isin's setup overhead."""
    mask = np.zeros(classification.shape, dtype=bool)
    for code in feature_codes.mapped_input_feature_codes:
        mask |= classification == code
    return mask


def assign_hag_classes(
    classification: np.ndarray,
    remaining: np.ndarray,
    hag: np.ndarray,
    hag_bands: tuple[float, float, float],
    feature_codes: FeatureCodes,
) -> None:
    """Assign the three vegetation classes from final HAG, using the HAG
    bands (low, medium, high; RunSettings.hag_bands). HAG is always in
    metres regardless of the input file's own coordinate unit."""
    low_veg_max, medium_veg_max, high_veg_max = hag_bands
    low_code, medium_code, high_code = (
        application_code(feature_codes, name)
        for name in ("low_vegetation", "medium_vegetation", "high_vegetation")
    )
    classification[remaining & (hag <= low_veg_max)] = low_code
    classification[
        remaining
        & (hag > low_veg_max)
        & (hag < medium_veg_max)
    ] = medium_code
    classification[
        remaining
        & (hag >= medium_veg_max)
        & (hag <= high_veg_max)
    ] = high_code


def classify_block(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    initial_classification: np.ndarray,
    settings: RunSettings,
    tin_out: list[np.ndarray] | None = None,
    tin_max_edge: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classify one block with the compiled core.

    The re-labelled input classes (settings.feature_codes.mapped_input_feature_codes)
    are reset and labelled ground / low, medium, high vegetation / noise,
    every other class is left alone, and the result is (classification,
    height above ground, ground point indices). Uses the engine settings,
    HAG bands and feature codes of `settings`.

    If tin_out is a list, the block's TIN is appended to it as an (m, 3)
    array of indices into x, y, z (see GroundRun.tin(); tin_max_edge is
    its max_edge, in metres).

    Raises InsufficientGroundSupport when there are too few points for a TIN.
    """
    feature_codes = settings.feature_codes
    high_veg_max = settings.hag_bands[2]
    classification = initial_classification.copy()
    unclassified_code = application_code(feature_codes, "unclassified")
    source = np.flatnonzero(reset_input_class_mask(classification, feature_codes))
    if source.size < 3:
        raise InsufficientGroundSupport(
            "Fewer than three eligible source-class candidates."
        )
    try:
        run = fast_ground.GroundRun(
            x[source], y[source], z[source], settings.engine
        ).run()
    except ValueError as exc:
        raise InsufficientGroundSupport(str(exc)) from exc
    ground = run.ground()
    height = run.heights()
    low = run.low()

    labels = np.full(source.size, unclassified_code, dtype=classification.dtype)
    labels[ground] = application_code(feature_codes, "ground")
    open_points = ~ground
    noise_code = application_code(feature_codes, "noise")
    labels[open_points & (height > high_veg_max)] = noise_code
    low_noise = open_points & low & (height < -LOW_NOISE_DEPTH)
    labels[low_noise] = noise_code
    assign_hag_classes(
        labels, labels == unclassified_code, height, settings.hag_bands, feature_codes
    )
    classification[source] = labels

    hag = np.full(x.size, np.nan, dtype=np.float32)
    hag[source] = height
    model = run.model()
    if tin_out is not None:
        tin_out.append(source[run.tin(tin_max_edge)])
    surface_count = int(np.count_nonzero(model))
    ground_count = int(np.count_nonzero(ground))
    log(
        f"      Fast engine: {ground_count:,} ground points of "
        f"{source.size:,} candidates ({surface_count:,} on the surface, "
        f"{ground_count - surface_count:,} within "
        f"{100 * float(run.settings.ground_band.width):g} cm of it); surface "
        f"followed past its edge +{run.followed:,}, corners turned "
        f"+{run.turned:,}, spikes taken out {run.spikes_removed:,}; "
        f"isolated low points kept out: {int(np.count_nonzero(low)):,} "
        f"({int(np.count_nonzero(low_noise)):,} labelled noise)"
    )
    if run.seed_square < float(settings.engine.seeds.max_building_size):
        log(
            "      Small block: starting squares reduced to "
            f"{run.seed_square:g} m so the surface has at least three "
            "starting points."
        )
    return classification, hag, source[model]
