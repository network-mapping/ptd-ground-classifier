"""Reading and writing LAS/LAZ for the classifier: header and CRS/unit handling, sets of tiles, whole-tile I/O.

Nothing else on the classify path imports laspy. CRS parsing needs pyproj, but only when a CRS is asked for.

``LasHeader`` and ``LasDataset`` describe files and sets of files (CRS, units, bounds); ``lasio`` reads and writes
points; ``lasutils`` holds the unit conversions and input expansion. See the module docstrings for examples.
"""
from .lasdataset import LasDataset, LasDatasetError, LasFile, LasUnitsError
from .lasheader import LasHeader, LasHeaderError
from .lasio import drop_extra_dimensions, iter_record_batches, read_tile, write_record_batches, write_tile
from .lasutils import (
    IMPERIAL_FEET,
    METRES,
    UNIT_LABELS,
    UNIT_OPTION_NAMES,
    UNITS_AUTO,
    UNITS_CHOICES,
    US_FEET,
    US_FEET_TO_METRES,
    IMPERIAL_FEET_TO_METRES,
    expand_inputs,
    get_conversion,
    lasfiles_in_dir,
    parse_units_option,
    unit_for_scale,
)
