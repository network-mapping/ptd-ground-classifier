from pathlib import Path
import numpy as np


US_FEET = 'survey_foot'
IMPERIAL_FEET = 'foot'
METRES = 'm'
US_FEET_TO_METRES = (1200 / 3937)
IMPERIAL_FEET_TO_METRES = 0.3048

LAS = 'las'
LAZ = 'laz'

# What a user may ask for: 'auto' reads the CRS of each file; the others state the unit of the coordinates.
UNITS_AUTO = 'auto'
UNITS_CHOICES = (UNITS_AUTO, 'usft', 'metres', 'foot')
_UNIT_OPTIONS = {
    'metres': METRES, 'meters': METRES, 'metre': METRES, 'meter': METRES, 'm': METRES,
    'usft': US_FEET, 'survey_foot': US_FEET,
    'foot': IMPERIAL_FEET, 'ift': IMPERIAL_FEET,
}
UNIT_OPTION_NAMES = {METRES: 'metres', US_FEET: 'usft', IMPERIAL_FEET: 'foot'}
UNIT_LABELS = {METRES: 'metres', US_FEET: 'US survey feet', IMPERIAL_FEET: 'international feet'}


def parse_units_option(value) -> str | None:
    """The unit ('m', 'survey_foot' or 'foot') a units option names, or None for 'auto'.

    Raises:
        ValueError: ``value`` is none of ``UNITS_CHOICES`` (or an alias such as 'meters').
    """
    text = str(value).strip().lower()
    if text == UNITS_AUTO:
        return None
    if text not in _UNIT_OPTIONS:
        raise ValueError(f"units must be one of {', '.join(UNITS_CHOICES)} (got {value!r})")
    return _UNIT_OPTIONS[text]


def get_conversion(units):

    if units == US_FEET:
        return US_FEET_TO_METRES
    elif units == IMPERIAL_FEET:
        return IMPERIAL_FEET_TO_METRES
    elif units == METRES:
        return 1
    else:
        raise ValueError(f'Unknown units: {units}')


def unit_for_scale(metres_per_unit: float) -> str:
    """The unit ('m', 'survey_foot' or 'foot') whose metres-per-unit is ``metres_per_unit``.

    Any other scale is taken to be US survey feet, as it always was before international feet were supported.
    """
    if abs(metres_per_unit - 1.0) < 1.0e-9:
        return METRES
    if abs(metres_per_unit - IMPERIAL_FEET_TO_METRES) < 1.0e-9:
        return IMPERIAL_FEET
    return US_FEET

def metres_to_dataset_units(length: float, units: str) -> float:
    
    return float(length) / get_conversion(units)


def convert_to_metres(x, units):

    return x * get_conversion(units)


def ppm(inter_point_spacing):
    return 1 / (np.pi * inter_point_spacing ** 2 / 4)


def lasfiles_in_dir(
        directory: str | Path,
        *,
        recursive: bool = False,
) -> list[Path]:

    if recursive:
        return sorted(Path(directory).rglob('*.la[sz]'))
    else:
        return sorted(Path(directory).glob('*.la[sz]'))


def expand_inputs(paths) -> list[Path]:
    """Resolve files and folders of LAS/LAZ files to a sorted list of distinct absolute paths.

    A folder contributes its ``*.las`` and ``*.laz`` files (not recursive).

    Raises:
        FileNotFoundError: a path does not exist, or nothing was found.
        ValueError: a file that is not LAS or LAZ.
    """
    found: list[Path] = []
    for path in paths:
        path = Path(path).resolve()
        if path.is_dir():
            found.extend(file.resolve() for file in lasfiles_in_dir(path))
        elif not path.exists():
            raise FileNotFoundError(f'Input does not exist: {path}')
        elif path.suffix.lower() not in {f'.{LAS}', f'.{LAZ}'}:
            raise ValueError(f'Input must be LAS or LAZ: {path}')
        else:
            found.append(path)
    unique = sorted(set(found), key=str)
    if not unique:
        raise FileNotFoundError('No LAS/LAZ input found.')
    return unique
