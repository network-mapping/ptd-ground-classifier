from pathlib import Path
import numpy as np


US_FEET = 'survey_foot'
IMPERIAL_FEET = 'foot'
METRES = 'm'
US_FEET_TO_METRES = (1200 / 3937)
IMPERIAL_FEET_TO_METRES = 0.3048

LAS = 'las'
LAZ = 'laz'

def get_conversion(units):

    if units == US_FEET:
        return US_FEET_TO_METRES
    elif units == IMPERIAL_FEET:
        return IMPERIAL_FEET_TO_METRES
    elif units == METRES:
        return 1
    else:
        raise ValueError(f'Unknown units: {units}')

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
