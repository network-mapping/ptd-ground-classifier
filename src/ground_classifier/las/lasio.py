import os
import tempfile
import uuid
import numpy as np
from pathlib import Path
from contextlib import contextmanager

import laspy

from .lasheader import LasHeader

class NoLazReadError(ValueError):
    pass


def read_tile(path):
    """Read a whole LAS/LAZ file: its header and every point record (a ``laspy.LasData``)."""
    return laspy.read(path)


def drop_extra_dimensions(tile, names):
    """Remove the named ExtraBytes dimensions from ``tile`` (a ``laspy.LasData``).

    Names match ignoring case, but the exact stored name is passed to laspy, whose
    ``remove_extra_dim`` updates both the point-record layout and the ExtraBytes VLR.
    Returns the names that were removed.
    """
    wanted = {str(name).casefold() for name in names}
    removed = []
    for name in [str(name) for name in tile.point_format.extra_dimension_names]:
        if name.casefold() in wanted:
            tile.remove_extra_dim(name)
            removed.append(name)
    return tuple(removed)


def write_tile(path, tile, *, generating_software=None, overwrite=True):
    """Write ``tile`` (a ``laspy.LasData``) to ``path`` as a whole, replacing any file there.

    The file is written beside the target under a temporary name and moved into place, so a
    failure never leaves a half-written ``path``. With ``overwrite=False`` an existing ``path``
    is an error. ``generating_software``, when given, is set in the header first.
    """
    path = Path(path)
    if not overwrite and (path.exists() or path.is_symlink()):
        raise FileExistsError(f'Refusing to overwrite {path}.')
    path.parent.mkdir(parents=True, exist_ok=True)
    if generating_software is not None:
        tile.header.generating_software = generating_software
    # Not mkstemp: that would create the file with mode 0600, and the output should get the usual permissions.
    temporary = path.with_name(f'.partial-{uuid.uuid4().hex}{path.suffix}')
    try:
        tile.write(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def create_temporary(path):
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f'Refusing to overwrite {path}.')
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix='.partial-', suffix=path.suffix, dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        yield temporary
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def iter_record_batches(path, *, batch_size=250_000):
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError('batch_size must be a positive integer.')
    with laspy.open(path) as reader:
        yield from reader.chunk_iterator(batch_size)


def write_record_batches(path, header, batches, *, compression=True, batch_size=250_000, overwrite=False):
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError('batch_size must be a positive integer.')
    path = Path(path)
    if not overwrite and (path.exists() or path.is_symlink()):
        raise FileExistsError(f'Refusing to overwrite {path}.')
    path.parent.mkdir(parents=True, exist_ok=True)
    native = LasHeader(header).native_copy()
    iterator = iter(batches)
    descriptor, temporary_name = tempfile.mkstemp(prefix='.partial-', suffix=path.suffix, dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    count = 0
    try:
        with laspy.open(temporary, mode='w', header=native, do_compress=compression) as writer:
            for records in iterator:
                if not isinstance(records, laspy.ScaleAwarePointRecord):
                    raise ValueError('Native LAS batches must be scale-aware point records.')
                if len(records) > batch_size:
                    raise ValueError('LAS batch exceeds the configured batch_size.')
                if (records.point_format != native.point_format
                        or not np.array_equal(records.scales, native.scales)
                        or not np.array_equal(records.offsets, native.offsets)):
                    raise ValueError('LAS batch schema or coordinate encoding differs from the output header.')
                writer.write_points(records)
                count += len(records)
            if native.evlrs:
                writer.write_evlrs(native.evlrs)
        (os.replace if overwrite else os.link)(temporary, path)
    finally:
        try:
            if hasattr(iterator, 'close'):
                iterator.close()
        finally:
            temporary.unlink(missing_ok=True)
    return count
