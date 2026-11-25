import os
import tempfile
import numpy as np
from pathlib import Path
from contextlib import contextmanager

import laspy

from .lasheader import LasHeader

class NoLazReadError(ValueError):
    pass


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


def write_record_batches(path, header, batches, *, compression=True, batch_size=250_000):
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError('batch_size must be a positive integer.')
    path = Path(path)
    if path.exists() or path.is_symlink():
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
        os.link(temporary, path)
    finally:
        try:
            if hasattr(iterator, 'close'):
                iterator.close()
        finally:
            temporary.unlink(missing_ok=True)
    return count
