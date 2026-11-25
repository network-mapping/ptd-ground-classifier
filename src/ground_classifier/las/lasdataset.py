"""A validated, immutable collection of LAS/LAZ tiles, usable independently of this application.

``LasDataset`` reads only the headers of its source files when constructed. It
validates that the usable (non-empty) tiles share a CRS, linear units and a sane
coordinate encoding, then exposes aggregate metadata (point count, bounds, CRS,
EPSG, units) and bounded, ordered iteration over native laspy records. No
application feature mapping is required; ``feature_codes`` is optional and is
simply carried along for callers that need it.

Typical usage::

    from las.lasdataset import LasDataset, LasDatasetError

    # A single file, a directory of *.las / *.laz, a glob, or an explicit list.
    dataset = LasDataset('data/tiles')
    dataset = LasDataset('data/tiles/*.laz')
    dataset = LasDataset(['a.laz', 'b.laz'], file_extent_warning_threshold='500 m')

    print(dataset.point_count)      # total points across non-empty tiles
    print(dataset.bounds)           # ((xmin, ymin, zmin), (xmax, ymax, zmax)) or None
    print(dataset.epsg, dataset.units, dataset.srs)   # e.g. 28356, 'm', WKT string
    for lasfile in dataset:         # LasFile(path, header, signature)
        print(lasfile.path, lasfile.header.point_count)

Source order is deterministic: directory and glob matches are sorted, while an
explicit sequence keeps the caller's order. Empty tiles are logged and skipped
(``dataset.files`` / iteration / ``point_count`` ignore them), and an all-empty
dataset is valid but has no CRS, units or reference header (accessing those
raises ``LasDatasetError``).

Streaming points without loading a whole dataset into memory::

    for lasfile, records in dataset.iter_record_batches(batch_size=250_000):
        xyz = records.xyz            # laspy ScaleAwarePointRecord, at most batch_size rows

Merging compatible tiles into one file (see ``las.lasio.write_record_batches``)::

    from las.lasio import write_record_batches

    records = (batch for _, batch in dataset.iter_record_batches())
    write_record_batches('merged.laz', dataset.raw_header(), records)

Writing derived/processed points needs a header describing the new schema::

    header = dataset.derived_header(dimensions=[laspy.ExtraBytesParams('cluster_id', 'int64')])

Detecting changed inputs for long-running applications::

    dataset = dataset.refreshed()   # same object if unchanged, re-read if any file changed

Errors: ``FileNotFoundError`` when nothing matches or a path is not a file;
``LasDatasetError`` (a ``ValueError``) for duplicate paths, waveform point
formats, CRS/unit mismatches, unsupported or non-projected CRS, invalid
scales/offsets/bounds, or an invalid ``file_extent_warning_threshold``.
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from glob import glob, has_magic
from pathlib import Path
import logging

import numpy as np
from pint import Quantity

from .lasheader import LasHeader, LasHeaderError


logger = logging.getLogger(__name__)


class LasDatasetError(ValueError):
    """Raised when a set of LAS files is invalid or inconsistent as a dataset."""


@dataclass(frozen=True)
class LasFile:
    """Immutable descriptor of one LAS/LAZ file.

    Attributes:
        path: Location of the file.
        header: Header read once when the descriptor was created.
        signature: ``(size_bytes, mtime_ns)`` at read time, used by
            ``LasDataset.refreshed()`` to detect modification.

    Example::

        lasfile = LasFile.from_file('tile.laz')
        lasfile.header.point_count
    """

    path: Path
    header: LasHeader
    signature: tuple[int, int]

    @classmethod
    def from_file(cls, path: str | Path) -> 'LasFile':
        """Read the header of ``path`` and record its size/mtime signature."""
        path = Path(path)
        header = LasHeader.from_file(path)
        status = path.stat()
        return cls(path, header, (status.st_size, status.st_mtime_ns))


class LasDataset:
    """Ordered, validated set of LAS/LAZ files that share a CRS and units.

    Args:
        data: One of a file path, a directory (its ``*.las``/``*.laz`` files),
            a glob pattern, or a sequence of file paths.
        feature_codes: Optional application-specific mapping, stored unchanged
            as ``self.feature_codes``. Not used by this class.
        file_extent_warning_threshold: Pint-compatible length string or quantity
            (default ``'1000 m'``). A warning is logged for any file whose largest
            extent exceeds it, since untiled data can exhaust memory.

    Raises:
        FileNotFoundError: no matching files, or a listed path is not a file.
        LasDatasetError: see the module docstring.

    Example::

        dataset = LasDataset('data/tiles', file_extent_warning_threshold='2 km')
        for lasfile in dataset:
            print(lasfile.path.name, lasfile.header.point_count)
    """

    def __init__(self, data, feature_codes=None, *, file_extent_warning_threshold='1000 m'):
        self.data = data
        self.feature_codes = feature_codes
        self.file_extent_warning_threshold = Quantity(file_extent_warning_threshold).to('m')
        if (not np.isfinite(self.file_extent_warning_threshold.magnitude)
                or self.file_extent_warning_threshold.magnitude <= 0):
            raise LasDatasetError('file_extent_warning_threshold must be positive and finite.')
        if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
            paths = [Path(path) for path in data]
        elif has_magic(str(data)):
            paths = [Path(path) for path in sorted(glob(str(data)))]
        elif Path(data).is_dir():
            paths = sorted(Path(data).glob('*.la[sz]'))
        else:
            paths = [Path(data)]
        if not paths or any(not path.is_file() for path in paths):
            raise FileNotFoundError(f'No matching LAS/LAZ input for {data}.')
        if len({path.resolve() for path in paths}) != len(paths):
            raise LasDatasetError('A LAS dataset cannot contain duplicate source paths.')
        self._sources = tuple(LasFile.from_file(path) for path in paths)
        for source in self._sources:
            if not source.header.point_count:
                logger.warning('%s: source tile contains no points and will be skipped.', source.path)
        self._files = tuple(source for source in self._sources if source.header.point_count)
        first = next(iter(self._files), None)
        for source in self._files:
            if source.header.point_format_id in (4, 5, 9, 10):
                raise LasDatasetError(f'{source.path}: PDAL processing does not support waveform point formats.')
            try:
                source.header.units
            except LasHeaderError as error:
                raise LasDatasetError(f'{source.path}: {error}') from error
            if not first.header.crs.equals(source.header.crs) or first.header.units != source.header.units:
                raise LasDatasetError(f'LAS CRS/units mismatch between {first.path} and {source.path}.')
            if (not np.isfinite(source.header.scales).all() or np.any(np.asarray(source.header.scales) <= 0)
                    or not np.isfinite(source.header.offsets).all()):
                raise LasDatasetError(f'{source.path}: invalid coordinate encoding.')
        self._bounds = None
        for source in self._files:
            if source.header.bounds is not None:
                minimum, maximum = source.header.bounds
                if not np.isfinite((minimum, maximum)).all() or np.any(np.asarray(minimum) > maximum):
                    raise LasDatasetError(f'{source.path}: invalid coordinate bounds.')
                maximum_extent = np.max(np.asarray(maximum) - minimum)
                threshold = self.file_extent_warning_threshold.to(source.header.units)
                if maximum_extent > threshold.magnitude:
                    logger.warning(
                        '%s: maximum file extent %g %s exceeds %g %s; the data does not appear to be tiled, '
                        'which may cause memory issues.',
                        source.path, maximum_extent, source.header.units, threshold.magnitude, source.header.units,
                    )
                if self._bounds is None:
                    self._bounds = minimum, maximum
                else:
                    self._bounds = (tuple(np.minimum(self._bounds[0], minimum)),
                                    tuple(np.maximum(self._bounds[1], maximum)))

    @property
    def files(self) -> tuple[LasFile, ...]:
        """Non-empty source files in dataset order."""
        return self._files

    @staticmethod
    def _signature(path):
        status = path.stat()
        return status.st_size, status.st_mtime_ns

    def refreshed(self):
        """Return ``self`` if no source changed on disk, else a newly validated dataset.

        Compares each source's size and mtime with the values captured at load
        time, so call it before starting work that must see current files.
        """
        if any(self._signature(source.path) != source.signature for source in self._sources):
            return LasDataset(
                [source.path for source in self._sources], feature_codes=self.feature_codes,
                file_extent_warning_threshold=self.file_extent_warning_threshold,
            )
        return self

    def __iter__(self) -> Iterator[LasFile]:
        return iter(self._files)

    def lasfile_paths(self) -> list[Path]:
        """Paths of the non-empty source files, e.g. to hand to PDAL."""
        return [source.path for source in self._files]

    @property
    def point_count(self):
        """Total number of points in all non-empty files (0 if all are empty)."""
        return sum(source.header.point_count for source in self._files)

    @property
    def bounds(self):
        """Aggregate ``((xmin, ymin, zmin), (xmax, ymax, zmax))`` in CRS units, or ``None``."""
        return self._bounds

    @property
    def _reference_header(self):
        if not self._files:
            raise LasDatasetError('An empty LAS dataset has no reference header.')
        return self._files[0].header

    @property
    def crs(self):
        """``pyproj.CRS`` shared by all files (raises if the dataset is empty)."""
        return self._reference_header.crs

    @property
    def srs(self):
        """The shared CRS as a WKT string, suitable for PDAL ``a_srs``/``out_srs``."""
        return self.crs.to_wkt()

    @property
    def units(self):
        """Linear units shared by all files: ``'m'``, ``'foot'`` or ``'survey_foot'``."""
        return self._reference_header.units

    @property
    def epsg(self):
        """EPSG code of the shared CRS, or ``None`` if it has no EPSG equivalent."""
        return self._reference_header.epsg

    def raw_header(self):
        """Header for writing source records unchanged, e.g. when merging tiles.

        Copies the first file's header and requires every other file to have the
        same version, point format, scales/offsets and VLR/EVLR metadata.

        Raises:
            LasDatasetError: files are not raw-compatible.
        """
        first = self._reference_header
        for source in self._files[1:]:
            if not first.raw_compatible(source.header):
                raise LasDatasetError(f'Raw LAS merge requires matching schemas and encodings: {source.path}.')
        return LasHeader(first)

    def derived_header(self, dimensions=(), *, scales=None, offsets=None):
        """Header for writing newly computed points (LAS 1.4, point format 6).

        Args:
            dimensions: ``laspy.ExtraBytesParams`` for each extra attribute to store.
            scales: Three coordinate scales; defaults to the finest scale per axis
                across the files.
            offsets: Three coordinate offsets; defaults to the centre of ``bounds``.

        The CRS is copied from the dataset, and the dataset bounds are checked to
        be representable with the chosen scales/offsets (``LasHeaderError`` if not).

        Example::

            header = dataset.derived_header([laspy.ExtraBytesParams('cluster_id', 'int64')])
        """
        first = self._reference_header
        if scales is None:
            scales = tuple(min(source.header.scales[axis] for source in self._files) for axis in range(3))
        if offsets is None:
            offsets = ((np.asarray(self.bounds[0]) / 2 + np.asarray(self.bounds[1]) / 2)
                       if self.bounds is not None else first.offsets)
        header = first.derived_header(dimensions, scales=scales, offsets=offsets)
        if self.bounds is not None:
            header.validate_coordinates(np.asarray(self.bounds))
        return header

    def iter_record_batches(self, batch_size=250_000):
        """Yield ``(LasFile, laspy.ScaleAwarePointRecord)`` for each file in order.

        Each batch has at most ``batch_size`` points, so memory use is bounded
        regardless of dataset size. Raises ``ValueError`` for a non-positive or
        non-integer ``batch_size``.

        Example::

            for lasfile, records in dataset.iter_record_batches(100_000):
                process(records.x, records.y, records.z)
        """
        from .lasio import iter_record_batches

        for source in self._files:
            for records in iter_record_batches(source.path, batch_size=batch_size):
                yield source, records