"""A validated, immutable collection of LAS/LAZ tiles, usable independently of this application.

``LasDataset`` reads only the headers of its source files when constructed. It
validates that the usable (non-empty) tiles share a CRS, linear units and a sane
coordinate encoding, then exposes aggregate metadata (point count, bounds, CRS,
EPSG, units) and bounded, ordered iteration over native laspy records. No
application feature mapping is required; ``feature_codes`` is optional and is
simply carried along for callers that need it.

Typical usage::

    from ground_classifier.las import LasDataset, LasDatasetError

    # A single file, a directory of *.las / *.laz, a glob, or an explicit list.
    dataset = LasDataset('data/tiles')
    dataset = LasDataset('data/tiles/*.laz')
    dataset = LasDataset(['a.laz', 'b.laz'], file_extent_warning_threshold=500)

    print(dataset.point_count)      # total points across non-empty tiles
    print(dataset.bounds)           # ((xmin, ymin, zmin), (xmax, ymax, zmax)) or None
    print(dataset.epsg, dataset.units, dataset.srs)   # e.g. 28356, 'm', WKT string
    for lasfile in dataset:         # LasFile(path, header, signature)
        print(lasfile.path, lasfile.header.point_count)

Units. By default (``units='auto'``) the units come from each file's CRS, which must be present,
projected and the same in every tile. Files often carry no CRS, so the units can be stated instead:
``units='metres'``, ``'usft'`` (US survey feet) or ``'foot'`` (international feet). A stated unit
is taken as given: the CRS is neither required nor compared. ``dataset.metres_per_unit`` converts
coordinates to metres.

Source order is deterministic: directory and glob matches are sorted, while an
explicit sequence keeps the caller's order. Empty tiles are logged and skipped
(``dataset.files`` / iteration / ``point_count`` ignore them, but ``dataset.sources``
lists them), and an all-empty dataset is valid but has no CRS, units or reference
header unless the units were stated (accessing those raises ``LasDatasetError``).

Neighbours and single members. A tile's surroundings are the other tiles in its folder, so
``LasDataset(files, siblings=True)`` holds the given files and every other LAS/LAZ file beside them
under one set of units; ``dataset.file(path)`` finds a member, and ``lasfile.read()`` or
``lasfile.iter_record_batches()`` reads its points.

Streaming points without loading a whole dataset into memory::

    for lasfile, records in dataset.iter_record_batches(batch_size=250_000):
        xyz = records.xyz            # laspy ScaleAwarePointRecord, at most batch_size rows

Merging compatible tiles into one file (see ``ground_classifier.las.lasio.write_record_batches``)::

    from ground_classifier.las import write_record_batches

    records = (batch for _, batch in dataset.iter_record_batches())
    write_record_batches('merged.laz', dataset.raw_header(), records)

Writing derived/processed points needs a header describing the new schema::

    header = dataset.derived_header(dimensions=[laspy.ExtraBytesParams('cluster_id', 'int64')])

Detecting changed inputs for long-running applications::

    dataset = dataset.refreshed()   # same object if unchanged, re-read if any file changed

Errors: ``FileNotFoundError`` when nothing matches or a path is not a file;
``LasDatasetError`` (a ``ValueError``) for duplicate paths, CRS/unit mismatches,
unsupported or non-projected CRS, invalid scales/offsets/bounds, or an invalid
``file_extent_warning_threshold``. ``LasUnitsError`` (a ``LasDatasetError``) is
raised when ``units='auto'`` finds no usable CRS: state the units to carry on.
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from glob import glob, has_magic
from pathlib import Path
import logging

import numpy as np

from .lasheader import LasHeader, LasHeaderError
from .lasio import iter_record_batches as _iter_record_batches
from .lasio import read_tile as _read_tile
from .lasutils import UNITS_AUTO, get_conversion, lasfiles_in_dir, parse_units_option


logger = logging.getLogger(__name__)


class LasDatasetError(ValueError):
    """Raised when a set of LAS files is invalid or inconsistent as a dataset."""


class LasUnitsError(LasDatasetError):
    """Raised when the units of a file cannot be told from its CRS and none were stated."""


@dataclass(frozen=True)
class LasFile:
    """Immutable descriptor of one LAS/LAZ file, and the way to read its points.

    Attributes:
        path: Location of the file.
        header: Header read once when the descriptor was created.
        signature: ``(size_bytes, mtime_ns)`` at read time, used by
            ``LasDataset.refreshed()`` to detect modification.

    Example::

        lasfile = LasFile.from_file('tile.laz')
        lasfile.header.point_count
        tile = lasfile.read()                            # the whole tile (a laspy.LasData)
        for records in lasfile.iter_record_batches():    # or in bounded batches
            ...
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

    def read(self):
        """Read the whole file: its header and every point record (a ``laspy.LasData``)."""
        return _read_tile(self.path)

    def iter_record_batches(self, batch_size=250_000):
        """Yield the file's points as ``laspy.ScaleAwarePointRecord`` batches of at most ``batch_size`` rows."""
        yield from _iter_record_batches(self.path, batch_size=batch_size)


class LasDataset:
    """Ordered, validated set of LAS/LAZ files that share a CRS and units.

    Args:
        data: One of a file path, a directory (its ``*.las``/``*.laz`` files),
            a glob pattern, or a sequence of file paths.
        feature_codes: Optional application-specific mapping, stored unchanged
            as ``self.feature_codes``. Not used by this class.
        units: ``'auto'`` (default) reads the units from the CRS, which every
            non-empty file must carry, be projected and share. ``'metres'``,
            ``'usft'`` or ``'foot'`` state the units instead and skip those CRS checks.
        siblings: Also include every other LAS/LAZ file in the folders of the given
            files, the tiles around them that give them their surroundings. The dataset
            is then in sorted path order, with absolute paths, and its units and CRS
            checks cover all of them.
        file_extent_warning_threshold: Length in metres (default 1000). A warning is
            logged for any file whose largest extent exceeds it, since untiled data
            can exhaust memory.

    Raises:
        FileNotFoundError: no matching files, or a listed path is not a file.
        LasDatasetError: see the module docstring.

    Example::

        dataset = LasDataset('data/tiles', units='usft', file_extent_warning_threshold=2000)
        for lasfile in dataset:
            print(lasfile.path.name, lasfile.header.point_count)
        dataset.file('data/tiles/a.laz').read()      # one member, by path
    """

    def __init__(self, data, feature_codes=None, *, units=UNITS_AUTO, siblings=False, file_extent_warning_threshold=1000.0):
        self.data = data
        self.feature_codes = feature_codes
        try:
            self._stated_units = parse_units_option(units)
        except ValueError as error:
            raise LasDatasetError(str(error)) from error
        threshold = file_extent_warning_threshold
        if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
                or not np.isfinite(threshold) or threshold <= 0):
            raise LasDatasetError('file_extent_warning_threshold must be a positive finite number of metres.')
        self.file_extent_warning_threshold = float(threshold)
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
        if siblings:
            found = {path.resolve() for path in paths}
            for folder in {path.parent for path in found}:
                found.update(file.resolve() for file in lasfiles_in_dir(folder))
            paths = sorted(found, key=str)
        self._sources = tuple(LasFile.from_file(path) for path in paths)
        self._by_path = {source.path.resolve(): source for source in self._sources}
        for source in self._sources:
            if not source.header.point_count:
                logger.warning('%s: source tile contains no points and will be skipped.', source.path)
        self._files = tuple(source for source in self._sources if source.header.point_count)
        first = next(iter(self._files), None)
        for source in self._files:
            if self._stated_units is None:
                try:
                    source.header.units
                except LasHeaderError as error:
                    raise LasUnitsError(
                        f'Cannot determine the units of {source.path.name}: {error} Guessing wrong silently '
                        f'corrupts every coordinate; state the units (metres, usft or foot) instead.'
                    ) from error
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
                units_name = self._stated_units or source.header.units
                threshold_in_units = self.file_extent_warning_threshold / get_conversion(units_name)
                if maximum_extent > threshold_in_units:
                    logger.warning(
                        '%s: maximum file extent %g %s exceeds %g %s; the data does not appear to be tiled, '
                        'which may cause memory issues.',
                        source.path, maximum_extent, units_name, threshold_in_units, units_name,
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

    @property
    def sources(self) -> tuple[LasFile, ...]:
        """Every source file in dataset order, including empty ones."""
        return self._sources

    def file(self, path) -> LasFile:
        """The member at ``path`` (any spelling of it; empty files too).

        Raises:
            KeyError: ``path`` is not in this dataset.
        """
        try:
            return self._by_path[Path(path).resolve()]
        except KeyError:
            raise KeyError(f'{path} is not in this LAS dataset.') from None

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
                units=self._stated_units or UNITS_AUTO,
                file_extent_warning_threshold=self.file_extent_warning_threshold,
            )
        return self

    def __iter__(self) -> Iterator[LasFile]:
        return iter(self._files)

    def lasfile_paths(self) -> list[Path]:
        """Paths of the non-empty source files."""
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
        """The shared CRS as a WKT string, or ``None`` if the files carry no CRS."""
        crs = self.crs
        return None if crs is None else crs.to_wkt()

    @property
    def units(self):
        """Linear units shared by all files: ``'m'``, ``'foot'`` or ``'survey_foot'``.

        The units stated in the constructor, else those of the CRS.
        """
        if self._stated_units is not None:
            return self._stated_units
        return self._reference_header.units

    @property
    def metres_per_unit(self):
        """Metres in one coordinate unit of the files (1 for metres)."""
        return float(get_conversion(self.units))

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
        for source in self._files:
            for records in source.iter_record_batches(batch_size):
                yield source, records