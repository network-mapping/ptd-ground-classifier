"""Wrapper around ``laspy.LasHeader`` with CRS/unit handling and header derivation.

``LasHeader`` keeps the complete native laspy header (version, point format,
scales/offsets, VLRs/EVLRs, ExtraBytes descriptors) and adds the queries and
transformations applications usually need. It does no point I/O.

Reading a header (cheap; no points are loaded, and the CRS is not parsed until asked for)::

    from ground_classifier.las import LasHeader, LasHeaderError

    header = LasHeader.from_file('tile.laz')
    header.version              # '1.4'
    header.point_format_id      # 6
    header.point_count          # 1_234_567
    header.dimension_names      # ('X', 'Y', 'Z', 'intensity', ..., 'HeightAboveGround')
    header.has_dimension('HeightAboveGround')
    header.scales, header.offsets
    header.bounds               # ((xmin, ymin, zmin), (xmax, ymax, zmax)); None if empty
    header.extent_xy            # (xmin, ymin, xmax, ymax) exactly as the header states it

Coordinate reference system (needs ``pyproj``)::

    header.crs                  # pyproj.CRS or None
    header.epsg                 # int or None
    try:
        header.units            # 'm', 'foot' or 'survey_foot'
    except LasHeaderError:
        ...                     # no CRS, not projected, or mixed/unsupported units

Querying ExtraBytes::

    header.extra_dimension('height_above_ground')   # name match ignores case and punctuation
    header.no_data_value('HeightAboveGround')       # declared sentinel, or None

Building headers for output::

    import laspy

    native = header.native_copy()                   # independent laspy.LasHeader
    with_extra = header.with_dimensions([laspy.ExtraBytesParams('cluster_id', 'int64')])
    new_header = header.derived_header([laspy.ExtraBytesParams('cluster_id', 'int64')],
                                       scales=(0.001, 0.001, 0.001), offsets=(500000, 6000000, 0))
    new_header.validate_coordinates(xyz)            # xyz: (n, 3) array; checks int32 range

Writing with the headers (see ``ground_classifier.las.lasio``)::

    from ground_classifier.las import write_record_batches

    write_record_batches('out.laz', header, scale_aware_record_batches)

``LasHeader`` instances are safe to share: the constructor and ``native_copy()``
copy the underlying laspy header, and no method mutates ``self``.
"""

from copy import deepcopy
from math import fma
import re

import laspy
import numpy as np

from .lasutils import METRES, IMPERIAL_FEET, US_FEET


class LasHeaderError(ValueError):
    """Raised for unusable CRS/units, invalid coordinate encodings or ambiguous ExtraBytes."""


class LasHeader:
    """Immutable-by-convention view of a LAS file header.

    Args:
        native: A ``laspy.LasHeader`` or another ``LasHeader``; it is copied.

    Use ``LasHeader.from_file(path)`` to read one from disk.
    """

    def __init__(self, native):
        self._native = native.native_copy() if isinstance(native, LasHeader) else native.copy()
        # Parsed on first use: it needs pyproj, and a header read only for its counts and bounds (the
        # scheduler, spawned workers) must not depend on the CRS being readable.
        self._crs = None
        self._crs_parsed = False

    @classmethod
    def from_file(cls, path):
        """Read the header of a LAS/LAZ file without loading its points."""
        with laspy.open(path) as reader:
            return cls(reader.header)

    def native_copy(self):
        """Return an independent ``laspy.LasHeader`` that callers may modify."""
        return self._native.copy()

    @property
    def version(self):
        """LAS version string, e.g. ``'1.4'``."""
        return str(self._native.version)

    @property
    def point_format_id(self):
        """LAS point data record format number."""
        return self._native.point_format.id

    @property
    def point_count(self):
        """Number of points declared by the header."""
        return self._native.point_count

    @property
    def dimension_names(self):
        """Names of all standard and extra dimensions in the point format."""
        return tuple(self._native.point_format.dimension_names)

    @property
    def extra_dimension_names(self):
        """Names of the ExtraBytes dimensions in the point format."""
        return tuple(str(name) for name in self._native.point_format.extra_dimension_names)

    @property
    def point_format_size(self):
        """Size in bytes of one point record."""
        return int(self._native.point_format.size)

    @property
    def generating_software(self):
        """The ``generating_software`` field of the header."""
        return str(self._native.generating_software)

    def has_dimension(self, name: str) -> bool:
        """True if ``name`` is an exact (case-sensitive) dimension name."""
        return name in self.dimension_names

    @property
    def scales(self):
        """Coordinate scale factors ``(x, y, z)``."""
        return tuple(self._native.scales)

    @property
    def offsets(self):
        """Coordinate offsets ``(x, y, z)``."""
        return tuple(self._native.offsets)

    @property
    def bounds(self):
        """``((xmin, ymin, zmin), (xmax, ymax, zmax))`` from the header, or ``None`` if empty."""
        if not self.point_count:
            return None
        return tuple(self._native.mins), tuple(self._native.maxs)

    @property
    def extent_xy(self):
        """``(xmin, ymin, xmax, ymax)`` exactly as the header states it, whether or not the file has points."""
        mins, maxs = self._native.mins, self._native.maxs
        return float(mins[0]), float(mins[1]), float(maxs[0]), float(maxs[1])

    @property
    def crs(self):
        """Parsed ``pyproj.CRS``, or ``None`` if the header carries none.

        Raises:
            LasHeaderError: the header's CRS records cannot be parsed.
        """
        if not self._crs_parsed:
            try:
                self._crs = self._native.parse_crs()
            except Exception as error:
                raise LasHeaderError(f'LAS CRS could not be parsed: {error}') from error
            self._crs_parsed = True
        return self._crs

    @property
    def epsg(self):
        """EPSG code of ``crs``, or ``None`` if absent or not representable."""
        return None if self.crs is None else self.crs.to_epsg()

    @property
    def units(self):
        """Linear unit of the projected CRS: ``'m'``, ``'foot'`` or ``'survey_foot'``.

        Raises:
            LasHeaderError: no CRS, a non-projected CRS, differing horizontal and
                vertical units, or units other than metres/feet/US survey feet.
        """
        if self.crs is None:
            raise LasHeaderError('LAS header has no resolvable coordinate reference system.')
        components = self.crs.sub_crs_list if self.crs.is_compound else [self.crs]
        horizontal = components[0]
        if not horizontal.is_projected:
            raise LasHeaderError('LAS processing requires a projected coordinate reference system.')
        factors = [axis.unit_conversion_factor for component in components for axis in component.axis_info]
        if len(factors) < 2 or not all(np.isfinite(value) and value > 0 for value in factors):
            raise LasHeaderError('LAS CRS has no supported linear coordinate units.')
        if not all(np.isclose(value, factors[0], rtol=0, atol=1e-12) for value in factors):
            raise LasHeaderError('LAS CRS uses different horizontal and vertical coordinate units.')
        for factor, unit in ((1.0, METRES), (0.3048, IMPERIAL_FEET), (1200 / 3937, US_FEET)):
            if np.isclose(factors[0], factor, rtol=0, atol=1e-12):
                return unit
        raise LasHeaderError('LAS CRS linear units are not supported.')

    def with_dimensions(self, dimensions):
        """Return a new header with extra ``laspy.ExtraBytesParams`` appended.

        Existing ExtraBytes descriptors (including NoData, scale and offset
        metadata) are preserved.
        """
        native = self.native_copy()
        dimensions = tuple(dimensions)
        if dimensions:
            extra_vlrs = native.vlrs.get('ExtraBytesVlr')
            original = deepcopy(extra_vlrs[0].extra_bytes_structs) if extra_vlrs else []
            native.add_extra_dims(dimensions)
            native.vlrs.get('ExtraBytesVlr')[0].extra_bytes_structs[:len(original)] = original
        return LasHeader(native)

    def extra_dimension(self, name):
        """Find a scalar ExtraBytes dimension, ignoring case and non-alphanumerics.

        ``'height_above_ground'`` matches ``'HeightAboveGround'``.

        Raises:
            LasHeaderError: no match, several matches, or a vector dimension.
        """
        normalized = re.sub(r'[^a-zA-Z0-9]', '', name).lower()
        dimensions = [dimension for dimension in self._native.point_format.extra_dimensions
                      if re.sub(r'[^a-zA-Z0-9]', '', dimension.name).lower() == normalized]
        if len(dimensions) != 1 or dimensions[0].num_elements != 1:
            raise LasHeaderError(f'LAS header has no unambiguous scalar {name}.')
        return dimensions[0]

    def no_data_value(self, name):
        """NoData sentinel declared for scalar dimension ``name``, or ``None``.

        The value is returned in physical units (the ExtraBytes scale/offset are
        applied). Standard dimensions never have a sentinel.

        Raises:
            LasHeaderError: ``name`` is not an unambiguous scalar ExtraBytes dimension.

        Example::

            nodata = header.no_data_value('HeightAboveGround')
            valid = values != nodata if nodata is not None else slice(None)
        """
        if name in self._native.point_format.standard_dimension_names:
            return None

        dimension = self.extra_dimension(name)
        descriptors = [entry for vlr in self._native.vlrs.get('ExtraBytesVlr')
                       for entry in vlr.extra_bytes_structs
                       if entry.format_name() == dimension.name]
        if len(descriptors) != 1:
            raise LasHeaderError(f'No unambiguous ExtraBytes descriptor for {name}.')
        descriptor = descriptors[0]
        if not descriptor.options & descriptor.NO_DATA_BIT_MASK:
            return None

        value = np.asarray(descriptor.no_data)[0].item()
        scale = 1.0 if dimension.scales is None else float(dimension.scales[0])
        offset = 0.0 if dimension.offsets is None else float(dimension.offsets[0])
        if scale == 1.0 and offset == 0.0:
            return value
        return fma(float(value), scale, offset)

    def raw_compatible(self, other):
        """True if records of ``other`` can be written unchanged under this header.

        Requires equal version, point format, scales, offsets and VLR/EVLR metadata.
        """
        def metadata(native):
            return tuple((entry.user_id, entry.record_id, entry.description, entry.record_data_bytes())
                         for entry in (*native.vlrs, *(native.evlrs or ()))
                         if entry.user_id != 'laszip encoded')

        return (self.version == other.version
                and self._native.point_format == other._native.point_format
                and self.scales == other.scales and self.offsets == other.offsets
                and metadata(self._native) == metadata(other._native))

    def derived_header(self, dimensions=(), *, scales=None, offsets=None):
        """New LAS 1.4, point format 6 header for computed points.

        Keeps this header's CRS and, unless overridden, its scales and offsets.
        ``dimensions`` are ``laspy.ExtraBytesParams`` added as ExtraBytes.

        Raises:
            LasHeaderError: ``scales``/``offsets`` are not three finite values with
                positive scales.
        """
        native = laspy.LasHeader(point_format=6, version='1.4')
        native.scales = np.asarray(self.scales if scales is None else scales, dtype=float)
        native.offsets = np.asarray(self.offsets if offsets is None else offsets, dtype=float)
        if (native.scales.shape != (3,) or native.offsets.shape != (3,)
                or not np.isfinite(native.scales).all() or not np.isfinite(native.offsets).all()
                or np.any(native.scales <= 0)):
            raise LasHeaderError('Coordinate encoding requires three positive finite scales and finite offsets.')
        if self.crs is not None:
            native.add_crs(self.crs)
        return LasHeader(native).with_dimensions(dimensions)

    def validate_coordinates(self, xyz):
        """Check that ``xyz`` (shape ``(n, 3)``, finite) encodes within int32 range.

        Raises:
            LasHeaderError: invalid shape/values or overflow with these scales/offsets.
        """
        xyz = np.asarray(xyz)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
            raise LasHeaderError('LAS coordinates must be finite with shape (n, 3).')
        encoded = np.rint((xyz - self.offsets) / self.scales)
        limits = np.iinfo(np.int32)
        if np.any(encoded < limits.min) or np.any(encoded > limits.max):
            raise LasHeaderError('Coordinates exceed the signed int32 range of the output encoding.')
