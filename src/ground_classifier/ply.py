"""Binary PLY triangle meshes: write_mesh() and read_mesh().

The layout is fixed so that CloudCompare, MeshLab and read_mesh() all take it as is:

    format binary_little_endian 1.0
    element vertex N:  double x, double y, double z
    element face M:    list uchar int vertex_indices

Coordinates are doubles in whatever units the caller gives (the pipeline writes the input file's own units, so a mesh
lies over its point cloud). PLY has no coordinate offset: CloudCompare proposes its own shift when it loads large
coordinates.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

_VERTEX = np.dtype("<f8")
_FACE = np.dtype([("n", "u1"), ("v", "<i4", (3,))])


def write_mesh(path, xyz, faces, comments=()) -> tuple[int, int]:
    """Write the triangles `faces` (m, 3 indices into `xyz`) with only the vertices they use.

    Parameters
    ----------
    xyz : (n, 3) float array
    faces : (m, 3) int array
    comments : strings written as `comment` lines of the header (one line each)

    Returns
    -------
    (vertices, faces) written."""
    xyz = np.asarray(xyz)
    faces = np.asarray(faces)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape (m, 3)")
    used, compact = np.unique(faces, return_inverse=True)
    vertices = np.ascontiguousarray(xyz[used], dtype=_VERTEX)
    records = np.empty(len(faces), _FACE)
    records["n"] = 3
    records["v"] = compact.reshape(-1, 3)
    header = ["ply", "format binary_little_endian 1.0"]
    for text in comments:
        if "\n" in text or "\r" in text:
            raise ValueError("a comment must be a single line")
        header.append(f"comment {text}")
    header += [
        f"element vertex {len(vertices)}", "property double x", "property double y", "property double z",
        f"element face {len(records)}", "property list uchar int vertex_indices", "end_header",
    ]
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        vertices.tofile(f)
        records.tofile(f)
    return len(vertices), len(records)


def read_mesh(path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Read a file written by `write_mesh`: (xyz (n, 3) float64, faces (m, 3) int32, comments).

    Raises ValueError for any other PLY layout."""
    data = Path(path).read_bytes()
    marker = b"end_header\n"
    end = data.find(marker)
    if end < 0:
        raise ValueError("not a PLY file: no end_header")
    lines = data[:end].decode("ascii").rstrip("\n").split("\n")
    comments = [line[len("comment "):] for line in lines if line.startswith("comment ")]
    structure = [line for line in lines if not line.startswith("comment ")]
    counts = {}
    for line in structure:
        parts = line.split()
        if len(parts) == 3 and parts[0] == "element":
            counts[parts[1]] = int(parts[2])
    expected = ["ply", "format binary_little_endian 1.0", f"element vertex {counts.get('vertex')}",
                "property double x", "property double y", "property double z",
                f"element face {counts.get('face')}", "property list uchar int vertex_indices"]
    if structure != expected:
        raise ValueError("unsupported PLY layout (only the one write_mesh writes)")
    n, m = counts["vertex"], counts["face"]
    body = data[end + len(marker):]
    if len(body) != n * 3 * _VERTEX.itemsize + m * _FACE.itemsize:
        raise ValueError("PLY body has the wrong size")
    xyz = np.frombuffer(body, _VERTEX, 3 * n).reshape(n, 3).copy()
    records = np.frombuffer(body, _FACE, m, offset=n * 3 * _VERTEX.itemsize)
    if m and not (records["n"] == 3).all():
        raise ValueError("only triangles are supported")
    return xyz, records["v"].astype(np.int32), comments
