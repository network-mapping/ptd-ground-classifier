"""Convert validation clip .npz files to .laz so they can be viewed in CloudCompare.

Usage: python tests/validation/npz_to_laz.py [clips_dir] [out_dir]

The Classification field holds ``c_input`` (what the engine receives). ``c_terra``
(reference classification) and ``core_mask`` (1 = core, 0 = buffer) are written as
extra dimensions, which CloudCompare shows as scalar fields.
"""
import sys
from pathlib import Path

import laspy
import numpy as np

HERE = Path(__file__).resolve().parent
CLIPS_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "clips"
OUT_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE.parents[1] / "output" / "validation_clips_laz"


def convert(npz: Path, out_dir: Path) -> Path:
    d = np.load(npz)
    x, y, z = (np.asarray(d[k], dtype=np.float64) for k in "xyz")

    # Clip coordinates are metres; 0.1 mm scale preserves the 0.009144 m (0.03 ft) FPL grid.
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.offsets = [x.min(), y.min(), z.min()]
    header.scales = [0.0001, 0.0001, 0.0001]
    las = laspy.LasData(header)
    las.x, las.y, las.z = x, y, z
    las.classification = np.asarray(d["c_input"], dtype=np.uint8)

    for name in ("c_terra", "core_mask"):
        las.add_extra_dim(laspy.ExtraBytesParams(name=name, type=np.uint8))
        las[name] = np.asarray(d[name], dtype=np.uint8)

    out = out_dir / f"{npz.stem}.laz"
    las.write(out)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for npz in sorted(CLIPS_DIR.glob("*.npz")):
        out = convert(npz, OUT_DIR)
        print(f"{npz.name} -> {out}")


if __name__ == "__main__":
    main()
