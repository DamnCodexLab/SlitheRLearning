"""
Make files/body_rib_V.stl: new_body_rib.stl + one M3 side mount on each side (for IR wall sensors).

Rib frame = world frame for a V module: x along body, y up, z lateral. Each side gets a flat
pad on the outer face of the arch wall (full band width, so it sits flush on the bed when the
rib is printed lying on its band face) with a horizontal M3 tapping hole pointing sideways.
"""
import os

import trimesh

import make_rib_H as R          # shared manifold helpers (box, to_mf, to_tm)
import manifold3d as mf

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = os.path.join(HERE, "files")

M3_TAP = 2.8                    # M3 screw cuts its own thread in PLA (use 3.2 for a loose fit + nut)
PAD_PROUD = 3.0                 # pad thickness outside the wall (wall is 2 mm -> ~5 mm of thread)
PAD_H = 10.0                    # pad height (y)
HOLE_Y = 4.0                    # hole height: above the wheel (top y -7.5), on the flat part of the wall
WALL_OUT = 21.6                 # outer face of the arch wall (|z|)


def cyl_z(x, y, r, z0, z1):
    return mf.Manifold.cylinder(z1 - z0, r, r, 48).translate([x, y, z0])


def build():
    rib = R.to_mf(trimesh.load(os.path.join(FILES, "new_body_rib.stl"), force="mesh"))
    x0, x1 = R.BAND_X
    for s in (+1, -1):
        zin, zout = WALL_OUT - 1.0, WALL_OUT + PAD_PROUD     # pad overlaps 1 mm into the wall
        lo, hi = sorted((s * zin, s * zout))
        rib = rib + R.box([x0, HOLE_Y - PAD_H / 2, lo], [x1, HOLE_Y + PAD_H / 2, hi])
        a, b = sorted((s * (WALL_OUT - 2.5), s * (zout + 1)))   # through pad and wall
        rib = rib - cyl_z(R.AXIS_X, HOLE_Y, M3_TAP / 2, a, b)
    return max(rib.decompose(), key=lambda p: p.volume())


def main():
    rib = R.to_tm(build())
    rib.merge_vertices(); rib.update_faces(rib.nondegenerate_faces()); rib.remove_unreferenced_vertices()
    assert rib.is_watertight, "rib not watertight"
    out = os.path.join(FILES, "body_rib_V.stl")
    rib.export(out)
    print(out, "faces", len(rib.faces), "volume", round(rib.volume, 1), "bounds", rib.bounds.round(2).tolist())


if __name__ == "__main__":
    main()
