"""
Make files/body_rib_H.stl: rib for H (pitch) modules.

The H module is the normal module rolled 90 deg about the body axis, so in the rib's own
frame (x along body, y = servo axis, z across) the world directions are
    world lateral  = local y      (world z = y - 18.715)
    world down     = local +z     (world y = 18.715 - z)

Changes vs new_body_rib.stl (all functional interfaces are kept as-is):
  * D-shaped arch + wheel tabs  ->  rounded-oval loop (superellipse, n = 4) in the same
    x 0.2..10.2 band; it touches the floor plate (pivot) and the top cap (horn) so both
    stay tied in.
  * no wheel axle tabs (H modules never touch the floor)
  * new wire collector on the world-bottom side (local +z): a 16 x 10.5 mm loop running
    along the body, centred under the module, with a 3 mm snap-in slit.
  * M3 tapping hole on a pad on each world side (IR wall sensors), blind, 6.5 mm deep.
Kept from new_body_rib: floor plate + pivot hole, top cap + horn screw hole, top bar,
rear leg with the 4-hole pattern.
"""
import os

import manifold3d as mf
import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = os.path.join(HERE, "files")

BAND_X = (0.2, 10.2)
AXIS_X = 5.2                    # pivot / horn hole centre (measured)
ROLL_Y = 18.715                 # roll line of the H mount (world body centre line)
# oval (superellipse) in the local (y, z) plane
Y_C, A_IN, B_IN, WALL, N = 14.9, 20.7, 19.6, 2.0, 4.0
# wire collector (outer box, local y/z), wall and slit
COL_Y = (ROLL_Y - 8.0, ROLL_Y + 8.0)
COL_Z = (20.0, 30.5)
COL_WALL = 1.6
SLIT = 3.0
# M3 side mounts for IR wall sensors. World lateral = local y, so the two sides are the two
# flat ends of the oval (floor-plate side and top-cap side). The pivot / horn holes sit at z = 0,
# so the M3 hole is moved 8.5 mm toward world-down (+z). Holes are blind so the screw tip
# cannot reach the horn or the coupler pin.
M3_TAP = 2.8
IR_Z = 8.5
IR_PAD_Z = (4.0, 13.0)
IR_PAD_PROUD = 3.0
IR_DEPTH = 6.5


def to_mf(m):
    m = m.copy(); m.merge_vertices()
    return mf.Manifold(mf.Mesh(vert_properties=np.asarray(m.vertices, np.float32),
                               tri_verts=np.asarray(m.faces, np.uint32)))


def to_tm(M):
    mesh = M.to_mesh()
    # manifold's output is already a closed, shared-vertex mesh; re-merging would collapse slivers
    return trimesh.Trimesh(np.asarray(mesh.vert_properties)[:, :3], np.asarray(mesh.tri_verts), process=False)


def box(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return mf.Manifold.cube(list(hi - lo)).translate(list(lo))


def superellipse(a, b, yc, n=N, k=256):
    t = np.linspace(0, 2 * np.pi, k, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    y = yc + a * np.sign(c) * np.abs(c) ** (2 / n)
    z = b * np.sign(s) * np.abs(s) ** (2 / n)
    return np.c_[y, z]


def prism(poly_yz, x0, x1):
    """Extrude a (y, z) polygon along x from x0 to x1."""
    cs = mf.CrossSection([poly_yz.tolist()])
    ext = mf.Manifold.extrude(cs, x1 - x0)            # in (u=y, v=z, w=x)
    T = [[0, 0, 1, x0], [1, 0, 0, 0], [0, 1, 0, 0]]   # x = w + x0, y = u, z = v
    return ext.transform(T)


def cyl_y(x, z, r, y0, y1):
    c = mf.Manifold.cylinder(y1 - y0, r, r, 64)        # along +w
    # proper rotation (det +1): (u, v, w) -> (x + u, y0 + w, z - v)
    T = [[1, 0, 0, x], [0, 0, 1, y0], [0, -1, 0, z]]
    return c.transform(T)


def build():
    orig = to_mf(trimesh.load(os.path.join(FILES, "new_body_rib.stl"), force="mesh"))
    outer = prism(superellipse(A_IN + WALL, B_IN + WALL, Y_C), *BAND_X)
    ring = outer - prism(superellipse(A_IN, B_IN, Y_C), BAND_X[0] - 1, BAND_X[1] + 1)

    # parts of new_body_rib to keep, as ONE clip region (separately clipped pieces would share
    # cut planes and leave non-manifold edges). The floor clip reaches 0.6 mm into the oval wall
    # so the plate overlaps the wall instead of just touching it.
    into_wall = prism(superellipse(A_IN + 0.6, B_IN + 0.6, Y_C), BAND_X[0] - 0.5, BAND_X[1] + 0.5)
    keep = ((box([0, -7, -25], [10.3, -2.5, 25]) ^ into_wall)           # pivot floor plate
            + box([0, 29, -9], [10.3, 38, 9])                            # horn top cap
            + box([10.15, 30, -9], [30, 40, 9]) + box([20, 0, -9], [30, 40, 9]))   # top bar + 4-hole leg
    core = orig ^ keep

    col = box([BAND_X[0], COL_Y[0], COL_Z[0]], [BAND_X[1], COL_Y[1], COL_Z[1]])
    col = col - box([BAND_X[0] - 1, COL_Y[0] + COL_WALL, 22.2], [BAND_X[1] + 1, COL_Y[1] - COL_WALL, COL_Z[1] - COL_WALL])
    col = col - box([BAND_X[0] - 1, ROLL_Y - SLIT / 2, COL_Z[1] - COL_WALL - 0.5], [BAND_X[1] + 1, ROLL_Y + SLIT / 2, COL_Z[1] + 1])

    rib = ring + core + col
    y_lo, y_hi = Y_C - (A_IN + WALL), Y_C + (A_IN + WALL)          # outer faces of the oval
    for face, out in ((y_lo, -1), (y_hi, +1)):
        a, b = sorted((face - out * 2.0, face + out * IR_PAD_PROUD))  # pad overlaps into the wall
        rib = rib + box([BAND_X[0], a, IR_PAD_Z[0]], [BAND_X[1], b, IR_PAD_Z[1]])
        top = face + out * IR_PAD_PROUD
        a, b = sorted((top + out * 1.0, top - out * IR_DEPTH))
        rib = rib - cyl_y(AXIS_X, IR_Z, M3_TAP / 2, a, b)
    rib = rib - cyl_y(AXIS_X, 0.0, 3.4, -10.0, -3.0)    # pivot: coupler pin passes through
    rib = rib - cyl_y(AXIS_X, 0.0, 1.25, 30.0, 40.0)    # horn screw
    return max(rib.decompose(), key=lambda p: p.volume())   # drop zero-volume clipping slivers


def main():
    rib = to_tm(build())
    # STL readers merge coincident vertices, which collapses a few sliver triangles; do that
    # here and drop the resulting zero-area faces so the file stays closed for every reader
    rib.merge_vertices()
    rib.update_faces(rib.nondegenerate_faces())
    rib.remove_unreferenced_vertices()
    assert rib.is_watertight, "rib not watertight"
    out = os.path.join(FILES, "body_rib_H.stl")
    rib.export(out)
    print(out, "faces", len(rib.faces), "volume", round(rib.volume, 1), "bounds", rib.bounds.round(2).tolist())


if __name__ == "__main__":
    main()
