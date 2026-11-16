"""
Make files/body_coupler_new.stl: the original coupler with its mounting block
re-done to bolt onto new_body_rib.stl's 4-hole leg.

new_body_rib leg (x 25.2..27.7) holes, axis along x, dia 3:
    (y, z) = (15.33, +-3.5) and (22.10, +-3.5)   -> pattern centre (18.715, 0)
Old coupler block: x -21.05..-16.05, y 12.8..20.8, z +-5.9, 2 diagonal holes.

New block: square 11.8 x 11.8 centred on the pattern centre, so the same coupler
bolts on upright (V module, servo axis vertical) or rolled 90 deg about the body
axis (H module, servo axis horizontal / lateral).
"""
import os

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = os.path.join(HERE, "files")

HOLE_R = 1.5
HOLES_YZ = [(15.33, 3.5), (15.33, -3.5), (22.10, 3.5), (22.10, -3.5)]
PATTERN_C = (18.715, 0.0)
HALF = 5.9
BLOCK_X = (-21.05, -16.05)


def box(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    b = trimesh.creation.box(extents=hi - lo)
    b.apply_translation((lo + hi) / 2)
    return b


def hole(y, z, x0=-23.0, x1=-15.0):
    c = trimesh.creation.cylinder(radius=HOLE_R, height=x1 - x0, sections=48)
    c.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    c.apply_translation([(x0 + x1) / 2, y, z])
    return c


def main():
    old = trimesh.load(os.path.join(FILES, "body_coupler.stl"), force="mesh")
    cy, cz = PATTERN_C
    # solid block: fills the old holes and grows the block up to y = 24.6
    block = box([BLOCK_X[0], cy - HALF, cz - HALF], [BLOCK_X[1], cy + HALF, cz + HALF])
    solid = trimesh.boolean.union([old, block], engine="manifold")
    new = trimesh.boolean.difference([solid] + [hole(y, z) for y, z in HOLES_YZ], engine="manifold")
    assert new.is_watertight
    out = os.path.join(FILES, "body_coupler_new.stl")
    new.export(out)
    print(out, "faces", len(new.faces), "bounds", new.bounds.round(2).tolist(),
          "volume", round(new.volume, 1), "(old", round(old.volume, 1), ")")


if __name__ == "__main__":
    main()
