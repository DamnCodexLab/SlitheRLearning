"""
Make files/caboose_new.stl: the original caboose with its front leg re-drilled to the
new_body_rib 4-hole pattern, so body_coupler_new bolts to it (upright, V).

The caboose leg is the old rib leg shifted by -48.9 mm in x:
  leg plate x -23.7..-21.2, old holes (y, z) = (16.0, +3.0), (18.0, -3.0), dia 3
New holes (same as new_body_rib): (15.33, +-3.5), (22.10, +-3.5), dia 3
"""
import os

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = os.path.join(HERE, "files")

PLATE_X = (-23.7, -21.2)
OLD_HOLES = [(16.0, 3.03), (18.03, -3.0)]
NEW_HOLES = [(15.33, 3.5), (15.33, -3.5), (22.10, 3.5), (22.10, -3.5)]
R = 1.5


def cyl_x(y, z, r, x0, x1):
    c = trimesh.creation.cylinder(radius=r, height=x1 - x0, sections=48)
    c.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    c.apply_translation([(x0 + x1) / 2, y, z])
    return c


def main():
    cab = trimesh.load(os.path.join(FILES, "caboose.stl"), force="mesh")
    plugs = [cyl_x(y, z, R + 0.05, *PLATE_X) for y, z in OLD_HOLES]
    filled = trimesh.boolean.union([cab] + plugs, engine="manifold")
    drills = [cyl_x(y, z, R, PLATE_X[0] - 3, PLATE_X[1] + 3) for y, z in NEW_HOLES]
    new = trimesh.boolean.difference([filled] + drills, engine="manifold")
    assert new.is_watertight
    out = os.path.join(FILES, "caboose_new.stl")
    new.export(out)
    print(out, "faces", len(new.faces), "volume", round(new.volume, 1), "(old", round(cab.volume, 1), ")",
          "bounds same:", np.allclose(new.bounds, cab.bounds, atol=0.01))


if __name__ == "__main__":
    main()
