"""
Build plate for one V module + one H module:  print.stl (one file) and print.3mf (same plate,
each part a separate object so the slicer can give them their own settings).

  V module : body_coupler_new, body_rib_V, body_rib_wire_guide, wheel x2
  H module : body_coupler_new, body_rib_H

Each part is turned onto the face that gives the least unsupported overhang (faces steeper
than 45 deg below horizontal that are not on the bed), ties broken by more bed contact,
then packed on the plate with 6 mm gaps. Z is up, units mm, everything sits on z = 0.
"""
import os

import numpy as np
import trimesh
from trimesh.transformations import rotation_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = os.path.join(HERE, "files")
BED = (220.0, 220.0)          # typical 220 x 220 mm bed; the plate must fit inside this
GAP = 6.0

PLATE = [  # (label, file)
    ("V coupler", "body_coupler_new"),
    ("V rib (IR mounts)", "body_rib_V"),
    ("V wire guide", "body_rib_wire_guide"),
    ("V wheel 1", "wheel"),
    ("V wheel 2", "wheel"),
    ("H coupler", "body_coupler_new"),
    ("H rib (oval, IR mounts)", "body_rib_H"),
]

CANDIDATES = {  # name -> rotation that puts that face of the part down onto the bed
    "-z down": np.eye(4),
    "+z down": rotation_matrix(np.pi, [1, 0, 0]),
    "-y down": rotation_matrix(np.pi / 2, [1, 0, 0]),
    "+y down": rotation_matrix(-np.pi / 2, [1, 0, 0]),
    "-x down": rotation_matrix(-np.pi / 2, [0, 1, 0]),
    "+x down": rotation_matrix(np.pi / 2, [0, 1, 0]),
}


def score(m):
    n, a = m.face_normals, m.area_faces
    zmin = m.triangles[:, :, 2].min(axis=1)
    on_bed = zmin < 0.05
    overhang = a[(n[:, 2] < -np.cos(np.radians(45))) & ~on_bed].sum()
    contact = a[(n[:, 2] < -0.99) & on_bed].sum()
    return overhang, contact


def orient(m):
    best = None
    for name, R in CANDIDATES.items():
        x = m.copy(); x.apply_transform(R); x.apply_translation(-x.bounds[0])
        ov, ct = score(x)
        key = (round(ov, 0), -ct)
        if best is None or key < best[0]:
            best = (key, name, x, ov, ct)
    return best[1], best[2], best[3], best[4]


def pack(parts):
    """Simple shelf packing, largest footprint first, rotating 90 deg about z if it helps."""
    order = sorted(range(len(parts)), key=lambda i: -np.prod(parts[i][1].extents[:2]))
    x = y = shelf_h = 0.0
    for i in order:
        m = parts[i][1]
        if m.extents[0] < m.extents[1]:
            m.apply_transform(rotation_matrix(np.pi / 2, [0, 0, 1])); m.apply_translation(-m.bounds[0])
        w, d = m.extents[:2]
        if x > 0 and x + w > BED[0] - 2 * GAP:
            x, y, shelf_h = 0.0, y + shelf_h + GAP, 0.0
        m.apply_translation([x + GAP, y + GAP, 0])
        x += w + GAP
        shelf_h = max(shelf_h, d)
    allb = np.vstack([p[1].bounds for p in parts])
    centre = (allb.min(0) + allb.max(0)) / 2
    for _, m, *_ in parts:
        m.apply_translation([BED[0] / 2 - centre[0], BED[1] / 2 - centre[1], 0])


def main():
    parts = []
    for label, f in PLATE:
        m = trimesh.load(os.path.join(FILES, f + ".stl"), force="mesh")
        assert m.is_watertight, f + " is not watertight"
        side, mo, ov, ct = orient(m)
        parts.append([label, mo, f, side, ov, ct])
    pack(parts)

    allb = np.vstack([p[1].bounds for p in parts])
    lo, hi = allb.min(0), allb.max(0)
    for label, m, f, side, ov, ct in parts:
        print(f"{label:30s} {f + '.stl':26s} {side:8s} size {m.extents.round(1).tolist()} mm, "
              f"overhang {ov:6.0f} mm2, bed contact {ct:5.0f} mm2")
    print("plate footprint", (hi - lo).round(1).tolist(), "within bed", BED, ":", bool(np.all(lo >= 0) and hi[0] <= BED[0] and hi[1] <= BED[1]))

    # no two parts may touch or overlap (bounding boxes are 6 mm apart by construction; double check)
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            a, b = parts[i][1].bounds, parts[j][1].bounds
            assert np.any(a[1][:2] + 1.0 < b[0][:2]) or np.any(b[1][:2] + 1.0 < a[0][:2]), (parts[i][0], parts[j][0])

    plate = trimesh.util.concatenate([p[1] for p in parts])
    out_stl = os.path.join(HERE, "print.stl")
    plate.export(out_stl)
    scene = trimesh.Scene()
    for label, m, *_ in parts:
        scene.add_geometry(m, node_name=label, geom_name=label)
    out_3mf = os.path.join(HERE, "print.3mf")
    scene.export(out_3mf)
    chk = trimesh.load(out_stl, force="mesh")
    print("wrote", out_stl, f"({len(chk.split(only_watertight=False))} parts, {len(chk.faces)} faces, "
          f"min z {chk.bounds[0][2]:.3f}, PLA volume {sum(p[1].volume for p in parts) / 1000:.1f} cm3)")
    print("wrote", out_3mf)
    return parts


if __name__ == "__main__":
    main()
