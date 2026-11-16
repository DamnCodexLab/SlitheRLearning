"""
Pack the Slitherbot (VH revision: new_body_rib + body_coupler_new) meshes for the browser viewer (assembly/viewer/snake_data.js).

Every geometry stays in the shared design frame (mm, x = snake axis, y = up),
except wheels, which are re-centred on their axle so they can spin.
Large meshes are decimated to keep the page light.
"""
import base64
import json
import os
import sys

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import assemble_snake as A  # noqa: E402
import snake_vh as VH  # noqa: E402

PLA_G_PER_MM3 = 1.24e-3   # solid PLA; printed parts are mostly thin walls
SERVO_G = 13.4            # MG90S datasheet mass

OUT = os.path.join(HERE, "assembly", "viewer")

# head (design frame of head_*.stl: z up, y long axis) -> body frame, bolted to the
# front rib's rear leg where the next coupler block would sit
HEAD_T = np.eye(4)
HEAD_T[:3, :3] = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
HEAD_T[:3, 3] = [27.85 + 8.92, -65 + 16.8, -83.7]


def dec(m, faces):
    if len(m.faces) <= faces:
        return m
    return m.simplify_quadric_decimation(face_count=faces)


def pack(m):
    m = m.copy()
    m.merge_vertices()
    v = np.asarray(m.vertices, np.float32)
    f = np.asarray(m.faces, np.uint32)
    return {
        "v": base64.b64encode(v.tobytes()).decode(),
        "f": base64.b64encode(f.tobytes()).decode(),
        "n": int(len(f)),
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    body, horn = A.mg90s()
    wheel = A.load("wheel")
    wheel.apply_translation(-wheel.bounds.mean(0))      # centred on axle, axis = z
    hb, ht = A.load("head_bottom"), A.load("head_top")
    for m in (hb, ht):
        m.apply_transform(HEAD_T)

    geoms = {
        "rib": (A.load("body_rib_V"), "flat"),
        "ribH": (A.load("body_rib_H"), "flat"),
        "coupler": (A.load("body_coupler_new"), "flat"),
        "wire_guide": (A.load("body_rib_wire_guide"), "flat"),
        "servo": (body, "flat"),
        "horn": (horn, "flat"),
        "wheel": (dec(wheel, 1400), "smooth"),
        "caboose": (A.load("caboose_new"), "flat"),
        "tail_guide": (A.load("tail_wire_guide"), "flat"),
        "tail": (dec(A.load("tail"), 9000), "smooth"),
        "head_bottom": (dec(hb, 9000), "smooth"),
        "head_top": (dec(ht, 9000), "smooth"),
    }
    data = {}
    for k, (m, s) in geoms.items():
        full = m
        if k == "servo":
            mass, com = SERVO_G, m.bounds.mean(0)
        elif k == "horn":
            mass, com = 1.0, m.bounds.mean(0)
        else:
            mass, com = abs(full.volume) * PLA_G_PER_MM3, full.center_mass
        data[k] = dict(pack(m), shade=s, mass=round(float(mass), 2), com=[round(float(c), 2) for c in com])
    data["_meta"] = {
        "pitch": A.PITCH, "axis_x": float(VH.AXIS_PT[0]), "wheel_axle": [float(VH.AXIS_PT[0]), -15.0, 16.5],
        "wheel_r": 7.5, "ground_y": -22.5,
        "caboose_axles": [-138.8, -39.0],
        "layout": VH.DEFAULT_LAYOUT, "roll_pt": VH.ROLL_PT.tolist(),
    }
    path = os.path.join(OUT, "snake_data.js")
    with open(path, "w") as fh:
        fh.write("window.SNAKE_DATA=" + json.dumps(data) + ";\n")
    print("faces", {k: v["n"] for k, v in data.items() if k != "_meta"})
    print(path, round(os.path.getsize(path) / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
