"""
Export the VH snake for Fusion: one STL per body in assembly/parts_vh/ (design frame, mm)
and assembly/parts_vh/poses.json with the 4x4 placement of every rigid body for a few poses.
Used by fusion_build_vh.py.
"""
import json
import os
import sys

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import assemble_snake as A  # noqa: E402
import snake_vh as S  # noqa: E402
from analyze_vh import gait  # noqa: E402
from export_viewer_data import HEAD_T  # noqa: E402

OUT = os.path.join(HERE, "assembly", "parts_vh")


def main():
    os.makedirs(OUT, exist_ok=True)
    wheel = A.load("wheel"); wc = wheel.bounds.mean(0)
    wheel = wheel.simplify_quadric_decimation(face_count=900)
    wr = wheel.copy(); wr.apply_translation([S.AXIS_PT[0] - wc[0], 0, 0])
    wl = wr.copy(); wl.apply_transform(np.diag([1, 1, -1, 1]))
    servo, horn = (trimesh.boolean.union(m.split(only_watertight=False), engine="manifold") for m in A.mg90s())
    heads = {}
    for n in ("head_bottom", "head_top"):
        h = A.load(n); h.apply_transform(HEAD_T)
        heads[n] = h.simplify_quadric_decimation(face_count=4000)
    tail = A.load("tail").simplify_quadric_decimation(face_count=5000)
    parts = {
        "rib": A.load("body_rib_V"), "rib_H": A.load("body_rib_H"), "coupler": A.load("body_coupler_new"),
        "wire_guide": A.load("body_rib_wire_guide"), "wheel_R": wr, "wheel_L": wl,
        "servo": servo, "horn": horn, "caboose": A.load("caboose_new"),
        "tail_guide": A.load("tail_wire_guide"), "tail": tail, **heads,
    }
    for ax_i, ax in enumerate((-138.8, -39.0)):
        for s, nm in ((1, "R"), (-1, "L")):
            w = (wr if s > 0 else wl).copy(); w.apply_translation([ax - S.AXIS_PT[0], 0, 0])
            parts[f"cab_wheel_{ax_i}{nm}"] = w
    for k, m in parts.items():
        m.export(os.path.join(OUT, k + ".stl"))

    comps = {
        "base": ["caboose", "tail_guide", "tail", "coupler", "servo",
                 "cab_wheel_0R", "cab_wheel_0L", "cab_wheel_1R", "cab_wheel_1L"],
        "coupler_servo": ["coupler", "servo"],
        "rib_V": ["rib", "wire_guide", "wheel_R", "wheel_L", "horn"],
        "rib_H": ["rib_H", "horn"],
        "rib_V_head": ["rib", "wire_guide", "wheel_R", "wheel_L", "horn", "head_bottom", "head_top"],
    }
    L = S.DEFAULT_LAYOUT

    def placements(th):
        C, R = S.chain(L, th)
        pl = [{"name": "base", "comp": "base", "T": np.eye(4).tolist()}]
        for k, t in enumerate(L):
            if k:
                pl.append({"name": f"m{k:02d}{t}_coupler", "comp": "coupler_servo", "T": C[k].tolist()})
            comp = "rib_V_head" if k == len(L) - 1 else f"rib_{t}"
            pl.append({"name": f"m{k:02d}{t}_rib", "comp": comp, "T": R[k].tolist()})
        return pl

    poses = {
        "straight": placements(np.zeros(len(L))),
        "gait_A30_headlift30": placements(gait(0.9, 30, 45, 1, 30, True)),
        "coil_A55_lag90": placements(gait(1.3, 55, 90, 1, 0, True)),
    }
    with open(os.path.join(OUT, "poses.json"), "w") as fh:
        json.dump({"layout": L, "components": comps, "poses": poses}, fh)
    print("wrote", len(parts), "parts and", len(poses), "poses to", OUT)


if __name__ == "__main__":
    main()
