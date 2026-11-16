"""
Validation of the VH snake (new_body_rib + body_coupler_new + caboose_new):

  1. full-gait collision sweep: every pair of rigid bodies, exact mesh booleans
     (manifold3d), over whole gait cycles and joint-limit corner cases
  2. floor clearance per frame (caboose flat on the floor, y = -22.5)
  3. lift capacity: gravity torque on each H servo when cantilevering the front
     (or rear) of the body, with and without head / caboose + tail

Writes assembly/vh_validation.json and prints a summary.
"""
import itertools
import json
import os
import sys
import time

import manifold3d as mf
import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import assemble_snake as A  # noqa: E402
import snake_vh as S  # noqa: E402
from export_viewer_data import HEAD_T  # noqa: E402

LAYOUT = S.DEFAULT_LAYOUT
NM = len(LAYOUT)
V = [k for k, t in enumerate(LAYOUT) if t == "V"]
H = [k for k, t in enumerate(LAYOUT) if t == "H"]
LIM = np.radians(55)
GROUND = -22.5
OVERLAP_MM3 = 0.5          # ignore numerical slivers at bolted / touching faces
PLA = 1.24e-3              # g/mm^3
SERVO_G = 13.4
STALL = {"4.8V": 1.8, "6V": 2.2}   # MG90S stall torque, kgf*cm


# ---------------------------------------------------------------- geometry
def to_mf(m):
    m = m.copy(); m.merge_vertices()
    return mf.Manifold(mf.Mesh(vert_properties=np.asarray(m.vertices, np.float32),
                               tri_verts=np.asarray(m.faces, np.uint32)))


def union_parts(m):
    parts = [to_mf(p) for p in m.split(only_watertight=False)]
    out = parts[0]
    for p in parts[1:]:
        out = out + p
    return out


def servo_solid():
    """Servo case (fixed to the coupler) and output spline + horn (turns with the rib)."""
    b, h = A.mg90s()
    return union_parts(b), union_parts(h)


def load_parts():
    wheel = A.load("wheel"); wc = wheel.bounds.mean(0)
    wr = wheel.copy(); wr.apply_translation([S.AXIS_PT[0] - wc[0], 0, 0])
    wl = wr.copy(); wl.apply_transform(np.diag([1, 1, -1, 1]))
    head = trimesh.util.concatenate([A.load("head_bottom"), A.load("head_top")])
    head.apply_transform(HEAD_T)
    head = head.simplify_quadric_decimation(face_count=20000)
    cab_w = []
    for ax in (-138.8, -39.0):
        for s in (1, -1):
            w = wheel.copy(); w.apply_translation([ax - wc[0], 0, 0])
            if s < 0:
                w.apply_transform(np.diag([1, 1, -1, 1]))
            cab_w.append(w)
    meshes = {
        "rib": A.load("body_rib_V"), "rib_H": A.load("body_rib_H"), "coupler": A.load("body_coupler_new"),
        "guide": A.load("body_rib_wire_guide"), "wheel_R": wr, "wheel_L": wl,
        "caboose": A.load("caboose_new"), "tail": A.load("tail"), "tail_guide": A.load("tail_wire_guide"),
        "head": head,
    }
    solids = {k: to_mf(m) for k, m in meshes.items()}
    solids["servo"], solids["horn"] = servo_solid()
    for i, w in enumerate(cab_w):
        solids[f"cab_wheel{i}"] = to_mf(w)
    # rigid bodies (parts that never move relative to each other), in local frames
    sv = solids["servo"]
    bodies = {
        "base": solids["caboose"] + solids["tail_guide"] + solids["coupler"] + sv
                + solids["cab_wheel0"] + solids["cab_wheel1"] + solids["cab_wheel2"] + solids["cab_wheel3"],
        "tail": solids["tail"],
        "cpl": solids["coupler"] + sv,
        "ribV": solids["rib"] + solids["guide"] + solids["wheel_R"] + solids["wheel_L"] + solids["horn"],
        "ribH": solids["rib_H"] + solids["horn"],
        "head": solids["head"],
    }
    return bodies, meshes


def mat34(T):
    return np.asarray(T[:3, :4], np.float64)


def pose_bodies(bodies, thetas):
    C, R = S.chain(LAYOUT, thetas)
    out = [("base", bodies["base"]), ("tail", bodies["tail"])]
    for k, t in enumerate(LAYOUT):
        if k > 0:
            out.append((f"cpl{k}", bodies["cpl"].transform(mat34(C[k]))))
        out.append((f"rib{k}{t}", bodies["rib" + t].transform(mat34(R[k]))))
    out.append(("head", bodies["head"].transform(mat34(R[-1]))))
    return out


def bbox(m):
    b = m.bounding_box()
    return np.array(b[:3]), np.array(b[3:])


def collisions(posed):
    hits = []
    boxes = [bbox(m) for _, m in posed]
    for i, j in itertools.combinations(range(len(posed)), 2):
        (a0, a1), (b0, b1) = boxes[i], boxes[j]
        if np.any(a1 < b0) or np.any(b1 < a0):
            continue
        v = (posed[i][1] ^ posed[j][1]).volume()
        if v > OVERLAP_MM3:
            hits.append((posed[i][0], posed[j][0], round(v, 2)))
    lowest = min(bbox(m)[0][1] for n, m in posed if n not in ("base", "tail"))
    return hits, lowest


# ---------------------------------------------------------------- gaits
def gait(phase, amp_deg, lag_deg, lift_from=1, lift_deg=0.0, level=True, bias_deg=0.0):
    th = np.zeros(NM)
    for i, k in enumerate(V):
        th[k] = np.radians(amp_deg) * np.sin(phase + i * np.radians(lag_deg)) + np.radians(bias_deg)
    hk = H[-lift_from]
    th[hk] += np.radians(lift_deg)
    if level and lift_from > 1:
        th[H[-1]] -= np.radians(lift_deg)
    return np.clip(th, -LIM, LIM)


SCENARIOS = [
    # name, amp, lag, lift_from, lift, level
    ("viewer default (A30, lag45, head lift 30 at H8)", 30, 45, 1, 30, True),
    ("max yaw amplitude (A55, lag45)", 55, 45, 1, 0, True),
    ("tight coil (A55, lag90)", 55, 90, 1, 0, True),
    ("tightest coil (A55, lag120)", 55, 120, 1, 0, True),
    ("max head lift while waving (A55, lag45, H8 55)", 55, 45, 1, 55, True),
    ("lift from H7 30, level (A30)", 30, 45, 2, 30, True),
    ("lift from H5 55, level (A40, lag60)", 40, 60, 4, 55, True),
]
FRAMES = 36


def corner_cases():
    """Every pair of neighbouring joints at their limits (+-55 each), others straight."""
    cases = []
    for k in range(NM - 1):
        for a, b in itertools.product((-55, 55), repeat=2):
            th = np.zeros(NM); th[k] = np.radians(a); th[k + 1] = np.radians(b)
            cases.append((f"{LAYOUT[k]}{k}={a:+d}, {LAYOUT[k+1]}{k+1}={b:+d}", th))
    # three in a row, same sign (tightest local bend)
    for k in range(NM - 2):
        for s in (-55, 55):
            th = np.zeros(NM); th[k:k + 3] = np.radians(s)
            cases.append((f"{LAYOUT[k:k+3]} at {k}..{k+2} all {s:+d}", th))
    return cases


# ---------------------------------------------------------------- torque
def masses(meshes):
    g = {k: abs(m.volume) * PLA for k, m in meshes.items() if k not in ("head",)}
    head = trimesh.util.concatenate([A.load("head_bottom"), A.load("head_top")]); head.apply_transform(HEAD_T)
    hb, ht = A.load("head_bottom"), A.load("head_top")
    head_mass = (abs(hb.volume) + abs(ht.volume)) * PLA
    head_com = (hb.center_mass * abs(hb.volume) + ht.center_mass * abs(ht.volume)) / (abs(hb.volume) + abs(ht.volume))
    head_com = trimesh.transformations.transform_points([head_com], HEAD_T)[0]
    def mc(name):
        m = meshes[name]
        return abs(m.volume) * PLA, m.center_mass
    b, h = A.mg90s()
    return {
        "cpl": [mc("coupler"), (SERVO_G, b.bounds.mean(0))],
        "ribV": [mc("rib"), mc("guide"), mc("wheel_R"), mc("wheel_L"), (1.0, h.bounds.mean(0))],
        "ribH": [mc("rib_H"), (1.0, h.bounds.mean(0))],
        "head": [(head_mass, head_com)],
    }


def lift_torques(M, with_head, from_front=True):
    """Holding torque (kgf*cm) on each H joint when it alone cantilevers everything
    beyond it horizontally (worst case), front or rear."""
    th = np.zeros(NM)
    C, R = S.chain(LAYOUT, th)
    res = []
    for hk in H:
        ax = trimesh.transformations.transform_points([S.AXIS_PT], R[hk])[0]
        items = []
        mods = range(hk, NM) if from_front else range(0, hk + 1)
        for k in mods:
            if from_front:
                if k > hk:
                    items += [(m, c, C[k]) for m, c in M["cpl"]]
                items += [(m, c, R[k]) for m, c in M["rib" + LAYOUT[k]]]
            else:  # rear part rotates with the coupler side of joint hk: modules < hk, and coupler hk
                if k < hk:
                    items += [(m, c, R[k]) for m, c in M["rib" + LAYOUT[k]]]
                if k > 0:
                    items += [(m, c, C[k]) for m, c in M["cpl"]]
        if from_front and with_head:
            items += [(m, c, R[-1]) for m, c in M["head"]]
        tq, mass = 0.0, 0.0
        for m, c, T in items:
            p = trimesh.transformations.transform_points([c], T)[0]
            tq += m * (p[0] - ax[0]); mass += m
        res.append({"joint": f"H{H.index(hk) + 1}", "module": hk, "mass_g": round(mass, 1),
                    "kgf_cm": round(abs(tq) / 10000, 2),
                    "modules_lifted": (NM - hk) if from_front else (hk + 1)})
    return res


def main():
    t0 = time.time()
    bodies, meshes = load_parts()
    report = {"scenarios": [], "corners": {}, "lift": {}}

    # straight pose sanity
    hits, low = collisions(pose_bodies(bodies, np.zeros(NM)))
    report["straight"] = {"collisions": hits, "lowest_y": round(float(low), 2)}
    print(f"straight: {len(hits)} collisions, lowest point y={low:.1f}")

    for name, amp, lag, lf, lift, level in SCENARIOS:
        worst, all_hits = GROUND + 100, {}
        for f in range(FRAMES):
            th = gait(2 * np.pi * f / FRAMES, amp, lag, lf, lift, level)
            hits, low = collisions(pose_bodies(bodies, th))
            worst = min(worst, low)
            for a, b, v in hits:
                key = f"{a}~{b}"
                all_hits[key] = max(all_hits.get(key, 0), v)
        report["scenarios"].append({"name": name, "frames": FRAMES, "collisions": all_hits,
                                    "lowest_y": round(float(worst), 2), "floor_hit": bool(worst < GROUND - 0.5)})
        print(f"{name}: {len(all_hits)} colliding pairs {all_hits if all_hits else ''}; lowest y {worst:.1f}"
              f"{'  ** below floor **' if worst < GROUND - 0.5 else ''}  [{time.time() - t0:.0f}s]")

    cc = corner_cases()
    bad = []
    for label, th in cc:
        hits, low = collisions(pose_bodies(bodies, th))
        if hits:
            bad.append({"case": label, "collisions": hits})
    report["corners"] = {"cases": len(cc), "colliding": bad}
    print(f"corner cases: {len(cc)} checked, {len(bad)} with collisions  [{time.time() - t0:.0f}s]")
    for b in bad[:20]:
        print("   ", b["case"], b["collisions"])

    M = masses(meshes)
    for label, wh, ff in (("front, with head", True, True), ("front, no head", False, True),
                          ("rear (no caboose/tail)", False, False)):
        report["lift"][label] = lift_torques(M, wh, ff)
    print("lift torques (kgf*cm, horizontal cantilever, worst case):")
    for label, rows in report["lift"].items():
        print("  ", label)
        for r in rows:
            print(f"      {r['joint']} lifts {r['modules_lifted']:2d} modules, {r['mass_g']:6.1f} g -> {r['kgf_cm']:5.2f} kgf*cm "
                  f"({r['kgf_cm'] / STALL['4.8V'] * 100:4.0f}% of 4.8V stall)")
    out = os.path.join(HERE, "assembly", "vh_validation.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=1)
    print("wrote", out, f"in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
