"""
VH snake kinematics on the new rib (new_body_rib.stl) + body_coupler_new.stl.

Module = coupler + MG90S + rib (the rib turns on the servo).
  V module : coupler upright  -> servo axis vertical  (yaw,   design-frame y = world up)
  H module : coupler rolled 90 deg about the body axis (x) through the leg-pattern
             centre (y = 18.715, z = 0)  -> servo axis horizontal across the body (pitch)
Positive H angle lifts everything downstream (towards the head).

Design frame: x = snake axis (head at +x), y = up, z = lateral, mm.
Ground: y = -22.5 (rib wheels), caboose sits flat on it.
"""
import numpy as np
from trimesh.transformations import rotation_matrix, translation_matrix

PITCH = 48.9
AXIS_PT = np.array([5.2, 0.0, 0.0])          # servo axis (rib pivot hole / coupler pin centre)
AXIS_DIR = np.array([0.0, 1.0, 0.0])
ROLL_PT = np.array([0.0, 18.715, 0.0])       # centre of the 4-hole leg pattern
ROLL_DIR = np.array([1.0, 0.0, 0.0])
DEFAULT_LAYOUT = "VHVHVHVHVHVHVHVHV"


def roll(prev_type, this_type):
    """Relative roll of this module's coupler vs the previous rib."""
    if prev_type == this_type:
        return np.eye(4)
    ang = np.pi / 2 if this_type == "H" else -np.pi / 2
    return rotation_matrix(ang, ROLL_DIR, ROLL_PT)


def joint(theta):
    return rotation_matrix(theta, AXIS_DIR, AXIS_PT)


def chain(layout, thetas, base=np.eye(4)):
    """Return lists of 4x4 world frames: couplers[k], ribs[k] (k = module index)."""
    couplers, ribs = [], []
    C = base.copy()
    prev = "V"  # the caboose leg is upright
    for k, t in enumerate(layout):
        C = C @ roll(prev, t) if k else C
        R = C @ joint(thetas[k])
        couplers.append(C); ribs.append(R)
        C = R @ translation_matrix([PITCH, 0, 0])
        prev = t
    return couplers, ribs


def build_scene(layout=DEFAULT_LAYOUT, thetas=None):
    """trimesh Scene of the VH snake (caboose + modules + head) at the given joint angles."""
    import trimesh
    import assemble_snake as A
    from export_viewer_data import HEAD_T
    thetas = np.zeros(len(layout)) if thetas is None else np.asarray(thetas)
    rib, ribH, cpl = A.load("body_rib_V"), A.load("body_rib_H"), A.load("body_coupler_new")
    guide, wheel = A.load("body_rib_wire_guide"), A.load("wheel")
    servo, horn = A.mg90s()
    wc = wheel.bounds.mean(0)
    wr = wheel.copy(); wr.apply_translation([AXIS_PT[0] - wc[0], 0, 0])
    wl = wr.copy(); wl.apply_transform(np.diag([1, 1, -1, 1]))
    head = trimesh.util.concatenate([A.load("head_bottom"), A.load("head_top")]); head.apply_transform(HEAD_T)
    sc = trimesh.Scene()

    def add(m, T, name, color):
        m = m.copy(); m.visual.face_colors = color
        sc.add_geometry(m, node_name=name, geom_name=name, transform=T)

    white, tan, grey, gold, black = [235, 232, 222, 255], [233, 220, 194, 255], [110, 110, 110, 255], [190, 160, 90, 255], [40, 40, 40, 255]
    add(A.load("caboose_new"), np.eye(4), "caboose", white)
    add(A.load("tail"), np.eye(4), "tail", white)
    C, R = chain(layout, thetas)
    for k, t in enumerate(layout):
        tag = f"m{k:02d}{t}"
        add(cpl, C[k], tag + "_coupler", white); add(servo, C[k], tag + "_mg90s", gold)
        add(ribH if t == "H" else rib, R[k], tag + "_rib", tan if t == "H" else white)
        add(horn, R[k], tag + "_horn", black)
        if t == "V":
            add(guide, R[k], tag + "_wire_guide", white)
            add(wr, R[k], tag + "_wheel_R", grey); add(wl, R[k], tag + "_wheel_L", grey)
    add(head, R[-1], "head", black)
    return sc


if __name__ == "__main__":
    import os
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assembly")
    L = DEFAULT_LAYOUT
    lifted = np.zeros(len(L)); h = [k for k, t in enumerate(L) if t == "H"]
    lifted[h[-2]], lifted[h[-1]] = np.radians(30), np.radians(-30)
    for name, th in (("straight", np.zeros(len(L))), ("head_lift", lifted)):
        sc = build_scene(L, th)
        stem = os.path.join(out, f"snake_vh17_{name}")
        sc.export(stem + ".glb"); sc.to_mesh().export(stem + ".stl")
        print("wrote", stem + ".{glb,stl}")
