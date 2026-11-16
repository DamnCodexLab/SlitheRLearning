"""
Assemble an N-node Slitherbot body from the original STLs (files/*.stl).

All body STLs share one design frame (mm):
  x : snake long axis (head toward +x, caboose/tail toward -x)
  y : up  (wheels touch ground at y = -22.5)
  z : lateral

Mating features found in the STLs:
  * Joint axis   : vertical line x = 5, z = 0 (coupler pin -> rib floor hole,
                   servo spline -> rib top-plate hole at y ~ 36).
  * Segment link : rib rear leg (x 25.2..27.7) carries the 2-hole pattern
                   (z=-3,y=18 / z=+3,y=16) that the next coupler block
                   (face at x=-21.05) bolts to -> pitch = 27.7 + 0.15 + 21.05 = 48.9
  * Wheels       : axle hole in rib bottom tabs at (x=5, y=-15), wheel z 14..19 (+ mirror)
  * Caboose      : same leg pattern ending at x=-21.2, i.e. it is the parent of node 0.

The MG90S servo is NOT in the STL set; a simplified stand-in body is generated.

Usage:
  python assemble_snake.py                       # 4 nodes, straight + serpentine pose
  python assemble_snake.py --nodes 4 --angles 30 -30 30 -30 --tail
"""
import argparse
import os

import numpy as np
import trimesh
from trimesh.transformations import rotation_matrix, translation_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = os.path.join(HERE, "files")
OUT = os.path.join(HERE, "assembly")

PITCH = 48.9                      # segment-to-segment distance along x
SERVO_X = 5.2   # servo output axis / rib pivot hole centre (measured on the rib and coupler pin)
AXIS_PT = np.array([SERVO_X, 0.0, 0.0])
AXIS_DIR = np.array([0.0, 1.0, 0.0])

COLORS = {
    "rib": [235, 232, 222, 255],
    "coupler": [215, 212, 200, 255],
    "wire_guide": [200, 200, 190, 255],
    "wheel": [110, 110, 110, 255],
    "servo": [190, 160, 90, 255],
    "servo_top": [30, 30, 30, 255],
    "caboose": [235, 232, 222, 255],
    "tail": [120, 170, 120, 160],
}


def load(name):
    return trimesh.load(os.path.join(FILES, name + ".stl"), force="mesh")


def box(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    b = trimesh.creation.box(extents=hi - lo)
    b.apply_translation((lo + hi) / 2)
    return b


def cyl_y(cx, cz, r, y0, y1, sections=40):
    c = trimesh.creation.cylinder(radius=r, height=y1 - y0, sections=sections)
    c.apply_transform(rotation_matrix(-np.pi / 2, [1, 0, 0]))  # z-axis -> y-axis
    c.apply_translation([cx, (y0 + y1) / 2, cz])
    return c


def mg90s():
    """Simplified MG90S placed where the coupler holds it.
    Body sits on coupler floor (y=0) between the walls (x=+-12),
    output spline coaxial with the joint axis (x=5, z=0)."""
    body = trimesh.util.concatenate([
        box([-11.4, 0.0, -6.1], [11.4, 22.7, 6.1]),       # case
        box([-16.2, 18.0, -6.1], [16.2, 20.5, 6.1]),      # mounting flange
        cyl_y(SERVO_X, 0.0, 6.0, 22.7, 27.0),                 # gear tower
        cyl_y(-1.0, 0.0, 3.0, 22.7, 26.0),                # secondary boss
    ])
    shaft = trimesh.util.concatenate([
        cyl_y(SERVO_X, 0.0, 2.4, 27.0, 31.5, 24),             # spline
        cyl_y(SERVO_X, 0.0, 3.5, 31.5, 34.5, 32),             # horn hub
        box([SERVO_X, 31.5, -2.5], [20.0, 33.0, 2.5]),        # single-arm horn
    ])
    return body, shaft


def joint(theta_deg):
    return rotation_matrix(np.radians(theta_deg), AXIS_DIR, AXIS_PT)


def build(n_nodes, angles, with_tail=False):
    rib, coupler = load("body_rib"), load("body_coupler")
    guide, wheel = load("body_rib_wire_guide"), load("wheel")
    servo_body, servo_shaft = mg90s()

    # wheel.stl is modelled at the caboose rear axle (x=-138.8); move it to the rib axle (x=5)
    wc = wheel.bounds.mean(0)
    wheel_r = wheel.copy(); wheel_r.apply_translation([5.0 - wc[0], 0, 0])
    wheel_l = wheel_r.copy(); wheel_l.apply_transform(np.diag([1, 1, -1, 1]))  # trimesh fixes winding

    scene = trimesh.Scene()

    def add(mesh, T, name, color):
        m = mesh.copy()
        m.visual.face_colors = COLORS[color]
        scene.add_geometry(m, node_name=name, geom_name=name, transform=T)

    W_c = np.eye(4)  # world transform of coupler_0 (fixed base)
    if with_tail:
        cab = load("caboose")
        add(cab, W_c, "caboose", "caboose")
        for ax_x in (-138.8, -39.0):  # caboose rear + front axle holes
            for side, w in (("R", wheel_r), ("L", wheel_l)):
                add(w, translation_matrix([ax_x - 5.0, 0, 0]), f"caboose_wheel_{ax_x:.0f}_{side}", "wheel")
        add(load("tail_wire_guide"), W_c, "tail_wire_guide", "wire_guide")
        add(load("tail"), W_c, "tail_skin", "tail")

    for k in range(n_nodes):
        th = angles[k] if k < len(angles) else 0.0
        W_r = W_c @ joint(th)
        add(coupler, W_c, f"n{k}_coupler", "coupler")
        add(servo_body, W_c, f"n{k}_mg90s", "servo")
        add(servo_shaft, W_r, f"n{k}_mg90s_horn", "servo_top")
        add(rib, W_r, f"n{k}_rib", "rib")
        add(guide, W_r, f"n{k}_wire_guide", "wire_guide")
        add(wheel_r, W_r, f"n{k}_wheel_R", "wheel")
        add(wheel_l, W_r, f"n{k}_wheel_L", "wheel")
        W_c = W_r @ translation_matrix([PITCH, 0, 0])
    return scene


def export(scene, stem):
    os.makedirs(OUT, exist_ok=True)
    scene.export(os.path.join(OUT, stem + ".glb"))
    scene.export(os.path.join(OUT, stem + ".3mf"))
    scene.to_mesh().export(os.path.join(OUT, stem + ".stl"))
    print("wrote", os.path.join(OUT, stem) + ".{glb,3mf,stl}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, default=4)
    ap.add_argument("--angles", type=float, nargs="*", default=None,
                    help="joint angles in degrees, one per node (default: straight + 25/-25 serpentine)")
    ap.add_argument("--tail", action="store_true", help="also attach caboose + tail skin behind node 0")
    a = ap.parse_args()
    suffix = "_with_tail" if a.tail else ""
    if a.angles is None:
        export(build(a.nodes, [0] * a.nodes, a.tail), f"snake_{a.nodes}node_straight{suffix}")
        serp = [25 * (-1) ** k for k in range(a.nodes)]
        export(build(a.nodes, serp, a.tail), f"snake_{a.nodes}node_serpentine{suffix}")
    else:
        export(build(a.nodes, a.angles, a.tail), f"snake_{a.nodes}node_custom{suffix}")
