"""
Generate the snake robot description for simulation, from the CAD in
hardware/mechanical/pla-mg90s-morphism (run export_fusion_vh.py there first if parts changed).

Outputs (next to this script):
    snake_vh17_full.urdf / .xml   17 joints, caboose + tail + head
    snake_vh17_bare.urdf / .xml   17 joints, caboose only (no tail shell, no head)
    snake_vh9_full.urdf  / .xml   9 joints  (VHVHVHVHV)
    snake_vh9_bare.urdf  / .xml
    meshes/*.stl                  visual meshes, metres, shared by all files

  .urdf  -> Isaac Sim / Isaac Lab (URDF importer); also any URDF tool
  .xml   -> MuJoCo (MJCF, with floor, position servos and TCRT5000 rangefinders)

Conventions
  World: x forward (towards the head), y left, z up, metres, floor at z = 0.
  Links: base (caboose), m01 .. mNN (module k = rib k + the next module's coupler and servo).
  Joints j01_V .. jNN_x numbered tail -> head; every servo joint turns about its link's +z.
         V joints: +angle turns the downstream body to the left (yaw).
         H joints: +angle lifts the downstream body (pitch up).
  Wheels: passive continuous joints (V modules and caboose); they give the sideways grip.
  IR (TCRT5000): one board per side on every rib's M3 pad. The sensing point is offset from
         the screw hole along the board (IR_OFFSET) and stands off the pad (IR_STANDOFF).
         URDF frame ir_*: +x points outward.  MJCF site ir_*: +z points outward (rangefinder axis).
  Collision: simple boxes / cylinders; visual: the CAD meshes. Robot self-collision is off in the
         MJCF (joint limits already keep the parts clear; checked in analyze_vh.py and Fusion).

Masses are ESTIMATES: printed parts as solid PLA (1240 kg/m^3), MG90S 13.4 g, horn 1 g,
TCRT5000 module 3 g. Replace with measured values in MASS_OVERRIDE_KG when available.
"""
import os
import sys

import numpy as np
import trimesh
from trimesh.transformations import quaternion_from_matrix, euler_from_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
MECH = os.path.join(HERE, "..", "mechanical", "pla-mg90s-morphism")
PARTS = os.path.join(MECH, "assembly", "parts_vh")
sys.path.insert(0, MECH)
import snake_vh as S  # noqa: E402

MM = 1e-3
VARIANTS = {"vh17": "VHVHVHVHVHVHVHVHV", "vh9": "VHVHVHVHV"}

# --- physical parameters (edit here) -----------------------------------------------------
PLA_DENSITY = 1240.0                       # kg/m^3, solid; real prints are lighter (infill)
PART_MASS_KG = {"servo": 0.0134, "horn": 0.001}
MASS_OVERRIDE_KG = {}                      # e.g. {"rib": 0.0042}: measured mass per part
JOINT_LIMIT = np.radians(55.0)             # rib hits its coupler at 60 deg
SERVO_EFFORT = 0.176                       # N*m, MG90S stall 1.8 kgf*cm at 4.8 V (0.216 at 6 V)
SERVO_VELOCITY = 10.5                      # rad/s, 0.1 s / 60 deg at 4.8 V
SERVO_KP = 2.0                             # N*m/rad position gain (saturates at ~5 deg error)
SERVO_DAMPING = 0.01                       # N*m*s/rad
SERVO_ARMATURE = 2e-4                      # kg*m^2, reflected rotor inertia (guess)
SERVO_FRICTION = 0.005                     # N*m
WHEEL_DAMPING = 1e-5
IR_MASS = 0.003                            # TCRT5000 module
IR_BOARD = (0.032, 0.014, 0.0016)          # board length, width, thickness (m)
IR_HOLE_FROM_END = 0.0035                  # M3 hole centre from the board end
IR_OFFSET = 0.025                          # screw hole -> TCRT5000 optical centre, along the board
IR_STANDOFF = 0.0016 + 0.007               # pad face -> sensor face (board + TCRT5000 body)
IR_RANGE = 0.025                           # TCRT5000 useful range ~1-25 mm

# design (CAD) frame -> world: x = x, y = -z_cad, z = y_cad, floor (y_cad = -22.5) -> z = 0
N = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1.0]])
M = N.copy(); M[2, 3] = 22.5 * MM


def T(xyz):
    t = np.eye(4); t[:3, 3] = xyz; return t


def mm(D):
    """CAD transform with mm translation -> metres."""
    D = np.array(D, float); D[:3, 3] *= MM; return D


AXIS = np.asarray(S.AXIS_PT) * MM
WHEEL_AXLE = np.array([5.2, -15.0, 16.5]) * MM
WHEEL_R, WHEEL_W = 7.5 * MM, 5.0 * MM
CAB_AXLES_X = (-138.8 * MM, -39.0 * MM)
# IR pads (rib CAD frame, mm): hole centre on the pad face, outward normal, "up along the board"
IR_V = [((5.2, 4.0, s * 24.6), (0, 0, s), (0, 1, 0)) for s in (1, -1)]
IR_H = [((5.2, -10.8, 8.5), (0, -1, 0), (0, 0, -1)), ((5.2, 40.6, 8.5), (0, 1, 0), (0, 0, -1))]


# --- meshes and mass properties ----------------------------------------------------------
_mesh_cache = {}


def mesh(name):
    if name not in _mesh_cache:
        m = trimesh.load(os.path.join(PARTS, name + ".stl"), force="mesh")
        m.apply_scale(MM)
        _mesh_cache[name] = m
    return _mesh_cache[name]


def part_mass(name):
    m = mesh(name)
    if name in MASS_OVERRIDE_KG:
        return MASS_OVERRIDE_KG[name]
    base = name.split("_")[0] if name.startswith("wheel") else name
    return PART_MASS_KG.get(base, PLA_DENSITY * m.volume)


def box_inertia(mass, size):
    a, b, c = size
    return mass / 12.0 * np.diag([b * b + c * c, a * a + c * c, a * a + b * b])


class Link:
    def __init__(self, name, W):
        self.name, self.W = name, W           # W: world pose at zero joint angles
        self.visuals, self.boxes, self.cylinders = [], [], []
        self.masses = []                      # (mass, com_local, inertia_at_com_local)

    def add_part(self, part, D, coll="aabb"):
        """Add CAD part `part` placed at CAD pose D (metres) in the CAD world."""
        L = np.linalg.inv(self.W) @ M @ D
        self.visuals.append((part, L))
        m = mesh(part).copy(); m.apply_transform(L)
        mass = part_mass(part)
        m.density = mass / m.volume
        self.masses.append((mass, m.center_mass, m.moment_inertia))
        if coll == "aabb":
            lo, hi = mesh(part).bounds
            self.boxes.append((L @ T((lo + hi) / 2), hi - lo))
        elif coll == "rib":   # band (with side pads) + leg
            lo, hi = mesh(part).bounds
            for x0, x1 in ((lo[0], 10.3 * MM), (20.0 * MM, hi[0])):
                if x1 > x0:
                    m2 = mesh(part).slice_plane([x0, 0, 0], [1, 0, 0]).slice_plane([x1, 0, 0], [-1, 0, 0])
                    l2, h2 = m2.bounds
                    self.boxes.append((L @ T((l2 + h2) / 2), h2 - l2))

    def add_box_mass(self, mass, L, size):
        R = L[:3, :3]
        self.masses.append((mass, L[:3, 3], R @ box_inertia(mass, size) @ R.T))

    def inertial(self):
        tot = sum(m for m, _, _ in self.masses)
        com = sum(m * c for m, c, _ in self.masses) / tot
        I = np.zeros((3, 3))
        for m, c, Ic in self.masses:
            d = c - com
            I += Ic + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
        return tot, com, I


class Joint:
    def __init__(self, name, kind, parent, child, axis=(0, 0, 1)):
        self.name, self.kind, self.parent, self.child, self.axis = name, kind, parent, child, axis


def build(layout, full):
    """Links and joints of one variant (all poses at zero joint angles)."""
    links, joints, irs = {}, [], []
    C, R = S.chain(layout, np.zeros(len(layout)))
    C = [mm(c) for c in C]; R = [mm(r) for r in R]

    def wheel_links(parent, Dframe, axle, tag):
        for side, z in (("L", -1), ("R", 1)):    # CAD +z is world right (y = -z_cad)
            a = np.array([axle[0], axle[1], z * axle[2]])
            W = M @ Dframe @ T(a) @ np.linalg.inv(N)
            w = Link(f"{tag}_wheel_{side}", W)
            src = "wheel_R" if side == "R" else "wheel_L"
            # wheel meshes sit at the rib axle (x = 5.2 mm); shift to this axle
            w.add_part(src, Dframe @ T([a[0] - AXIS[0], 0, 0]), coll=None)
            w.cylinders.append((np.eye(4) @ _rot_x(np.pi / 2), WHEEL_R, WHEEL_W))
            links[w.name] = w
            joints.append(Joint(f"{tag}_wheel_{side}", "continuous", parent, w.name, axis=(0, 1, 0)))

    # base = caboose (+ tail) + first coupler and servo
    base = Link("base", M @ np.linalg.inv(N))
    base.add_part("caboose", np.eye(4))
    if full:
        base.add_part("tail", np.eye(4)); base.add_part("tail_guide", np.eye(4), coll=None)
    base.add_part("coupler", C[0]); base.add_part("servo", C[0], coll=None)
    links["base"] = base
    for i, x in enumerate(CAB_AXLES_X):
        wheel_links("base", np.eye(4), np.array([x, WHEEL_AXLE[1], WHEEL_AXLE[2]]),
                    f"base_{'rear' if i == 0 else 'front'}")

    parent = "base"
    n = len(layout)
    for k, t in enumerate(layout):
        name = f"m{k + 1:02d}"
        lk = Link(name, M @ R[k] @ T(AXIS) @ np.linalg.inv(N))
        lk.add_part("rib" if t == "V" else "rib_H", R[k], coll="rib")
        lk.add_part("horn", R[k], coll=None)
        if t == "V":
            lk.add_part("wire_guide", R[k], coll=None)
        if k + 1 < n:
            lk.add_part("coupler", C[k + 1]); lk.add_part("servo", C[k + 1], coll=None)
        elif full:
            lk.add_part("head_bottom", R[k]); lk.add_part("head_top", R[k])
        links[name] = lk
        joints.append(Joint(f"j{k + 1:02d}_{t}", "revolute", parent, name))
        if t == "V":
            wheel_links(name, R[k], WHEEL_AXLE, name)
        # IR boards, one per side
        for hole, normal, up in (IR_V if t == "V" else IR_H):
            hole, normal, up = np.array(hole) * MM, np.array(normal, float), np.array(up, float)
            Rw = (M @ R[k])[:3, :3]
            side = "L" if (Rw @ normal)[1] > 0 else "R"
            ex, ey = normal, up                       # frame: x outward, y along the board, z = x cross y
            F = np.eye(4); F[:3, :3] = np.c_[ex, ey, np.cross(ex, ey)]
            sensor = R[k] @ T(hole + normal * IR_STANDOFF + up * IR_OFFSET) @ F
            board_c = R[k] @ T(hole + normal * IR_BOARD[2] / 2 + up * (IR_BOARD[0] / 2 - IR_HOLE_FROM_END)) @ F
            Ls = np.linalg.inv(lk.W) @ M @ sensor
            Lb = np.linalg.inv(lk.W) @ M @ board_c
            lk.add_box_mass(IR_MASS, Lb, (IR_BOARD[2], IR_BOARD[0], IR_BOARD[1]))
            irs.append({"name": f"{name}_ir_{side}", "link": name, "L": Ls, "board": Lb})
        parent = name
    return links, joints, irs


def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1.0]])


def fmt(v):
    return " ".join(f"{x:.6g}" for x in v)


def rpy(L):
    return euler_from_matrix(L[:3, :3], "sxyz")


# --- URDF ------------------------------------------------------------------------------------
def urdf(name, links, joints, irs):
    out = [f'<?xml version="1.0"?>', f'<!-- generated by make_description.py; do not edit by hand -->',
           f'<robot name="{name}">']
    for lk in links.values():
        mass, com, I = lk.inertial()
        out.append(f'  <link name="{lk.name}">')
        out.append(f'    <inertial><origin xyz="{fmt(com)}" rpy="0 0 0"/><mass value="{mass:.6g}"/>'
                   f'<inertia ixx="{I[0,0]:.6g}" ixy="{I[0,1]:.6g}" ixz="{I[0,2]:.6g}" iyy="{I[1,1]:.6g}" '
                   f'iyz="{I[1,2]:.6g}" izz="{I[2,2]:.6g}"/></inertial>')
        for part, L in lk.visuals:
            out.append(f'    <visual><origin xyz="{fmt(L[:3, 3])}" rpy="{fmt(rpy(L))}"/>'
                       f'<geometry><mesh filename="meshes/{part}.stl"/></geometry></visual>')
        for L, size in lk.boxes:
            out.append(f'    <collision><origin xyz="{fmt(L[:3, 3])}" rpy="{fmt(rpy(L))}"/>'
                       f'<geometry><box size="{fmt(size)}"/></geometry></collision>')
        for L, r, w in lk.cylinders:
            out.append(f'    <collision><origin xyz="{fmt(L[:3, 3])}" rpy="{fmt(rpy(L))}"/>'
                       f'<geometry><cylinder radius="{r:.6g}" length="{w:.6g}"/></geometry></collision>')
        for ir in irs:
            if ir["link"] == lk.name:
                b = ir["board"]
                out.append(f'    <visual><origin xyz="{fmt(b[:3, 3])}" rpy="{fmt(rpy(b))}"/>'
                           f'<geometry><box size="{fmt((IR_BOARD[2], IR_BOARD[0], IR_BOARD[1]))}"/></geometry></visual>')
        out.append('  </link>')
    for ir in irs:
        out.append(f'  <link name="{ir["name"]}"/>')
    for j in joints:
        L = np.linalg.inv(links[j.parent].W) @ links[j.child].W
        out.append(f'  <joint name="{j.name}" type="{j.kind}"><parent link="{j.parent}"/><child link="{j.child}"/>'
                   f'<origin xyz="{fmt(L[:3, 3])}" rpy="{fmt(rpy(L))}"/><axis xyz="{fmt(j.axis)}"/>')
        if j.kind == "revolute":
            out.append(f'    <limit lower="{-JOINT_LIMIT:.6g}" upper="{JOINT_LIMIT:.6g}" effort="{SERVO_EFFORT}" velocity="{SERVO_VELOCITY}"/>'
                       f'<dynamics damping="{SERVO_DAMPING}" friction="{SERVO_FRICTION}"/>')
        else:
            out.append(f'    <dynamics damping="{WHEEL_DAMPING}"/>')
        out.append('  </joint>')
    for ir in irs:
        L = ir["L"]
        out.append(f'  <joint name="{ir["name"]}_joint" type="fixed"><parent link="{ir["link"]}"/><child link="{ir["name"]}"/>'
                   f'<origin xyz="{fmt(L[:3, 3])}" rpy="{fmt(rpy(L))}"/></joint>')
    out.append('</robot>')
    return "\n".join(out) + "\n"


# --- MJCF ------------------------------------------------------------------------------------
def quat(L):
    return quaternion_from_matrix(L)          # w x y z, MuJoCo order


def mjcf(name, links, joints, irs):
    children = {}
    for j in joints:
        children.setdefault(j.parent, []).append(j)
    meshes = sorted({p for lk in links.values() for p, _ in lk.visuals})
    out = [f'<!-- generated by make_description.py; do not edit by hand -->',
           f'<mujoco model="{name}">',
           '  <compiler angle="radian" meshdir="meshes" autolimits="true"/>',
           '  <option timestep="0.002" integrator="implicitfast"/>',
           '  <default>',
           f'    <joint armature="{SERVO_ARMATURE}" damping="{SERVO_DAMPING}" frictionloss="{SERVO_FRICTION}"/>',
           '    <geom contype="1" conaffinity="0" condim="3" friction="1 0.005 0.0001"/>',
           '    <default class="visual"><geom type="mesh" contype="0" conaffinity="0" group="1" density="0" rgba="0.92 0.9 0.85 1"/></default>',
           '    <default class="collision"><geom group="3" density="0" rgba="0.3 0.5 0.9 0.4"/></default>',
           '    <default class="wheel"><joint armature="1e-7" damping="' + f"{WHEEL_DAMPING}" + '" frictionloss="0"/></default>',
           '  </default>',
           '  <asset>']
    out += [f'    <mesh name="{m}" file="{m}.stl"/>' for m in meshes]
    out += ['    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1="0.2 0.3 0.4" rgb2="0.3 0.4 0.5"/>',
            '    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.1"/>',
            '  </asset>',
            '  <worldbody>',
            '    <light pos="0 0 2" dir="0 0 -1" directional="true"/>',
            '    <geom name="floor" type="plane" size="0 0 0.05" material="grid" contype="0" conaffinity="1"/>']

    def body(lname, L, joint, ind):
        lk = links[lname]
        pad = "  " * ind
        mass, com, I = lk.inertial()
        out.append(f'{pad}<body name="{lname}" pos="{fmt(L[:3, 3])}" quat="{fmt(quat(L))}">')
        if joint is None:
            out.append(f'{pad}  <freejoint name="root"/>')
        elif joint.kind == "revolute":
            out.append(f'{pad}  <joint name="{joint.name}" axis="{fmt(joint.axis)}" range="{-JOINT_LIMIT:.6g} {JOINT_LIMIT:.6g}"/>')
        else:
            out.append(f'{pad}  <joint name="{joint.name}" class="wheel" axis="{fmt(joint.axis)}"/>')
        out.append(f'{pad}  <inertial pos="{fmt(com)}" mass="{mass:.6g}" fullinertia="{I[0,0]:.6g} {I[1,1]:.6g} {I[2,2]:.6g} {I[0,1]:.6g} {I[0,2]:.6g} {I[1,2]:.6g}"/>')
        for part, Lv in lk.visuals:
            out.append(f'{pad}  <geom class="visual" mesh="{part}" pos="{fmt(Lv[:3, 3])}" quat="{fmt(quat(Lv))}"/>')
        for Lb, size in lk.boxes:
            out.append(f'{pad}  <geom class="collision" type="box" size="{fmt(np.asarray(size) / 2)}" pos="{fmt(Lb[:3, 3])}" quat="{fmt(quat(Lb))}"/>')
        for Lc, r, w in lk.cylinders:
            out.append(f'{pad}  <geom class="collision" type="cylinder" size="{r:.6g} {w / 2:.6g}" pos="{fmt(Lc[:3, 3])}" quat="{fmt(quat(Lc))}"/>')
        for ir in irs:
            if ir["link"] == lname:
                b, s = ir["board"], ir["L"]
                out.append(f'{pad}  <geom class="visual" type="box" size="{fmt(np.array((IR_BOARD[2], IR_BOARD[0], IR_BOARD[1])) / 2)}" '
                           f'pos="{fmt(b[:3, 3])}" quat="{fmt(quat(b))}" rgba="0.1 0.3 0.8 1"/>')
                # site z must point outward: rotate the (x-outward) sensor frame by +90 deg about y
                sz = s @ np.array([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1.0]])
                out.append(f'{pad}  <site name="{ir["name"]}" pos="{fmt(sz[:3, 3])}" quat="{fmt(quat(sz))}" size="0.002" rgba="1 0 0 1"/>')
        for j in children.get(lname, []):
            body(j.child, np.linalg.inv(lk.W) @ links[j.child].W, j, ind + 1)
        out.append(f'{pad}</body>')

    body("base", links["base"].W @ np.eye(4), None, 2)
    out.append('  </worldbody>')
    out.append('  <actuator>')
    for j in joints:
        if j.kind == "revolute":
            out.append(f'    <position name="{j.name}" joint="{j.name}" kp="{SERVO_KP}" ctrlrange="{-JOINT_LIMIT:.6g} {JOINT_LIMIT:.6g}" '
                       f'forcerange="{-SERVO_EFFORT} {SERVO_EFFORT}"/>')
    out.append('  </actuator>')
    out.append('  <sensor>')
    for j in joints:
        if j.kind == "revolute":
            out.append(f'    <jointpos name="{j.name}_pos" joint="{j.name}"/>')
    for ir in irs:
        out.append(f'    <rangefinder name="{ir["name"]}" site="{ir["name"]}" cutoff="{IR_RANGE}"/>')
    out.append('    <framequat name="base_quat" objtype="body" objname="base"/>')
    out.append('  </sensor>')
    out.append('</mujoco>')
    return "\n".join(out) + "\n"


def main():
    os.makedirs(os.path.join(HERE, "meshes"), exist_ok=True)
    used = set()
    for var, layout in VARIANTS.items():
        for full in (True, False):
            name = f"snake_{var}_{'full' if full else 'bare'}"
            links, joints, irs = build(layout, full)
            with open(os.path.join(HERE, name + ".urdf"), "w", encoding="utf-8") as fh:
                fh.write(urdf(name, links, joints, irs))
            with open(os.path.join(HERE, name + ".xml"), "w", encoding="utf-8") as fh:
                fh.write(mjcf(name, links, joints, irs))
            used |= {p for lk in links.values() for p, _ in lk.visuals}
            tot = sum(lk.inertial()[0] for lk in links.values()) + 0
            nj = sum(j.kind == "revolute" for j in joints)
            print(f"{name}: {len(links)} links, {nj} servo joints, {len(joints) - nj} wheel joints, "
                  f"{len(irs)} IR sensors, mass {tot * 1000:.0f} g")
    for p in sorted(used):
        mesh(p).export(os.path.join(HERE, "meshes", p + ".stl"))
    print("meshes:", sorted(used))


if __name__ == "__main__":
    main()
