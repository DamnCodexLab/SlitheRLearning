"""
Fusion 360 script: play the Slitherbot lateral-undulation gait on the assembly
built by fusion_build_snake.py (same motion model as the browser viewer).

The base (caboose) is un-grounded while it plays, moved over the floor by the
wheel model, and the four servo joints are driven each frame. At the end the
snake is put back to its straight start pose and re-grounded.
"""
import math
import time

import adsk.core
import adsk.fusion

DURATION_S = 12.0
AMP_DEG, FREQ_HZ, PHASE_DEG, BIAS_DEG = 35.0, 0.5, 70.0, 0.0
DIRECTION = 1           # 1 = head first, -1 = tail first
SIM_SPEED = 1.0         # 1 = real time
RESET_AT_END = True     # put the snake back to its straight start pose
N, PITCH, AX = 4, 48.9, 5.0
CABOOSE_AXLES = (-138.8, -39.0)
CT = 0.3                # rolling vs side-slip resistance
LIMIT = math.radians(60)


def gait(t):
    A, ph, b = math.radians(AMP_DEG), math.radians(PHASE_DEG), math.radians(BIAS_DEG)
    return [max(-LIMIT, min(LIMIT, A * math.sin(2 * math.pi * FREQ_HZ * t + DIRECTION * k * ph) + b))
            for k in range(N)]


def contacts(q, th):
    c, s = math.cos(q[2]), math.sin(q[2])
    out = [(q[0] + a * c, q[1] + a * s, q[2]) for a in CABOOSE_AXLES]
    cx, cz, cp = q
    for k in range(N):
        px, pz = cx + AX * math.cos(cp), cz + AX * math.sin(cp)
        pk = cp + th[k]
        rx, rz = px - AX * math.cos(pk), pz - AX * math.sin(pk)
        out.append((px, pz, pk))
        cx, cz, cp = rx + PITCH * math.cos(pk), rz + PITCH * math.sin(pk), pk
    return out


def vel(c0, c1):
    r = []
    for (x0, z0, a), (x1, z1, _) in zip(c0, c1):
        dx, dz = x1 - x0, z1 - z0
        r += [-math.sin(a) * dx + math.cos(a) * dz, CT * (math.cos(a) * dx + math.sin(a) * dz)]
    return r


def solve3(A, b):
    for i in range(3):
        p = max(range(i, 3), key=lambda r: abs(A[r][i]))
        A[i], A[p], b[i], b[p] = A[p], A[i], b[p], b[i]
        for r in range(i + 1, 3):
            f = A[r][i] / A[i][i]
            for c in range(i, 3):
                A[r][c] -= f * A[i][c]
            b[r] -= f * b[i]
    x = [0.0] * 3
    for i in (2, 1, 0):
        x[i] = (b[i] - sum(A[i][c] * x[c] for c in range(i + 1, 3))) / A[i][i]
    return x


def step(q, th0, th1):
    c0 = contacts(q, th0)
    b = [-v for v in vel(c0, contacts(q, th1))]
    J = []
    for i in range(3):
        dq = list(q); dq[i] += 1e-5
        J.append([v / 1e-5 for v in vel(c0, contacts(dq, th0))])
    A = [[sum(J[i][r] * J[j][r] for r in range(len(b))) for j in range(3)] for i in range(3)]
    y = [sum(J[i][r] * b[r] for r in range(len(b))) for i in range(3)]
    v = solve3(A, y)
    return [q[0] + v[0], q[1] + v[1], q[2] + v[2]]


def base_matrix(q):
    m = adsk.core.Matrix3D.create()
    m.setToRotation(-q[2], adsk.core.Vector3D.create(0, 1, 0), adsk.core.Point3D.create(0, 0, 0))
    m.translation = adsk.core.Vector3D.create(q[0] / 10, 0, q[1] / 10)   # mm -> cm
    return m


def run(_context: str):
    app = adsk.core.Application.get()
    design = adsk.fusion.Design.cast(app.activeProduct)
    root = design.rootComponent
    base = root.occurrences.itemByName("base_caboose:1")
    joints = [root.joints.itemByName(f"servo_{k}") for k in range(N)]
    vp = app.activeViewport

    base.isGrounded = False
    q, t, frames = [0.0, 0.0, 0.0], 0.0, 0
    t_wall0 = time.time()
    last = t_wall0
    while time.time() - t_wall0 < DURATION_S:
        now = time.time()
        dt = min(0.1, now - last) * SIM_SPEED
        last = now
        for i in range(4):
            h = dt / 4
            q = step(q, gait(t), gait(t + h))
            t += h
        th = gait(t)
        base.transform2 = base_matrix(q)
        for k, j in enumerate(joints):
            j.jointMotion.rotationValue = -th[k]   # Fusion +Y rotation = -theta in the planar model
        vp.refresh()
        adsk.doEvents()
        frames += 1

    dist = math.hypot(q[0], q[1])
    print(f"played {DURATION_S:.0f} s, {frames} frames ({frames / DURATION_S:.1f} fps), "
          f"travelled {dist:.0f} mm, heading {math.degrees(q[2]):.0f} deg")
    if not RESET_AT_END:
        return
    for j in joints:
        j.jointMotion.rotationValue = 0.0
    base.transform2 = adsk.core.Matrix3D.create()
    base.isGrounded = True
    vp.refresh()
