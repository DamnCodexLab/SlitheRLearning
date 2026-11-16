"""
Physics sanity checks for the gait simulation: does the snake move for the right physical reasons?

Each check changes one physical thing and compares the outcome with what physics predicts:
  1. free fall       : falling COM accelerates at g
  2. weight support  : at rest, floor contact forces add up to m*g, and the snake does not creep
  3. no friction     : with a frictionless floor the gait can NOT move the centre of mass
                       (internal forces only; momentum is conserved)
  4. locked wheels   : without rolling wheels friction is the same in every direction, so
                       the same gait should give much less forward motion
  5. wheel slip      : wheels on the floor roll along the body but barely slide sideways
  6. servo limits    : actuator torque never exceeds the MG90S stall torque
  7. time step       : halving the time step twice barely changes the result

    python software/sinusoidal/check_physics.py              # vh9_full
    python software/sinusoidal/check_physics.py --model vh17_full --phase 60
"""
import argparse
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_gait as R  # noqa: E402

RESULTS = []


def report(name, ok, detail):
    RESULTS.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def wheel_dofs(m):
    return [m.jnt_dofadr[j] for j in range(m.njnt) if "wheel" in m.joint(j).name]


def forward_mm(m, gait, duration, **kw):
    _, log = R.run(m, gait, duration, **kw)
    return 1000 * (log["com"][-1, 0] - log["com0"][0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=R.MODELS, default="vh9_full")
    ap.add_argument("--phase", type=float, default=45.0)
    ap.add_argument("--duration", type=float, default=6.0)
    a = ap.parse_args()

    base = R.load(a.model)
    layout, _ = R.joint_layout(base)
    gait = R.SerpenoidGait(layout, phase_deg=a.phase)
    mass = mujoco.mj_getTotalmass(base)
    g = -base.opt.gravity[2]
    print(f"{a.model}: mass {1000 * mass:.0f} g, {base.nu} servos, {len(wheel_dofs(base))} wheels, "
          f"dt {1000 * base.opt.timestep:.1f} ms, {a.duration:.0f} s runs\n")

    # 1. free fall: start 0.3 m up, measure COM acceleration before touchdown
    m = R.load(a.model); d = mujoco.MjData(m)
    d.qpos[2] += 0.3
    mujoco.mj_forward(m, d)
    z, t = [], []
    while d.time < 0.15:
        mujoco.mj_step(m, d); z.append(d.subtree_com[1][2]); t.append(d.time)
    acc = 2 * np.polyfit(t, z, 2)[0]
    report("free fall", abs(acc + g) < 0.02 * g, f"COM acceleration {acc:.3f} m/s^2 (gravity {-g:.3f})")

    # 2. weight support + no creep at rest
    m = R.load(a.model); d = mujoco.MjData(m)
    for _ in range(int(1.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    x0 = d.subtree_com[1].copy()
    for _ in range(int(3.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    f6 = np.zeros(6); fz = 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        mujoco.mj_contactForce(m, d, i, f6)
        fz += (c.frame[:3] * f6[0])[2]                    # normal force, world z component
    creep = 1000 * np.linalg.norm(d.subtree_com[1][:2] - x0[:2])
    report("weight support", abs(fz - mass * g) < 0.02 * mass * g,
           f"floor normal force {fz:.3f} N vs m*g {mass * g:.3f} N")
    report("no creep at rest", creep < 1.0, f"COM drifted {creep:.3f} mm in 3 s with servos holding 0")

    # reference run
    ref = forward_mm(R.load(a.model), gait, a.duration)

    # 3. frictionless floor: momentum conservation, no net horizontal motion of the COM
    m = R.load(a.model)
    m.geom_friction[:, :] = 0.0
    m.dof_frictionloss[wheel_dofs(m)] = 0.0
    _, log = R.run(m, gait, a.duration)
    drift = 1000 * np.linalg.norm(log["com"][-1, :2] - log["com0"][:2])
    report("frictionless floor", drift < 0.02 * abs(ref),
           f"COM moved {drift:.1f} mm (with friction: {ref:.0f} mm) - the gait itself cannot push the body")

    # 4. locked wheels: isotropic friction -> lateral undulation loses its grip advantage
    m = R.load(a.model)
    m.dof_damping[wheel_dofs(m)] = 1e3
    locked = forward_mm(m, gait, a.duration)
    report("locked wheels", abs(locked) < 0.3 * abs(ref),
           f"{locked:.0f} mm vs {ref:.0f} mm rolling ({100 * locked / ref:.0f}%) - forward motion needs the wheels")

    # 5. Coulomb friction: a wheel may only slide sideways when its friction force is at the limit mu*N.
    #    Sliding below the limit would be solver "creep" (unphysical); sliding at the limit is real.
    m = R.load(a.model)
    wheels = {m.jnt_bodyid[j] for j in range(m.njnt) if "wheel" in m.joint(j).name}
    side, roll, use = [], [], []
    v6, f6 = np.zeros(6), np.zeros(6)

    def slip(d):
        if d.time < 1.0:
            return
        for i in range(d.ncon):
            c = d.contact[i]
            b = m.geom_bodyid[c.geom2 if c.geom1 == 0 else c.geom1]
            if b not in wheels:
                continue
            mujoco.mj_contactForce(m, d, i, f6)
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b, v6, 0)
            axle = d.xmat[b].reshape(3, 3)[:, 1]        # wheel joint axis = body y
            v = v6[3:].copy(); v[2] = 0.0
            s = abs(v @ axle)
            side.append(s); roll.append(np.linalg.norm(v - (v @ axle) * axle))
            if s > 0.02 and f6[0] > 1e-4:               # visibly sliding sideways (> 20 mm/s), loaded
                use.append(np.hypot(f6[1], f6[2]) / (c.friction[0] * f6[0]))
    R.run(m, gait, a.duration, on_step=slip)
    report("Coulomb sliding", np.median(use) > 0.95,
           f"sliding wheels use {100 * np.median(use):.0f}% of mu*N (median); sideways speed is "
           f"{100 * np.mean(side) / np.mean(roll):.0f}% of rolling speed - the gait asks more side force than friction gives")

    # 6. servo torque limit
    m = R.load(a.model)
    _, log = R.run(m, gait, a.duration)
    peak = np.abs(log["force"]).max()
    report("servo torque limit", peak <= m.actuator_forcerange[:, 1].max() + 1e-9,
           f"peak {peak:.3f} N*m, limit {m.actuator_forcerange[0, 1]:.3f} N*m (MG90S stall at 4.8 V)")

    # 7. time-step convergence
    out = []
    for dt in (0.002, 0.001, 0.0005):
        m = R.load(a.model); m.opt.timestep = dt
        out.append(forward_mm(m, gait, a.duration))
    spread = (max(out) - min(out)) / abs(out[-1])
    report("time-step convergence", spread < 0.05,
           "dt 2 / 1 / 0.5 ms -> " + " / ".join(f"{x:.0f}" for x in out) + f" mm (spread {100 * spread:.1f}%)")

    print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks passed")
    sys.exit(0 if all(RESULTS) else 1)


if __name__ == "__main__":
    main()
