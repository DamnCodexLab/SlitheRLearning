"""
Hard-coded sinusoidal (serpenoid) gait for the Slitherbot snake in MuJoCo.

Every V (yaw) joint follows
    theta_i(t) = A * sin(2*pi*f*t + i*phi) + bias        i = 0, 1, ... counted tail -> head
which is the same law as the browser viewer (hardware/.../assembly/viewer/index.html).
With +phi the wave travels from head to tail, which pushes the snake head-first
(the passive wheels roll along the body but grip sideways). H (pitch) joints hold 0,
except the last one, which can lift the head.

Examples (run from anywhere, inside the slitherbot-sw env):
    python software/sinusoidal/run_gait.py                         # vh9_full, 10 s, prints a summary
    python software/sinusoidal/run_gait.py --video run.mp4 --plot run.png
    python software/sinusoidal/run_gait.py --viewer                # interactive window (WSLg / desktop)
    python software/sinusoidal/run_gait.py --model vh17_full --amp 35 --phase 40
"""
import argparse
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DESCRIPTION = os.path.normpath(os.path.join(HERE, "..", "..", "hardware", "description"))
MODELS = ("vh9_full", "vh9_bare", "vh17_full", "vh17_bare")


# ---------------------------------------------------------------- gait
class SerpenoidGait:
    """Open-loop gait: joint targets depend only on time."""

    def __init__(self, layout, amp_deg=30.0, freq_hz=0.5, phase_deg=45.0, bias_deg=0.0,
                 head_lift_deg=0.0, direction=1, ramp_s=1.0, limit_rad=np.radians(55.0)):
        self.layout = layout
        self.v_idx = [k for k, c in enumerate(layout) if c == "V"]
        self.h_idx = [k for k, c in enumerate(layout) if c == "H"]
        self.A, self.f = np.radians(amp_deg), freq_hz
        self.phi, self.bias = np.radians(phase_deg), np.radians(bias_deg)
        self.lift, self.direction = np.radians(head_lift_deg), direction
        self.ramp_s, self.limit = ramp_s, limit_rad

    def __call__(self, t):
        s = min(1.0, t / self.ramp_s) if self.ramp_s > 0 else 1.0   # fade in, no jerk at start
        th = np.zeros(len(self.layout))
        for i, k in enumerate(self.v_idx):
            th[k] = s * (self.A * np.sin(2 * np.pi * self.f * t + self.direction * i * self.phi) + self.bias)
        if self.h_idx:
            th[self.h_idx[-1]] = s * self.lift
        return np.clip(th, -self.limit, self.limit)


# ---------------------------------------------------------------- simulation
def load(model_name, cone="elliptic", impratio=10.0):
    import mujoco
    m = mujoco.MjModel.from_xml_path(os.path.join(DESCRIPTION, f"snake_{model_name}.xml"))
    if cone == "elliptic":
        # Elliptic friction cones + high impratio: wheels resist sideways slip much more
        # faithfully than the default pyramidal cone (which lets contacts creep).
        m.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        m.opt.impratio = impratio
    return m


def joint_layout(m):
    import mujoco
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in range(m.nu)]
    return "".join(n.split("_")[1] for n in names), names


def run(m, gait, duration, on_step=None, control_dt=0.02):
    """Simulate; servo targets are updated at 1/control_dt Hz (MG90S PWM is 50 Hz)."""
    import mujoco
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    com0 = d.subtree_com[1].copy()                      # body 1 = base; subtree = whole snake
    log = {"t": [], "com": [], "q": [], "target": [], "force": []}
    qadr = m.jnt_qposadr[m.actuator_trnid[:, 0]]
    n_sub = max(1, round(control_dt / m.opt.timestep))
    steps = int(duration / m.opt.timestep)
    for i in range(steps):
        if i % n_sub == 0:
            d.ctrl[:] = gait(d.time)
            log["t"].append(d.time); log["com"].append(d.subtree_com[1].copy())
            log["q"].append(d.qpos[qadr].copy()); log["target"].append(d.ctrl.copy())
            log["force"].append(d.actuator_force.copy())
        mujoco.mj_step(m, d)
        if on_step is not None and on_step(d) is False:
            break
    log = {k: np.array(v) for k, v in log.items()}
    log["com0"] = com0
    return d, log


def summarize(m, log, layout):
    com = log["com"] - log["com0"]
    t = log["t"]
    half = t >= t[-1] / 2                                # steady state: second half of the run
    v = np.polyfit(t[half], com[half, 0], 1)[0] if half.sum() > 2 else float("nan")
    sat = np.abs(log["force"]) >= 0.99 * m.actuator_forcerange[:, 1]
    err = np.degrees(np.abs(log["q"] - log["target"]))
    v_cols = [k for k, c in enumerate(layout) if c == "V"]
    return {
        "forward_m": com[-1, 0], "sideways_m": com[-1, 1],
        "speed_mm_s": 1000 * v,
        "track_err_deg_mean": err[:, v_cols].mean(), "track_err_deg_max": err[:, v_cols].max(),
        "torque_saturated_pct": 100 * sat.mean(),
    }


# ---------------------------------------------------------------- outputs
def record_video(m, gait, duration, path, fps=30, width=640, height=480):
    import imageio
    import mujoco
    r = mujoco.Renderer(m, height, width)
    r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False    # shadow maps alias badly at this scale
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.azimuth, cam.elevation = 0.5 + 0.06 * m.nu, 135.0, -40.0   # longer snake, further out
    frames, next_t = [], [0.0]

    def grab(d):
        if d.time >= next_t[0]:
            cam.lookat[:] = d.subtree_com[1]
            r.update_scene(d, cam)
            frames.append(r.render().copy())
            next_t[0] += 1.0 / fps

    _, log = run(m, gait, duration, on_step=grab)
    imageio.mimwrite(path, frames, fps=fps, quality=6, macro_block_size=8)
    return log


def plot(log, layout, names, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    com = (log["com"] - log["com0"]) * 1000
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(com[:, 0], com[:, 1], lw=1.5)
    ax[0].plot(0, 0, "o", ms=5, label="start"); ax[0].plot(com[-1, 0], com[-1, 1], "s", ms=5, label="end")
    ax[0].set_aspect("equal", "datalim"); ax[0].set_xlabel("forward x (mm)"); ax[0].set_ylabel("left y (mm)")
    ax[0].set_title("centre of mass path (top view)"); ax[0].legend(); ax[0].grid(alpha=0.3)
    for k in [k for k, c in enumerate(layout) if c == "V"][:3]:
        ax[1].plot(log["t"], np.degrees(log["target"][:, k]), "--", lw=1, label=f"{names[k]} target")
        ax[1].plot(log["t"], np.degrees(log["q"][:, k]), lw=1.2, label=f"{names[k]} actual")
    ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("angle (deg)"); ax[1].set_title("first three V joints")
    ax[1].legend(fontsize=7, ncol=3); ax[1].grid(alpha=0.3)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=100)


def viewer(m, gait):
    import mujoco
    import mujoco.viewer
    d = mujoco.MjData(m)
    n_sub = max(1, round(0.02 / m.opt.timestep))
    with mujoco.viewer.launch_passive(m, d) as v:
        v.cam.distance, v.cam.elevation = 0.9, -35.0
        while v.is_running():
            t0 = time.time()
            for _ in range(n_sub):
                d.ctrl[:] = gait(d.time)
                mujoco.mj_step(m, d)
            v.cam.lookat[:] = d.subtree_com[1]
            v.sync()
            time.sleep(max(0.0, n_sub * m.opt.timestep - (time.time() - t0)))


# ---------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=MODELS, default="vh9_full")
    ap.add_argument("--amp", type=float, default=30.0, help="V joint amplitude, deg")
    ap.add_argument("--freq", type=float, default=0.5, help="wave frequency, Hz")
    ap.add_argument("--phase", type=float, default=45.0, help="phase lag between neighbouring V joints, deg")
    ap.add_argument("--bias", type=float, default=0.0, help="constant V offset (turning), deg")
    ap.add_argument("--lift", type=float, default=0.0,
                    help="head lift on the last H joint, deg (on vh9 this tips the snake so ribs and caboose drag)")
    ap.add_argument("--backward", action="store_true", help="reverse the wave (tail first)")
    ap.add_argument("--duration", type=float, default=10.0, help="simulated seconds")
    ap.add_argument("--cone", choices=("elliptic", "pyramidal"), default="elliptic", help="friction cone")
    ap.add_argument("--video", help="write an mp4 of the run (needs MUJOCO_GL=egl when headless)")
    ap.add_argument("--plot", help="write a png: centre-of-mass path + joint tracking")
    ap.add_argument("--viewer", action="store_true", help="open the interactive MuJoCo viewer instead")
    a = ap.parse_args()

    m = load(a.model, a.cone)
    layout, names = joint_layout(m)
    gait = SerpenoidGait(layout, a.amp, a.freq, a.phase, a.bias, a.lift, -1 if a.backward else 1)
    if a.viewer:
        return viewer(m, gait)

    t0 = time.time()
    if a.video:
        log = record_video(m, gait, a.duration, a.video)
    else:
        _, log = run(m, gait, a.duration)
    wall = time.time() - t0
    s = summarize(m, log, layout)
    print(f"{a.model} ({layout}), {a.duration:.0f} s, amp {a.amp} deg, {a.freq} Hz, phase {a.phase} deg, "
          f"lift {a.lift} deg, {'backward' if a.backward else 'forward'} wave, {a.cone} cone")
    print(f"  moved {1000 * s['forward_m']:+.0f} mm forward, {1000 * s['sideways_m']:+.0f} mm sideways; "
          f"steady speed {s['speed_mm_s']:+.1f} mm/s")
    print(f"  V tracking error mean {s['track_err_deg_mean']:.1f} deg, max {s['track_err_deg_max']:.1f} deg; "
          f"servo torque saturated {s['torque_saturated_pct']:.0f}% of the time")
    print(f"  simulated in {wall:.1f} s wall ({a.duration / wall:.1f}x real time)")
    if a.plot:
        plot(log, layout, names, a.plot, f"{a.model}: A={a.amp} deg, f={a.freq} Hz, phi={a.phase} deg")
    for p in (a.video, a.plot):
        if p:
            print("  wrote", os.path.abspath(p))


if __name__ == "__main__":
    main()
