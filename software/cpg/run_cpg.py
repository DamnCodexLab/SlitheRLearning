"""
Drive the Slitherbot in MuJoCo with the CPG (cpg.py).

    python software/cpg/run_cpg.py                       # constant parameters, like run_gait.py
    python software/cpg/run_cpg.py --random-phase        # start from random phases, watch them lock
    python software/cpg/run_cpg.py --demo --plot demo.png   # parameter changes mid-run: CPG vs sine
    python software/cpg/run_cpg.py --viewer --demo       # watch it (WSLg / desktop)

--demo runs this schedule (20 s), once with the CPG and once with the closed-form sine gait
switched at the same instants, and reports how large the joint-target jumps are:
     0 s  amp 30, 0.5 Hz, phase 45            forward
     5 s  amp 40, 0.8 Hz                      faster
    10 s  bias +10                            turn left
    15 s  bias 0, phase -45                   wave reversed: backward
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "sinusoidal"))
import run_gait as R          # noqa: E402  (model loading, simulation loop, summary, video)
from cpg import CPG           # noqa: E402

DEMO = [  # (time s, parameter changes, label)
    (0.0, dict(v_amp=30, freq=0.5, v_phase=45, v_bias=0), "forward"),
    (5.0, dict(v_amp=40, freq=0.8), "faster"),
    (10.0, dict(v_bias=10), "turn left"),
    (15.0, dict(v_bias=0, v_phase=-45), "backward"),
]


class Scheduled:
    """Applies DEMO-style parameter changes at given times, then asks the CPG for targets."""

    def __init__(self, cpg, schedule):
        self.cpg, self.schedule, self.i = cpg, schedule, 0

    def __call__(self, t):
        while self.i < len(self.schedule) and t >= self.schedule[self.i][0]:
            self.cpg.set_params(**self.schedule[self.i][1]); self.i += 1
        return self.cpg(t)


class ScheduledSine:
    """The closed-form sine gait with the same schedule: parameters simply switch."""

    def __init__(self, layout, schedule):
        self.layout, self.schedule = layout, schedule
        self.v = [k for k, c in enumerate(layout) if c == "V"]

    def __call__(self, t):
        p = {}
        for t0, ch, _ in self.schedule:
            if t >= t0:
                p.update(ch)
        th = np.zeros(len(self.layout))
        s = min(1.0, t)                                      # same 1 s fade-in as run_gait
        for i, k in enumerate(self.v):
            th[k] = s * np.radians(p["v_amp"] * np.sin(2 * np.pi * p["freq"] * t + i * np.radians(p["v_phase"]))
                                   + p["v_bias"])
        return np.clip(th, -np.radians(55), np.radians(55))


def segment_report(log, schedule, duration):
    """Per schedule segment: mean speed and heading change, measured on the centre of mass
    sampled once per gait period (removes the side-to-side wiggle)."""
    com = (log["com"] - log["com0"])[:, :2]; t = log["t"]
    edges = [s[0] for s in schedule] + [duration]
    freq, out = None, []
    for (t0, ch, label), t1 in zip(schedule, edges[1:]):
        freq = ch.get("freq", freq)
        T = 1.0 / freq
        ts = np.arange(t0 + 1.0, t1 + 1e-9, T)               # skip 1 s of transition
        if len(ts) < 3:
            continue
        p = np.array([com[min(np.searchsorted(t, x), len(t) - 1)] for x in ts])
        dp = np.diff(p, axis=0)
        hd = np.degrees(np.unwrap(np.arctan2(dp[:, 1], dp[:, 0])))
        rate = (hd[-1] - hd[0]) / (ts[-2] - ts[0])           # left turn positive
        out.append(f"    {t0:4.0f}-{t1:4.0f} s {label:10s}: {1000 * np.linalg.norm(dp, axis=1).mean() / T:5.0f} mm/s, "
                   f"heading {(hd[0] + 180) % 360 - 180:+5.0f} deg, turning {rate:+5.1f} deg/s")
    return out


def max_jump(log):
    """Largest change of any joint target between two 50 Hz control updates (deg)."""
    return np.degrees(np.abs(np.diff(log["target"], axis=0)).max())


def plot_demo(logs, layout, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for (name, log), col in zip(logs.items(), ("C0", "C3")):
        com = (log["com"] - log["com0"]) * 1000
        ax[0].plot(com[:, 0], com[:, 1], color=col, lw=1.5, label=name)
        for t0, _, label in DEMO[1:]:
            i = np.searchsorted(log["t"], t0)
            ax[0].plot(com[i, 0], com[i, 1], "o", color=col, ms=4)
            if name == "CPG":
                ax[0].annotate(label, com[i, :2], fontsize=8, xytext=(4, 4), textcoords="offset points")
        k = [i for i, c in enumerate(layout) if c == "V"][2]
        ax[1].plot(log["t"], np.degrees(log["target"][:, k]), color=col, lw=1.2, label=f"{name}, joint {k + 1:02d} target")
    for t0, _, _ in DEMO[1:]:
        ax[1].axvline(t0, color="0.6", lw=0.8, ls="--")
    ax[0].set_aspect("equal", "datalim"); ax[0].grid(alpha=0.3); ax[0].legend()
    ax[0].set_xlabel("forward x (mm)"); ax[0].set_ylabel("left y (mm)"); ax[0].set_title("centre of mass path (dots = parameter change)")
    ax[1].set_xlim(3, 18); ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8)
    ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("deg"); ax[1].set_title("joint target around the parameter changes")
    fig.tight_layout(); fig.savefig(path, dpi=100)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=R.MODELS, default="vh9_full")
    ap.add_argument("--amp", type=float, default=30.0, help="V amplitude, deg")
    ap.add_argument("--freq", type=float, default=0.5, help="Hz")
    ap.add_argument("--phase", type=float, default=45.0, help="V phase lag between neighbours, deg (negative = backward)")
    ap.add_argument("--bias", type=float, default=0.0, help="V offset (turning), deg")
    ap.add_argument("--lift", type=float, default=0.0, help="head lift on the last H joint, deg")
    ap.add_argument("--coupling", type=float, default=4.0, help="phase coupling strength w, 1/s")
    ap.add_argument("--conv", type=float, default=10.0, help="amplitude/offset convergence rate a, 1/s")
    ap.add_argument("--random-phase", action="store_true", help="start the oscillators at random phases")
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--demo", action="store_true", help="run the parameter-change schedule (20 s)")
    ap.add_argument("--video", help="mp4 of the CPG run (MUJOCO_GL=egl when headless)")
    ap.add_argument("--plot", help="png: path + joint targets")
    ap.add_argument("--viewer", action="store_true")
    a = ap.parse_args()

    m = R.load(a.model)
    layout, names = R.joint_layout(m)

    def make_cpg():
        c = CPG(layout, freq=a.freq, v_amp=a.amp, v_phase=a.phase, v_bias=a.bias, head_lift=a.lift,
                coupling=a.coupling, conv=a.conv, init_phase="random" if a.random_phase else None, seed=0)
        return Scheduled(c, DEMO) if a.demo else c

    duration = 20.0 if a.demo else a.duration
    if a.viewer:
        return R.viewer(m, make_cpg())

    gait = make_cpg()
    t0 = time.time()
    log = R.record_video(m, gait, duration, a.video) if a.video else R.run(m, gait, duration)[1]
    wall = time.time() - t0
    cpg = gait.cpg if a.demo else gait
    print(f"{a.model} ({layout}), CPG w={a.coupling}/s a={a.conv}/s, {duration:.0f} s "
          f"({duration / wall:.1f}x real time), final phase-lag error {np.degrees(cpg.phase_error()):.2f} deg")

    if a.demo:
        sine_log = R.run(R.load(a.model), ScheduledSine(layout, DEMO), duration)[1]
        for name, lg in (("CPG", log), ("sine", sine_log)):
            print(f"  {name}: largest joint-target jump between 50 Hz updates {max_jump(lg):.1f} deg")
            print("\n".join(segment_report(lg, DEMO, duration)))
        if a.plot:
            plot_demo({"CPG": log, "sine (switched)": sine_log}, layout, a.plot)
    else:
        s = R.summarize(m, log, layout)
        print(f"  moved {1000 * s['forward_m']:+.0f} mm forward, {1000 * s['sideways_m']:+.0f} mm sideways; "
              f"steady speed {s['speed_mm_s']:+.1f} mm/s; servo torque saturated {s['torque_saturated_pct']:.0f}%")
        print(f"  largest joint-target jump between 50 Hz updates {max_jump(log):.1f} deg")
        if a.plot:
            R.plot(log, layout, names, a.plot, f"CPG {a.model}: A={a.amp}, f={a.freq} Hz, phi={a.phase}")
    for p in (a.video, a.plot):
        if p:
            print("  wrote", os.path.abspath(p))


if __name__ == "__main__":
    main()
