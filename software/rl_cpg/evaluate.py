"""
Evaluate a trained policy on fixed goals, against the open-loop CPG (default gait, no steering).

    python software/rl_cpg/evaluate.py                                   # runs/vh9_bare/best_model.zip
    python software/rl_cpg/evaluate.py --policy runs/vh9_bare/final_model.zip --goals 50
    MUJOCO_GL=egl python software/rl_cpg/evaluate.py --plot eval.png --video eval.mp4
    python software/rl_cpg/evaluate.py --viewer                          # watch, new goal each episode

Goals are 1-2 m away at evenly spread directions around the snake (same set for every
controller), so results are comparable between runs.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from snake_env import GOAL_DIST, SnakeGoalEnv, add_goal_marker  # noqa: E402


def goal_set(n, seed=123):
    rng = np.random.default_rng(seed)
    b = np.linspace(-np.pi, np.pi, n, endpoint=False) + np.pi / n
    r = rng.uniform(*GOAL_DIST, n)
    return [np.array([ri * np.cos(bi), ri * np.sin(bi)]) for ri, bi in zip(r, b)]


def episode(env, act, goal, on_step=None):
    """Run one episode to `goal` (relative to the start). Paths are returned relative to the start."""
    obs, _ = env.reset(seed=0, options={"goal": goal})
    start = env.com()
    path, params, done, info = [env.com()], [], False, {}
    while not done:
        a = act(obs)
        params.append(env.to_params(a))
        obs, _, term, trunc, info = env.step(a)
        path.append(env.com())
        if on_step:
            on_step(env)
        done = term or trunc
    return bool(info["is_success"]), info["time"], np.array(path) - start, params


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="vh9_bare")
    ap.add_argument("--policy", default=None, help="zip from train_sac.py (default runs/<model>/best_model.zip)")
    ap.add_argument("--goals", type=int, default=24)
    ap.add_argument("--plot", help="png with all paths")
    ap.add_argument("--video", help="mp4 of a few episodes")
    ap.add_argument("--viewer", action="store_true")
    a = ap.parse_args()

    from stable_baselines3 import SAC
    path = a.policy or os.path.join(HERE, "runs", a.model, "best_model.zip")
    if not os.path.isabs(path) and not os.path.exists(path):
        path = os.path.join(HERE, path)
    agent = SAC.load(path, device="cpu")
    policy = lambda o: agent.predict(o, deterministic=True)[0]          # noqa: E731
    open_loop = lambda o: np.zeros(4, np.float32)                       # noqa: E731  default gait, never steers
    env = SnakeGoalEnv(a.model)

    if a.viewer:
        import mujoco
        import mujoco.viewer
        rng = np.random.default_rng()

        def show(e):                                                    # every 20 ms of sim time
            v.user_scn.ngeom = 0
            add_goal_marker(v.user_scn, e.goal)
            v.cam.lookat[:] = [*e.com(), 0]
            v.sync(); time.sleep(0.02)
        env.on_control = show
        with mujoco.viewer.launch_passive(env.m, env.d) as v:
            v.cam.distance, v.cam.elevation = 2.6, -60.0
            while v.is_running():
                b, r = rng.uniform(-np.pi, np.pi), rng.uniform(*GOAL_DIST)
                goal = r * np.array([np.cos(b), np.sin(b)])
                ok, t, _, _ = episode(env, policy, goal)
                print(f"goal {goal.round(2)}: {'reached' if ok else 'missed'} in {t:.1f} s")
                if not v.is_running():
                    break
        return

    goals = goal_set(a.goals)
    res = {}
    for name, act in (("RL + CPG", policy), ("open-loop CPG", open_loop)):
        runs = [episode(env, act, g) for g in goals]
        res[name] = runs
        ok = np.array([r[0] for r in runs]); t = np.array([r[1] for r in runs])
        final = [np.linalg.norm(r[2][-1] - g) for r, g in zip(runs, goals)]
        print(f"{name:14s}: reached {ok.sum()}/{len(ok)} goals ({100 * ok.mean():.0f}%), "
              f"mean time when reached {t[ok].mean() if ok.any() else float('nan'):.1f} s, "
              f"median final distance {np.median(final):.2f} m")
    rl = res["RL + CPG"]
    allp = np.array([list(p.values()) for r in rl for p in r[3]])
    print("RL action use (min / mean / max):  " + ", ".join(
        f"{k} {allp[:, i].min():.2f}/{allp[:, i].mean():.2f}/{allp[:, i].max():.2f}" for i, k in enumerate(rl[0][3][0])))

    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        for axi, (name, runs) in zip(ax, res.items()):
            for (ok, _, p, _), g in zip(runs, goals):
                col = "C2" if ok else "C3"
                axi.plot(p[:, 0], p[:, 1], color=col, lw=1)
                axi.add_patch(plt.Circle(g, 0.15, color=col, alpha=0.3))
            axi.plot(0, 0, "ko", ms=4)
            n_ok = sum(r[0] for r in runs)
            axi.set_title(f"{name}: {n_ok}/{len(runs)} goals reached (green)")
            axi.set_aspect("equal"); axi.grid(alpha=0.3); axi.set_xlabel("x (m)"); axi.set_ylabel("y (m)")
            axi.set_xlim(-2.4, 2.4); axi.set_ylim(-2.4, 2.4)
        fig.tight_layout(); fig.savefig(a.plot, dpi=100)
        print("wrote", os.path.abspath(a.plot))

    if a.video:
        import imageio
        frames, k = [], [0]
        env_v = SnakeGoalEnv(a.model)

        def grab(e):                                                    # every 6th 20 ms step: 8.3 Hz
            k[0] += 1
            if k[0] % 6 == 0:
                frames.append(e.render())
        env_v.on_control = grab
        for g in goal_set(4, seed=7):
            episode(env_v, policy, g)
        imageio.mimwrite(a.video, frames, fps=25, quality=6, macro_block_size=8)      # 3x real time
        print("wrote", os.path.abspath(a.video))


if __name__ == "__main__":
    main()
