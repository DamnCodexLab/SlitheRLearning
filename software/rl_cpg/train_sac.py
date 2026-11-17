"""
Train SAC (Stable-Baselines3) on SnakeGoalEnv, CPU only.

    python software/rl_cpg/train_sac.py                          # vh9_bare, 150k steps, 6 envs
    python software/rl_cpg/train_sac.py --steps 300000 --envs 6 --name long
    python software/rl_cpg/train_sac.py --model vh17_bare        # later

Writes software/rl_cpg/runs/<name>/:
    best_model.zip    best policy by evaluation success rate / reward
    final_model.zip   policy at the end of training
    progress.csv      training log (episode reward, success rate, losses)
    evaluations.npz   periodic deterministic evaluations (20 goals each)
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def make_env(model, seed):
    def _init():
        from stable_baselines3.common.monitor import Monitor
        from snake_env import SnakeGoalEnv
        env = Monitor(SnakeGoalEnv(model), info_keywords=("is_success",))
        env.reset(seed=seed)
        return env
    return _init


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="vh9_bare")
    ap.add_argument("--steps", type=int, default=150_000, help="policy steps (0.5 s each)")
    ap.add_argument("--envs", type=int, default=6, help="parallel simulations (processes)")
    ap.add_argument("--threads", type=int, default=1, help="PyTorch threads (1 is fastest for this small net)")
    ap.add_argument("--utd", type=float, default=1 / 3,
                    help="gradient updates per collected step; SAC updates (~15 ms each on this CPU) are the bottleneck")
    ap.add_argument("--name", default=None, help="run folder name (default: model name)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import torch
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    torch.set_num_threads(a.threads)
    out = os.path.join(HERE, "runs", a.name or a.model)
    os.makedirs(out, exist_ok=True)

    venv = SubprocVecEnv([make_env(a.model, a.seed + i) for i in range(a.envs)], start_method="fork")
    eval_env = DummyVecEnv([make_env(a.model, 10_000 + a.seed)])
    agent = SAC("MlpPolicy", venv, learning_rate=3e-4, buffer_size=300_000, batch_size=256,
                gamma=0.98, tau=0.005, train_freq=1, gradient_steps=max(1, round(a.utd * a.envs)), learning_starts=3_000,
                policy_kwargs=dict(net_arch=[256, 256]), seed=a.seed, device="cpu", verbose=0)
    agent.set_logger(configure(out, ["csv", "stdout"]))
    cb = EvalCallback(eval_env, n_eval_episodes=20, eval_freq=max(1, 10_000 // a.envs),
                      best_model_save_path=out, log_path=out, deterministic=True, verbose=1)
    t0 = time.time()
    agent.learn(total_timesteps=a.steps, callback=cb, log_interval=20)
    agent.save(os.path.join(out, "final_model"))
    print(f"done: {a.steps} steps in {(time.time() - t0) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
