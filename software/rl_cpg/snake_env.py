"""
Gymnasium environment: the Slitherbot reaches goal points on a flat floor, with an RL policy
that sets CPG parameters (the RL-over-CPG layering of Jiang et al., arXiv 2312.03223).

Layers
    policy  (2 Hz)  : action = 4 CPG parameters  -> CPG.set_params
    CPG     (1 kHz) : coupled oscillators         -> joint targets, sent to the servos at 50 Hz
    MuJoCo  (500 Hz): position servos, wheels, contacts

Task: a goal appears 1-2 m away in a random direction around the snake. Reward is progress
towards it; reaching it (centre of mass within GOAL_RADIUS) ends the episode with a bonus.
Episodes last at most EPISODE_S seconds.

Action (each in [-1, 1], mapped linearly to)        Observation (float32)
    freq     0.3 .. 1.0  Hz                             goal: distance, cos/sin of bearing (3)
    v_amp    10  .. 45   deg                            body: gravity in base frame, base gyro (6)
    v_phase  30  .. 75   deg   (> 0: head first)        joints: angles / limit, velocities / 10 (2n)
    v_bias  -15  .. 15   deg   (turning)                gait: cos/sin of the CPG phase (2)
                                                        previous action (4)
"""
import os
import sys

import gymnasium as gym
import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "sinusoidal"))
sys.path.insert(0, os.path.join(HERE, "..", "cpg"))
import run_gait as R      # noqa: E402  (model loading)
from cpg import CPG       # noqa: E402

ACTION_RANGES = {"freq": (0.3, 1.0), "v_amp": (10.0, 45.0), "v_phase": (30.0, 75.0), "v_bias": (-15.0, 15.0)}
POLICY_DT = 0.5          # s between policy decisions (2 Hz)
CONTROL_DT = 0.02        # s between servo target updates (50 Hz, MG90S PWM)
EPISODE_S = 30.0
GOAL_DIST = (1.0, 2.0)   # m
GOAL_RADIUS = 0.15       # m
PROGRESS_SCALE = 10.0    # reward per metre of progress = 10
SUCCESS_BONUS = 10.0
SMOOTH_PENALTY = 0.05    # x |a_t - a_{t-1}|^2


class SnakeGoalEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, model="vh9_bare", render_mode=None):
        self.model_name = model
        self.m = R.load(model)
        self.d = mujoco.MjData(self.m)
        self.layout, _ = R.joint_layout(self.m)
        self.n = len(self.layout)
        self.qadr = self.m.jnt_qposadr[self.m.actuator_trnid[:, 0]]
        self.vadr = self.m.jnt_dofadr[self.m.actuator_trnid[:, 0]]
        self.head = self.m.body(f"m{self.n:02d}").id
        self.v0 = self.layout.index("V")
        self.limit = self.m.actuator_ctrlrange[0, 1]
        self.n_ctrl = round(POLICY_DT / CONTROL_DT)
        self.n_sub = round(CONTROL_DT / self.m.opt.timestep)
        self.max_steps = round(EPISODE_S / POLICY_DT)
        self.lo = np.array([r[0] for r in ACTION_RANGES.values()])
        self.hi = np.array([r[1] for r in ACTION_RANGES.values()])
        self.action_space = gym.spaces.Box(-1.0, 1.0, (len(ACTION_RANGES),), np.float32)
        n_obs = 3 + 6 + 2 * self.n + 2 + len(ACTION_RANGES)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (n_obs,), np.float32)
        self.render_mode = render_mode
        self._renderer = None
        self.on_control = None

    # ------------------------------------------------------------ helpers
    def to_params(self, action):
        a = np.clip(np.asarray(action, float), -1, 1)
        return dict(zip(ACTION_RANGES, self.lo + (a + 1) / 2 * (self.hi - self.lo)))

    def com(self):
        return self.d.subtree_com[1][:2].copy()

    def heading(self):
        """Body axis direction (rad): from the tail (base) to the head module."""
        v = self.d.xpos[self.head][:2] - self.d.xpos[1][:2]
        return np.arctan2(v[1], v[0])

    def goal_dist(self):
        return float(np.linalg.norm(self.goal - self.com()))

    def _obs(self):
        rel = self.goal - self.com()
        bearing = np.arctan2(rel[1], rel[0]) - self.heading()
        R_b = self.d.xmat[1].reshape(3, 3)
        grav = R_b.T @ np.array([0.0, 0.0, -1.0])
        gyro = self.d.qvel[3:6]                                      # free joint: angular velocity, base frame
        q = self.d.qpos[self.qadr] / self.limit
        qd = self.d.qvel[self.vadr] / 10.0
        ph = self.cpg.theta[self.v0]
        return np.concatenate([[np.linalg.norm(rel), np.cos(bearing), np.sin(bearing)], grav, gyro,
                               q, qd, [np.cos(ph), np.sin(ph)], self.prev_action]).astype(np.float32)

    # ------------------------------------------------------------ gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.m, self.d)
        mujoco.mj_forward(self.m, self.d)
        self.cpg = CPG(self.layout)                                  # default gait, amplitude ramps from 0
        self.prev_action = np.zeros(len(ACTION_RANGES), np.float32)
        self.cpg.set_params(**self.to_params(self.prev_action))
        if options and "goal" in options:                             # (x, y) relative to the start COM
            self.goal = self.com() + np.asarray(options["goal"], float)
        else:
            r = self.np_random.uniform(*GOAL_DIST)
            b = self.np_random.uniform(-np.pi, np.pi)
            self.goal = self.com() + r * np.array([np.cos(b), np.sin(b)])
        self.steps = 0
        self.prev_dist = self.goal_dist()
        return self._obs(), {"goal": self.goal.copy()}

    def step(self, action):
        action = np.clip(np.asarray(action, np.float32), -1, 1)
        self.cpg.set_params(**self.to_params(action))
        for _ in range(self.n_ctrl):
            self.d.ctrl[:] = self.cpg(self.d.time)
            for _ in range(self.n_sub):
                mujoco.mj_step(self.m, self.d)
            if self.on_control is not None:                           # e.g. video capture at 50 Hz
                self.on_control(self)
        self.steps += 1
        dist = self.goal_dist()
        reward = PROGRESS_SCALE * (self.prev_dist - dist) - SMOOTH_PENALTY * float(np.sum((action - self.prev_action) ** 2))
        self.prev_dist, self.prev_action = dist, action
        success = dist < GOAL_RADIUS
        upright = self.d.xmat[1][8] > 0.5                            # base z axis still pointing up
        if success:
            reward += SUCCESS_BONUS
        terminated = bool(success or not upright)
        truncated = self.steps >= self.max_steps
        info = {"is_success": success, "dist": dist, "time": self.d.time}
        return self._obs(), float(reward), terminated, truncated, info

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.m, 480, 640)
            self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
            self._cam = mujoco.MjvCamera()
            self._cam.distance, self._cam.azimuth, self._cam.elevation = 2.6, 90.0, -60.0
        mid = (self.com() + self.goal) / 2
        self._cam.lookat[:] = [mid[0], mid[1], 0.0]
        self._renderer.update_scene(self.d, self._cam)
        add_goal_marker(self._renderer.scene, self.goal)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close(); self._renderer = None


def add_goal_marker(scene, goal):
    """Draw the goal (a red disc of GOAL_RADIUS) into a mjvScene."""
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CYLINDER, np.array([GOAL_RADIUS, GOAL_RADIUS, 0.002]),
                        np.array([goal[0], goal[1], 0.002]), np.eye(3).flatten(), np.array([0.9, 0.2, 0.2, 0.6], np.float32))
    scene.ngeom += 1
