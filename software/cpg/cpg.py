"""
Central pattern generator (CPG) for the Slitherbot: one phase oscillator per servo joint,
in the amplitude-controlled phase-oscillator form of Ijspeert & Crespi (ICRA 2007), as used
for snake robots by Bing et al. (ICRA 2017) and Jiang et al. (arXiv 2312.03223).

Per joint i, with the joints split into two chains (V = yaw joints, H = pitch joints):

    dθ_i/dt  = 2π·f + Σ_j w·sin(θ_j − θ_i − φ_ij)        phase; j = chain neighbours of i,
                                                          φ_ij = +φ for the head-side neighbour,
                                                          −φ for the tail-side one
    d²r_i/dt² = a·( a/4·(R − r_i) − dr_i/dt )             amplitude, critically damped to R
    d²δ_i/dt² = a·( a/4·(B − δ_i) − dδ_i/dt )             offset, critically damped to B
    q_i       = r_i·sin(θ_i) + δ_i                        joint target

At steady state the V chain gives q_i = R·sin(2πf·t + i·φ) + B, i.e. exactly the serpenoid
gait of software/sinusoidal. What the CPG adds:
  * changing R, f, φ or B while running gives smooth joint targets (no jumps), because the
    parameters enter through the dynamics instead of the closed-form formula;
  * the phase lags are an attractor: any starting phases converge to the travelling wave;
  * a handful of parameters (set_params) is a small, smooth action space for RL.

Usage: cpg = CPG(layout); cpg.set_params(v_amp=40, freq=0.8); q = cpg(t)  (t increasing)
"""
import numpy as np

DEG = np.pi / 180.0


class CPG:
    PARAMS = ("freq", "v_amp", "v_phase", "v_bias", "h_amp", "h_phase", "h_bias", "head_lift")

    def __init__(self, layout, freq=0.5, v_amp=30.0, v_phase=45.0, v_bias=0.0,
                 h_amp=0.0, h_phase=45.0, h_bias=0.0, head_lift=0.0,
                 coupling=4.0, conv=10.0, dt=0.001, limit_deg=55.0, init_phase=None, seed=None):
        """
        layout     : joint types tail -> head, e.g. "VHVHVHVHV"
        freq       : Hz, shared by all oscillators
        v_* / h_*  : amplitude, phase lag between chain neighbours, offset (degrees) for V / H joints
        head_lift  : extra offset on the last H joint (degrees)
        coupling   : w, phase coupling strength (1/s); larger = faster phase locking
        conv       : a, amplitude/offset convergence rate (1/s); settle time ~ 8/a seconds
        dt         : internal Euler step (s)
        init_phase : None = already on the travelling wave (theta = chain index * lag),
                     "random" = uniform random (the coupling locks them), or an array (rad)
        """
        self.layout, self.n = layout, len(layout)
        self.w, self.a, self.dt = coupling, conv, dt
        self.limit = limit_deg * DEG
        self.chains = {c: [k for k, t in enumerate(layout) if t == c] for c in "VH"}
        self.last_h = self.chains["H"][-1] if self.chains["H"] else None
        self.p = {}
        self.set_params(freq=freq, v_amp=v_amp, v_phase=v_phase, v_bias=v_bias,
                        h_amp=h_amp, h_phase=h_phase, h_bias=h_bias, head_lift=head_lift)
        rng = np.random.default_rng(seed)
        if init_phase is None:
            self.theta = np.zeros(self.n)
            for idx in self.chains.values():
                self.theta[idx] = np.arange(len(idx)) * self.lag[idx]
        elif isinstance(init_phase, str) and init_phase == "random":
            self.theta = rng.uniform(0, 2 * np.pi, self.n)
        else:
            self.theta = np.asarray(init_phase, float).copy()
        self.r, self.dr = np.zeros(self.n), np.zeros(self.n)        # amplitudes start at 0: soft start
        self.d, self.dd = np.zeros(self.n), np.zeros(self.n)        # offsets
        self.t = 0.0

    # ------------------------------------------------------------ parameters
    def set_params(self, **kw):
        """Change any of PARAMS (degrees / Hz). Takes effect smoothly through the dynamics."""
        for k, v in kw.items():
            if k not in self.PARAMS:
                raise KeyError(f"unknown CPG parameter {k!r}; use one of {self.PARAMS}")
            self.p[k] = float(v)
        p = self.p
        self.R = np.zeros(self.n); self.B = np.zeros(self.n); self.lag = np.zeros(self.n)
        for c in "VH":
            idx = self.chains[c]
            self.R[idx] = p[f"{c.lower()}_amp"] * DEG
            self.B[idx] = p[f"{c.lower()}_bias"] * DEG
            self.lag[idx] = p[f"{c.lower()}_phase"] * DEG
        if self.last_h is not None:
            self.B[self.last_h] += p["head_lift"] * DEG

    def params(self):
        return dict(self.p)

    # ------------------------------------------------------------ dynamics
    def _phase_rate(self, theta):
        rate = np.full(self.n, 2 * np.pi * self.p["freq"])
        for idx in self.chains.values():
            for a, b in zip(idx[:-1], idx[1:]):              # a = tail side, b = head side
                phi = self.lag[a]
                rate[a] += self.w * np.sin(theta[b] - theta[a] - phi)
                rate[b] += self.w * np.sin(theta[a] - theta[b] + phi)
        return rate

    def step(self, dt):
        a = self.a
        self.theta += dt * self._phase_rate(self.theta)
        ddr = a * (a / 4 * (self.R - self.r) - self.dr)
        ddd = a * (a / 4 * (self.B - self.d) - self.dd)
        self.r += dt * self.dr; self.dr += dt * ddr
        self.d += dt * self.dd; self.dd += dt * ddd
        self.t += dt

    def advance(self, t):
        """Integrate up to time t (seconds)."""
        while self.t < t - 1e-12:
            self.step(min(self.dt, t - self.t))

    def output(self):
        return np.clip(self.r * np.sin(self.theta) + self.d, -self.limit, self.limit)

    def __call__(self, t):
        """Joint targets (rad) at time t; same interface as sinusoidal.SerpenoidGait."""
        self.advance(t)
        return self.output()

    def phase_error(self):
        """Largest deviation (rad) of neighbour phase differences from their targets."""
        err = 0.0
        for idx in self.chains.values():
            for a, b in zip(idx[:-1], idx[1:]):
                e = np.angle(np.exp(1j * (self.theta[b] - self.theta[a] - self.lag[a])))
                err = max(err, abs(e))
        return err
