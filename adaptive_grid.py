"""Adaptive gridlines for register-encoded random walks.

Everything in this file is CLASSICAL: the walks are simulated on a classical computer and nothing is run as a
quantum circuit. They are designed for a quantum implementation (an n-bit register would be n qubits; the
moves are ones a register circuit can make), but no quantum advantage is claimed or measured here.

Each parameter lives in an n-bit register: 2^n grid points spread over a window [lo, hi]. With a fixed
window equal to the full allowed range, precision is limited by the spacing (hi - lo) / 2^n. An adaptive grid
moves and resizes the window as the walk learns where the probability is, so a few bits can still reach
high precision. The danger is zooming in on the wrong place (a "decoy") and never seeing the right answer,
so some strategies are also allowed to zoom back out.

The walk itself is a Metropolis walk on the grid. Each step proposes either a local move (every parameter
by -1, 0 or +1 grid point, the lazy three-point coin of the QCLT circuit) or, with probability P_GLOBAL, a
global jump to a uniformly random grid point of the current window (on a quantum computer, a Hadamard on every
register qubit:
cheap for a small register, and the only way to cross an empty gap between separate peaks). Both proposals
are symmetric, so the usual Metropolis rule applies. The
target density is evaluated only at grid points, as a quantum oracle evaluating the likelihood at the
register's value would. Re-gridding happens between blocks of steps (in a circuit: between coherent
segments, using the block's samples), and the walker is moved to the nearest grid point of the new grid.

Strategies (see STRATEGIES):
  fixed       window = full range, never changes                      (baseline, n bits)
  fixed16     window = full range with 16 bits per parameter          (reference: "just add bits/qubits")
  fixed16_step  as fixed16, but the local step is 2^m grid points with m re-chosen every block by the same
              spread rule the zoom strategies use (a fair "more bits + adaptive step" baseline)
  zoom_in     shrink the window to K x recent spread around the recent mean; never widen
  zoom_inout  resize the window to K x recent spread (larger or smaller); also widen x2 when the walk
              presses against the window edges
  restart     zoom_inout, plus every R re-grids spend one block on the full range ("look around"),
              then re-zoom on wherever the walk is

Test problems (see PROBLEMS): a narrow 2-D peak, a 2-D decoy trap, and a real 5-parameter wavelet posterior.

Run:   python3.12 adaptive_grid.py              # all experiments + figures/previous/adaptive_grid_*.pdf
       python3.12 adaptive_grid.py --plot-only
"""
import math
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "figures" / "previous"
DATA = Path(__file__).parent / "data"            # run data and logs (figures/ holds PDFs only)

# ============================================================================ test problems
# All problems live on the unit cube [0, 1]^d (the full allowed range of every parameter).


def wrap_diff(a, b, per):
    """a - b, with the periodic components (e.g. the wavelet phase) mapped to [-0.5, 0.5)."""
    d = np.asarray(a, float) - b
    return np.where(per, (d + 0.5) % 1.0 - 0.5, d)


WAVELET_PERIODIC = np.array([False, False, False, False, True])      # t0, f0, Q, lnA, phase


class Gauss2D:
    """Mixture of axis-aligned 2-D Gaussians; exact posterior known."""

    def __init__(self, comps):
        self.comps = comps                  # list of (weight, mean (2,), sigma (2,))
        self.d = 2
        self.periodic = np.zeros(2, bool)

    def logp(self, x):
        tot = 0.0
        for w, m, s in self.comps:
            z = (x - m) / s
            tot += w * math.exp(-0.5 * float(z @ z)) / (2 * math.pi * s[0] * s[1])
        return math.log(tot) if tot > 0 else -1e300


NARROW = Gauss2D([(1.0, np.array([0.713, 0.331]), np.array([0.0040, 0.0060]))])
DECOY = Gauss2D([(0.7, np.array([0.840, 0.180]), np.array([0.0120, 0.0120])),     # true peak: most probability
                 (0.3, np.array([0.420, 0.560]), np.array([0.0600, 0.0600]))])    # decoy: broad, near the start


class Wavelet5D:
    """Posterior of a single sine-Gaussian in white noise (5 parameters, SNR 40), point-evaluated."""

    def __init__(self, snr=40.0, seed=7):
        import tdrjmcmc_qclt as T
        self.T = T
        self.d = 5
        self.periodic = WAVELET_PERIODIC.copy()             # the phase wraps around
        assert np.array_equal(self.periodic, T.PERIODIC)
        self.true_phys = np.array([0.52, 140.0, 20.0, 0.0, 1.3])
        th = self.true_phys.copy()
        i0, i1 = T.wave_window(th)
        h = np.zeros(T.N)
        h[i0:i1] = T.wave_on(th, i0, i1)
        h *= snr / np.linalg.norm(h)
        self.true_phys[3] = math.log(snr / np.linalg.norm(T.wave_on(np.r_[th[:3], 0.0, th[4]], i0, i1)))
        rng = np.random.default_rng(seed)
        self.d_data = h + rng.normal(size=T.N)
        self.true = (self.true_phys - T.LO) / (T.HI - T.LO)

    def to_phys(self, x):
        return self.T.LO + x * (self.T.HI - self.T.LO)

    def tf_peak(self):
        """(t, f) in unit coordinates of the brightest cell of the sampler's time-frequency birth map."""
        birth, self.T.BIRTH = self.T.BIRTH, "tf"
        P, _ = self.T.tf_proposal(self.d_data)
        self.T.BIRTH = birth
        i, j = np.unravel_index(int(np.argmax(P)), P.shape)
        return (np.array([i, j]) + 0.5) / P.shape[0]

    def logp(self, x):
        th = self.to_phys(x)
        i0, i1 = self.T.wave_window(th)
        if i1 <= i0:
            return -0.5 * float(self.d_data @ self.d_data)
        r = self.d_data[i0:i1] - self.T.wave_on(th, i0, i1)
        return -0.5 * (float(r @ r) - float(self.d_data[i0:i1] @ self.d_data[i0:i1]))


# ============================================================================ grid walk
class Grid:
    """2^n grid points per parameter over the window [lo, lo + w]. For a periodic parameter the window may wrap
    around the seam at 0 = 1, and a full-range window is a ring (local steps wrap)."""

    def __init__(self, d, n_bits, periodic=None):
        self.d, self.n = d, 1 << n_bits
        self.per = np.zeros(d, bool) if periodic is None else np.asarray(periodic, bool)
        self.lo = np.zeros(d)
        self.w = np.ones(d)

    def x(self, i):
        x = self.lo + (i + 0.5) * self.w / self.n
        return np.where(self.per, x % 1.0, x)

    def offset(self, x):
        """Position of x relative to the window start (periodic: via the nearest image of the window centre)."""
        return np.where(self.per, wrap_diff(x, self.lo + self.w / 2, self.per) + self.w / 2, np.asarray(x) - self.lo)

    def index(self, x):
        return np.clip(np.floor(self.offset(x) / self.w * self.n).astype(int), 0, self.n - 1)

    def set(self, lo, w):
        w = np.minimum(w, 1.0)
        self.lo = np.where(self.per, np.asarray(lo, float) % 1.0, np.clip(lo, 0.0, 1.0 - w))
        self.w = w

    def stats(self, blk):
        """Mean and spread of a block of samples (circular for periodic parameters)."""
        c = self.lo + self.w / 2
        dl = wrap_diff(blk, c, self.per)
        return c + dl.mean(0), dl.std(0)


STRATEGIES = {
    "fixed": dict(adapt=False),
    "fixed16": dict(adapt=False, bits=16),
    "fixed16_step": dict(adapt=False, bits=16, adapt_step=True),
    "zoom_in": dict(adapt=True, widen=False, edge=False, restart=0),
    "zoom_inout": dict(adapt=True, widen=True, edge=True, restart=0),
    "restart": dict(adapt=True, widen=True, edge=True, restart=4),
    "multiwin": dict(adapt=True, multi=True, restart=4, max_windows=4),
}


P_GLOBAL = 0.05
P_GLOBAL_LOOK = 0.5      # during look-around blocks: search the full range hard (global jumps half the time)


def walk(problem, strategy, n_bits, n_steps, block, K, seed, x0=None, w0=None, p_global=P_GLOBAL,
         zoom_factor=2.0, pan_fraction=1.0, regrid_blocks=1, edge_threshold=0.2,
         restart_period=None):
    """Metropolis walk on an adaptive grid. Returns samples (continuous values) and the window history."""
    st = STRATEGIES[strategy]
    if st.get("multi"):
        return walk_multi(problem, n_bits, n_steps, block, K, seed, x0, w0, p_global, st)
    rng = np.random.default_rng(seed)
    g = Grid(problem.d, st.get("bits", n_bits), getattr(problem, "periodic", None))
    if w0 is not None:
        g.set(np.asarray(x0) - np.asarray(w0) / 2, np.asarray(w0))
    i = g.index(np.full(problem.d, 0.5) if x0 is None else np.asarray(x0))
    lp = problem.logp(g.x(i))
    step = np.ones(problem.d, dtype=np.int64)       # local step in grid points (fixed16_step adapts it)

    def pow2_step(length):                          # step of about `length`, as a power of two grid points
        return (2 ** np.round(np.log2(np.maximum(length / (g.w / g.n), 1.0)))).astype(np.int64)

    if st.get("adapt_step"):                        # start with the step a zoom strategy's first window has
        step = pow2_step((np.ones(problem.d) if w0 is None else np.asarray(w0)) / (1 << n_bits))
    samples = np.empty((n_steps, problem.d))
    lps = np.empty(n_steps)
    explore = np.zeros(n_steps, dtype=bool)        # True during look-around blocks (not used in estimates)
    windows = [(0, g.lo.copy(), g.w.copy())]
    n_acc = 0
    saved = None                                   # zoomed window kept while "looking around"
    n_regrid = 0
    for t in range(n_steps):
        if rng.random() < (P_GLOBAL_LOOK if saved is not None else p_global):
            j = rng.integers(0, g.n, size=problem.d)            # global jump (uniform superposition)
        else:
            j = i + rng.integers(-1, 2, size=problem.d) * step   # local lazy step
            j = np.where(g.per & (g.w >= 1.0), j % g.n, j)           # full-range periodic window: a ring
        if np.all((j >= 0) & (j < g.n)) and np.any(j != i):
            lpj = problem.logp(g.x(j))
            if math.log(rng.random()) < lpj - lp:
                i, lp = j, lpj
                n_acc += 1
        samples[t] = g.x(i)
        lps[t] = lp
        explore[t] = saved is not None
        if st.get("adapt_step") and (t + 1) % block == 0:
            # same rule as zoom in & out: window K x spread (floor: half the current step) holds 2^n_bits points
            blk = samples[t + 1 - block:t + 1]
            sd = np.maximum(wrap_diff(blk, samples[t], g.per).std(0), step * (g.w / g.n) / 2)
            step = pow2_step(K * sd / (1 << n_bits))
        if st["adapt"] and (t + 1) % (block * regrid_blocks) == 0:
            n_regrid += 1
            blk = samples[t + 1 - block:t + 1]
            mu, sd = g.stats(blk)
            cell = g.w / g.n
            sd = np.maximum(sd, cell / 2)
            target_w = K * sd
            if saved is not None:
                # end of a look-around block: move only if it found a better point than the zoomed walk had
                prev_lo, prev_w, prev_best_x, prev_best_lp = saved
                k = int(np.argmax(lps[t + 1 - block:t + 1]))
                if lps[t + 1 - block + k] > prev_best_lp:
                    g.set(blk[k] - prev_w / 2, prev_w)
                    samples[t] = blk[k]
                else:
                    g.set(prev_lo, prev_w)
                    samples[t] = prev_best_x
                saved = None
            elif ((restart_period if restart_period is not None else st["restart"])
                  and n_regrid % (restart_period if restart_period is not None else st["restart"]) == 0):
                k = int(np.argmax(lps[t + 1 - block:t + 1]))
                keep_w = np.maximum(target_w, 1e-12)
                saved = (mu - keep_w / 2, np.minimum(keep_w, 1.0), blk[k].copy(), lps[t + 1 - block + k])
                g.set(np.zeros(problem.d), np.ones(problem.d))
            else:
                new_w = target_w if st["widen"] else np.minimum(target_w, g.w)
                upper = g.w * zoom_factor if st["widen"] else g.w
                new_w = np.clip(new_w, g.w / zoom_factor, upper)
                if st["edge"]:
                    idx = np.floor(g.offset(blk) / g.w * g.n).astype(int)
                    edge_frac = np.mean((idx <= 0) | (idx >= g.n - 1), axis=0)
                    new_w = np.where(edge_frac > edge_threshold, np.maximum(new_w, zoom_factor * g.w), new_w)
                old_center = g.lo + g.w / 2
                center_shift = wrap_diff(mu, old_center, g.per)
                new_center = old_center + pan_fraction * center_shift
                g.set(new_center - new_w / 2, new_w)
            i = g.index(samples[t])
            lp = problem.logp(g.x(i))
            windows.append((t + 1, g.lo.copy(), g.w.copy()))
    return samples, windows, n_acc / n_steps, explore


def walk_multi(problem, n_bits, n_steps, block, K, seed, x0, w0, p_global, st):
    """Several adaptive windows at once (an extra 'which window' register).

    State = (window a, grid point i). Local moves stay inside window a; global jumps pick a window uniformly
    and a uniform grid point in it (symmetric while the windows are fixed). The discrete target of a grid
    point is density x cell volume of its window, so time spent in each window tracks probability mass.
    Each window re-grids from its own samples (zoom in & out). Every few blocks a temporary full-range
    window is added for one block; if it finds a point as good as the best in the existing windows and
    outside all of them, a new window is opened there (up to max_windows)."""
    rng = np.random.default_rng(seed)
    n = 1 << n_bits
    d = problem.d

    per = getattr(problem, "periodic", np.zeros(d, bool))

    def mk(lo, w):
        w = np.minimum(np.maximum(w, 1e-12), 1.0)
        return [np.where(per, np.asarray(lo, float) % 1.0, np.clip(lo, 0.0, 1.0 - w)), w]

    def off(a, x):                                   # position relative to window a's start (periodic-aware)
        c = wins[a][0] + wins[a][1] / 2
        return np.where(per, wrap_diff(x, c, per) + wins[a][1] / 2, np.asarray(x) - wins[a][0])

    wins = [mk(np.asarray(x0) - np.asarray(w0) / 2, np.asarray(w0)) if w0 is not None else mk(np.zeros(d), np.ones(d))]
    xs = lambda a, i: np.where(per, (wins[a][0] + (i + 0.5) * wins[a][1] / n) % 1.0, wins[a][0] + (i + 0.5) * wins[a][1] / n)
    idx = lambda a, x: np.clip(np.floor(off(a, x) / wins[a][1] * n).astype(int), 0, n - 1)
    inside = lambda a, x: np.all((off(a, x) >= 0) & (off(a, x) <= wins[a][1]))
    lvol = lambda a: float(np.sum(np.log(wins[a][1] / n)))
    a = 0
    i = idx(0, np.full(d, 0.5) if x0 is None else np.asarray(x0))
    lp = problem.logp(xs(a, i))
    samples = np.empty((n_steps, d)); lps = np.empty(n_steps)
    which = np.empty(n_steps, dtype=int); explore = np.zeros(n_steps, dtype=bool)
    windows = [(0, wins[0][0].copy(), wins[0][1].copy())]
    n_acc, n_regrid, temp = 0, 0, False
    for t in range(n_steps):
        if temp and rng.random() < P_GLOBAL_LOOK:              # look-around: search the full-range window
            b = len(wins) - 1
            j = rng.integers(0, n, size=d)
        elif rng.random() < p_global:
            b = int(rng.integers(len(wins)))
            j = rng.integers(0, n, size=d)
        else:
            b, j = a, i + rng.integers(-1, 2, size=d)
            j = np.where(per & (wins[a][1] >= 1.0), j % n, j)        # full-range periodic window: a ring
        if np.all((j >= 0) & (j < n)) and (b != a or np.any(j != i)):
            lpj = problem.logp(xs(b, j))
            if math.log(rng.random()) < (lpj + lvol(b)) - (lp + lvol(a)):
                a, i, lp = b, j, lpj
                n_acc += 1
        samples[t], lps[t], which[t] = xs(a, i), lp, a
        explore[t] = temp and a == len(wins) - 1
        if (t + 1) % block == 0:
            n_regrid += 1
            sl = slice(t + 1 - block, t + 1)
            if temp:                                   # end of a look-around: maybe open a new window
                tw = len(wins) - 1
                in_t = which[sl] == tw
                best_old = lps[sl][~in_t].max() if (~in_t).any() else -np.inf
                if in_t.any():
                    k = int(np.argmax(np.where(in_t, lps[sl], -np.inf)))
                    xb = samples[sl][k]
                    new_ok = (lps[sl][k] >= best_old - 2.0 and len(wins) - 1 < st["max_windows"]
                              and not any(inside(q, xb) for q in range(tw)))
                else:
                    new_ok = False
                cur_x = xs(a, i)
                wins.pop()
                if new_ok:
                    wins.append(mk(xb - 2.0 / n, np.full(d, 4.0 / n)))
                temp = False
                if a >= len(wins) or not inside(a, cur_x):    # walker was in the removed temporary window
                    a = len(wins) - 1 if new_ok else int(np.argmax([lps[sl][~in_t].max() if (~in_t).any() else 0]))
                    tgt = xb if new_ok else samples[sl][~in_t][int(np.argmax(lps[sl][~in_t]))] if (~in_t).any() else cur_x
                    i = idx(a, tgt)
                else:
                    i = idx(a, cur_x)
            else:
                for q in range(len(wins)):             # zoom in & out, each window from its own samples
                    m = which[sl] == q
                    if m.sum() < 20:
                        continue
                    blk = samples[sl][m]
                    cell = wins[q][1] / n
                    c = wins[q][0] + wins[q][1] / 2
                    dl = wrap_diff(blk, c, per)                       # circular for the phase
                    sd = np.maximum(dl.std(0), cell / 2)
                    new_w = K * sd
                    ii = np.floor(off(q, blk) / wins[q][1] * n).astype(int)
                    edge = np.mean((ii <= 0) | (ii >= n - 1), axis=0)
                    new_w = np.where(edge > 0.2, np.maximum(new_w, 2 * wins[q][1]), new_w)
                    cur = xs(a, i) if q == a else None
                    wins[q] = mk(c + dl.mean(0) - new_w / 2, new_w)
                    if q == a:
                        i = idx(a, cur)
                if st["restart"] and n_regrid % st["restart"] == 0:
                    wins.append(mk(np.zeros(d), np.ones(d)))
                    temp = True
            lp = problem.logp(xs(a, i))
            for q in range(len(wins) - (1 if temp else 0)):
                windows.append((t + 1, wins[q][0].copy(), wins[q][1].copy()))
    return samples, windows, n_acc / n_steps, explore


# ============================================================================ experiments
N_SEEDS = 12
CONFIGS = {
    "narrow": dict(problem="narrow", n_bits=4, n_steps=20_000, block=200, K=8.0, x0=(0.5, 0.5)),
    "decoy": dict(problem="decoy", n_bits=4, n_steps=40_000, block=200, K=8.0, x0=(0.45, 0.55)),
    "wavelet": dict(problem="wavelet", n_bits=5, n_steps=30_000, block=300, K=8.0, x0=None),
}
K_SWEEP = [3.0, 5.0, 8.0, 12.0, 20.0, 32.0, 50.0]
BITS_SWEEP = [3, 4, 5, 6]


def make_problem(name):
    return {"narrow": NARROW, "decoy": DECOY}.get(name) or Wavelet5D()


def _job(args):
    cfg_name, strategy, seed, overrides = args
    cfg = dict(CONFIGS[cfg_name], **overrides)
    prob = make_problem(cfg["problem"])
    x0, w0 = cfg["x0"], None
    if cfg["problem"] == "wavelet":
        # start like a data-driven birth: at the peak of the data's time-frequency map (not at the truth),
        # full range in the other parameters
        x0 = np.r_[prob.tf_peak(), 0.5, 0.5, 0.5]
        w0 = np.r_[0.08, 0.25, 1.0, 1.0, 1.0] if strategy not in ("fixed", "fixed16") else None
    t = time.process_time()
    s, win, acc, ex = walk(prob, strategy, cfg["n_bits"], cfg["n_steps"], cfg["block"], cfg["K"], seed, x0, w0)
    thin = max(1, cfg["n_steps"] // 4000)
    return dict(cfg=cfg_name, strategy=strategy, seed=seed, overrides=overrides, samples=s[::thin].astype(np.float32),
                explore=ex[::thin], thin=thin, windows=win, acc=acc, cpu=time.process_time() - t)


def run_all():
    from multiprocessing import Pool
    jobs = []
    for c in CONFIGS:
        for st in STRATEGIES:
            for s in range(N_SEEDS):
                jobs.append((c, st, s, {}))
    for K in K_SWEEP:                                       # criterion sweep: window = K x spread
        for st in ("zoom_in", "zoom_inout", "restart", "multiwin"):
            for s in range(N_SEEDS):
                jobs.append(("decoy", st, 100 + s, {"K": K}))
                jobs.append(("narrow", st, 100 + s, {"K": K}))
    for b in BITS_SWEEP:                                    # register bits per parameter
        for st in ("fixed", "zoom_inout", "restart", "multiwin"):
            for s in range(N_SEEDS):
                jobs.append(("narrow", st, 200 + s, {"n_bits": b}))
    t = time.time()
    with Pool(8) as pool:                     # 8 workers: memory headroom
        res = pool.map(_job, jobs, chunksize=2)
    print(f"{len(jobs)} walks in {time.time() - t:.0f}s", flush=True)
    with open(DATA / "adaptive_grid.pkl", "wb") as f:
        pickle.dump(res, f)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    DATA.mkdir(exist_ok=True)
    if "--plot-only" not in sys.argv:
        run_all()


# ============================================================================ reference for the wavelet problem
def wavelet_reference(n_steps=200_000, seed=0):
    """Continuous Metropolis with a Fisher-shaped Gaussian proposal, started at the truth: the 'right answer'
    (posterior mean and spread) that the grid walks are scored against."""
    prob = Wavelet5D()
    x = prob.true.copy()
    J = np.empty((0,))
    eps = 1e-6
    th = prob.to_phys(x)
    i0, i1 = prob.T.wave_window(th)
    cols = []
    for k in range(5):
        dx = np.zeros(5); dx[k] = eps
        cols.append((prob.T.wave_on(prob.to_phys(x + dx), i0, i1) - prob.T.wave_on(prob.to_phys(x - dx), i0, i1)) / (2 * eps))
    J = np.stack(cols, 1)
    C = np.linalg.inv(J.T @ J + 1e-6 * np.eye(5))
    L = np.linalg.cholesky(C * 2.38**2 / 5)
    rng = np.random.default_rng(seed)
    lp = prob.logp(x)
    out = np.empty((n_steps, 5))
    for t in range(n_steps):
        y = x + L @ rng.normal(size=5)
        y = np.where(prob.periodic, y % 1.0, y)                 # the phase wraps
        if np.all((y >= 0) & (y <= 1)):
            ly = prob.logp(y)
            if math.log(rng.random()) < ly - lp:
                x, lp = y, ly
        out[t] = x
    dl = wrap_diff(out[n_steps // 5:], prob.true, prob.periodic)
    return np.where(prob.periodic, (prob.true + dl.mean(0)) % 1.0, prob.true + dl.mean(0)), dl.std(0)


# ============================================================================ analysis
def truth(cfg):
    if cfg == "narrow":
        return NARROW.comps[0][1], NARROW.comps[0][2]
    if cfg == "wavelet":
        with open(DATA / "adaptive_grid_wavelet_ref.pkl", "rb") as f:
            return pickle.load(f)
    return None


def checkpoints(r, n_pts=40):
    n = len(r["samples"])
    ks = np.unique(np.geomspace(20, n, n_pts).astype(int))
    return ks, ks * r["thin"]


def error_curve(r, mu, sd):
    """max over parameters of |running posterior mean - true mean| / true spread (second-half window)."""
    ks, its = checkpoints(r)
    s, ex = r["samples"], r.get("explore", np.zeros(len(r["samples"]), bool))
    err = []
    for k in ks:
        keep = ~ex[k // 2:k]
        per = WAVELET_PERIODIC if r["cfg"] == "wavelet" else np.zeros(len(mu), bool)
        err.append(np.max(np.abs(wrap_diff(s[k // 2:k][keep], mu, per).mean(0)) / sd) if keep.any() else np.nan)
    return its, np.array(err)


def decoy_curve(r):
    ks, its = checkpoints(r)
    s, ex = r["samples"], r.get("explore", np.zeros(len(r["samples"]), bool))
    near = np.linalg.norm(s - DECOY.comps[0][1], axis=1) < 0.06
    return its, np.array([near[k // 2:k][~ex[k // 2:k]].mean() if (~ex[k // 2:k]).any() else np.nan for k in ks])


LAB = {"fixed": "Fixed grid", "fixed16": "Fixed grid, 16 bits",
       "fixed16_step": "16 bits, adaptive step", "zoom_in": "Zoom in",
       "zoom_inout": "Zoom in and out", "restart": "Zoom with look-around",
       "multiwin": "Multiple windows with look-around"}
COL = {"fixed": "#456990", "fixed16": "#7b9bb5", "fixed16_step": "#287271",
       "zoom_in": "#287271", "zoom_inout": "#762a83", "restart": "#2166ac",
       "multiwin": "#008c95"}
LS = {"fixed": "-", "fixed16": "--", "fixed16_step": ":", "zoom_in": "-", "zoom_inout": "-", "restart": "-", "multiwin": "-."}
INK, MUTED, GRIDC = "#222222", "#666666", "#dddddd"


def style():
    import paper_style
    paper_style.apply()


def finish(ax):
    ax.grid(color=GRIDC, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    if os.environ.get("FIG_PREVIEW_DIR"):           # PNG previews only outside the repo (PDF is the figure)
        fig.savefig(Path(os.environ["FIG_PREVIEW_DIR"]) / f"{name}.png", dpi=110, bbox_inches="tight")
    print("saved", OUT / f"{name}.pdf")


def band(ax, x, ys, color, ls, label, logy=True):
    ys = np.array(ys)
    lo, med, hi = np.nanpercentile(ys, [25, 50, 75], axis=0)
    ax.fill_between(x, lo, hi, color=color, alpha=0.16, lw=0)
    ax.plot(x, med, ls, color=color, lw=1.3, label=label)


def standalone_figures(res):
    """Save one axes per PDF so the paper can compose panels in LaTeX."""
    from matplotlib import pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle
    import paper_style
    style()

    # Window examples: one target and one AMR strategy per file.
    target_cmap = LinearSegmentedColormap.from_list("target_density", ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5"])
    seq = LinearSegmentedColormap.from_list("window_time", ["#efedf5", "#bcbddc", "#807dba", "#4a1486"])
    examples = [("narrow", "zoom_inout"), ("decoy", "zoom_in"),
                ("decoy", "restart"), ("decoy", "multiwin")]
    grid_axis = np.linspace(0, 1, 250)
    for cfg, strategy in examples:
        prob = NARROW if cfg == "narrow" else DECOY
        z = np.array([[prob.logp(np.array([x, y])) for x in grid_axis] for y in grid_axis])
        fig, ax = plt.subplots(figsize=(4.0, 3.5))
        ax.pcolormesh(grid_axis, grid_axis, np.exp(z - z.max()) ** .35,
                      cmap=target_cmap, shading="auto", rasterized=True)
        r = next(r for r in res if r["cfg"] == cfg and r["strategy"] == strategy
                 and r["seed"] == 1 and not r["overrides"])
        wins = r["windows"]
        for k, (_, lo, w) in enumerate(wins):
            if k % max(1, len(wins) // 12) and k != len(wins) - 1:
                continue
            ax.add_patch(Rectangle(lo, w[0], w[1], fill=False, lw=.85,
                                   ec=seq(k / max(1, len(wins) - 1))))
        samples = r["samples"][-400:]
        ax.plot(samples[:, 0], samples[:, 1], ".", ms=1.3, color="#b2182b", alpha=.6)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Parameter 1", ylabel="Parameter 2", aspect="equal")
        finish(ax)
        sm = plt.cm.ScalarMappable(cmap=seq, norm=plt.Normalize(0, 1))
        cb = fig.colorbar(sm, ax=ax, fraction=.046, pad=.04)
        cb.set_label("Re-grid order")
        fig.tight_layout()
        fig.savefig(OUT / f"adaptive_grid_example_{cfg}_{strategy}.pdf", bbox_inches="tight")
        plt.close(fig)

    base = [r for r in res if not r["overrides"]]
    mu_n, sd_n = truth("narrow")
    mu_w, sd_w = truth("wavelet")

    # Time histories on each target.
    for cfg, mu, sd, filename in (("narrow", mu_n, sd_n, "adaptive_grid_narrow_error"),
                                  ("wavelet", mu_w, sd_w, "adaptive_grid_wavelet_error")):
        fig, ax = plt.subplots(figsize=(4.8, 3.4))
        for strategy in STRATEGIES:
            curves = [error_curve(r, mu, sd) for r in base
                      if r["cfg"] == cfg and r["strategy"] == strategy]
            band(ax, curves[0][0], [c[1] for c in curves], COL[strategy], LS[strategy], LAB[strategy])
        ax.set(xscale="log", yscale="log", xlabel="Walk steps",
               ylabel="Mean error (true spreads)")
        ax.axhline(1, color=MUTED, lw=.6, ls=":")
        finish(ax)
        ax.legend(frameon=False, ncol=2, loc="best")
        fig.tight_layout()
        fig.savefig(OUT / f"{filename}.pdf", bbox_inches="tight")
        plt.close(fig)

    # Decoy recovery over time.
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    for strategy in STRATEGIES:
        curves = [decoy_curve(r) for r in base if r["cfg"] == "decoy" and r["strategy"] == strategy]
        x = curves[0][0]
        found = np.array([c[1] >= .5 for c in curves]).mean(0)
        ax.plot(x, found, LS[strategy], color=COL[strategy], lw=1.25, label=LAB[strategy])
    ax.set(xscale="log", ylim=(-.03, 1.03), xlabel="Walk steps", ylabel="Runs in target peak")
    finish(ax); ax.legend(frameon=False, ncol=2, loc="best")
    fig.tight_layout(); fig.savefig(OUT / "adaptive_grid_decoy_recovery.pdf", bbox_inches="tight"); plt.close(fig)

    # Resolution against total register bits.
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    for strategy in ("fixed", "zoom_inout", "restart", "multiwin"):
        med, lo, hi = [], [], []
        for bits in BITS_SWEEP:
            err = [error_curve(r, mu_n, sd_n)[1][-1] for r in res if r["cfg"] == "narrow"
                   and r["strategy"] == strategy and r["overrides"].get("n_bits") == bits]
            med.append(np.median(err)); lo.append(np.percentile(err, 25)); hi.append(np.percentile(err, 75))
        ax.errorbar(BITS_SWEEP, med, yerr=[np.array(med)-lo, np.array(hi)-med], marker="o", ms=3.5,
                    color=COL[strategy], lw=1.15, capsize=0, label=LAB[strategy])
    ax.set(yscale="log", xticks=BITS_SWEEP, xlabel="Register bits per parameter",
           ylabel="Final mean error (true spreads)")
    ax.axhline(1, color=MUTED, lw=.6, ls=":")
    finish(ax); ax.legend(frameon=False, ncol=2, loc="best")
    fig.tight_layout(); fig.savefig(OUT / "adaptive_grid_register_precision.pdf", bbox_inches="tight"); plt.close(fig)

    # Window-width criterion K, with a separate figure for each measured outcome.
    for case, filename in (("decoy", "adaptive_grid_K_decoy"), ("narrow", "adaptive_grid_K_narrow")):
        fig, ax = plt.subplots(figsize=(4.8, 3.4))
        for strategy in ("zoom_in", "zoom_inout", "restart", "multiwin"):
            vals = []
            for kval in K_SWEEP:
                cur = [decoy_curve(r)[1][-1] for r in res if case == "decoy" and r["cfg"] == case
                       and r["strategy"] == strategy and r["overrides"].get("K") == kval]
                if case == "decoy":
                    vals.append(np.mean(np.asarray(cur) >= .5))
                else:
                    cur = [error_curve(r, mu_n, sd_n)[1][-1] for r in res if r["cfg"] == case
                           and r["strategy"] == strategy and r["overrides"].get("K") == kval]
                    vals.append(np.median(cur))
            ax.plot(K_SWEEP, vals, "o-", color=COL[strategy], ms=3.5, lw=1.15, label=LAB[strategy])
        if case == "decoy":
            ax.set(ylabel="Runs in target peak", ylim=(-.03, 1.03))
        else:
            ax.set_yscale("log"); ax.set_ylabel("Final mean error (true spreads)")
            ax.axhline(1, color=MUTED, lw=.6, ls=":")
        ax.set_xscale("log"); ax.set_xticks(K_SWEEP, [str(int(k)) for k in K_SWEEP])
        ax.set_xlabel("Window width / recent spread (K)")
        finish(ax); ax.legend(frameon=False, ncol=2, loc="best")
        fig.tight_layout(); fig.savefig(OUT / f"{filename}.pdf", bbox_inches="tight"); plt.close(fig)


def plot_all():
    with open(DATA / "adaptive_grid.pkl", "rb") as f:
        res = pickle.load(f)
    standalone_figures(res)
    summary = {}
    for st in STRATEGIES:
        rr = [r for r in res if not r["overrides"] and r["strategy"] == st]
        for cfg in ("narrow", "wavelet"):
            mu, sd = truth(cfg)
            summary[(cfg, st)] = np.median([error_curve(r, mu, sd)[1][-1] for r in rr if r["cfg"] == cfg])
        curves = [decoy_curve(r) for r in rr if r["cfg"] == "decoy"]
        found = np.array([c[1] >= .5 for c in curves]).mean(0)
        summary[("decoy", st)] = (found[-1], np.median([c[1][-1] for c in curves]))
    print("\nfinal error (true spreads, median)       narrow    wavelet | decoy: runs that found peak, time in peak")
    for st in STRATEGIES:
        print(f"  {LAB[st]:32s} {summary[('narrow', st)]:8.3f} {summary[('wavelet', st)]:9.3f} | "
              f"{summary[('decoy', st)][0]:5.0%}  {summary[('decoy', st)][1]:5.2f} (correct 0.70)")


if __name__ == "__main__":
    if "--plot-only" not in sys.argv:
        mu, sd = wavelet_reference()
        with open(DATA / "adaptive_grid_wavelet_ref.pkl", "wb") as f:
            pickle.dump((mu, sd), f)
        print("wavelet reference mean", np.round(mu, 5), "spread", np.round(sd, 6), flush=True)
    plot_all()
