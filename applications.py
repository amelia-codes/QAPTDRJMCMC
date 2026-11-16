"""Preliminary checks: QCLT + AMR (qclt_amr.py) on posteriors other than gravitational-wave wavelets.

Six problems, all with SIMULATED data (no real measurements), each with a reference answer that does not come
from any of the samplers being compared:
  gauss     6-D Gaussian, strongly correlated (condition number 10^4)        exact
  banana    4-D "twisted Gaussian" of Haario et al. (1999), b = 0.05         exact (direct draws)
  logistic  Bayesian logistic regression, 5 coefficients, 300 data points    importance sampling around the mode
  rv        exoplanet radial velocities, one Keplerian orbit (P, K, e, w, M0) importance sampling around the mode
  hep       particle physics: a resonance on a falling background in a binned invariant-mass spectrum
            (mass, resolution, signal yield, background yield, background slope; Poisson likelihood)
  cm        condensed matter: low-temperature heat capacity C = gamma T + beta T^3 + Schottky anomaly
            (gamma, beta, two-level fraction n, gap Delta)
Samplers (qclt_amr.SAMPLERS), at equal likelihood-evaluation budgets: tuned random walk (classical), Adaptive
Metropolis (classical), QCLT on 16-bit registers (quantum kick, simulated), QCLT + AMR (ours; quantum kick,
simulated). No tempering and no trans-dimensional moves here: these are fixed-dimension, single-mode problems.

Run:   python3.12 applications.py              # references + 192 runs + figures/previous/applications_*.pdf
       python3.12 applications.py --plot-only
"""
import math
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

import qclt_amr as Q

OUT = Path(__file__).parent / "figures" / "previous"
DATA = Path(__file__).parent / "data"
SEEDS = 8
N_ITER, N_ADAPT = 60_000, 30_000
INK, MUTED, GRIDC = "#0b0b0b", "#52514e", "#e4e3df"
COL = {"rw": "#52514e", "am": "#1baf7a", "qclt": "#e87ba4", "qamr_axis": "#eda100", "qamr": "#d62f2f"}
LS = {"rw": "--", "am": "-", "qclt": "-", "qamr_axis": "-.", "qamr": "-"}


class Box:
    """Uniform prior on a box; the samplers work in unit-cube coordinates x."""
    periodic = None

    def to_phys(self, x):
        return self.lo + np.asarray(x) * (self.hi - self.lo)

    def to_unit(self, th):
        return (np.asarray(th) - self.lo) / (self.hi - self.lo)

    def start(self, rng):                         # random start in the central half of the box
        return rng.uniform(0.25, 0.75, self.d)


# ----------------------------------------------------------------------------- problems
class Gauss(Box):
    key, field, title = "gauss", "test", "Correlated Gaussian (6-D)"
    names = [f"$x_{i + 1}$" for i in range(6)]

    def __init__(self):
        rng = np.random.default_rng(11)
        self.d = 6
        self.lo, self.hi = np.zeros(6), np.ones(6)
        self.periodic = np.zeros(6, bool)
        Qm, _ = np.linalg.qr(rng.normal(size=(6, 6)))
        self.C = Qm @ np.diag(np.geomspace(7e-4, 7e-2, 6) ** 2) @ Qm.T
        self.mu = rng.uniform(0.4, 0.6, 6)
        self.P = np.linalg.inv(self.C)

    def logp(self, x):
        z = x - self.mu
        return -0.5 * z @ self.P @ z

    def reference(self, rng, n):
        s = rng.multivariate_normal(self.mu, self.C, size=2 * n)
        return s[np.all((s > 0) & (s < 1), axis=1)][:n], None


class Banana(Box):
    key, field, title = "banana", "test", "Twisted Gaussian, Haario et al. (4-D)"
    names = ["$y_1$", "$y_2$", "$y_3$", "$y_4$"]
    B = 0.05

    def __init__(self):
        self.d = 4
        self.lo, self.hi = np.array([-40.0, -70.0, -6.0, -6.0]), np.array([40.0, 12.0, 6.0, 6.0])
        self.periodic = np.zeros(4, bool)

    def logp(self, x):
        y = self.to_phys(x)
        return -y[0] ** 2 / 200 - 0.5 * (y[1] + self.B * (y[0] ** 2 - 100)) ** 2 - 0.5 * (y[2] ** 2 + y[3] ** 2)

    def reference(self, rng, n):
        z = rng.normal(size=(3 * n, 4))
        y = np.column_stack([10 * z[:, 0], z[:, 1] - self.B * (100 * z[:, 0] ** 2 - 100), z[:, 2], z[:, 3]])
        x = self.to_unit(y)
        return x[np.all((x > 0) & (x < 1), axis=1)][:n], None


class Logistic(Box):
    key, field, title = "logistic", "statistics", "Bayesian logistic regression (5-D)"
    names = [r"$\beta_0$", r"$\beta_1$", r"$\beta_2$", r"$\beta_3$", r"$\beta_4$"]

    def __init__(self):
        rng = np.random.default_rng(12)
        self.d = 5
        self.lo, self.hi = np.full(5, -4.0), np.full(5, 4.0)
        self.periodic = np.zeros(5, bool)
        C = 0.5 * np.ones((4, 4)) + 0.5 * np.eye(4)
        self.X = np.column_stack([np.ones(300), rng.multivariate_normal(np.zeros(4), C, size=300)])
        self.truth = np.array([-0.5, 1.0, -1.5, 0.8, 0.3])
        self.y = (rng.random(300) < 1 / (1 + np.exp(-self.X @ self.truth))).astype(float)

    def logp(self, x):
        eta = self.X @ self.to_phys(x)
        return float(self.y @ eta - np.logaddexp(0, eta).sum())


class RadialVelocity(Box):
    key, field, title = "rv", "astronomy", "Exoplanet radial velocity (5-D)"
    names = ["$P$ [d]", "$K$ [m/s]", "$e$", r"$\omega$", "$M_0$"]

    def __init__(self):
        rng = np.random.default_rng(13)
        self.d = 5
        # P searched around the periodogram peak (as a data-driven start would), angles periodic
        self.lo, self.hi = np.array([23.0, 0.0, 0.0, 0.0, 0.0]), np.array([25.6, 30.0, 0.8, 2 * np.pi, 2 * np.pi])
        self.periodic = np.array([False, False, False, True, True])
        self.t = np.sort(rng.uniform(0, 200, 40))
        self.sigma = 2.0
        self.truth = np.array([24.3, 12.0, 0.3, 1.2, 2.0])
        self.v = self.model(self.truth) + rng.normal(size=40) * self.sigma

    def model(self, th):
        P, K, e, w, M0 = th
        M = M0 + 2 * np.pi * self.t / P
        E = M.copy()
        for _ in range(30):                          # Newton on Kepler's equation
            dE = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
            E -= dE
            if np.max(np.abs(dE)) < 1e-12:
                break
        nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))
        return K * (np.cos(nu + w) + e * np.cos(w))

    def logp(self, x):
        r = self.v - self.model(self.to_phys(np.mod(x, 1.0)))
        return -0.5 * float(r @ r) / self.sigma**2


class Resonance(Box):
    key, field, title = "hep", "particle physics", "Resonance on a falling background (5-D)"
    names = [r"$m$ [GeV]", r"$\sigma_m$ [GeV]", "$S$ [events]", "$B$ [events]", r"$\lambda$ [1/GeV]"]

    def __init__(self):
        rng = np.random.default_rng(14)
        self.d = 5
        self.lo, self.hi = np.array([115.0, 0.5, 0.0, 18000.0, 0.02]), np.array([135.0, 5.0, 1500.0, 22000.0, 0.04])
        self.periodic = np.zeros(5, bool)
        self.edges = np.arange(100.0, 161.0, 1.0)              # 60 bins of 1 GeV, 100-160 GeV
        self.mid = 0.5 * (self.edges[1:] + self.edges[:-1])
        self.truth = np.array([125.0, 2.0, 600.0, 20000.0, 0.03])
        self.n = rng.poisson(self.expected(self.truth)).astype(float)

    def expected(self, th):
        m, s, S, B, lam = th
        bkg = B * lam * np.exp(-lam * (self.mid - 100.0)) / (1 - np.exp(-60.0 * lam))
        sig = S * np.exp(-0.5 * ((self.mid - m) / s) ** 2) / (np.sqrt(2 * np.pi) * s)
        return bkg + sig                                     # per 1-GeV bin (midpoint rule)

    def logp(self, x):
        mu = self.expected(self.to_phys(x))
        return float(self.n @ np.log(mu) - mu.sum())


class HeatCapacity(Box):
    key, field, title = "cm", "condensed matter", "Heat capacity with a Schottky anomaly (4-D)"
    names = [r"$\gamma$ [mJ/mol K$^2$]", r"$\beta$ [mJ/mol K$^4$]", "$n$", r"$\Delta$ [K]"]
    R = 8314.46                                              # mJ / (mol K)

    def __init__(self):
        rng = np.random.default_rng(15)
        self.d = 4
        self.lo, self.hi = np.array([0.0, 0.2, 0.0, 1.0]), np.array([15.0, 0.6, 0.06, 15.0])
        self.periodic = np.zeros(4, bool)
        self.T = np.geomspace(1.0, 20.0, 60)
        self.truth = np.array([5.0, 0.4, 0.02, 6.0])
        c = self.model(self.truth)
        self.sig = 0.02 * c + 0.5
        self.c = c + rng.normal(size=60) * self.sig

    def model(self, th):
        g, b, n, D = th
        z = D / self.T
        return g * self.T + b * self.T**3 + n * self.R * z**2 * np.exp(-z) / (1 + np.exp(-z)) ** 2

    def logp(self, x):
        r = (self.c - self.model(self.to_phys(x))) / self.sig
        return -0.5 * float(r @ r)


PROBLEMS = [Gauss, Banana, Logistic, RadialVelocity, Resonance, HeatCapacity]
PAIRS = {"gauss": None, "banana": (0, 1), "logistic": None, "rv": (2, 3), "hep": (0, 2), "cm": (0, 1)}


# ----------------------------------------------------------------------------- references
def importance_reference(prob, rng, n):
    """Posterior draws by importance sampling: Student-t (5 d.o.f.) proposal around the mode, covariance 1.5^2 x
    inverse Hessian; periodic parameters handled in local (unwrapped) coordinates. Returns resampled draws (unit
    coordinates) and the importance-sampling effective sample size."""
    from scipy.optimize import minimize
    x0 = prob.to_unit(prob.truth)
    f = lambda x: -prob.logp(np.where(prob.periodic, x, np.clip(x, 1e-9, 1 - 1e-9)))
    m = minimize(f, x0, method="Nelder-Mead", options=dict(xatol=1e-10, fatol=1e-10, maxiter=40_000)).x
    m = minimize(f, m, method="BFGS").x
    h = 1e-5
    H = np.empty((prob.d, prob.d))
    for i in range(prob.d):
        for j in range(prob.d):
            ei, ej = np.eye(prob.d)[i] * h, np.eye(prob.d)[j] * h
            H[i, j] = (f(m + ei + ej) - f(m + ei - ej) - f(m - ei + ej) + f(m - ei - ej)) / (4 * h * h)
    C = 1.5**2 * np.linalg.inv(0.5 * (H + H.T))
    L = np.linalg.cholesky(C)
    nu = 5
    z = rng.normal(size=(n, prob.d)) / np.sqrt(rng.chisquare(nu, size=(n, 1)) / nu)
    x = m + z @ L.T
    inside = np.all(prob.periodic | ((x > 0) & (x < 1)), axis=1)
    logq = -0.5 * (nu + prob.d) * np.log1p(np.einsum("ij,ij->i", z, z) / nu)
    logw = np.full(n, -np.inf)
    logw[inside] = np.array([prob.logp(np.mod(xi, 1.0)) for xi in x[inside]]) - logq[inside]
    w = np.exp(logw - logw.max())
    ess = w.sum() ** 2 / (w @ w)
    idx = rng.choice(n, size=n // 4, p=w / w.sum())
    return np.where(prob.periodic, np.mod(x[idx], 1.0), x[idx]), ess


def references():
    rng = np.random.default_rng(2026)
    out = {}
    for P in PROBLEMS:
        prob = P()
        draws, ess = prob.reference(rng, 200_000) if hasattr(prob, "reference") else importance_reference(prob, rng, 400_000)
        out[prob.key] = dict(draws=draws.astype(np.float32), is_ess=ess)
        print(f"{prob.key:9s} reference: {len(draws)} draws" + (f", importance-sampling ESS {ess:.0f}" if ess else " (exact)"),
              flush=True)
    return out


def ref_stats(draws, per):
    """Mean and spread (circular about the mean for periodic parameters)."""
    c = np.where(per, np.angle(np.exp(2j * np.pi * draws).mean(0)) / (2 * np.pi) % 1.0, draws.mean(0))
    dl = Q.wrapd(draws - c, per, 1.0)
    return c, dl.std(0)


# ----------------------------------------------------------------------------- runs
def _job(args):
    k, sampler, seed = args
    prob = PROBLEMS[k]()
    t = time.process_time()
    r = Q.run(prob, sampler, 700 + seed, N_ITER, N_ADAPT)
    r.update(problem=prob.key, sampler=sampler, seed=seed, cpu=time.process_time() - t)
    return r


def run_all():
    from multiprocessing import Pool
    tv, nq = Q.validate_kick()
    print(f"QCLT kick vs qcov circuit: TV distance {tv:.1e} ({nq} qubits)", flush=True)
    assert tv < 1e-9
    refs = references()
    jobs = [(k, s, seed) for k in range(len(PROBLEMS)) for s in Q.SAMPLERS for seed in range(SEEDS)]
    t = time.time()
    with Pool(int(os.environ.get("N_WORKERS", 6))) as pool:
        res = pool.map(_job, jobs, chunksize=1)
    print(f"{len(jobs)} runs in {time.time() - t:.0f}s", flush=True)
    with open(DATA / "applications.pkl", "wb") as f:
        pickle.dump(dict(refs=refs, runs=res), f)


# ----------------------------------------------------------------------------- analysis
def error_curve(r, mu, sd, per):
    """max over parameters of |posterior mean so far (second half) - reference mean| / reference spread."""
    x = r["x"].astype(float)
    ks = np.unique(np.geomspace(20, len(x), 60).astype(int))
    err = [np.max(np.abs(Q.wrapd(x[k // 2:k] - mu, per, 1.0).mean(0)) / sd) for k in ks]
    return r["evals"][ks - 1].astype(float), np.array(err)


def ess_per_1k(r, mu, per):
    """Smallest effective sample size over the parameters per 1000 likelihood evaluations, after adaptation."""
    a = r["n_adapt"] // r["thin"]
    x = Q.wrapd(r["x"][a:].astype(float) - mu, per, 1.0)
    ess = min(len(x) / Q.iact(x[:, p]) for p in range(x.shape[1]))
    return 1000 * ess / max(r["evals"][-1] - r["evals"][a], 1)


def window_outline(win, i, j):
    """Corners of the projection onto parameters (i, j) of a QCLT + AMR window (a box in principal-axis
    coordinates), in unit coordinates, as a closed convex polygon."""
    import itertools
    from scipy.spatial import ConvexHull
    hw = win["s"] * win["nreg"] / 2
    corners = np.array([win["c"] + win["V"] @ (np.array(sg) * hw) for sg in itertools.product((-1, 1), repeat=len(hw))])
    P = corners[:, [i, j]]
    return P[ConvexHull(P).vertices]


def most_correlated_pair(draws):
    c = np.abs(np.triu(np.corrcoef(np.asarray(draws, float).T), 1))
    return tuple(int(v) for v in np.unravel_index(np.argmax(c), c.shape))


def style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8,
        "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def finish(ax):
    ax.grid(color=GRIDC, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


QC_NOTE = ("Simulated data throughout. Classical: runs entirely on a classical computer. Quantum kick, simulated: the "
           "adaptive step is drawn from the QCLT circuit,\nsimulated exactly on a classical computer (no quantum "
           "hardware); acceptance, history and windows are classical. Reference answers come from exact draws or "
           "importance sampling,\nnot from any of these samplers.")


def save(fig, name):
    fig.text(0.01, -0.01, QC_NOTE, fontsize=6.0, color=MUTED, va="top")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    if os.environ.get("FIG_PREVIEW_DIR"):
        fig.savefig(Path(os.environ["FIG_PREVIEW_DIR"]) / f"{name}.png", dpi=110, bbox_inches="tight")
    print("saved", OUT / f"{name}.pdf")


def plot(data):
    from matplotlib import pyplot as plt
    style()
    refs, runs = data["refs"], data["runs"]
    probs = [P() for P in PROBLEMS]
    # ---- figure 1: the posteriors, with one QCLT + AMR run and its final windows
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 5.0))
    for ax, prob in zip(axes.ravel(), probs):
        ref = refs[prob.key]["draws"]
        i, j = PAIRS[prob.key] or most_correlated_pair(ref)
        R = prob.to_phys(ref[:40_000])
        ax.hist2d(R[:, i], R[:, j], bins=60, cmap="Greys", cmin=1, rasterized=True)
        r = next(x for x in runs if x["problem"] == prob.key and x["sampler"] == "qamr" and x["seed"] == 0)
        S = prob.to_phys(r["x"][r["n_adapt"] // r["thin"]:][::4])
        ax.plot(S[:, i], S[:, j], ".", ms=1.6, color=COL["qamr"], alpha=0.45, rasterized=True)
        hull = window_outline(r["window"], i, j)          # exact projection of the final window onto (i, j)
        H = np.column_stack([prob.lo[[i, j]] + hull * (prob.hi - prob.lo)[[i, j]]])
        ax.plot(*np.vstack([H, H[:1]]).T, "--", color=COL["qamr"], lw=0.9)
        ax.set_xlabel(prob.names[i])
        ax.set_ylabel(prob.names[j])
        ax.set_title(f"{prob.title}\n({prob.field})", loc="left", fontsize=7.6)
        finish(ax)
    axes[0, 0].plot([], [], "s", color="#9d9c97", label="reference posterior")
    axes[0, 0].plot([], [], ".", color=COL["qamr"], label="QCLT + AMR samples (one run)")
    axes[0, 0].plot([], [], "--", color=COL["qamr"], label="final 6-bit register window (projection)")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "applications_posteriors")

    # ---- figure 2: accuracy against likelihood evaluations, and efficiency after adaptation
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 5.0))
    summary = {}
    for ax, prob in zip(axes.ravel(), probs):
        mu, sd = ref_stats(refs[prob.key]["draws"].astype(float), prob.periodic)
        for s in Q.SAMPLERS:
            rr = [x for x in runs if x["problem"] == prob.key and x["sampler"] == s]
            curves = [error_curve(x, mu, sd, prob.periodic) for x in rr]
            grid = np.geomspace(max(c[0][0] for c in curves), min(c[0][-1] for c in curves), 80)
            ys = np.array([np.interp(grid, c[0], c[1]) for c in curves])
            lo_, med, hi_ = np.percentile(ys, [25, 50, 75], axis=0)
            ax.fill_between(grid, lo_, hi_, color=COL[s], alpha=0.15, lw=0)
            ax.plot(grid, med, LS[s], color=COL[s], lw=1.3, label=Q.LABEL[s])
            summary[(prob.key, s)] = dict(err=float(np.median([c[1][-1] for c in curves])),
                                          ess=float(np.median([ess_per_1k(x, mu, prob.periodic) for x in rr])),
                                          acc=float(np.nanmedian([x["acc"]["ad"] for x in rr])))
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.axhline(0.1, color=MUTED, lw=0.6, ls=":")
        ax.set_xlabel("likelihood evaluations")
        ax.set_ylabel("error of posterior mean\n(reference spreads)")
        ax.set_title(f"{prob.title}\n({prob.field})", loc="left", fontsize=7.6)
        finish(ax)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "applications_accuracy")

    print(f"\n{'problem':9s} {'sampler':5s} {'final error':>11s} {'ESS/1k evals':>12s} {'adaptive acc.':>13s} {'kick qubits':>11s}")
    for prob in probs:
        for s in Q.SAMPLERS:
            v = summary[(prob.key, s)]
            nq = {"qclt": Q.kick_qubits(prob.d, Q.NB), "qamr": Q.kick_qubits(prob.d, Q.AMR_BITS)}.get(s, 0)
            print(f"{prob.key:9s} {s:5s} {v['err']:11.3f} {v['ess']:12.2f} {v['acc']:13.3f} {nq:11d}")
    return summary


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    DATA.mkdir(exist_ok=True)
    if "--plot-only" not in sys.argv:
        run_all()
    with open(DATA / "applications.pkl", "rb") as f:
        plot(pickle.load(f))
