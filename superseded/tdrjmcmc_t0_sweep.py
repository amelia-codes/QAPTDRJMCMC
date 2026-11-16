"""Is the adaptive (APT) move worth it, and when should it switch on?

The adaptive move (AP: Tian et al.; QCLT-AP: the quantum version) switches on at iteration t0. Before t0,
and as the fixed fallback afterwards, the sampler uses the same untuned random step as the naive sampler
(Tian et al.'s conventional proposal), so t0 -> never is exactly the naive sampler and the comparison
isolates the adaptive move. Nothing here uses derivatives, except the Fisher reference.

Scored two ways:
  * model updates per chain until the match with the injected signal first reaches 0.95
    (one of our iterations = one update per chain; one BayesWave iteration = 100);
  * waveform evaluations until then: the cost of an oracle call on a quantum computer. Fisher moves need
    about 20 extra waveform evaluations each for the derivatives. For BayesWave the count is approximated
    as model updates x chains (one likelihood per update), a lower bound.

References: naive steps, Fisher (derivatives), and real BayesWave (data/snr_sweep.pkl for the easy case,
data/bayeswave_hard.pkl for the hard case; BayesWave's match is the posterior median at saved checkpoints).

Run:   python3.12 tdrjmcmc_t0_sweep.py              # runs + figures/tdrjmcmc_t0_sweep.pdf
       python3.12 tdrjmcmc_t0_sweep.py --plot-only
"""
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

import tdrjmcmc_qclt as T

T0S = [10, 30, 100, 300, 1000, 3000]
METHODS = ["ap_nb", "qclt_nb"]
REFS = ["rw", "fisher"]
N_SEEDS = 8
THRESH = 0.95
CONDITIONS = [
    dict(name="easy", label="Easy case: short chirp, SNR 40, new wavelets placed where the data has power",
         data="chirp", snr=40.0, birth="tf", kmax=10, tmax=800.0, n_iter=12_000, bw="snr_sweep"),
    dict(name="hard", label="Hard case: wide chirp + extra burst, SNR 40, new wavelets placed at random",
         data="hard", snr=40.0, birth="prior", kmax=20, tmax=1600.0, n_iter=30_000, bw="hard"),
]
OUT = Path(__file__).parent / "figures"
DATA = Path(__file__).parent / "data"            # run data and logs (figures/ holds PDFs only)
NAME = {"ap_nb": "AP (Tian et al.)", "qclt_nb": "QCLT-AP (quantum)", "rw": "Naive (fixed random steps)",
        "fisher": "Fisher (uses derivatives)", "bw": "BayesWave (real)"}
COL = {"ap_nb": "#1baf7a", "qclt_nb": "#6250d6", "rw": "#52514e", "fisher": "#2a78d6", "bw": "#0b0b0b"}


def configure(c):
    T.DATASET, T.SNR_INJ, T.BIRTH, T.K_MAX, T.T_MAX = c["data"], c["snr"], c["birth"], c["kmax"], c["tmax"]
    T.N_MIN_HIST, T.AP_FALLBACK = 10, "rw"


def _job(args):
    cond, method, t0, seed = args
    configure(cond)
    if t0 is not None:
        T.T0 = t0
    n = cond["n_iter"] if method != "fisher" else 12_000
    r = T.run(method, 500 + seed, n, n, record_waves=False)
    step = max(1, n // 4000)
    return dict(cond=cond["name"], method=method, t0=t0, seed=seed, match=r["match"][::step].astype(np.float32),
                evals=r["evals"][::step], iters=np.arange(0, n, step))


def run_all():
    from multiprocessing import Pool
    jobs = []
    for c in CONDITIONS:
        jobs += [(c, m, None, s) for m in REFS for s in range(N_SEEDS)]
        jobs += [(c, m, t0, s) for m in METHODS for t0 in T0S for s in range(N_SEEDS)]
    jobs.sort(key=lambda j: -j[0]["n_iter"])
    t = time.time()
    with Pool(int(os.environ.get("N_WORKERS", 8))) as pool:   # default 8 workers: memory headroom
        res = pool.map(_job, jobs, chunksize=1)
    print(f"{len(jobs)} runs in {time.time() - t:.0f}s", flush=True)
    with open(DATA / "tdrjmcmc_t0_sweep_v2.pkl", "wb") as f:
        pickle.dump(res, f)


def first_hit(r, key):
    i = np.argmax(r["match"] >= THRESH)
    return float(r[key][i]) + 1 if r["match"][i] >= THRESH else np.nan


def bayeswave_refs(cond):
    """Per BayesWave run: (updates per chain, waveform evaluations) when its median match first reaches
    THRESH, plus its checkpoint curves. Evaluations = updates x chains, a lower bound."""
    if cond["bw"] == "snr_sweep":
        with open(DATA / "snr_sweep.pkl", "rb") as f:
            runs = [r for r in pickle.load(f)["bw"] if r["snr"] == 40]
    else:
        p = DATA / "bayeswave_hard.pkl"
        if not p.exists():
            return None
        with open(p, "rb") as f:
            runs = pickle.load(f)
    nchain = 20
    hits, curves = [], []
    for r in runs:
        its = sorted(k for k, v in r["checkpoints"].items() if v is not None and len(v))
        med = [float(np.median(r["checkpoints"][k])) for k in its]
        curves.append((np.array(its) * 100.0, np.array(its) * 100.0 * nchain, np.array(med)))
        ok = [k for k, m in zip(its, med) if m >= THRESH]
        hits.append((ok[0] * 100.0, ok[0] * 100.0 * nchain) if ok else (np.nan, np.nan))
    return np.array(hits), curves


def plot():
    import matplotlib as mpl
    from matplotlib import pyplot as plt
    with open(DATA / "tdrjmcmc_t0_sweep_v2.pkl", "rb") as f:
        res = pickle.load(f)
    mpl.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8,
        "axes.titlesize": 8.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.edgecolor": T.MUTED, "axes.labelcolor": T.INK, "xtick.color": T.MUTED, "ytick.color": T.MUTED,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(len(CONDITIONS), 3, figsize=(7.6, 2.9 * len(CONDITIONS)),
                             gridspec_kw=dict(width_ratios=[1, 1, 1.15]))
    for row, c in enumerate(CONDITIONS):
        rc = [r for r in res if r["cond"] == c["name"]]
        bw = bayeswave_refs(c)
        print(f"[{c['name']}] median cost to reach match {THRESH}: updates per chain | waveform evaluations")
        summ = {}
        for ref in REFS:
            rr = [r for r in rc if r["method"] == ref]
            summ[ref] = (np.nanmedian([first_hit(r, "iters") for r in rr]), np.nanmedian([first_hit(r, "evals") for r in rr]),
                         sum(np.isfinite(first_hit(r, "iters")) for r in rr), len(rr))
        for col, key, ylab in ((0, "iters", "model updates per chain"), (1, "evals", "waveform evaluations")):
            ax = axes[row, col]
            for ref in REFS:
                ax.axhline(summ[ref][col], color=COL[ref], lw=1.1, ls="--" if ref == "rw" else "-", label=NAME[ref])
            if bw is not None:
                ax.axhline(np.nanmedian(bw[0][:, col]), color=COL["bw"], lw=1.1, ls=":",
                           label=NAME["bw"] + (" (lower bound)" if col else ""))
            for m, off in zip(METHODS, (0.94, 1.06)):
                st = []
                for t0 in T0S:
                    h = np.array([first_hit(r, key) for r in rc if r["method"] == m and r["t0"] == t0])
                    st.append(np.nanpercentile(h, [25, 50, 75]) if np.isfinite(h).any() else [np.nan] * 3)
                st = np.array(st)
                ax.errorbar(np.array(T0S) * off, st[:, 1], yerr=[st[:, 1] - st[:, 0], st[:, 2] - st[:, 1]],
                            fmt="o-" if m == "qclt_nb" else "s-", ms=4, color=COL[m], lw=1.1, capsize=0,
                            mec="white", mew=0.6, label=NAME[m])
                summ[(m, col)] = st[:, 1]
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel(r"$t_0$: iteration the adaptive move switches on")
            ax.set_ylabel(f"{ylab}\nto reach match {THRESH} (lower = cheaper)")
            ax.set_title(f"({'abcdef'[3 * row + col]}) " + ("Cost in iterations" if col == 0 else
                         "Cost in waveform evaluations"), loc="left", fontsize=7.8)
        for ref in REFS:
            print(f"   {NAME[ref]:28s} {summ[ref][0]:10.0f} | {summ[ref][1]:10.3g}   ({summ[ref][2]}/{summ[ref][3]} runs)")
        if bw is not None:
            print(f"   {NAME['bw']:28s} {np.nanmedian(bw[0][:, 0]):10.0f} | {np.nanmedian(bw[0][:, 1]):10.3g}   (lower bound)")
        for m in METHODS:
            print(f"   {NAME[m]:28s} " + "  ".join(f"t0={t0}: {a:.0f} | {b:.3g}" for t0, a, b in
                                                  zip(T0S, summ[(m, 0)], summ[(m, 1)])))
        # (c) match vs waveform evaluations
        ax = axes[row, 2]
        grid = np.geomspace(1e3, max(r["evals"][-1] for r in rc), 250)
        for m, t0, ls, col in [("rw", None, "--", COL["rw"]), ("fisher", None, "-", COL["fisher"]),
                               ("qclt_nb", 10, ":", "#b7aef2"), ("qclt_nb", 300, "-", COL["qclt_nb"]),
                               ("qclt_nb", 3000, "-.", "#3a2d9c"), ("ap_nb", 300, "-", COL["ap_nb"])]:
            rr = [r for r in rc if r["method"] == m and r["t0"] == t0]
            ys = np.array([np.interp(grid, r["evals"], r["match"], right=np.nan) for r in rr])
            ax.plot(grid, np.nanmedian(ys, 0), ls, color=col, lw=1.2,
                    label=NAME[m] + (f", t0 = {t0}" if t0 else ""))
        if bw is not None:
            for k, (u, e, mm) in enumerate(bw[1]):
                ax.plot(e, mm, "D-", ms=2.5, lw=0.5, color=COL["bw"], alpha=0.55,
                        label="BayesWave runs (lower bound)" if k == 0 else None)
        ax.axhline(THRESH, color=T.MUTED, lw=0.6, ls=":")
        ax.set_xscale("log"); ax.set_ylim(0, 1)
        ax.set_xlabel("waveform evaluations")
        ax.set_ylabel("match with the injected signal\n(median of runs)")
        ax.set_title(f"({'abcdef'[3 * row + 2]}) Match vs waveform evaluations", loc="left", fontsize=7.8)
        ax.legend(loc="lower right", frameon=False, fontsize=5.8)
        axes[row, 0].text(0.0, 1.2, c["label"], transform=axes[row, 0].transAxes, fontsize=8.5, color=T.INK,
                          fontweight="bold")
        for a in axes[row]:
            a.grid(color=T.GRIDC, lw=0.6); a.set_axisbelow(True)
            for sp in ("top", "right"):
                a.spines[sp].set_visible(False)
    h, l = axes[0, 1].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.03), fontsize=7)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.08, wspace=0.45, hspace=0.85)
    fig.savefig(OUT / "tdrjmcmc_t0_sweep.pdf", bbox_inches="tight")
    if os.environ.get("FIG_PREVIEW_DIR"):           # PNG previews only outside the repo (PDF is the figure)
        fig.savefig(Path(os.environ["FIG_PREVIEW_DIR"]) / "tdrjmcmc_t0_sweep.png", dpi=110, bbox_inches="tight")
    print("saved", OUT / "tdrjmcmc_t0_sweep.pdf")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    DATA.mkdir(exist_ok=True)
    if "--plot-only" not in sys.argv:
        run_all()
    plot()
