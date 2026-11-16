"""Cost-fair comparison: accuracy vs number of waveform evaluations.

On a quantum computer each evaluation of the waveform/likelihood is an oracle call, so that, not the number
of iterations, is the cost that matters. A Fisher move needs the waveform's derivatives (about 20 extra
waveform evaluations per move), while naive, AP and QCLT-AP moves need one. Every sampler gets a similar
budget of waveform evaluations (~1.3 million, summed over all temperatures); AP and QCLT-AP use Tian et al.'s
fixed untuned proposal (not Fisher) for their warm-up and fallback moves, so they are derivative-free.

Samplers: rw (naive: fixed random steps of 1% of each range), fisher, ap_nb (AP, Tian et al.), qclt_nb (QCLT-AP).
Conditions: easy (short chirp, data-driven birth) and hard (wide chirp + burst, random birth), both at SNR 40.

Run:   python3.12 cost_benchmark.py              -> figures/cost_benchmark.pdf
       python3.12 cost_benchmark.py --plot-only
"""
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

import tdrjmcmc_qclt as T

SEEDS = 8
ITERS = {"rw": 150_000, "fisher": 12_000, "ap_nb": 150_000, "qclt_nb": 150_000}
METHODS = list(ITERS)
CONDS = [
    dict(name="easy", label="Easy case: short chirp, SNR 40,\nnew wavelets placed where the data has power",
         data="chirp", snr=40.0, birth="tf", kmax=10, tmax=800.0),
    dict(name="hard", label="Hard case: wide chirp + extra burst, SNR 40,\nnew wavelets placed at random",
         data="hard", snr=40.0, birth="prior", kmax=20, tmax=1600.0),
]
OUT = Path(__file__).parent / "figures"
DATA = Path(__file__).parent / "data"
STORE = DATA / "cost_benchmark.pkl"
THRESH = 0.95
NAME = {"rw": "Naive (fixed random steps)", "fisher": "Fisher (uses derivatives)",
        "ap_nb": "AP, Tian et al. (adaptive)", "qclt_nb": "QCLT-AP (quantum adaptive)"}


def _job(args):
    cond, method, seed = args
    T.DATASET, T.SNR_INJ, T.BIRTH, T.K_MAX, T.T_MAX = cond["data"], cond["snr"], cond["birth"], cond["kmax"], cond["tmax"]
    T.T0, T.N_MIN_HIST, T.AP_FALLBACK = 500, 10, "rw"
    n = ITERS[method]
    t0 = time.process_time()
    r = T.run(method, 700 + seed, n, n // 2, record_waves=False)
    step = max(1, n // 6000)                                 # keep ~6000 points per run
    return dict(cond=cond["name"], method=method, seed=seed, match=r["match"][::step].astype(np.float32),
                evals=r["evals"][::step], iters=np.arange(0, n, step), cpu=time.process_time() - t0,
                n=n, total_evals=int(r["evals"][-1]))


def run_all():
    from multiprocessing import Pool
    jobs = [(c, m, s) for c in CONDS for m in METHODS for s in range(SEEDS)]
    jobs.sort(key=lambda j: -ITERS[j[1]])                    # long jobs first
    t = time.time()
    with Pool(8) as pool:                     # 8 workers: memory headroom
        res = pool.map(_job, jobs, chunksize=1)
    print(f"{len(jobs)} runs in {time.time() - t:.0f}s", flush=True)
    with open(STORE, "wb") as f:
        pickle.dump(res, f)


def hit(r):
    i = np.argmax(r["match"] >= THRESH)
    return (r["evals"][i], r["iters"][i], r["cpu"] * r["iters"][i] / r["n"]) if r["match"][i] >= THRESH else (np.nan,) * 3


def plot():
    import matplotlib as mpl
    from matplotlib import pyplot as plt
    with open(STORE, "rb") as f:
        res = pickle.load(f)
    mpl.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8,
        "axes.titlesize": 8.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.edgecolor": T.MUTED, "axes.labelcolor": T.INK, "xtick.color": T.MUTED, "ytick.color": T.MUTED,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(len(CONDS), 3, figsize=(7.4, 2.7 * len(CONDS)),
                             gridspec_kw=dict(width_ratios=[1.3, 1.3, 1]))
    print(f"\nwaveform evaluations to reach match {THRESH} (median over {SEEDS} runs; runs that got there):")
    for row, c in enumerate(CONDS):
        for m in METHODS:
            rr = [r for r in res if r["cond"] == c["name"] and r["method"] == m]
            grid = np.geomspace(1e3, max(r["total_evals"] for r in rr), 300)
            ys = np.array([np.interp(grid, r["evals"], r["match"], left=np.nan, right=np.nan) for r in rr])
            lo, med, hi = np.nanpercentile(ys, [25, 50, 75], axis=0)
            col, ls = T.COLORS[m], T.LSTYLE.get(m, "-")
            axes[row, 0].fill_between(grid, lo, hi, color=col, alpha=0.15, lw=0)
            axes[row, 0].plot(grid, med, ls, color=col, lw=1.3, label=NAME[m])
            ycpu = np.array([np.interp(np.geomspace(0.5, r["cpu"], 300), r["cpu"] * r["iters"] / r["n"], r["match"])
                             for r in rr])
            xcpu = np.geomspace(0.5, np.median([r["cpu"] for r in rr]), 300)
            axes[row, 1].plot(xcpu, np.median(ycpu, axis=0), ls, color=col, lw=1.3)
            h = np.array([hit(r) for r in rr])
            print(f"  [{c['name']}] {NAME[m]:30s} {np.nanmedian(h[:, 0]):12.3g} evals "
                  f"({np.isfinite(h[:, 0]).sum()}/{len(rr)}), {np.nanmedian(h[:, 2]):6.1f} CPU s")
            bp = axes[row, 2].boxplot([h[np.isfinite(h[:, 0]), 0]], positions=[METHODS.index(m)], widths=0.55,
                                      patch_artist=True, showfliers=False, medianprops=dict(color=T.INK, lw=1),
                                      whiskerprops=dict(color=T.MUTED), capprops=dict(color=T.MUTED))
            bp["boxes"][0].set(facecolor=col, alpha=0.85, edgecolor="white")
        # real BayesWave on the same signal (evaluations = model updates x chains: a lower bound)
        import bw_ref
        case = 40 if c["name"] == "easy" else "hard"
        bw_cs = bw_ref.curves(case)
        if bw_cs:
            for k, bc in enumerate(bw_cs):
                axes[row, 0].plot(bc["evals"], bc["match"], "D-", ms=2.5, lw=0.6, color=bw_ref.COLOR, alpha=0.6,
                                  label="BayesWave (real; evaluations ≥ updates × chains)" if k == 0 else None)
                axes[row, 1].plot(bc["cpu"], bc["match"], "D-", ms=2.5, lw=0.6, color=bw_ref.COLOR, alpha=0.6)
            hb = bw_ref.first_hit(case, THRESH, "evals")
            hc = bw_ref.first_hit(case, THRESH, "cpu")
            print(f"  [{c['name']}] {'BayesWave (real, lower bound)':30s} {np.nanmedian(hb):12.3g} evals "
                  f"({np.isfinite(hb).sum()}/{len(hb)}), {np.nanmedian(hc):6.1f} CPU s (compiled C)")
            if np.isfinite(hb).any():                     # few runs at shared checkpoints: show each run
                ok = hb[np.isfinite(hb)]
                axes[row, 2].plot(len(METHODS) + np.linspace(-0.12, 0.12, len(ok)), ok, "D", ms=4,
                                  color=bw_ref.COLOR, mec="white", mew=0.5)
        for ax in axes[row, :2]:
            ax.set_xscale("log")
            ax.set_ylim(0, 1)
            ax.axhline(THRESH, color=T.MUTED, lw=0.6, ls=":")
            ax.set_ylabel("match with the injected signal\n(median of runs)")
        axes[row, 0].set_xlabel("waveform evaluations (all temperatures)")
        axes[row, 1].set_xlabel("CPU seconds, one core (ours: Python; BayesWave: C)")
        axes[row, 0].set_title(f"({'ad'[row]}) {c['label']}", loc="left", fontsize=7.6)
        axes[row, 1].set_title(f"({'be'[row]}) same runs vs CPU time", loc="left", fontsize=7.6)
        ax = axes[row, 2]
        ax.set_yscale("log")
        ax.set_xticks(range(len(METHODS) + 1))
        ax.set_xticklabels(["Naive", "Fisher", "AP", "QCLT-\nAP", "Bayes-\nWave*"])
        ax.text(1.0, -0.32, "*lower bound", transform=ax.transAxes, ha="right", fontsize=6, color=T.MUTED)
        ax.set_ylabel(f"evaluations to reach match {THRESH}\n(lower = cheaper)")
        ax.set_title(f"({'cf'[row]}) cost to reach {THRESH}", loc="left", fontsize=7.6)
        for ax in axes[row]:
            ax.grid(color=T.GRIDC, lw=0.6); ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.04), fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=1.5, w_pad=1.0)
    fig.savefig(OUT / "cost_benchmark.pdf", bbox_inches="tight")
    if os.environ.get("FIG_PREVIEW_DIR"):           # PNG previews only outside the repo (PDF is the figure)
        fig.savefig(Path(os.environ["FIG_PREVIEW_DIR"]) / "cost_benchmark.png", dpi=110, bbox_inches="tight")
    print("saved", OUT / "cost_benchmark.pdf")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    DATA.mkdir(exist_ok=True)
    if "--plot-only" not in sys.argv:
        run_all()
    plot()
