"""Is adaptive parallel tempering (APT) worth it, how should it be switched on, and what does it cost?

One set of runs answers three questions (figures/previous/apt_cost.pdf,
figures/previous/apt_t0.pdf, figures/previous/apt_tempering.pdf):
  1. cost:      waveform evaluations (oracle calls on a quantum computer) and CPU time needed to reach a
                given accuracy, for naive, Fisher, AP (Tian et al.) and QCLT-AP, against real BayesWave;
  2. t0:        when should the adaptive proposal switch on?
  3. tempering: what does the adaptive temperature ladder (the other half of "APT") contribute?

Honesty conventions (see the audit notes in figures/previous/README.md):
  * every run uses its own noise realisation (data_seed = 1000 + seed), as BayesWave's runs do;
  * all waveform evaluations are counted, including the 8192 used to build the time-frequency birth map when
    that map is used (easy case); the hard case uses prior births and computes no map;
  * Fisher moves use finite-difference derivatives (20 extra waveform evaluations per move). For sine-Gaussian
    wavelets the derivatives are also available in closed form, so Fisher's cost is shown two ways: as run,
    and "if derivatives were free" (finite-difference evaluations removed);
  * "reaches 0.95" means the median match of the cold chain over the second half of the run so far reaches
    0.95: the same definition as BayesWave's checkpoint values (median over the saved posterior samples in
    the second half of the chain);
  * AP and QCLT-AP are derivative-free here: before t0, and on 20% of moves after it, they use the naive step.

Run:   python3.12 apt_study.py              # ~1.5-2 h with 8 workers
       python3.12 apt_study.py --plot-only
"""
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

import bw_ref
import tdrjmcmc_qclt as T

OUT = Path(__file__).parent / "figures" / "previous"
DATA = Path(__file__).parent / "data"
SEEDS = 8
THRESH = 0.95
T0S = [10, 100, 300, 1000, 3000]
T0_MAIN = 300                                      # t0 used for AP / QCLT-AP in the cost figure
CONDS = [
    dict(name="easy", label="Easy case: short chirp, SNR 40, new wavelets placed where the data has power",
         data="chirp", snr=40.0, birth="tf", kmax=10, tmax=800.0, iters=30_000, iters_fisher=12_000, bw=40),
    dict(name="hard", label="Hard case: wide chirp + extra burst, SNR 40, new wavelets placed at random",
         data="hard", snr=40.0, birth="prior", kmax=20, tmax=1600.0, iters=150_000, iters_fisher=40_000, bw="hard"),
]
# (method, t0, adaptive temperatures)
CONFIGS = ([("rw", None, True), ("fisher", None, True)]
           + [(m, t0, True) for m in ("ap_nb", "qclt_nb", "qamr_nb") for t0 in T0S]
           + [("rw", None, False), ("fisher", None, False)]
           + [(m, T0_MAIN, False) for m in ("ap_nb", "qclt_nb", "qamr_nb")])
NAME = {"rw": "Naive, fixed random steps (classical)", "fisher": "Fisher, finite-difference derivatives (classical)",
        "fisher_free": "Fisher, if derivatives were free (classical)", "ap_nb": "AP, Tian et al. (classical)",
        "qclt_nb": "QCLT-AP (quantum kick, simulated)",
        "qamr_nb": "QCLT + AMR, ours (quantum kick, simulated; 6-bit registers)"}
COL = {"rw": "#52514e", "fisher": "#2a78d6", "fisher_free": "#2a78d6", "ap_nb": "#1baf7a", "qclt_nb": "#6250d6",
       "qamr_nb": "#d62f2f"}
LS = {"rw": "--", "fisher": "-", "fisher_free": ":", "ap_nb": "-", "qclt_nb": "-", "qamr_nb": "-"}
BWC = {"default": "#0b0b0b", "matched": "#9d9c97"}


def _job(args):
    cond, method, t0, adapt, seed = args
    T.DATASET, T.SNR_INJ, T.BIRTH, T.K_MAX, T.T_MAX = cond["data"], cond["snr"], cond["birth"], cond["kmax"], cond["tmax"]
    T.N_MIN_HIST, T.AP_FALLBACK, T.ADAPT_TEMPS = 10, "rw", adapt
    if t0 is not None:
        T.T0 = t0
    n = cond["iters_fisher"] if method == "fisher" else cond["iters"]
    t = time.process_time()
    r = T.run(method, 500 + seed, n, n // 2, data_seed=1000 + seed, record_waves=False)
    cpu = time.process_time() - t
    step = max(1, n // 4000)
    return dict(cond=cond["name"], method=method, t0=t0, adapt=adapt, seed=seed, n=n, step=step,
                match=r["match"][::step].astype(np.float32), evals=r["evals"][::step], evals_fd=r["evals_fd"][::step],
                evals_setup=r["evals_setup"], iters=np.arange(0, n, step) + 1, cpu=cpu)


def run_all(methods=None):
    """Run every config (or only those of `methods`, merged into the existing data/apt_study.pkl)."""
    from multiprocessing import Pool
    jobs = [(c, m, t0, a, s) for c in CONDS for (m, t0, a) in CONFIGS for s in range(SEEDS)
            if methods is None or m in methods]
    jobs.sort(key=lambda j: -(j[0]["iters_fisher"] if j[1] == "fisher" else j[0]["iters"]))
    t = time.time()
    with Pool(int(os.environ.get("N_WORKERS", 8))) as pool:
        res = pool.map(_job, jobs, chunksize=1)
    print(f"{len(jobs)} runs in {time.time() - t:.0f}s", flush=True)
    if methods is not None:
        with open(DATA / "apt_study.pkl", "rb") as f:
            res = [r for r in pickle.load(f) if r["method"] not in methods] + res
    with open(DATA / "apt_study.pkl", "wb") as f:
        pickle.dump(res, f)


# ----------------------------------------------------------------------------- analysis
def running_posterior_match(r, n_pts=250):
    """(positions k, median of match over the second half of [0, k]) on a geometric grid of k."""
    m = r["match"]
    ks = np.unique(np.geomspace(4, len(m), n_pts).astype(int))
    return ks, np.array([np.median(m[k // 2:k]) for k in ks])


def cost_axes(r, ks):
    """Cost of reaching position k: iterations, waveform evaluations (as run / derivatives free), CPU."""
    i = ks - 1
    return dict(iters=r["iters"][i].astype(float), evals=r["evals"][i].astype(float),
                evals_free=(r["evals"][i] - r["evals_fd"][i]).astype(float), cpu=r["cpu"] * r["iters"][i] / r["n"])


def reach(r, key):
    ks, pm = running_posterior_match(r)
    ok = np.nonzero(pm >= THRESH)[0]
    return cost_axes(r, ks)[key][ok[0]] if len(ok) else np.nan


def select(res, cond, method, t0=None, adapt=True):
    return [r for r in res if r["cond"] == cond and r["method"] == method and r["t0"] == t0 and r["adapt"] == adapt]


def style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8,
        "axes.titlesize": 8.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.edgecolor": T.MUTED, "axes.labelcolor": T.INK, "xtick.color": T.MUTED, "ytick.color": T.MUTED,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def finish(ax):
    ax.grid(color=T.GRIDC, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def logy(ax):
    """Log y axis (labelled minor ticks when it spans little more than a decade), unless nothing finite was
    plotted (e.g. no run reached the threshold)."""
    from matplotlib import ticker
    if any(np.isfinite(np.asarray(l.get_ydata(), float)).any() for l in ax.lines):
        ax.set_yscale("log")
        ax.yaxis.set_minor_formatter(ticker.LogFormatterSciNotation(labelOnlyBase=False, minor_thresholds=(1.5, 0.6)))


def save(fig, name):
    fig.text(0.01, -0.01, T.QC_NOTE, fontsize=6.2, color=T.MUTED, va="top")     # classical vs quantum, every figure
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    if os.environ.get("FIG_PREVIEW_DIR"):           # PNG previews only outside the repo (PDF is the figure)
        fig.savefig(Path(os.environ["FIG_PREVIEW_DIR"]) / f"{name}.png", dpi=110, bbox_inches="tight")
    print("saved", OUT / f"{name}.pdf")


def median_curve(rr, key):
    """Median over runs of the running posterior match, against cost `key` (on a common log grid)."""
    curves = [(cost_axes(r, running_posterior_match(r)[0])[key], running_posterior_match(r)[1]) for r in rr]
    lo = max(min(c[0][c[0] > 0].min() for c in curves), 1.0)
    grid = np.geomspace(lo, max(c[0].max() for c in curves), 200)
    ys = np.array([np.interp(grid, x, y, left=np.nan, right=np.nan) for x, y in curves])
    return grid, np.nanmedian(ys, 0)


def bw_points(case, key):
    out = {}
    for v in ("default", "matched"):
        cs = bw_ref.curves(case, v)
        if cs:
            out[v] = cs
    return out


def fig_cost(res):
    from matplotlib import pyplot as plt
    style()
    fig, axes = plt.subplots(len(CONDS), 3, figsize=(7.8, 2.9 * len(CONDS)), gridspec_kw=dict(width_ratios=[1.2, 1.2, 1.25]))
    lines = []
    for row, c in enumerate(CONDS):
        series = [("rw", "rw", None, "evals"), ("fisher", "fisher", None, "evals"),
                  ("fisher_free", "fisher", None, "evals_free"), ("ap_nb", "ap_nb", T0_MAIN, "evals"),
                  ("qclt_nb", "qclt_nb", T0_MAIN, "evals"), ("qamr_nb", "qamr_nb", T0_MAIN, "evals")]
        box = []
        for lab, m, t0, ekey in series:
            rr = select(res, c["name"], m, t0, True)
            if not rr:                                   # sampler not run (yet)
                continue
            for col, key in ((0, ekey), (1, "cpu")):
                if lab == "fisher_free" and col == 1:
                    continue
                g, y = median_curve(rr, key)
                axes[row, col].plot(g, y, LS[lab], color=COL[lab], lw=1.3, label=NAME[lab] if col == 0 else None)
            h = np.array([reach(r, ekey) for r in rr])
            box.append((lab, h))
            lines.append(f"[{c['name']}] {NAME[lab]:38s} reach {THRESH}: {np.nanmedian(h):10.3g} evals "
                         f"({np.isfinite(h).sum()}/{len(h)} runs)"
                         + ("" if lab == "fisher_free" else f", {np.nanmedian([reach(r, 'cpu') for r in rr]):6.1f} CPU s"))
        for v, cs in bw_points(c["bw"], "evals").items():
            for k, bc in enumerate(cs):
                axes[row, 0].plot(bc["evals"], bc["match"], "D-", ms=2.3, lw=0.5, color=BWC[v], alpha=0.7,
                                  label=bw_ref.LABELS[v] + " (≈ updates × chains)" if k == 0 else None)
                axes[row, 1].plot(bc["cpu"], bc["match"], "D-", ms=2.3, lw=0.5, color=BWC[v], alpha=0.7)
            hb = bw_ref.first_hit(c["bw"], THRESH, "evals", v)
            box.append((v, hb))
            lines.append(f"[{c['name']}] {bw_ref.LABELS[v]:38s} reach {THRESH}: {np.nanmedian(hb):10.3g} evals "
                         f"({np.isfinite(hb).sum()}/{len(hb)} runs), "
                         f"{np.nanmedian(bw_ref.first_hit(c['bw'], THRESH, 'cpu', v)):6.1f} CPU s (C)")
        for col in (0, 1):
            ax = axes[row, col]
            ax.set_xscale("log"); ax.set_ylim(0, 1)
            ax.axhline(THRESH, color=T.MUTED, lw=0.6, ls=":")
            ax.set_ylabel(f"posterior median match so far\n(median of {SEEDS} runs)")
            finish(ax)
        axes[row, 0].set_xlabel("waveform evaluations (all temperatures)")
        axes[row, 1].set_xlabel("CPU seconds, one core (ours: Python; BayesWave: compiled)")
        axes[row, 0].set_title(f"({'ad'[row]}) Match vs waveform evaluations", loc="left", fontsize=7.8)
        axes[row, 1].set_title(f"({'be'[row]}) Match vs CPU time", loc="left", fontsize=7.8)
        ax = axes[row, 2]
        for k, (lab, h) in enumerate(box):
            ok = h[np.isfinite(h)]
            color = COL.get(lab, BWC.get(lab))
            ax.plot(k + np.linspace(-0.15, 0.15, len(ok)), ok, "o", ms=3, color=color, mec="white", mew=0.4)
            if len(ok):
                ax.plot([k - 0.25, k + 0.25], [np.median(ok)] * 2, color=T.INK, lw=1.0)
            ax.text(k, 1.01, f"{len(ok)}/{len(h)}", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=5.8, color=T.MUTED)
        logy(ax)
        ax.set_xticks(range(len(box)))
        ax.set_xticklabels({"rw": "Naive · classical", "fisher": "Fisher FD · classical",
                            "fisher_free": "Fisher free · classical", "ap_nb": "AP · classical",
                            "qclt_nb": "QCLT-AP · quantum", "qamr_nb": "QCLT+AMR (ours) · quantum",
                            "default": "BW default · classical",
                            "matched": "BW matched · classical"}[l] for l, _ in box)
        ax.tick_params(axis="x", labelsize=6.0, pad=2, labelrotation=90)
        ax.set_ylabel(f"waveform evaluations to reach {THRESH}\n(lower = cheaper; dots = runs)")
        ax.set_title(f"({'cf'[row]}) Cost to reach {THRESH}\n(numbers: runs that got there)", loc="left",
                     fontsize=7.8, pad=9)
        finish(ax)
        axes[row, 0].text(0.0, 1.17, c["label"], transform=axes[row, 0].transAxes, fontsize=8.3, color=T.INK,
                          fontweight="bold")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.05), fontsize=6.8)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.12, wspace=0.45, hspace=0.95)
    save(fig, "apt_cost")
    print("\n".join(lines))


def fig_t0(res):
    from matplotlib import pyplot as plt
    style()
    fig, axes = plt.subplots(len(CONDS), 2, figsize=(7.1, 2.8 * len(CONDS)))
    for row, c in enumerate(CONDS):
        print(f"[{c['name']}] median cost to reach {THRESH} vs t0 (iterations | evaluations):")
        for col, key in ((0, "iters"), (1, "evals")):
            ax = axes[row, col]
            for lab, m, ekey in (("rw", "rw", key), ("fisher", "fisher", key)) + ((("fisher_free", "fisher", "evals_free"),) if key == "evals" else ()):
                h = np.array([reach(r, ekey) for r in select(res, c["name"], m, None, True)])
                n_ok = np.isfinite(h).sum()
                if n_ok:
                    ax.axhline(np.nanmedian(h), color=COL[lab], ls=LS[lab], lw=1.1,
                               label=NAME[lab] + (f" ({n_ok}/{len(h)} runs reached)" if n_ok < len(h) else ""))
                print(f"   {NAME[lab]:38s} {key:6s} {np.nanmedian(h) if n_ok else np.nan:.3g} ({n_ok}/{len(h)} runs reached)")
            for m, off, mk in (("ap_nb", 0.92, "s"), ("qclt_nb", 1.0, "o"), ("qamr_nb", 1.08, "*")):
                if not select(res, c["name"], m, T0S[0], True):
                    continue
                st, nok = [], []
                for t0 in T0S:
                    h = np.array([reach(r, key) for r in select(res, c["name"], m, t0, True)])
                    st.append(np.nanpercentile(h, [25, 50, 75]) if np.isfinite(h).any() else [np.nan] * 3)
                    nok.append(int(np.isfinite(h).sum()))
                st = np.array(st)
                ax.errorbar(np.array(T0S) * off, st[:, 1], yerr=[st[:, 1] - st[:, 0], st[:, 2] - st[:, 1]], fmt=mk + "-",
                            ms=4, color=COL[m], lw=1.1, capsize=0, mec="white", mew=0.6, label=NAME[m])
                print(f"   {NAME[m]:38s} {key:6s} " + "  ".join(f"t0={t}: {v:.3g} ({k}/{SEEDS})" for t, v, k in zip(T0S, st[:, 1], nok)))
            for v in ("default", "matched"):                 # BayesWave: no t0, so a horizontal line
                hb = bw_ref.first_hit(c["bw"], THRESH, "updates" if key == "iters" else "evals", v)
                if np.isfinite(hb).any():
                    nb = int(np.isfinite(hb).sum())
                    ax.axhline(np.nanmedian(hb), color=BWC[v], ls="-.", lw=1.1,
                               label=bw_ref.LABELS[v] + (f" ({nb}/{len(hb)} runs reached)" if nb < len(hb) else ""))
                    print(f"   {bw_ref.LABELS[v]:38s} {key:6s} {np.nanmedian(hb):.3g} ({nb}/{len(hb)} runs reached)")
            ax.set_xscale("log")
            logy(ax)
            ax.set_xlabel(r"$t_0$: iteration the adaptive proposal switches on")
            ax.set_ylabel(("model updates per chain" if key == "iters" else "waveform evaluations")
                          + f" to reach {THRESH}\n(lower = cheaper)")
            ax.set_title(f"({'abcd'[2 * row + col]}) " + ("Cost in model updates per chain (ours: iterations)"
                                                          if key == "iters" else "Cost in waveform evaluations"),
                         loc="left", fontsize=7.8)
            finish(ax)
        axes[row, 0].text(0.0, 1.18, c["label"], transform=axes[row, 0].transAxes, fontsize=8.3, color=T.INK,
                          fontweight="bold")
    h, l = axes[0, 1].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.07), fontsize=7)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.86, bottom=0.08, wspace=0.35, hspace=0.75)
    save(fig, "apt_t0")


def fig_tempering(res):
    from matplotlib import pyplot as plt
    style()
    methods = [(m, t0) for m, t0 in (("rw", None), ("fisher", None), ("ap_nb", T0_MAIN), ("qclt_nb", T0_MAIN),
                                      ("qamr_nb", T0_MAIN)) if any(r["method"] == m for r in res)]
    fig, axes = plt.subplots(1, len(CONDS), figsize=(7.6, 2.9))
    for col, c in enumerate(CONDS):
        ax = axes[col]
        print(f"[{c['name']}] evaluations to reach {THRESH}: adaptive vs fixed temperature ladder")
        for k, (m, t0) in enumerate(methods):
            for j, adapt in enumerate((True, False)):
                h = np.array([reach(r, "evals") for r in select(res, c["name"], m, t0, adapt)])
                ok = h[np.isfinite(h)]
                x = k + (-0.18 if adapt else 0.18)
                ax.plot(x + np.linspace(-0.07, 0.07, len(ok)), ok, "o" if adapt else "s", ms=3,
                        color=COL[m] if adapt else "white", mec=COL[m], mew=0.8)
                if len(ok):
                    ax.plot([x - 0.12, x + 0.12], [np.median(ok)] * 2, color=T.INK, lw=1.0)
                print(f"   {NAME[m]:38s} {'adaptive' if adapt else 'fixed':9s} {np.nanmedian(h):10.3g} "
                      f"({np.isfinite(h).sum()}/{len(h)} runs reached)")
        k = len(methods)                                 # BayesWave, matched settings: its own ladder adapted / fixed
        for adapt, v in ((True, "matched"), (False, "matched_fixedT")):
            h = bw_ref.first_hit(c["bw"], THRESH, "evals", v)
            if not len(h):
                continue
            ok = h[np.isfinite(h)]
            x = k + (-0.18 if adapt else 0.18)
            ax.plot(x + np.linspace(-0.07, 0.07, len(ok)), ok, "D" if adapt else "d", ms=3,
                    color=BWC["default"] if adapt else "white", mec=BWC["default"], mew=0.8)
            if len(ok):
                ax.plot([x - 0.12, x + 0.12], [np.median(ok)] * 2, color=T.INK, lw=1.0)
            if len(ok) < len(h):
                ax.text(x, np.max(ok) * 1.5 if len(ok) else 1e6, f"{len(ok)}/{len(h)}", ha="center", fontsize=5.8,
                        color=T.MUTED)
            print(f"   {bw_ref.LABELS[v]:38s} {'adaptive' if adapt else 'fixed':9s} {np.nanmedian(h):10.3g} "
                  f"({len(ok)}/{len(h)} runs reached)")
        logy(ax)
        ax.set_xticks(range(k + 1))
        ax.set_xticklabels([{"rw": "Naive\nclassical", "fisher": "Fisher (FD)\nclassical", "ap_nb": "AP\nclassical",
                             "qclt_nb": "QCLT-AP\nquantum kick", "qamr_nb": "QCLT+AMR\n(ours)\nquantum kick"}[m]
                            for m, _ in methods] + ["BayesWave\n(matched, ≈)\nclassical"], fontsize=6.0)
        ax.set_ylabel(f"waveform evaluations to reach {THRESH}")
        ax.set_title(f"({'ab'[col]}) " + c["label"].split(":")[0], loc="left", fontsize=7.8)
        finish(ax)
    hs = [axes[0].plot([], [], "o", color=T.MUTED)[0], axes[0].plot([], [], "s", color="white", mec=T.MUTED)[0]]
    fig.legend(hs, ["adaptive temperature ladder (ours: tuned in burn-in, then frozen; BayesWave: its own)",
                    "fixed: each sampler's initial geometric ladder"],
               loc="upper left", bbox_to_anchor=(0.01, 0.94), ncol=2, frameon=False, fontsize=6.8)
    fig.suptitle("What the adaptive temperature ladder (the tempering half of APT) contributes; dots = runs, "
                 "bars = medians", x=0.01, ha="left", fontsize=8, color=T.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    save(fig, "apt_tempering")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    DATA.mkdir(exist_ok=True)
    if "--add" in sys.argv:                       # e.g. --add qamr_nb
        run_all(sys.argv[sys.argv.index("--add") + 1].split(","))
    elif "--plot-only" not in sys.argv:
        run_all()
    with open(DATA / "apt_study.pkl", "rb") as f:
        res = pickle.load(f)
    fig_cost(res)
    fig_t0(res)
    fig_tempering(res)
