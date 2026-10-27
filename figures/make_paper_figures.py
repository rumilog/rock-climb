"""Result figures for the RA-L paper, in one consistent style.

  fig_grasp_selection.pdf (single column)  Fig. 3: (a) final hand poses from the eval reset pose,
                                           (b) correct-grasp rate (eval_results/reset_pose_probe.py)
  fig_mechanism.pdf  (single column)  superseded: (a) PCA of predicted actions, (b) within-input spread
  fig_hardware.pdf   (full width)     (a) paired slip force + stats, (b) failure-mode bins

Usage:
  python figures/make_paper_figures.py --session eval_results/paired_session_20260515_074256.json
  python figures/make_paper_figures.py --data figures/figure_data.json   # mechanism
  python figures/make_paper_figures.py --placeholder   # layout check only, never for the paper

figure_data.json (export it from the eval JSON / analysis scripts):
  {
    "hardware": {                      # slip force in N, index i = pair i (paired order)
      "crimp": {"with": [20 floats], "without": [20 floats]}, ... sloper, pinch, jug
    },
    "mechanism": {
      "pca":           {"crimp": {"with": [[pc1, pc2], ...], "without": [[pc1, pc2], ...]}, ...},
      "within_spread": {"crimp": {"with": [8 floats], "without": [8 floats]}, ...}  # one per input
    }
  }

Encoding, shared by every figure: color = model (orange = with taxonomy, matching the
grasp-type branch in Fig. 1; gray = without); grasp type = position, marker shape, and
direct labels, never color.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from scipy.stats import wilcoxon

OUT_DIR = Path(__file__).resolve().parent

TYPES = ["crimp", "sloper", "pinch", "jug"]
MARKERS = {"crimp": "o", "sloper": "s", "pinch": "^", "jug": "D"}
C_WITH = "#DD6B20"
C_WITHOUT = "#8A8984"
INK = "#1A202C"
MUTED = "#52514E"
GRID = "#E2E1DC"
# Failure-mode bins (N), from the paper: complete <= 5.3 < weak <= 13.5 < moderate <= 23.0 < strong
BIN_EDGES = [5.3, 13.5, 23.0]
BIN_NAMES = ["Complete failure", "Weak", "Moderate", "Strong"]
BIN_COLORS = ["#D6E4F5", "#8DB4E6", "#3E7FCF", "#1D4F91"]  # one hue, light -> dark

COL_W, FULL_W = 3.5, 7.16  # IEEE two-column widths (in)

plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.edgecolor": MUTED,
    "axes.linewidth": 0.6,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "pdf.fonttype": 42,
})


def panel_label(ax, s):
    ax.text(0, 1.03, f"({s})", transform=ax.transAxes, ha="left", va="bottom",
            fontweight="bold", color=INK)


def ygrid(ax):
    ax.yaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def paired_stats(w, wo):
    w, wo = np.asarray(w, float), np.asarray(wo, float)
    diff = w - wo
    p = wilcoxon(w, wo, alternative="two-sided").pvalue if np.any(diff) else 1.0
    # ddof=0 matches eval_results/generate_ratchet_figures.py:cohen_d and the paper's
    # reported d values; the textbook d_z uses ddof=1 (~3% smaller at n=20).
    sd = diff.std(ddof=0)
    d = diff.mean() / sd if sd > 0 else float("inf")
    return {"d_median": np.median(w) - np.median(wo), "p": p, "d": d,
            "wins": int((diff > 0).sum()), "ties": int((diff == 0).sum()),
            "losses": int((diff < 0).sum())}


def fmt_p(p):
    return "$p<0.001$" if p < 0.001 else f"$p={p:.2g}$"


def watermark(fig):
    fig.text(0.5, 0.5, "PLACEHOLDER DATA", ha="center", va="center", fontsize=26,
             color="#E34948", alpha=0.25, rotation=20, fontweight="bold")


# ---------------------------------------------------------------- Fig. 3
def fig_mechanism(mech, placeholder):
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(COL_W, 4.3),
                                     gridspec_kw={"height_ratios": [1.35, 1], "hspace": 0.42})

    # (a) Predicted actions in PCA: with = filled orange, without = hollow gray.
    for t in TYPES:
        w = np.asarray(mech["pca"][t]["with"])
        wo = np.asarray(mech["pca"][t]["without"])
        ax_a.scatter(wo[:, 0], wo[:, 1], s=14, marker=MARKERS[t], facecolors="none",
                     edgecolors=C_WITHOUT, linewidths=0.7, zorder=2)
        ax_a.scatter(w[:, 0], w[:, 1], s=14, marker=MARKERS[t], color=C_WITH,
                     edgecolors="white", linewidths=0.3, alpha=0.85, zorder=3)
        allpts = np.vstack([w, wo])
        cx, top = allpts[:, 0].mean(), allpts[:, 1].max()
        ax_a.annotate(t, (cx, top), xytext=(0, 4), textcoords="offset points",
                      ha="center", va="bottom", color=INK, fontsize=8)
    ax_a.set_xlabel("PC 1")
    ax_a.set_ylabel("PC 2")
    ax_a.set_xticks([])
    ax_a.set_yticks([])
    ax_a.margins(0.12)
    ax_a.legend(handles=[
        plt.Line2D([], [], marker="o", ls="", color=C_WITH, label="With taxonomy"),
        plt.Line2D([], [], marker="o", ls="", markerfacecolor="none",
                   markeredgecolor=C_WITHOUT, label="Without taxonomy"),
    ], loc="lower right", bbox_to_anchor=(1, 1.0), ncol=2, handletextpad=0.2,
        columnspacing=1.0, borderaxespad=0.1)
    panel_label(ax_a, "a")

    # (b) Within-input sampling spread: mean +/- SE over inputs, per type.
    x = np.arange(len(TYPES))
    bw = 0.36
    tops = []
    for off, key, col in [(-bw / 2, "with", C_WITH), (bw / 2, "without", C_WITHOUT)]:
        vals = [np.asarray(mech["within_spread"][t][key], float) for t in TYPES]
        mean = np.array([v.mean() for v in vals])
        se = np.array([v.std(ddof=1) / np.sqrt(len(v)) for v in vals])
        ax_b.bar(x + off, mean, width=bw - 0.03, color=col, zorder=2)
        ax_b.errorbar(x + off, mean, yerr=se, fmt="none", ecolor=INK, elinewidth=0.6,
                      capsize=1.5, zorder=3)
        tops.append(mean + se)
    for i, t in enumerate(TYPES):
        w = np.mean(mech["within_spread"][t]["with"])
        wo = np.mean(mech["within_spread"][t]["without"])
        ax_b.text(i, max(tops[0][i], tops[1][i]) * 1.03, f"{100 * (wo / w - 1):+.0f}%",
                  ha="center", va="bottom", fontsize=7, color=MUTED)
    ax_b.set_xticks(x, TYPES)
    ax_b.set_ylabel("Within-input spread (L2, rad)")
    ax_b.set_ylim(0, max(np.max(tops[0]), np.max(tops[1])) * 1.22)
    ygrid(ax_b)
    ax_b.tick_params(axis="x", length=0)
    panel_label(ax_b, "b")

    if placeholder:
        watermark(fig)
    return fig


# ---------------------------------------------------------------- Fig. 3 (reset-pose probe)
def fig_grasp_selection(gs):
    """(a) final hand poses from the eval reset pose in demo-PCA space, (b) correct-grasp rate."""
    from sklearn.decomposition import PCA
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(COL_W, 4.3),
                                     gridspec_kw={"height_ratios": [1.35, 1], "hspace": 0.42})

    # (a) PCA fit on demonstrator final hand poses; rollouts projected into it.
    demo = {t: np.asarray(gs["demo_final_hand"][t]) for t in TYPES}
    pca = PCA(n_components=2).fit(np.vstack(list(demo.values())))
    true = np.asarray(gs["true_type"])
    for t in TYPES:
        d = pca.transform(demo[t])
        ax_a.scatter(d[:, 0], d[:, 1], s=10, marker=MARKERS[t], color=GRID, lw=0, zorder=1)
        c = np.median(d, 0)
        ax_a.annotate(t, c, xytext=(0, 9), textcoords="offset points", ha="center",
                      va="bottom", color=INK, fontsize=8, zorder=5,
                      bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    for key in ["without", "with"]:
        xy = pca.transform(np.asarray(gs["reset"][key]["final_hand"]))
        for t in TYPES:
            m = xy[true == t]
            if key == "with":
                ax_a.scatter(m[:, 0], m[:, 1], s=12, marker=MARKERS[t], color=C_WITH,
                             edgecolors="white", linewidths=0.3, alpha=0.8, zorder=3)
            else:
                ax_a.scatter(m[:, 0], m[:, 1], s=12, marker=MARKERS[t], facecolors="none",
                             edgecolors=C_WITHOUT, linewidths=0.6, alpha=0.8, zorder=2)
    ax_a.set_xlabel("PC 1 (final hand pose)")
    ax_a.set_ylabel("PC 2")
    ax_a.set_xticks([])
    ax_a.set_yticks([])
    ax_a.margins(0.1)
    ax_a.legend(handles=[
        plt.Line2D([], [], marker="o", ls="", color=C_WITH, label="With taxonomy"),
        plt.Line2D([], [], marker="o", ls="", markerfacecolor="none",
                   markeredgecolor=C_WITHOUT, label="Without taxonomy"),
        plt.Line2D([], [], marker="o", ls="", color=GRID, label="Demonstrations"),
    ], loc="lower right", bbox_to_anchor=(1.02, 1.0), ncol=2, handletextpad=0.05,
        columnspacing=0.5, borderaxespad=0.1, fontsize=7)
    panel_label(ax_a, "a")

    # (b) Correct-grasp rate from the reset pose (bars) vs the recorded first frame (ticks).
    x = np.arange(len(TYPES))
    bw = 0.36
    for off, key, col in [(-bw / 2, "with", C_WITH), (bw / 2, "without", C_WITHOUT)]:
        reset = np.array([100 * gs["reset"][key]["correct_rate"][t] for t in TYPES])
        first = np.array([100 * gs["first_frame"][key]["correct_rate"][t] for t in TYPES])
        ax_b.bar(x + off, reset, width=bw - 0.03, color=col, zorder=2)
        ax_b.scatter(x + off, first, marker="_", s=60, color=INK, linewidths=1.0, zorder=3)
        for xi, v in zip(x + off, reset):
            ax_b.text(xi, v + 2, f"{v:.0f}", ha="center", va="bottom", fontsize=6.5, color=MUTED)
    ax_b.set_xticks(x, TYPES)
    ax_b.set_ylabel("Correct grasp (%)")
    ax_b.set_ylim(0, 118)
    ax_b.set_yticks([0, 25, 50, 75, 100])
    ygrid(ax_b)
    ax_b.tick_params(axis="x", length=0)
    ax_b.legend(handles=[Patch(color=C_WITH, label="With, from reset"),
                         Patch(color=C_WITHOUT, label="Without, from reset"),
                         plt.Line2D([], [], marker="_", ls="", ms=8, mew=1.0, color=INK,
                                    label="From recorded first frame")],
                loc="upper left", bbox_to_anchor=(0, -0.1), ncol=2, handlelength=1.0,
                columnspacing=0.8)
    panel_label(ax_b, "b")
    return fig


# ---------------------------------------------------------------- Fig. 4
def fig_hardware(hw, placeholder):
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(FULL_W, 2.55),
                                     gridspec_kw={"width_ratios": [1.55, 1], "wspace": 0.28})
    rng = np.random.default_rng(0)

    # (a) Paired slip force: boxes per model, pair lines between them, stats above.
    ymax = max(max(hw[t]["with"] + hw[t]["without"]) for t in TYPES)
    for i, t in enumerate(TYPES):
        w, wo = np.asarray(hw[t]["with"], float), np.asarray(hw[t]["without"], float)
        xs = {"without": i - 0.19, "with": i + 0.19}
        for key, vals, col in [("without", wo, C_WITHOUT), ("with", w, C_WITH)]:
            ax_a.boxplot(vals, positions=[xs[key]], widths=0.3, patch_artist=True,
                         showfliers=False, zorder=2,
                         boxprops=dict(facecolor=col, edgecolor=col, alpha=0.35, lw=0.8),
                         medianprops=dict(color=col, lw=1.6),
                         whiskerprops=dict(color=col, lw=0.8), capprops=dict(color=col, lw=0.8))
        jit = rng.uniform(-0.05, 0.05, len(w))
        for a, b, j in zip(wo, w, jit):
            ax_a.plot([xs["without"] + j, xs["with"] + j], [a, b], color=MUTED, lw=0.35,
                      alpha=0.35, zorder=1)
        ax_a.scatter(xs["without"] + jit, wo, s=5, color=C_WITHOUT, zorder=3, lw=0)
        ax_a.scatter(xs["with"] + jit, w, s=5, color=C_WITH, zorder=3, lw=0)
        s = paired_stats(w, wo)
        ax_a.text(i, ymax * 1.05, f"$\\Delta${s['d_median']:+.1f} N\n$d={s['d']:.2f}$\n{fmt_p(s['p'])}",
                  ha="center", va="bottom", fontsize=6.8, color=INK, linespacing=1.15)
    ax_a.set_xticks(range(len(TYPES)), TYPES)
    ax_a.tick_params(axis="x", length=0)
    ax_a.set_ylabel("Slip force (N)")
    ax_a.set_ylim(0, ymax * 1.45)
    ygrid(ax_a)
    ax_a.legend(handles=[Patch(color=C_WITHOUT, label="Without taxonomy"),
                         Patch(color=C_WITH, label="With taxonomy")],
                loc="upper left", bbox_to_anchor=(0, -0.1), ncol=2, handlelength=1.0)
    panel_label(ax_a, "a")

    # (b) Failure-mode bins: stacked share of trials, W/o and W per type.
    # Grasp types label the groups on the axis; W/o and W label each bar above it.
    for i, t in enumerate(TYPES):
        for j, (key, short) in enumerate([("without", "W/o"), ("with", "W")]):
            vals = np.asarray(hw[t][key], float)
            counts = np.bincount(np.digitize(vals, BIN_EDGES, right=True), minlength=4)
            x = i * 2.6 + j
            bottom = 0
            for k in range(4):
                h = 100 * counts[k] / len(vals)
                if h > 0:
                    ax_b.bar(x, h, bottom=bottom, width=0.82, color=BIN_COLORS[k],
                             edgecolor="white", linewidth=0.8, zorder=2)
                bottom += h
            ax_b.text(x, 101.5, short, ha="center", va="bottom", fontsize=6.5, color=MUTED)
    ax_b.set_xticks([i * 2.6 + 0.5 for i in range(len(TYPES))], TYPES)
    ax_b.tick_params(axis="x", length=0)
    ax_b.set_ylabel("Trials (%)")
    ax_b.set_ylim(0, 100)
    ax_b.legend(handles=[Patch(color=c, label=n) for c, n in zip(BIN_COLORS, BIN_NAMES)],
                loc="upper left", bbox_to_anchor=(0, -0.1), ncol=2, handlelength=1.0,
                columnspacing=0.8)
    panel_label(ax_b, "b")
    ax_b.texts[-1].set_y(1.09)  # clear the W/o / W bar labels

    if placeholder:
        watermark(fig)
    return fig


def print_stats(hw):
    """Recomputed stats, to check against the paper's numbers before submitting."""
    print(f"{'type':8s} {'med W':>6s} {'med W/o':>7s} {'delta':>6s} {'p':>8s} {'d':>5s}  W/T/L")
    allw, allwo = [], []
    for t in TYPES:
        w, wo = hw[t]["with"], hw[t]["without"]
        allw += w
        allwo += wo
        s = paired_stats(w, wo)
        print(f"{t:8s} {np.median(w):6.1f} {np.median(wo):7.1f} {s['d_median']:+6.1f} "
              f"{s['p']:8.4f} {s['d']:5.2f}  {s['wins']}/{s['ties']}/{s['losses']}")
    s = paired_stats(allw, allwo)
    print(f"{'overall':8s} {np.median(allw):6.1f} {np.median(allwo):7.1f} {s['d_median']:+6.1f} "
          f"{s['p']:8.4f} {s['d']:5.2f}  {s['wins']}/{s['ties']}/{s['losses']}")
    for key in ["without", "with"]:
        vals = np.concatenate([hw[t][key] for t in TYPES])
        share = np.bincount(np.digitize(vals, BIN_EDGES, right=True), minlength=4) / len(vals)
        print(f"bins {key:8s} " + "  ".join(f"{n}: {100 * v:.0f}%" for n, v in zip(BIN_NAMES, share)))


def print_mech_stats(mech):
    """Per-type within-input spread, W/o vs W, with a paired Wilcoxon over inputs."""
    print(f"{'type':8s} {'spread W':>9s} {'spread W/o':>11s} {'W/o vs W':>9s} {'p':>7s}  n")
    for t in TYPES:
        w = np.asarray(mech["within_spread"][t]["with"], float)
        wo = np.asarray(mech["within_spread"][t]["without"], float)
        p = wilcoxon(wo, w, alternative="two-sided").pvalue
        print(f"{t:8s} {w.mean():9.3f} {wo.mean():11.3f} {100 * (wo.mean() / w.mean() - 1):+8.0f}% "
              f"{p:7.3f}  {len(w)}")
    print("note: n is the number of inputs per type; with n=8 the smallest possible "
          "two-sided Wilcoxon p is 0.0078, and a non-significant p is not evidence of no difference.")


def placeholder_data():
    """Synthetic numbers for checking layout ONLY. Output files are suffixed _PLACEHOLDER."""
    rng = np.random.default_rng(1)
    hw, pca, std = {}, {}, {}
    centers = {"crimp": (-2, 1), "sloper": (2, 1.5), "pinch": (-1, -2), "jug": (2.5, -1.5)}
    for t in TYPES:
        wo = np.clip(rng.normal(9, 4, 20), 5.2, 30)
        w = np.clip(wo + rng.normal(4, 3, 20), 5.2, 33)
        hw[t] = {"with": w.round(1).tolist(), "without": wo.round(1).tolist()}
        c = np.array(centers[t])
        pca[t] = {"with": (c + rng.normal(0, 0.35, (32, 2))).tolist(),
                  "without": (c + rng.normal(0, 0.45, (32, 2))).tolist()}
        std[t] = {"with": rng.normal(0.05, 0.01, 32).tolist(),
                  "without": rng.normal(0.062, 0.012, 32).tolist()}
    return {"hardware": hw, "mechanism": {"pca": pca, "within_spread": std}}


def load_session(path):
    """Hardware data straight from a paired_eval session JSON, kept in pair order."""
    session = json.loads(Path(path).read_text())
    hw = {t: {"with": [], "without": []} for t in TYPES}
    for p in sorted(session["pairs"], key=lambda p: p["pair"]):
        if p.get("aborted"):
            continue
        stats = p["pc_stats"]
        hw[p["grasp_type"]]["with"].append(stats["WITH_TAXONOMY"]["ratchet"]["force_N"])
        hw[p["grasp_type"]]["without"].append(stats["WITHOUT_TAXONOMY"]["ratchet"]["force_N"])
    return hw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", type=Path, help="paired_session_*.json -> fig_hardware")
    ap.add_argument("--data", type=Path,
                    help="figure_data.json: 'grasp_selection' -> fig_grasp_selection, 'mechanism' -> fig_mechanism")
    ap.add_argument("--placeholder", action="store_true", help="layout check only")
    args = ap.parse_args()
    if not (args.session or args.data or args.placeholder):
        ap.error("pass --session and/or --data, or --placeholder")

    figs, hw = [], None
    if args.placeholder:
        data = placeholder_data()
        hw = data["hardware"]
        figs = [("fig_mechanism_PLACEHOLDER", fig_mechanism(data["mechanism"], True)),
                ("fig_hardware_PLACEHOLDER", fig_hardware(hw, True))]
    else:
        if args.session:
            hw = load_session(args.session)
            figs.append(("fig_hardware", fig_hardware(hw, False)))
        if args.data:
            data = json.loads(args.data.read_text())
            if "grasp_selection" in data:
                figs.append(("fig_grasp_selection", fig_grasp_selection(data["grasp_selection"])))
            if "mechanism" in data:  # superseded old Fig. 3 (within-input spread)
                figs.append(("fig_mechanism", fig_mechanism(data["mechanism"], False)))
                print_mech_stats(data["mechanism"])
    for name, fig in figs:
        for ext in ("pdf", "png"):
            fig.savefig(OUT_DIR / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02, dpi=300)
        print("wrote", OUT_DIR / f"{name}.pdf")
    if hw:
        print_stats(hw)


if __name__ == "__main__":
    main()
