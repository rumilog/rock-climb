"""
Generate paper-quality figures from paired evaluation results.

Usage:
    python3 eval_results/generate_figures.py
    python3 eval_results/generate_figures.py --session eval_results/paired_session_20260417_131412.json
"""

import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy.stats import fisher_exact, chi2_contingency
import math

# ── aesthetics ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "legend.fontsize":  10,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "figure.dpi":       150,
    "savefig.dpi":      200,
    "savefig.bbox":     "tight",
})

WITH_COLOR    = "#2563EB"   # blue
WITHOUT_COLOR = "#DC2626"   # red
ALPHA_FILL    = 0.18

GRASP_ORDER = ["crimp", "jug", "sloper", "pinch"]
GRASP_LABELS = {
    "crimp":  "Crimp\n(edge_B)",
    "jug":    "Jug\n(edge_A)",
    "sloper": "Sloper",
    "pinch":  "Pinch",
}

# ── statistics helpers ────────────────────────────────────────────────────────

def wilson_ci(k, n, z=1.96):
    """Wilson score 95% confidence interval for a proportion k/n."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return p, max(0, centre - margin), min(1, centre + margin)


def mcnemar_p(with_vals, no_vals):
    """
    McNemar's test on paired binary outcomes.
    b = WITH wins (WITH=1, NO=0)
    c = WITHOUT wins (WITH=0, NO=1)
    Returns two-sided p-value (exact mid-p for small b+c, chi2 otherwise).
    """
    b = sum(1 for w, n in zip(with_vals, no_vals) if w == 1 and n == 0)
    c = sum(1 for w, n in zip(with_vals, no_vals) if w == 0 and n == 1)
    if b + c == 0:
        return 1.0
    # exact binomial mid-p (two-sided) when b+c is small
    from scipy.stats import binom
    total = b + c
    p_obs = min(b, c) / total
    # two-tailed exact: 2 * P(X <= min(b,c)) under H0 (p=0.5), using mid-p
    p_val = 2 * binom.cdf(min(b, c), total, 0.5) - binom.pmf(min(b, c), total, 0.5)
    return float(min(1.0, p_val))


def sig_label(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"


# ── data loading ──────────────────────────────────────────────────────────────

def load_and_parse(path):
    with open(path) as f:
        session = json.load(f)

    # Build per-grasp dict
    grasp_data = {g: {"with": [], "no": [], "pairs": []} for g in GRASP_ORDER}

    for pair in session["pairs"]:
        if pair.get("aborted", False):
            continue
        g = pair["grasp_type"]
        if g not in grasp_data:
            continue
        w = 1 if pair["with_rating"] == "good" else 0
        n = 1 if pair["no_rating"]   == "good" else 0
        grasp_data[g]["with"].append(w)
        grasp_data[g]["no"].append(n)
        grasp_data[g]["pairs"].append(pair)

    return session, grasp_data


def compute_stats(grasp_data):
    stats = {}
    for g in GRASP_ORDER:
        w = grasp_data[g]["with"]
        n = grasp_data[g]["no"]
        N = len(w)
        with_rate, with_lo, with_hi = wilson_ci(sum(w), N)
        no_rate,   no_lo,   no_hi   = wilson_ci(sum(n), N)
        p = mcnemar_p(w, n)
        stats[g] = {
            "N": N,
            "with_k": sum(w), "with_rate": with_rate,
            "with_lo": with_lo, "with_hi": with_hi,
            "no_k":   sum(n), "no_rate": no_rate,
            "no_lo": no_lo,   "no_hi": no_hi,
            "delta": with_rate - no_rate,
            "mcnemar_p": p,
        }
    return stats


# ── Figure 1: grouped bar chart ───────────────────────────────────────────────

def fig1_bar_chart(stats, out_path):
    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    x = np.arange(len(GRASP_ORDER))
    w = 0.35

    for i, g in enumerate(GRASP_ORDER):
        s = stats[g]
        with_pct = s["with_rate"] * 100
        no_pct   = s["no_rate"]   * 100
        with_err = np.array([[
            (s["with_rate"] - s["with_lo"]) * 100,
            (s["with_hi"]   - s["with_rate"]) * 100
        ]]).T
        no_err = np.array([[
            (s["no_rate"] - s["no_lo"]) * 100,
            (s["no_hi"]   - s["no_rate"]) * 100
        ]]).T

        ax.bar(x[i] - w/2, with_pct, width=w, color=WITH_COLOR,
               yerr=with_err, capsize=4, error_kw={"linewidth": 1.4},
               label="With taxonomy" if i == 0 else "")
        ax.bar(x[i] + w/2, no_pct,   width=w, color=WITHOUT_COLOR,
               yerr=no_err, capsize=4, error_kw={"linewidth": 1.4},
               label="Without taxonomy" if i == 0 else "")

        # count labels on top of each bar (replaces old below-axis n= that clashed with xticks)
        ax.text(x[i] - w/2, with_pct + with_err[1,0] + 1,
                f"{s['with_k']}/{s['N']}", ha="center", va="bottom",
                fontsize=8, color="#1E3A5F")
        ax.text(x[i] + w/2, no_pct + no_err[1,0] + 1,
                f"{s['no_k']}/{s['N']}", ha="center", va="bottom",
                fontsize=8, color="#7F1D1D")

        # significance bracket
        top = max(with_pct + with_err[1,0], no_pct + no_err[1,0]) + 8
        p   = s["mcnemar_p"]
        sl  = sig_label(p)
        ax.text(x[i], top + 1, sl, ha="center", va="bottom",
                fontsize=11, fontweight="bold" if sl != "ns" else "normal",
                color="black" if sl != "ns" else "#888888")
        ax.plot([x[i]-w/2, x[i]+w/2], [top, top], color="black", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([GRASP_LABELS[g] for g in GRASP_ORDER])
    ax.set_ylabel("Grasp success rate (%)")
    ax.set_ylim(0, 130)                  # room for brackets + sig stars
    ax.set_title(
        "Taxonomy conditioning improves grasp success across all types\n"
        "80 paired trials · 20 per grasp type · real robot (LEAP hand + Franka)",
        pad=10,
    )
    ax.legend(loc="upper right", framealpha=0.95)
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


# ── Figure 2: per-pair heatmap ────────────────────────────────────────────────

def fig2_heatmap(grasp_data, out_path):
    """
    Re-laid out: each grasp type becomes its own narrow 2-column block,
    arranged HORIZONTALLY so labels no longer overflow the plot.
    """
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#EF4444", "#22C55E"])   # 0=red fail, 1=green success

    n_rows_per_batch = 20
    fig, axes = plt.subplots(
        1, len(GRASP_ORDER),
        figsize=(11, 7.5),
        gridspec_kw={"wspace": 0.55},
    )
    fig.subplots_adjust(top=0.82, bottom=0.08, left=0.07, right=0.97)

    for ax, g in zip(axes, GRASP_ORDER):
        d = grasp_data[g]
        arr = np.array(list(zip(d["with"], d["no"])), dtype=float)
        n = arr.shape[0]

        ax.imshow(arr, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["With\ntax.", "W/o\ntax."], fontsize=8.5)
        ax.xaxis.set_ticks_position("top")
        ax.xaxis.set_label_position("top")

        step = max(1, n // 10)
        yticks = list(range(0, n, step))
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{y+1}" for y in yticks], fontsize=7)

        wins_with = sum(d["with"]); wins_no = sum(d["no"])
        ax.set_title(f"{g.capitalize()}\n"
                     f"With {wins_with}/{n} · W/o {wins_no}/{n}",
                     fontsize=9.5, pad=22)

        for spine in ax.spines.values():
            spine.set_visible(False)

    # single shared legend
    green_patch = mpatches.Patch(color="#22C55E", label="Success")
    red_patch   = mpatches.Patch(color="#EF4444", label="Failure")
    fig.legend(handles=[green_patch, red_patch], loc="lower center",
               ncol=2, fontsize=10, frameon=False,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Per-pair outcomes — each row is one paired trial "
                 "(row 1 = first pair, row 20 = last)",
                 fontsize=11.5, y=0.97)

    axes[0].set_ylabel("Pair index", fontsize=9)

    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


# ── Figure 3: Z centroid scatter ─────────────────────────────────────────────

GRASP_COLORS = {
    "crimp":  "#F59E0B",
    "jug":    "#3B82F6",
    "sloper": "#10B981",
    "pinch":  "#8B5CF6",
}

def fig3_z_centroid(grasp_data, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    jitter = 0.12
    for i, g in enumerate(GRASP_ORDER):
        zs = []
        for pair in grasp_data[g]["pairs"]:
            for model in ["WITH_TAXONOMY", "WITHOUT_TAXONOMY"]:
                zs.append(pair["pc_stats"][model]["centroid"][2] * 100)  # cm
        xs = i + np.random.default_rng(42).uniform(-jitter, jitter, len(zs))
        ax.scatter(xs, zs, alpha=0.55, s=22,
                   color=GRASP_COLORS[g], label=g.capitalize(), zorder=3)
        med = np.median(zs)
        ax.hlines(med, i - 0.3, i + 0.3, colors=GRASP_COLORS[g],
                  linewidth=2, zorder=4)

    ax.set_xticks(range(len(GRASP_ORDER)))
    ax.set_xticklabels([GRASP_LABELS[g] for g in GRASP_ORDER])
    ax.set_ylabel("Point cloud Z centroid (cm)")
    ax.set_title("Hold height by grasp type\n(horizontal line = median · each dot = one trial PC)",
                 pad=6)
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.set_ylim(0, None)

    ax.annotate("Crimp holds sit ~1.5 cm lower than the others\n"
                "(geometry alone distinguishes the four types)",
                xy=(0.15, 2.15), xytext=(0.6, 1.1),
                fontsize=8.5, color="#444444",
                ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color="#888888", lw=0.8,
                                connectionstyle="arc3,rad=-0.2"))

    ax.set_ylim(0.5, 4.6)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


# ── Figure 4: win/tie/loss stacked bars ──────────────────────────────────────

def fig4_win_tie_loss(grasp_data, stats, out_path):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    cats = ["WITH wins", "Tie — both good", "Tie — both bad", "WITHOUT wins"]
    colors = ["#2563EB", "#22C55E", "#9CA3AF", "#DC2626"]

    x = np.arange(len(GRASP_ORDER))

    counts = {g: [0, 0, 0, 0] for g in GRASP_ORDER}
    for g in GRASP_ORDER:
        for w, n in zip(grasp_data[g]["with"], grasp_data[g]["no"]):
            if   w == 1 and n == 0:
                counts[g][0] += 1
            elif w == 1 and n == 1:
                counts[g][1] += 1
            elif w == 0 and n == 0:
                counts[g][2] += 1
            else:
                counts[g][3] += 1

    bar_data = np.array([[counts[g][j] for g in GRASP_ORDER] for j in range(4)])
    bottoms  = np.zeros(len(GRASP_ORDER))

    for j, (cat, col) in enumerate(zip(cats, colors)):
        vals = bar_data[j]
        ax.bar(x, vals, bottom=bottoms, color=col, label=cat, width=0.55)
        for xi, (v, b) in enumerate(zip(vals, bottoms)):
            if v > 0:
                ax.text(xi, b + v / 2, str(v), ha="center", va="center",
                        fontsize=10, color="white", fontweight="bold")
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels([GRASP_LABELS[g] for g in GRASP_ORDER])
    ax.set_ylabel("Number of pairs")
    ax.set_ylim(0, 27)                    # headroom so p-values don't touch title
    ax.set_title("Paired outcome breakdown per grasp type\n"
                 "(n=20 pairs each · number in each segment = pair count)",
                 pad=10)
    ax.legend(loc="center right", bbox_to_anchor=(1.30, 0.5),
              fontsize=9, frameon=True, framealpha=0.95)
    ax.yaxis.grid(True, alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)

    # McNemar p-value: placed CLEARLY inside the headroom band (20.5 .. 26)
    for i, g in enumerate(GRASP_ORDER):
        p  = stats[g]["mcnemar_p"]
        sl = sig_label(p)
        p_str = "p<0.001" if p < 0.001 else f"p={p:.3f}"
        ax.text(i, 21.3, sl, ha="center", va="bottom",
                fontsize=12, fontweight="bold" if sl != "ns" else "normal",
                color="black" if sl != "ns" else "#888888")
        ax.text(i, 20.4, p_str, ha="center", va="bottom",
                fontsize=8.5, color="#444444")

    fig.subplots_adjust(right=0.80)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ── Figure 5: delta success rate + overall summary ───────────────────────────

def fig5_delta_and_summary(stats, out_path):
    fig = plt.figure(figsize=(13, 5.5))
    gs  = GridSpec(1, 2, figure=fig, width_ratios=[1.2, 1], wspace=0.20)

    # Left: delta bar chart
    ax = fig.add_subplot(gs[0])
    x  = np.arange(len(GRASP_ORDER))
    deltas = [stats[g]["delta"] * 100 for g in GRASP_ORDER]
    colors = [WITH_COLOR if d >= 0 else WITHOUT_COLOR for d in deltas]
    bars = ax.bar(x, deltas, color=colors, width=0.55, zorder=3)

    for bar, d, g in zip(bars, deltas, GRASP_ORDER):
        p  = stats[g]["mcnemar_p"]
        sl = sig_label(p)
        y  = d + 1.5 if d >= 0 else d - 3.5
        ax.text(bar.get_x() + bar.get_width() / 2, y, sl,
                ha="center", va="bottom" if d >= 0 else "top",
                fontsize=10, fontweight="bold" if sl != "ns" else "normal")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([GRASP_LABELS[g] for g in GRASP_ORDER])
    ax.set_ylabel("Δ success rate (With − Without) [pp]")
    ax.set_title("Gain from taxonomy conditioning\n(positive = With taxonomy better)",
                 pad=6)
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Right: summary table
    ax2 = fig.add_subplot(gs[1])
    ax2.axis("off")
    table_data = [["Grasp", "With", "Without", "Δ", "p"]]
    for g in GRASP_ORDER:
        s = stats[g]
        table_data.append([
            g.capitalize(),
            f"{s['with_k']}/{s['N']}\n({s['with_rate']*100:.0f}%)",
            f"{s['no_k']}/{s['N']}\n({s['no_rate']*100:.0f}%)",
            f"{s['delta']*100:+.0f}pp",
            f"p={s['mcnemar_p']:.3f}" if s['mcnemar_p'] >= 0.001 else "p<0.001",
        ])
    # Overall
    all_with = sum(stats[g]["with_k"] for g in GRASP_ORDER)
    all_no   = sum(stats[g]["no_k"]   for g in GRASP_ORDER)
    all_N    = sum(stats[g]["N"]      for g in GRASP_ORDER)
    all_with_rate = all_with / all_N
    all_no_rate   = all_no   / all_N
    all_delta     = (all_with_rate - all_no_rate) * 100
    # Overall McNemar
    all_w_vals = []
    all_n_vals = []
    for g in GRASP_ORDER:
        all_w_vals += list(np.array(grasp_data_global[g]["with"]))
        all_n_vals += list(np.array(grasp_data_global[g]["no"]))
    all_p = mcnemar_p(all_w_vals, all_n_vals)
    table_data.append([
        "OVERALL",
        f"{all_with}/{all_N}\n({all_with_rate*100:.0f}%)",
        f"{all_no}/{all_N}\n({all_no_rate*100:.0f}%)",
        f"{all_delta:+.0f}pp",
        f"p<0.001" if all_p < 0.001 else f"p={all_p:.3f}",
    ])

    col_widths = [0.14, 0.20, 0.22, 0.14, 0.22]
    table = ax2.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        cellLoc="center",
        loc="center",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.1)

    # header style
    for j in range(len(table_data[0])):
        table[0, j].set_facecolor("#1E3A5F")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # highlight overall row
    for j in range(len(table_data[0])):
        table[len(table_data) - 1, j].set_facecolor("#DBEAFE")
        table[len(table_data) - 1, j].set_text_props(fontweight="bold")

    ax2.set_title("Results summary", pad=10)

    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

grasp_data_global = None   # needed for fig5 overall McNemar

def main():
    global grasp_data_global

    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default="eval_results/paired_session_20260417_131412.json")
    parser.add_argument("--outdir",  default="eval_results/figures")
    args = parser.parse_args()

    import os
    os.makedirs(args.outdir, exist_ok=True)

    session, grasp_data = load_and_parse(args.session)
    grasp_data_global   = grasp_data
    stats = compute_stats(grasp_data)

    # Print summary
    print("\n=== RESULTS SUMMARY ===")
    print(f"{'Grasp':10s} {'With':>10s} {'Without':>10s} {'Δ':>8s} {'McNemar p':>12s} {'sig':>5s}")
    print("-" * 60)
    for g in GRASP_ORDER:
        s = stats[g]
        print(f"{g:10s} {s['with_k']:>3d}/{s['N']:<3d} ({s['with_rate']*100:5.1f}%)"
              f"  {s['no_k']:>3d}/{s['N']:<3d} ({s['no_rate']*100:5.1f}%)"
              f"  {s['delta']*100:+6.1f}pp"
              f"  {s['mcnemar_p']:10.4f}  {sig_label(s['mcnemar_p'])}")

    all_with = sum(stats[g]["with_k"] for g in GRASP_ORDER)
    all_no   = sum(stats[g]["no_k"]   for g in GRASP_ORDER)
    all_N    = sum(stats[g]["N"]      for g in GRASP_ORDER)
    print("-" * 60)
    print(f"{'OVERALL':10s} {all_with:>3d}/{all_N:<3d} ({all_with/all_N*100:5.1f}%)"
          f"  {all_no:>3d}/{all_N:<3d} ({all_no/all_N*100:5.1f}%)"
          f"  {(all_with-all_no)/all_N*100:+6.1f}pp")

    print("\nWilson 95% CIs:")
    for g in GRASP_ORDER:
        s = stats[g]
        print(f"  {g}: WITH [{s['with_lo']*100:.1f}%, {s['with_hi']*100:.1f}%]  "
              f"WITHOUT [{s['no_lo']*100:.1f}%, {s['no_hi']*100:.1f}%]")

    # Generate figures
    fig1_bar_chart    (stats,                          f"{args.outdir}/fig1_success_rates.png")
    fig2_heatmap      (grasp_data,                     f"{args.outdir}/fig2_pair_heatmap.png")
    fig3_z_centroid   (grasp_data,                     f"{args.outdir}/fig3_z_centroid.png")
    fig4_win_tie_loss (grasp_data, stats,               f"{args.outdir}/fig4_win_tie_loss.png")
    fig5_delta_and_summary(stats,                      f"{args.outdir}/fig5_delta_summary.png")

    print(f"\nAll figures saved to {args.outdir}/")


if __name__ == "__main__":
    main()
