#!/usr/bin/env python3
"""
Display paired evaluation results in a nicely formatted terminal table
and generate a publication-quality results table figure (PNG).

Usage:
    python3 eval_results/display_results.py
    python3 eval_results/display_results.py \\
        --session eval_results/paired_session_20260515_074256.json
    python3 eval_results/display_results.py --no-figure   # terminal only
"""

import json
import argparse
import os
import math
import numpy as np
import sys
from pathlib import Path

GRASP_ORDER  = ["crimp", "jug", "sloper", "pinch"]
RATCHET_MAX_N = 33.9


# ── statistics ────────────────────────────────────────────────────────────────

def wilcoxon_p(a, b):
    from scipy import stats
    diffs = np.asarray(a) - np.asarray(b)
    if len(diffs) < 2 or np.all(diffs == 0):
        return 1.0
    try:
        _, p = stats.wilcoxon(a, b, alternative="two-sided")
        return float(p)
    except ValueError:
        return 1.0


def cohen_d(a, b):
    diff = np.asarray(a) - np.asarray(b)
    return float(diff.mean() / diff.std()) if diff.std() > 0 else float("nan")


def bootstrap_ci(a, b, stat_fn, n_boot=2000, seed=42):
    """Bootstrap 95% CI for any paired statistic."""
    rng = np.random.default_rng(seed)
    boot = []
    a, b = np.asarray(a), np.asarray(b)
    for _ in range(n_boot):
        idx = rng.integers(0, len(a), size=len(a))
        boot.append(stat_fn(a[idx], b[idx]))
    boot = np.sort(boot)
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def iqr(vals):
    return float(np.percentile(vals, 75) - np.percentile(vals, 25))


# ── data loading ──────────────────────────────────────────────────────────────

def load_sessions(paths):
    all_pairs = []
    for path in paths:
        with open(path) as f:
            d = json.load(f)
        for pair in d["pairs"]:
            if pair.get("aborted"):
                continue
            g = pair.get("grasp_type")
            if g not in GRASP_ORDER:
                continue
            wr = pair["pc_stats"].get("WITH_TAXONOMY",    {}).get("ratchet")
            nr = pair["pc_stats"].get("WITHOUT_TAXONOMY", {}).get("ratchet")
            if wr is None or nr is None:
                continue
            all_pairs.append({
                "grasp_type":    g,
                "orientation":   pair.get("orientation_deg"),
                "with_force_N":  wr["force_N"],
                "no_force_N":    nr["force_N"],
                "with_teeth":    wr["teeth"],
                "no_teeth":      nr["teeth"],
            })
    return all_pairs


# ── terminal display ──────────────────────────────────────────────────────────

BOLD  = "\033[1m"
RESET = "\033[0m"
GREEN = "\033[32m"
RED   = "\033[31m"
CYAN  = "\033[36m"
YELLOW= "\033[33m"
DIM   = "\033[2m"

def _color_p(p):
    if p < 0.001: return f"{GREEN}{BOLD}p<0.001 ***{RESET}"
    if p < 0.01:  return f"{GREEN}p={p:.4f} ** {RESET}"
    if p < 0.05:  return f"{YELLOW}p={p:.4f} *  {RESET}"
    return f"{DIM}p={p:.4f} ns {RESET}"

def _color_d(d):
    if abs(d) >= 1.2: return f"{GREEN}{BOLD}{d:+.2f}{RESET}"
    if abs(d) >= 0.8: return f"{GREEN}{d:+.2f}{RESET}"
    if abs(d) >= 0.5: return f"{YELLOW}{d:+.2f}{RESET}"
    return f"{d:+.2f}"


def print_results(pairs):
    by_g = {g: [p for p in pairs if p["grasp_type"] == g] for g in GRASP_ORDER}

    W = 74
    print()
    print("╔" + "═" * W + "╗")
    print("║" + f"{'TAXONOMY-CONDITIONED vs UNCONDITIONED DIFFUSION POLICY':^{W}}" + "║")
    print("║" + f"{'Spring testbed · ratchet slip force · Wilcoxon signed-rank (paired)':^{W}}" + "║")
    print("╠" + "═" * W + "╣")
    print("║" + f"  {'Grasp':8s}  {'n':>3s}  {'WITH med±IQR':>14s}  {'W/O med±IQR':>14s}  "
               f"{'Δ med':>6s}  p-value       d" + " " * 4 + "║")
    print("╠" + "─" * W + "╣")

    all_w, all_n = [], []
    for g in GRASP_ORDER:
        gp = by_g[g]
        if not gp:
            continue
        wf = [p["with_force_N"] for p in gp]
        nf = [p["no_force_N"]   for p in gp]
        all_w.extend(wf); all_n.extend(nf)
        pv  = wilcoxon_p(wf, nf)
        d   = cohen_d(wf, nf)
        wmed = np.median(wf); wiqr = iqr(wf)
        nmed = np.median(nf); niqr = iqr(nf)
        dmed = wmed - nmed
        # plain row (colour applied inline)
        row = (f"  {g.capitalize():8s}  {len(gp):>3d}  "
               f"{wmed:>5.1f}±{wiqr:>4.1f} N  "
               f"{nmed:>5.1f}±{niqr:>4.1f} N  "
               f"{dmed:>+5.1f} N  ")
        print("║" + row + _color_p(pv) + "  " + _color_d(d) + " " * (6 - len(f"{d:+.2f}")) + "║")

    print("╠" + "─" * W + "╣")
    pv  = wilcoxon_p(all_w, all_n)
    d   = cohen_d(all_w, all_n)
    wmed = np.median(all_w); wiqr = iqr(all_w)
    nmed = np.median(all_n); niqr = iqr(all_n)
    dmed = wmed - nmed
    row  = (f"  {'OVERALL':8s}  {len(pairs):>3d}  "
            f"{wmed:>5.1f}±{wiqr:>4.1f} N  "
            f"{nmed:>5.1f}±{niqr:>4.1f} N  "
            f"{dmed:>+5.1f} N  ")
    print("║" + BOLD + row + RESET + _color_p(pv) + "  " + _color_d(d) + " " * (6 - len(f"{d:+.2f}")) + "║")
    print("╚" + "═" * W + "╝")

    print()
    print(f"{DIM}  Force ref: 0 teeth=5.2N  3t=13.1N  5t=18.3N  7t=23.5N  9t=28.7N  11t=33.9N (max){RESET}")
    print(f"{DIM}  Effect size: |d|≥0.8=large, ≥0.5=medium, ≥0.2=small{RESET}")

    # Pegged note
    pw = sum(1 for p in pairs if p["with_teeth"] >= 11)
    pn = sum(1 for p in pairs if p["no_teeth"]   >= 11)
    if pw or pn:
        print(f"\n  {YELLOW}Note:{RESET} {pw} WITH and {pn} WITHOUT trials pegged at 11 teeth "
              f"(≥{RATCHET_MAX_N:.0f} N — true force may be higher).")
    print()


# ── results table figure ──────────────────────────────────────────────────────

def make_table_figure(pairs, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "figure.dpi": 150, "savefig.dpi": 220, "savefig.bbox": "tight",
    })

    WITH_COLOR    = "#2563EB"
    WITHOUT_COLOR = "#DC2626"

    by_g = {g: [p for p in pairs if p["grasp_type"] == g] for g in GRASP_ORDER}

    rows = []
    all_w, all_n = [], []
    for g in GRASP_ORDER:
        gp = by_g[g]
        if not gp:
            continue
        wf = [p["with_force_N"] for p in gp]
        nf = [p["no_force_N"]   for p in gp]
        all_w.extend(wf); all_n.extend(nf)
        pv  = wilcoxon_p(wf, nf)
        d   = cohen_d(wf, nf)
        d_lo, d_hi = bootstrap_ci(wf, nf,
                                   lambda a, b: cohen_d(list(a), list(b)))
        rows.append({
            "grasp":    g.capitalize(),
            "n":        len(gp),
            "with_med": np.median(wf),
            "with_iqr": iqr(wf),
            "no_med":   np.median(nf),
            "no_iqr":   iqr(nf),
            "delta":    np.median(wf) - np.median(nf),
            "p":        pv,
            "d":        d,
            "d_lo":     d_lo,
            "d_hi":     d_hi,
        })

    # Overall row
    pv_all = wilcoxon_p(all_w, all_n)
    d_all  = cohen_d(all_w, all_n)
    d_lo_all, d_hi_all = bootstrap_ci(all_w, all_n,
                                       lambda a, b: cohen_d(list(a), list(b)))
    rows.append({
        "grasp":    "OVERALL",
        "n":        len(pairs),
        "with_med": np.median(all_w),
        "with_iqr": iqr(all_w),
        "no_med":   np.median(all_n),
        "no_iqr":   iqr(all_n),
        "delta":    np.median(all_w) - np.median(all_n),
        "p":        pv_all,
        "d":        d_all,
        "d_lo":     d_lo_all,
        "d_hi":     d_hi_all,
    })

    # ── layout: table on left, bar chart on right ──────────────────────────────
    fig = plt.figure(figsize=(14, 5.5))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.6, 1.0], wspace=0.12)

    # Table
    ax_t = fig.add_subplot(gs[0])
    ax_t.axis("off")

    col_labels = ["Grasp", "n",
                  "With taxonomy\nmedian ± IQR (N)",
                  "Without taxonomy\nmedian ± IQR (N)",
                  "Δ median\n(N)",
                  "Wilcoxon p",
                  "Cohen's d\n[95% CI]"]

    cell_text = []
    cell_colors = []
    for r in rows:
        stars = sig_stars(r["p"])
        p_str = ("p<0.001" if r["p"] < 0.001 else f"p={r['p']:.4f}") + f" {stars}"
        d_str = f"{r['d']:+.2f} [{r['d_lo']:+.2f}, {r['d_hi']:+.2f}]"
        cell_text.append([
            r["grasp"],
            str(r["n"]),
            f"{r['with_med']:.1f} ± {r['with_iqr']:.1f}",
            f"{r['no_med']:.1f} ± {r['no_iqr']:.1f}",
            f"{r['delta']:+.1f}",
            p_str,
            d_str,
        ])
        # row background: highlight overall
        bg = "#DBEAFE" if r["grasp"] == "OVERALL" else "white"
        cell_colors.append([bg] * len(col_labels))

    table = ax_t.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        cellColours=cell_colors,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.2)

    # Header style
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#1E3A5F")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # p-value cell colour by significance
    for i, r in enumerate(rows):
        p = r["p"]
        if p < 0.001:
            c = "#BBFAD1"
        elif p < 0.01:
            c = "#D1FAE5"
        elif p < 0.05:
            c = "#FEF3C7"
        else:
            c = "#FEE2E2"
        table[i + 1, 5].set_facecolor(c)

    ax_t.set_title("Taxonomy conditioning vs. unconditioned policy — slip force at ratchet\n"
                   "(Wilcoxon signed-rank, paired, n=20 pairs per grasp type)",
                   pad=12, fontsize=10.5, fontweight="bold")

    # Bar chart of Cohen's d
    ax_b = fig.add_subplot(gs[1])
    non_overall = [r for r in rows if r["grasp"] != "OVERALL"]
    xs     = np.arange(len(non_overall))
    ds     = [r["d"] for r in non_overall]
    lo_err = [r["d"] - r["d_lo"] for r in non_overall]
    hi_err = [r["d_hi"] - r["d"] for r in non_overall]
    gnames = [r["grasp"] for r in non_overall]

    bars = ax_b.bar(xs, ds, color=WITH_COLOR, alpha=0.72, width=0.52, zorder=3)
    ax_b.errorbar(xs, ds, yerr=[lo_err, hi_err],
                  fmt="none", color="black", capsize=5, capthick=1.4,
                  linewidth=1.4, zorder=4)
    for xi, d in zip(xs, ds):
        ax_b.text(xi, d + hi_err[xi] + 0.05, f"d={d:.2f}",
                  ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    for level, lbl, ls in [(0.5, "medium", "--"), (0.8, "large", "-.")]:
        ax_b.axhline(level, color="#AAA", linewidth=0.9, linestyle=ls,
                     label=f"|d|={level} ({lbl})", zorder=0)

    ax_b.set_xticks(xs)
    ax_b.set_xticklabels(gnames, fontsize=9.5)
    ax_b.set_ylabel("Cohen's d  (with 95% bootstrap CI)")
    ax_b.set_title("Effect size per grasp type", pad=8, fontsize=10)
    ax_b.set_ylim(0, max(ds) * 1.4)
    ax_b.legend(fontsize=8, loc="upper left")
    ax_b.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax_b.set_axisbelow(True)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Display paired eval results in terminal + save table figure")
    parser.add_argument(
        "--session", nargs="+",
        default=["eval_results/paired_session_20260515_074256.json"],
        help="One or more paired_session_*.json paths")
    parser.add_argument("--outdir", default="eval_results/figures")
    parser.add_argument("--no-figure", action="store_true",
                        help="Print terminal table only, skip PNG generation")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Resolve paths
    script_dir = Path(__file__).resolve().parent.parent
    resolved = []
    for p in args.session:
        pp = Path(p)
        if not pp.exists():
            pp = script_dir / p
        resolved.append(str(pp))

    pairs = load_sessions(resolved)
    if not pairs:
        print("ERROR: no ratchet data found.")
        sys.exit(1)

    print(f"\nLoaded {len(pairs)} pairs from {len(resolved)} session(s).")
    print_results(pairs)

    if not args.no_figure:
        make_table_figure(pairs, f"{args.outdir}/fig_results_table.png")


if __name__ == "__main__":
    main()
