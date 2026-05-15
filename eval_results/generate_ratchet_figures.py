#!/usr/bin/env python3
"""
Ratchet-force figure generator for paired evaluation results.

Uses continuous slip-force data from the spring testbed ratchet instead of
binary success ratings.  The primary metric is force_N per trial, derived from:
  F_total = (1.18 + 1.6 * displacement_in) * 4.448  [N]
  displacement_in = teeth * 9.3 mm / 25.4

Usage:
    python3 eval_results/generate_ratchet_figures.py
    python3 eval_results/generate_ratchet_figures.py \\
        --session eval_results/paired_session_<ts>.json
    # Merge multiple sessions (e.g. if you resumed across days):
    python3 eval_results/generate_ratchet_figures.py \\
        --session eval_results/paired_session_<ts1>.json \\
                  eval_results/paired_session_<ts2>.json
"""

import json
import argparse
import os
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from pathlib import Path

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

WITH_COLOR    = "#2563EB"
WITHOUT_COLOR = "#DC2626"

GRASP_ORDER  = ["crimp", "jug", "sloper", "pinch"]
GRASP_LABELS = {
    "crimp":  "Crimp\n(edge_B)",
    "jug":    "Jug\n(edge_A)",
    "sloper": "Sloper",
    "pinch":  "Pinch",
}
GRASP_COLORS = {
    "crimp":  "#F59E0B",
    "jug":    "#3B82F6",
    "sloper": "#10B981",
    "pinch":  "#8B5CF6",
}

RATCHET_MAX_N = 33.9   # force at 11 teeth (≈ "at least this strong")


# ── data loading ──────────────────────────────────────────────────────────────

def load_sessions(paths):
    """Load one or more session JSONs and return a flat list of pair dicts.

    Pairs without ratchet data for either model (user skipped the prompt)
    are silently dropped.  Teeth=0 (no hold movement) is kept — it is a
    valid data point meaning the grip produced no measurable force.
    """
    all_pairs = []
    for path in paths:
        with open(path) as f:
            session = json.load(f)
        for pair in session["pairs"]:
            if pair.get("aborted"):
                continue
            g = pair.get("grasp_type")
            if g not in GRASP_ORDER:
                continue
            with_r = pair["pc_stats"].get("WITH_TAXONOMY",    {}).get("ratchet")
            no_r   = pair["pc_stats"].get("WITHOUT_TAXONOMY", {}).get("ratchet")
            if with_r is None or no_r is None:
                continue
            all_pairs.append({
                "grasp_type":     g,
                "orientation_deg": pair.get("orientation_deg"),
                "pair_idx":        pair.get("pair", len(all_pairs)),
                "with_teeth":      with_r["teeth"],
                "no_teeth":        no_r["teeth"],
                "with_force_N":    with_r["force_N"],
                "no_force_N":      no_r["force_N"],
                "with_force_lbf":  with_r["force_lbf"],
                "no_force_lbf":    no_r["force_lbf"],
                "with_disp_mm":    with_r["displacement_mm"],
                "no_disp_mm":      no_r["displacement_mm"],
                # pegged = ratchet hit its max (grip may be even stronger)
                "with_pegged":     with_r["teeth"] >= 11,
                "no_pegged":       no_r["teeth"] >= 11,
            })
    return all_pairs


def group_by_grasp(pairs):
    by_g = {g: [] for g in GRASP_ORDER}
    for p in pairs:
        if p["grasp_type"] in by_g:
            by_g[p["grasp_type"]].append(p)
    return by_g


# ── statistics ────────────────────────────────────────────────────────────────

def wilcoxon_p(a, b):
    """Wilcoxon signed-rank test (paired, two-sided). Returns p-value."""
    diffs = np.asarray(a, float) - np.asarray(b, float)
    if len(diffs) < 2 or np.all(diffs == 0):
        return 1.0
    try:
        _, p = stats.wilcoxon(a, b, alternative="two-sided")
        return float(p)
    except ValueError:
        return 1.0


def sig_label(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def cohen_d(a, b):
    """Cohen's d for paired data (effect size)."""
    diff = np.asarray(a) - np.asarray(b)
    if diff.std() == 0:
        return float("nan")
    return float(diff.mean() / diff.std())


# ── Figure 1: force box plots per grasp type ─────────────────────────────────

def fig1_force_boxplots(by_grasp, out_path):
    """Box plots of slip force (N) — WITH vs WITHOUT — per grasp type."""
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    x = np.arange(len(GRASP_ORDER))
    bw = 0.32

    def _bp(data, pos, color):
        return ax.boxplot(
            data, positions=[pos], widths=bw * 0.9, patch_artist=True,
            boxprops=dict(facecolor=color, alpha=0.65),
            medianprops=dict(color="white", linewidth=2.2),
            whiskerprops=dict(color=color, linewidth=1.2),
            capprops=dict(color=color, linewidth=1.2),
            flierprops=dict(marker="o", color=color, alpha=0.55, markersize=4),
        )

    for i, g in enumerate(GRASP_ORDER):
        pairs = by_grasp[g]
        if not pairs:
            continue
        with_f = [p["with_force_N"] for p in pairs]
        no_f   = [p["no_force_N"]   for p in pairs]

        _bp(with_f, x[i] - bw / 2, WITH_COLOR)
        _bp(no_f,   x[i] + bw / 2, WITHOUT_COLOR)

        # individual points (jittered)
        rng = np.random.default_rng(i)
        for vals, xi in [(with_f, x[i] - bw / 2), (no_f, x[i] + bw / 2)]:
            jit = rng.uniform(-0.06, 0.06, len(vals))
            col = WITH_COLOR if xi < x[i] else WITHOUT_COLOR
            ax.scatter(xi + jit, vals, color=col, alpha=0.35, s=12, zorder=4)

        # significance bracket
        p_val = wilcoxon_p(with_f, no_f)
        sl    = sig_label(p_val)
        top   = max(max(with_f), max(no_f)) + 1.0
        ax.plot([x[i] - bw / 2, x[i] + bw / 2], [top, top],
                color="black", linewidth=0.9)
        ax.text(x[i], top + 0.4, sl, ha="center", va="bottom",
                fontsize=11,
                fontweight="bold" if sl != "ns" else "normal",
                color="black" if sl != "ns" else "#888")

        # n label
        ax.text(x[i], -2.5, f"n={len(pairs)}", ha="center", va="top",
                fontsize=8, color="#555")

    ax.axhline(RATCHET_MAX_N, color="#AAA", linewidth=0.8, linestyle=":",
               zorder=0)
    ax.text(len(GRASP_ORDER) - 0.45, RATCHET_MAX_N + 0.3,
            "ratchet max (11 teeth)", fontsize=7.5, color="#999", va="bottom")

    with_p  = mpatches.Patch(facecolor=WITH_COLOR,    alpha=0.7, label="With taxonomy")
    no_p    = mpatches.Patch(facecolor=WITHOUT_COLOR,  alpha=0.7, label="Without taxonomy")
    ax.legend(handles=[with_p, no_p], loc="upper right", framealpha=0.95)
    ax.set_xticks(x)
    ax.set_xticklabels([GRASP_LABELS[g] for g in GRASP_ORDER])
    ax.set_ylabel("Slip force (N)  [lower = grip failed sooner]")
    ax.set_title(
        "Grip strength at slip: taxonomy-conditioned vs. unconditioned policy\n"
        "Spring testbed · Wilcoxon signed-rank test (paired) · box = IQR, line = median",
        pad=10,
    )
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.set_ylim(-4, None)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


# ── Figure 2: per-pair scatter WITH vs WITHOUT ────────────────────────────────

def fig2_paired_scatter(pairs, out_path):
    """Scatter WITH force (y) vs WITHOUT force (x) — one dot per pair."""
    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    for g in GRASP_ORDER:
        g_p = [p for p in pairs if p["grasp_type"] == g]
        if not g_p:
            continue
        xs = [p["no_force_N"]   for p in g_p]
        ys = [p["with_force_N"] for p in g_p]
        ax.scatter(xs, ys, c=GRASP_COLORS[g], label=g.capitalize(),
                   s=60, alpha=0.75, zorder=3, edgecolors="white", linewidth=0.5)

    lim = max(
        max((p["with_force_N"] for p in pairs), default=35),
        max((p["no_force_N"]   for p in pairs), default=35),
    ) + 3
    ax.plot([0, lim], [0, lim], "k--", linewidth=0.9, alpha=0.45, label="tie (y=x)")
    ax.fill_between([0, lim], [0, lim], [lim, lim],
                    color=WITH_COLOR,    alpha=0.04)
    ax.fill_between([0, lim], [0, 0],   [0, lim],
                    color=WITHOUT_COLOR, alpha=0.04)
    ax.text(1.2, lim - 3,    "WITH stronger",    color=WITH_COLOR,    fontsize=9, fontstyle="italic")
    ax.text(lim - 11.5, 1.8, "WITHOUT stronger", color=WITHOUT_COLOR, fontsize=9, fontstyle="italic")

    # overall force bias
    all_delta = [p["with_force_N"] - p["no_force_N"] for p in pairs]
    med_d = np.median(all_delta)
    pv    = wilcoxon_p([p["with_force_N"] for p in pairs],
                       [p["no_force_N"]   for p in pairs])
    ax.text(0.03, 0.97,
            f"Median Δ = {med_d:+.1f} N  (Wilcoxon {sig_label(pv)}, p={pv:.3f})",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#CCC"))

    ax.set_xlabel("Without taxonomy — slip force (N)")
    ax.set_ylabel("With taxonomy — slip force (N)")
    ax.set_title("Head-to-head pair comparison\n(each dot = one paired trial)", pad=6)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
    ax.xaxis.grid(True, alpha=0.3, linestyle="--")
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


# ── Figure 3: force by orientation ───────────────────────────────────────────

def fig3_force_by_orientation(pairs, out_path):
    """Mean slip force vs hold orientation for WITH and WITHOUT per grasp type."""
    orients = sorted({p["orientation_deg"] for p in pairs
                      if p["orientation_deg"] is not None})
    if len(orients) < 2:
        print(f"  Skipping fig3 — only {len(orients)} orientation(s) found")
        return

    fig, axes = plt.subplots(1, len(GRASP_ORDER),
                              figsize=(4 * len(GRASP_ORDER), 4.5),
                              sharey=True)
    if len(GRASP_ORDER) == 1:
        axes = [axes]

    for ax, g in zip(axes, GRASP_ORDER):
        g_p = [p for p in pairs
               if p["grasp_type"] == g and p["orientation_deg"] is not None]

        for key, label, color, ls in [
            ("with_force_N", "With",    WITH_COLOR,    "-"),
            ("no_force_N",   "Without", WITHOUT_COLOR, "--"),
        ]:
            means, sems, xs_valid = [], [], []
            for ori in orients:
                vals = [p[key] for p in g_p if p["orientation_deg"] == ori]
                if vals:
                    means.append(np.mean(vals))
                    sems.append(np.std(vals) / math.sqrt(len(vals)) if len(vals) > 1 else 0)
                    xs_valid.append(ori)
            if means:
                ax.plot(xs_valid, means, color=color, linestyle=ls,
                        marker="o", linewidth=1.8, markersize=6, label=label)
                ax.fill_between(xs_valid,
                                np.array(means) - np.array(sems),
                                np.array(means) + np.array(sems),
                                color=color, alpha=0.12)

        ax.set_title(g.capitalize(), fontsize=11)
        ax.set_xticks(orients)
        ax.set_xticklabels([f"{int(o)}°" if o == int(o) else f"{o}°"
                            for o in orients], fontsize=8)
        ax.xaxis.grid(True, alpha=0.3, linestyle="--")
        ax.yaxis.grid(True, alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        if ax == axes[0]:
            ax.set_ylabel("Mean slip force (N)")
        ax.set_xlabel("Hold orientation")

    axes[0].legend(fontsize=9)
    fig.suptitle("Slip force vs hold orientation  (solid=With taxonomy, dashed=Without)\n"
                 "error band = ±1 SEM",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ── Figure 4: per-pair force delta (signed bar chart) ────────────────────────

def fig4_force_delta(pairs, out_path):
    """Signed bar per pair: WITH − WITHOUT force, sorted by grasp type."""
    sorted_p = sorted(pairs,
                      key=lambda p: (GRASP_ORDER.index(p["grasp_type"]),
                                     p.get("pair_idx", 0)))
    deltas = [p["with_force_N"] - p["no_force_N"] for p in sorted_p]
    colors = [WITH_COLOR if d >= 0 else WITHOUT_COLOR for d in deltas]

    fig, ax = plt.subplots(figsize=(max(10, len(deltas) * 0.32), 4.8))
    xs = np.arange(len(deltas))
    ax.bar(xs, deltas, color=colors, width=0.8, alpha=0.72)
    ax.axhline(0, color="black", linewidth=0.9)

    # grasp type bands + labels
    pos = 0
    for g in GRASP_ORDER:
        cnt = sum(1 for p in sorted_p if p["grasp_type"] == g)
        if cnt == 0:
            continue
        ax.axvline(pos - 0.5, color="#CCC", linewidth=0.8, zorder=0)
        ax.text(pos + cnt / 2 - 0.5, ax.get_ylim()[0] - 0.5,
                g.capitalize(), ha="center", va="top", fontsize=9, color="#444")
        pos += cnt
    ax.axvline(pos - 0.5, color="#CCC", linewidth=0.8, zorder=0)

    # median line
    med = np.median(deltas)
    ax.axhline(med, color="#555", linewidth=1.2, linestyle=":",
               label=f"Overall median Δ = {med:+.1f} N")

    with_p = mpatches.Patch(facecolor=WITH_COLOR,    alpha=0.7, label="WITH stronger")
    no_p   = mpatches.Patch(facecolor=WITHOUT_COLOR,  alpha=0.7, label="WITHOUT stronger")
    ax.legend(handles=[with_p, no_p,
                       plt.Line2D([0],[0], color="#555", linestyle=":", linewidth=1.2,
                                  label=f"Median Δ = {med:+.1f} N")],
              loc="upper right", fontsize=9)

    ax.set_xticks([])
    ax.set_xlabel("Pairs (sorted by grasp type)")
    ax.set_ylabel("Force delta: WITH − WITHOUT (N)")
    ax.set_title("Per-pair force advantage of taxonomy conditioning\n"
                 "(bar above zero = WITH taxonomy achieved higher slip force)", pad=8)
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


# ── Terminal summary ──────────────────────────────────────────────────────────

def print_force_summary(by_grasp, pairs):
    print("\n" + "=" * 72)
    print("  RATCHET FORCE SUMMARY  (primary metric for this evaluation)")
    print("=" * 72)
    hdr = f"  {'Grasp':10s} {'n':>4s}  {'WITH med':>9s}  {'W/O med':>8s}  {'Δmed':>7s}  {'Wilcoxon p':>11s}  {'sig':>4s}  {'d':>6s}"
    print(hdr)
    print("  " + "─" * 68)

    all_w, all_n = [], []
    for g in GRASP_ORDER:
        gp = by_grasp[g]
        if not gp:
            print(f"  {g:10s}    0   (no ratchet data)")
            continue
        wf = [p["with_force_N"] for p in gp]
        nf = [p["no_force_N"]   for p in gp]
        all_w.extend(wf); all_n.extend(nf)
        pv   = wilcoxon_p(wf, nf)
        dmed = np.median(wf) - np.median(nf)
        cd   = cohen_d(wf, nf)
        print(f"  {g:10s} {len(gp):>4d}  "
              f"{np.median(wf):>7.1f} N  "
              f"{np.median(nf):>6.1f} N  "
              f"{dmed:>+5.1f} N  "
              f"{pv:>11.4f}  {sig_label(pv):>4s}  {cd:>+5.2f}")

    print("  " + "─" * 68)
    if all_w:
        pv   = wilcoxon_p(all_w, all_n)
        dmed = np.median(all_w) - np.median(all_n)
        cd   = cohen_d(all_w, all_n)
        print(f"  {'OVERALL':10s} {len(pairs):>4d}  "
              f"{np.median(all_w):>7.1f} N  "
              f"{np.median(all_n):>6.1f} N  "
              f"{dmed:>+5.1f} N  "
              f"{pv:>11.4f}  {sig_label(pv):>4s}  {cd:>+5.2f}")

    n_pegged_w = sum(1 for p in pairs if p["with_pegged"])
    n_pegged_n = sum(1 for p in pairs if p["no_pegged"])
    if n_pegged_w or n_pegged_n:
        print(f"\n  Note: {n_pegged_w} WITH and {n_pegged_n} WITHOUT trials pegged at 11 teeth "
              f"(≥{RATCHET_MAX_N:.1f} N — true force may be higher).")

    print("\n  Force reference: 0 teeth=5.2N  3t=13.1N  5t=18.3N  7t=23.5N  9t=28.7N  11t=33.9N")
    print("=" * 72)


# ── Figure 5: CDF of slip forces ─────────────────────────────────────────────

def fig5_force_cdf(by_grasp, pairs, out_path):
    """Empirical CDF of slip force, per grasp type + overall panel."""
    fig, axes = plt.subplots(1, 5, figsize=(16, 4.2), sharey=True)

    def _ecdf(vals):
        s = np.sort(vals)
        y = np.arange(1, len(s) + 1) / len(s)
        # prepend 0 at x=0 so CDF starts from the origin
        return np.concatenate([[0], s]), np.concatenate([[0], y])

    panels = [(g, by_grasp[g]) for g in GRASP_ORDER] + [("overall", None)]

    for ax, (label, gp) in zip(axes, panels):
        if label == "overall":
            wf = [p["with_force_N"] for p in pairs]
            nf = [p["no_force_N"]   for p in pairs]
            title = "OVERALL"
            lw = 2.4
        else:
            wf = [p["with_force_N"] for p in gp] if gp else []
            nf = [p["no_force_N"]   for p in gp] if gp else []
            title = label.capitalize()
            lw = 1.9

        if wf:
            xw, yw = _ecdf(wf)
            xn, yn = _ecdf(nf)
            ax.step(xw, yw, color=WITH_COLOR,    linewidth=lw, where="post",
                    label="With taxonomy")
            ax.step(xn, yn, color=WITHOUT_COLOR,  linewidth=lw, where="post",
                    label="Without taxonomy")

            # median lines
            ax.axvline(np.median(wf), color=WITH_COLOR,    linestyle=":", alpha=0.7)
            ax.axvline(np.median(nf), color=WITHOUT_COLOR,  linestyle=":", alpha=0.7)

            pv = wilcoxon_p(wf, nf)
            ax.set_title(f"{title}\n(Wilcoxon {sig_label(pv)}, p={pv:.3f})",
                         fontsize=10)

        ax.set_xlabel("Slip force (N)")
        ax.set_xlim(0, RATCHET_MAX_N + 2)
        ax.set_ylim(0, 1.05)
        ax.xaxis.grid(True, alpha=0.3, linestyle="--")
        ax.yaxis.grid(True, alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        if ax == axes[0]:
            ax.set_ylabel("Cumulative fraction")
            ax.legend(fontsize=8.5, loc="lower right")

    axes[2].set_xlabel("Slip force (N)")
    fig.suptitle("Cumulative distribution of slip force — taxonomy conditioning shifts distribution rightward",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ── Figure 6: Cohen's d effect size ──────────────────────────────────────────

def _bootstrap_d_ci(a, b, n_boot=2000, seed=42):
    """Percentile bootstrap 95% CI for Cohen's d of paired differences."""
    diffs = np.asarray(a) - np.asarray(b)
    rng   = np.random.default_rng(seed)
    boot  = []
    for _ in range(n_boot):
        s = rng.choice(diffs, size=len(diffs), replace=True)
        if s.std() > 0:
            boot.append(s.mean() / s.std())
    boot = np.sort(boot)
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def fig6_effect_sizes(by_grasp, out_path):
    """Cohen's d per grasp type with bootstrap 95% CI."""
    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    ds, lo_errs, hi_errs, colors_bar, labels = [], [], [], [], []
    for g in GRASP_ORDER:
        gp = by_grasp[g]
        if not gp:
            continue
        wf = [p["with_force_N"] for p in gp]
        nf = [p["no_force_N"]   for p in gp]
        d  = cohen_d(wf, nf)
        lo, hi = _bootstrap_d_ci(wf, nf)
        ds.append(d)
        lo_errs.append(d - lo)
        hi_errs.append(hi - d)
        colors_bar.append(WITH_COLOR if d > 0 else WITHOUT_COLOR)
        labels.append(g.capitalize())

    x = np.arange(len(ds))
    err = np.array([lo_errs, hi_errs])
    bars = ax.bar(x, ds, color=colors_bar, width=0.5, alpha=0.75, zorder=3)
    ax.errorbar(x, ds, yerr=err, fmt="none", color="black",
                capsize=6, capthick=1.5, linewidth=1.5, zorder=4)

    for xi, d in zip(x, ds):
        ax.text(xi, d + (0.06 if d >= 0 else -0.1), f"d = {d:.2f}",
                ha="center", va="bottom" if d >= 0 else "top",
                fontsize=10.5, fontweight="bold")

    # Reference lines
    for level, label, style in [(0.2, "small", ":"), (0.5, "medium", "--"), (0.8, "large", "-.")]:
        ax.axhline(level, color="#AAA", linewidth=0.9, linestyle=style,
                   label=f"|d|={level} ({label})", zorder=0)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Cohen's d  (paired differences)")
    ax.set_title("Effect size of taxonomy conditioning on grip strength\n"
                 "(bars = Cohen's d, error bars = bootstrap 95% CI, n=20 pairs each)",
                 pad=8)
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(ds) * 1.35)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


# ── Figure 7: 4-panel per-grasp scatter ──────────────────────────────────────

def fig7_per_grasp_scatter(by_grasp, out_path):
    """4-panel scatter (one per grasp type) WITH vs WITHOUT force per pair."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2), sharey=True, sharex=True)

    global_lim = max(
        max((p["with_force_N"] for g in by_grasp.values() for p in g), default=35),
        max((p["no_force_N"]   for g in by_grasp.values() for p in g), default=35),
    ) + 2

    for ax, g in zip(axes, GRASP_ORDER):
        gp = by_grasp[g]
        wf = [p["with_force_N"] for p in gp]
        nf = [p["no_force_N"]   for p in gp]

        ax.scatter(nf, wf, c=GRASP_COLORS[g], s=55, alpha=0.75,
                   zorder=3, edgecolors="white", linewidth=0.5)
        ax.plot([0, global_lim], [0, global_lim], "k--",
                linewidth=0.9, alpha=0.4, zorder=2)

        # Regression line
        if len(nf) > 2:
            m, b = np.polyfit(nf, wf, 1)
            xs = np.linspace(0, global_lim, 100)
            ax.plot(xs, m * xs + b, color=GRASP_COLORS[g],
                    linewidth=1.5, alpha=0.6, zorder=2)

        pv = wilcoxon_p(wf, nf)
        d  = cohen_d(wf, nf)
        ax.set_title(f"{g.capitalize()}\n"
                     f"d={d:.2f}  {sig_label(pv)} (p={pv:.3f})",
                     fontsize=10)
        ax.set_xlim(0, global_lim)
        ax.set_ylim(0, global_lim)
        ax.set_xlabel("Without taxonomy (N)")
        ax.xaxis.grid(True, alpha=0.3, linestyle="--")
        ax.yaxis.grid(True, alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

    axes[0].set_ylabel("With taxonomy (N)")
    fig.suptitle("Per-grasp-type head-to-head: WITH vs WITHOUT taxonomy grip force\n"
                 "(each dot = one paired trial · dashed = tie · solid = regression)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ── Figure 8: orientation × grasp heatmap ────────────────────────────────────

def fig8_orientation_heatmap(by_grasp, out_path):
    """Heatmap: mean slip force for WITH and WITHOUT at each orientation × grasp type."""
    orients = [-45.0, -22.5, 0.0, 22.5, 45.0]
    orient_labels = [f"{int(o)}°" if o == int(o) else f"{o}°" for o in orients]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    for ax, (key, model_label, cmap) in zip(axes, [
        ("with_force_N", "With taxonomy",    "Blues"),
        ("no_force_N",   "Without taxonomy", "Reds"),
    ]):
        data = np.full((len(GRASP_ORDER), len(orients)), np.nan)
        for gi, g in enumerate(GRASP_ORDER):
            for oi, ori in enumerate(orients):
                vals = [p[key] for p in by_grasp[g]
                        if p.get("orientation_deg") == ori]
                if vals:
                    data[gi, oi] = np.mean(vals)

        im = ax.imshow(data, cmap=cmap, vmin=0, vmax=RATCHET_MAX_N,
                       aspect="auto")
        plt.colorbar(im, ax=ax, label="Mean slip force (N)", shrink=0.85)

        ax.set_xticks(range(len(orients)))
        ax.set_xticklabels(orient_labels)
        ax.set_yticks(range(len(GRASP_ORDER)))
        ax.set_yticklabels([g.capitalize() for g in GRASP_ORDER])
        ax.set_xlabel("Hold orientation (relative to pull axis)")
        ax.set_title(model_label, fontsize=11, pad=8)

        # annotate cells
        for gi in range(len(GRASP_ORDER)):
            for oi in range(len(orients)):
                v = data[gi, oi]
                if not np.isnan(v):
                    ax.text(oi, gi, f"{v:.0f}", ha="center", va="center",
                            fontsize=9,
                            color="white" if v > RATCHET_MAX_N * 0.55 else "black")

    fig.suptitle("Mean slip force (N) by hold orientation × grasp type\n"
                 "(darker = stronger grip; each cell = mean of 4 paired trials)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ratchet-force figures from paired eval session(s)")
    parser.add_argument(
        "--session", nargs="+",
        default=["eval_results/paired_session_20260515_074256.json"],
        help="One or more paired_session_*.json paths (merged automatically)")
    parser.add_argument("--outdir", default="eval_results/figures")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Resolve paths relative to repo root if running from elsewhere
    script_dir = Path(__file__).resolve().parent.parent  # tele/
    resolved = []
    for p in args.session:
        pp = Path(p)
        if not pp.exists():
            pp = script_dir / p
        resolved.append(str(pp))

    pairs = load_sessions(resolved)
    if not pairs:
        print("ERROR: no ratchet data found in the specified session(s). "
              "Make sure you entered ratchet tooth counts during evaluation.")
        return

    by_grasp = group_by_grasp(pairs)
    print(f"Loaded {len(pairs)} pairs with ratchet data "
          f"from {len(resolved)} session file(s).")

    print_force_summary(by_grasp, pairs)

    fig1_force_boxplots      (by_grasp, f"{args.outdir}/fig_ratchet_1_boxplots.png")
    fig2_paired_scatter      (pairs,    f"{args.outdir}/fig_ratchet_2_scatter.png")
    fig3_force_by_orientation(pairs,    f"{args.outdir}/fig_ratchet_3_by_orientation.png")
    fig4_force_delta         (pairs,    f"{args.outdir}/fig_ratchet_4_per_pair_delta.png")
    fig5_force_cdf           (by_grasp, pairs, f"{args.outdir}/fig_ratchet_5_cdf.png")
    fig6_effect_sizes        (by_grasp, f"{args.outdir}/fig_ratchet_6_effect_sizes.png")
    fig7_per_grasp_scatter   (by_grasp, f"{args.outdir}/fig_ratchet_7_per_grasp_scatter.png")
    fig8_orientation_heatmap (by_grasp, f"{args.outdir}/fig_ratchet_8_orientation_heatmap.png")

    print(f"\nAll figures saved to {args.outdir}/")


if __name__ == "__main__":
    main()
