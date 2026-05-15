#!/usr/bin/env python3
"""
Deep observation analysis from the paired evaluation session.

Surfaces patterns reviewers will ask about and the paper's Discussion section needs:
  - Failure-mode breakdown (complete failure vs weak grip vs strong grip)
  - Variance reduction (precision-regularizer evidence)
  - Pegged-ratchet (right-censored) count
  - Order-effect check (did alternation work?)
  - Per-orientation × grasp deep dive
  - Strong-grip prevalence

Outputs:
  - eval_results/figures/fig_obs_1_failure_modes.png — stacked bar by grasp type
  - eval_results/figures/fig_obs_2_variance.png      — IQR comparison WITH vs WITHOUT
  - eval_results/figures/fig_obs_3_order_check.png   — order effect validation
  - eval_results/figures/fig_obs_4_orientation_dive.png — per-orientation breakdown
  - Terminal: a long structured printout suitable for pasting into OBSERVATIONS.md
"""

import json
import argparse
import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 10, "figure.dpi": 150,
    "savefig.dpi": 200, "savefig.bbox": "tight",
})

WITH_COLOR    = "#2563EB"
WITHOUT_COLOR = "#DC2626"

GRASP_ORDER  = ["crimp", "jug", "sloper", "pinch"]
GRASP_COLORS = {
    "crimp":  "#F59E0B",
    "jug":    "#3B82F6",
    "sloper": "#10B981",
    "pinch":  "#8B5CF6",
}

# Force thresholds for failure-mode bucketing (Newtons)
THRESH_COMPLETE_FAILURE = 5.3   # ≤ preload (0 teeth) = no displacement
THRESH_WEAK            = 13.5   # ≤ ~3 teeth = grip slipped early
THRESH_STRONG          = 23.0   # ≥ ~7 teeth = grip held well

WITH_LABEL    = "WITH"
WITHOUT_LABEL = "WITHOUT"


def load_pairs(path):
    with open(path) as f:
        d = json.load(f)
    out = []
    for pair in d["pairs"]:
        if pair.get("aborted"):
            continue
        wr = pair["pc_stats"].get("WITH_TAXONOMY",    {}).get("ratchet")
        nr = pair["pc_stats"].get("WITHOUT_TAXONOMY", {}).get("ratchet")
        if wr is None or nr is None:
            continue
        out.append({
            "grasp":        pair["grasp_type"],
            "orient":       pair.get("orientation_deg"),
            "order":        pair["order"],            # ["WITH_TAXONOMY", ...] or ["WITHOUT_TAXONOMY", ...]
            "with_force":   wr["force_N"],
            "no_force":     nr["force_N"],
            "with_teeth":   wr["teeth"],
            "no_teeth":     nr["teeth"],
        })
    return out


def bucket(force_N):
    if force_N <= THRESH_COMPLETE_FAILURE:
        return "complete failure"
    if force_N <= THRESH_WEAK:
        return "weak grip"
    if force_N <= THRESH_STRONG:
        return "moderate grip"
    return "strong grip"


BUCKETS       = ["complete failure", "weak grip", "moderate grip", "strong grip"]
BUCKET_COLORS = {
    "complete failure": "#7F1D1D",
    "weak grip":        "#EF4444",
    "moderate grip":    "#FBBF24",
    "strong grip":      "#16A34A",
}


# ── Observation 1: failure-mode breakdown ────────────────────────────────────

def fig_failure_modes(pairs, out_path):
    """Stacked bar of failure modes for WITH and WITHOUT, per grasp type."""
    fig, ax = plt.subplots(figsize=(10, 5.2))

    width  = 0.36
    xs     = np.arange(len(GRASP_ORDER))

    counts = {model: {g: Counter() for g in GRASP_ORDER}
              for model in [WITH_LABEL, WITHOUT_LABEL]}
    for p in pairs:
        counts[WITH_LABEL][p["grasp"]][bucket(p["with_force"])] += 1
        counts[WITHOUT_LABEL][p["grasp"]][bucket(p["no_force"])]  += 1

    for i, model in enumerate([WITH_LABEL, WITHOUT_LABEL]):
        offset = -width / 2 if model == WITH_LABEL else width / 2
        bottoms = np.zeros(len(GRASP_ORDER))
        for bk in BUCKETS:
            heights = np.array([counts[model][g][bk] for g in GRASP_ORDER])
            bars = ax.bar(xs + offset, heights, width=width * 0.9,
                          bottom=bottoms, color=BUCKET_COLORS[bk],
                          edgecolor="white", linewidth=0.5)
            for bar, h in zip(bars, heights):
                if h > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_y() + h / 2, str(int(h)),
                            ha="center", va="center",
                            fontsize=8.5, color="white", fontweight="bold")
            bottoms += heights

    # Add WITH/WITHOUT labels under each pair of bars
    for xi in xs:
        ax.text(xi - width / 2, -1.2, "W",  ha="center", va="top",
                fontsize=8, color=WITH_COLOR,    fontweight="bold")
        ax.text(xi + width / 2, -1.2, "W/o", ha="center", va="top",
                fontsize=8, color=WITHOUT_COLOR, fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels([g.capitalize() for g in GRASP_ORDER])
    ax.set_ylabel("Trials (out of 20 per model per grasp)")
    ax.set_ylim(-2.5, 24)
    ax.set_yticks(range(0, 25, 5))
    ax.set_title("Failure-mode breakdown — WITH vs WITHOUT taxonomy\n"
                 f"complete failure ≤{THRESH_COMPLETE_FAILURE:.1f}N · "
                 f"weak ≤{THRESH_WEAK:.1f}N · "
                 f"moderate ≤{THRESH_STRONG:.1f}N · strong > {THRESH_STRONG:.1f}N",
                 pad=10)

    handles = [plt.Rectangle((0, 0), 1, 1, color=BUCKET_COLORS[b]) for b in BUCKETS]
    ax.legend(handles, BUCKETS, loc="upper right", framealpha=0.95, fontsize=9)
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")
    return counts


# ── Observation 2: variance comparison ───────────────────────────────────────

def fig_variance(pairs, out_path):
    """IQR per model per grasp type — precision-regularizer evidence."""
    fig, ax = plt.subplots(figsize=(9, 5))

    xs    = np.arange(len(GRASP_ORDER))
    width = 0.36

    w_iqrs, n_iqrs = [], []
    for g in GRASP_ORDER:
        gp = [p for p in pairs if p["grasp"] == g]
        wf = [p["with_force"] for p in gp]
        nf = [p["no_force"]   for p in gp]
        w_iqrs.append(np.percentile(wf, 75) - np.percentile(wf, 25))
        n_iqrs.append(np.percentile(nf, 75) - np.percentile(nf, 25))

    bw = ax.bar(xs - width / 2, w_iqrs, width=width * 0.9,
                color=WITH_COLOR,    alpha=0.78, label="With taxonomy")
    bn = ax.bar(xs + width / 2, n_iqrs, width=width * 0.9,
                color=WITHOUT_COLOR,  alpha=0.78, label="Without taxonomy")

    for bars, vals in [(bw, w_iqrs), (bn, n_iqrs)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.3,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    # Delta annotation
    for xi, w, n in zip(xs, w_iqrs, n_iqrs):
        d_pct = (n - w) / n * 100 if n > 0 else 0
        ax.text(xi, max(w, n) + 2.0,
                ("↓" if w < n else "↑") + f"{abs(d_pct):.0f}%",
                ha="center", va="bottom", fontsize=9,
                color="#16A34A" if w < n else "#DC2626",
                fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels([g.capitalize() for g in GRASP_ORDER])
    ax.set_ylabel("Inter-quartile range of slip force (N)")
    ax.set_title("Precision-regularizer evidence: WITH-taxonomy has lower force variance\n"
                 "(↓X% = WITH is tighter than WITHOUT by X percent of WITHOUT's IQR)",
                 pad=8)
    ax.legend(loc="upper right", framealpha=0.95)
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(max(w_iqrs), max(n_iqrs)) * 1.30)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")
    return list(zip(GRASP_ORDER, w_iqrs, n_iqrs))


# ── Observation 3: order effect verification ─────────────────────────────────

def fig_order_check(pairs, out_path):
    """Did alternation control for order bias? Compare force when WITH went first vs second."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

    with_first  = [p for p in pairs if p["order"][0] == "WITH_TAXONOMY"]
    no_first    = [p for p in pairs if p["order"][0] == "WITHOUT_TAXONOMY"]

    # Per-model split: WITH model performance when it went first vs second
    for ax, model_key, model_label, color in [
        (axes[0], "with_force", "With-taxonomy model", WITH_COLOR),
        (axes[1], "no_force",   "Without-taxonomy model", WITHOUT_COLOR),
    ]:
        # When this model went FIRST in its pair
        if model_key == "with_force":
            first_vals  = [p["with_force"] for p in with_first]
            second_vals = [p["with_force"] for p in no_first]
        else:
            first_vals  = [p["no_force"]   for p in no_first]
            second_vals = [p["no_force"]   for p in with_first]

        positions = [0.85, 1.15]
        bp = ax.boxplot([first_vals, second_vals], positions=positions,
                        widths=0.22, patch_artist=True,
                        boxprops=dict(facecolor=color, alpha=0.6),
                        medianprops=dict(color="white", linewidth=2),
                        whiskerprops=dict(color=color),
                        capprops=dict(color=color))

        rng = np.random.default_rng(0)
        ax.scatter(rng.uniform(0.78, 0.92, len(first_vals)),  first_vals,
                   color=color, alpha=0.45, s=18, zorder=4)
        ax.scatter(rng.uniform(1.08, 1.22, len(second_vals)), second_vals,
                   color=color, alpha=0.45, s=18, zorder=4)

        # Mann–Whitney U test (between subjects, not paired since different pair sets)
        try:
            from scipy.stats import mannwhitneyu
            _, pv = mannwhitneyu(first_vals, second_vals, alternative="two-sided")
            star = "*" if pv < 0.05 else "ns"
        except Exception:
            pv = float("nan"); star = "?"

        ax.set_xticks(positions)
        ax.set_xticklabels([f"Went first\nn={len(first_vals)}",
                            f"Went second\nn={len(second_vals)}"])
        ax.set_title(f"{model_label}\nMann–Whitney p={pv:.3f} ({star})", fontsize=10)
        ax.set_xlim(0.65, 1.35)
        ax.yaxis.grid(True, alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        if ax == axes[0]:
            ax.set_ylabel("Slip force (N)")

    fig.suptitle("Order-effect check: did the model going first matter?\n"
                 "(if alternation worked, the two boxes per model should look similar)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ── Observation 4: per-orientation × grasp deep dive ─────────────────────────

def fig_orientation_dive(pairs, out_path):
    """Show median force and #pairs at each (grasp, orientation) cell, per model."""
    orients = [-45.0, -22.5, 0.0, 22.5, 45.0]
    fig, axes = plt.subplots(len(GRASP_ORDER), 1,
                              figsize=(10, 2.4 * len(GRASP_ORDER)),
                              sharex=True)

    for ax, g in zip(axes, GRASP_ORDER):
        gp     = [p for p in pairs if p["grasp"] == g]
        xs     = np.arange(len(orients))
        w_meds, n_meds, w_lo, w_hi, n_lo, n_hi = [], [], [], [], [], []

        for ori in orients:
            cell = [p for p in gp if p["orient"] == ori]
            wf   = [p["with_force"] for p in cell]
            nf   = [p["no_force"]   for p in cell]
            w_meds.append(np.median(wf) if wf else np.nan)
            n_meds.append(np.median(nf) if nf else np.nan)
            w_lo.append(np.min(wf)   if wf else np.nan)
            w_hi.append(np.max(wf)   if wf else np.nan)
            n_lo.append(np.min(nf)   if nf else np.nan)
            n_hi.append(np.max(nf)   if nf else np.nan)

        w_meds = np.array(w_meds); n_meds = np.array(n_meds)
        # range whiskers
        ax.vlines(xs - 0.12, w_lo, w_hi, color=WITH_COLOR,    alpha=0.4, linewidth=4)
        ax.vlines(xs + 0.12, n_lo, n_hi, color=WITHOUT_COLOR,  alpha=0.4, linewidth=4)
        ax.scatter(xs - 0.12, w_meds, color=WITH_COLOR,    s=70, zorder=5,
                   edgecolors="white", linewidth=1, label="With" if g == GRASP_ORDER[0] else "")
        ax.scatter(xs + 0.12, n_meds, color=WITHOUT_COLOR,  s=70, zorder=5,
                   edgecolors="white", linewidth=1, label="Without" if g == GRASP_ORDER[0] else "")

        ax.set_ylabel(g.capitalize(), fontweight="bold")
        ax.set_ylim(0, 36)
        ax.yaxis.grid(True, alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{int(o)}°" if o == int(o) else f"{o}°" for o in orients])

    axes[0].legend(loc="upper right", fontsize=9)
    axes[-1].set_xlabel("Hold orientation relative to pull axis")
    fig.suptitle("Per-orientation × grasp-type slip force\n"
                 "dot = median, whisker = min–max across 4 paired trials per cell",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ── Terminal observations report ─────────────────────────────────────────────

def print_observations(pairs):
    n = len(pairs)
    print("\n" + "═" * 78)
    print(f"  OBSERVATION REPORT  —  {n} pairs / {n*2} rollouts")
    print("═" * 78)

    # ── §1 Failure-mode breakdown
    print("\n§1. Failure-mode breakdown\n" + "-" * 50)
    overall = {model: Counter() for model in [WITH_LABEL, WITHOUT_LABEL]}
    by_grasp = {g: {model: Counter() for model in [WITH_LABEL, WITHOUT_LABEL]}
                for g in GRASP_ORDER}
    for p in pairs:
        wb = bucket(p["with_force"])
        nb = bucket(p["no_force"])
        overall[WITH_LABEL][wb] += 1
        overall[WITHOUT_LABEL][nb] += 1
        by_grasp[p["grasp"]][WITH_LABEL][wb] += 1
        by_grasp[p["grasp"]][WITHOUT_LABEL][nb] += 1

    print(f"  {'Bucket':22s} {'WITH':>10s} {'WITHOUT':>10s}    Δ")
    for bk in BUCKETS:
        w = overall[WITH_LABEL][bk]
        wo = overall[WITHOUT_LABEL][bk]
        arrow = "↓" if (bk in ("complete failure", "weak grip") and w < wo) else \
                "↑" if (bk in ("moderate grip", "strong grip") and w > wo) else " "
        print(f"  {bk:22s} {w:>3d}/{n} ({w/n*100:>4.0f}%) "
              f"{wo:>3d}/{n} ({wo/n*100:>4.0f}%)  {arrow}{abs(w-wo)/n*100:>4.0f}pp")

    # ── §2 Variance reduction
    print("\n§2. Variance reduction (precision-regularizer evidence)\n" + "-" * 50)
    print(f"  {'Grasp':10s} {'WITH IQR':>10s} {'W/O IQR':>10s}  {'Δ':>8s}")
    overall_w, overall_n = [], []
    for g in GRASP_ORDER:
        gp = [p for p in pairs if p["grasp"] == g]
        wf = [p["with_force"] for p in gp]
        nf = [p["no_force"]   for p in gp]
        overall_w.extend(wf); overall_n.extend(nf)
        w_iqr = np.percentile(wf, 75) - np.percentile(wf, 25)
        n_iqr = np.percentile(nf, 75) - np.percentile(nf, 25)
        pct = (n_iqr - w_iqr) / n_iqr * 100 if n_iqr > 0 else 0
        print(f"  {g:10s} {w_iqr:>7.2f} N  {n_iqr:>7.2f} N  {pct:>+6.0f}%")
    w_iqr = np.percentile(overall_w, 75) - np.percentile(overall_w, 25)
    n_iqr = np.percentile(overall_n, 75) - np.percentile(overall_n, 25)
    pct = (n_iqr - w_iqr) / n_iqr * 100 if n_iqr > 0 else 0
    print(f"  {'─'*10} {'─'*10} {'─'*10}  {'─'*8}")
    print(f"  {'OVERALL':10s} {w_iqr:>7.2f} N  {n_iqr:>7.2f} N  {pct:>+6.0f}%")

    # ── §3 Pegged-ratchet (right-censored) count
    print("\n§3. Pegged-ratchet (right-censored) trials\n" + "-" * 50)
    pw = sum(1 for p in pairs if p["with_teeth"] >= 11)
    pn = sum(1 for p in pairs if p["no_teeth"]   >= 11)
    print(f"  WITH    : {pw}/{n} trials pegged at 11 teeth (≥33.9 N)")
    print(f"  WITHOUT : {pn}/{n} trials pegged at 11 teeth (≥33.9 N)")
    if pw + pn > 0:
        print("  → Reported median forces are conservative for these trials")

    # ── §4 Order-effect check
    print("\n§4. Order-effect check\n" + "-" * 50)
    with_first = [p for p in pairs if p["order"][0] == "WITH_TAXONOMY"]
    no_first   = [p for p in pairs if p["order"][0] == "WITHOUT_TAXONOMY"]
    print(f"  Pairs where WITH went first: {len(with_first)}")
    print(f"  Pairs where W/O  went first: {len(no_first)}")
    try:
        from scipy.stats import mannwhitneyu
        for tag, key in [("WITH model", "with_force"),
                          ("WITHOUT model", "no_force")]:
            if key == "with_force":
                a = [p[key] for p in with_first]
                b = [p[key] for p in no_first]
            else:
                a = [p[key] for p in no_first]
                b = [p[key] for p in with_first]
            _, pv = mannwhitneyu(a, b, alternative="two-sided")
            verdict = ("no order bias" if pv >= 0.05 else
                       "ORDER BIAS DETECTED (p<0.05)")
            print(f"  {tag}: when-first med={np.median(a):.1f}N, "
                  f"when-second med={np.median(b):.1f}N → p={pv:.3f} ({verdict})")
    except ImportError:
        print("  (scipy.stats.mannwhitneyu unavailable — skipping test)")

    # ── §5 Hardest orientations per grasp
    print("\n§5. Hardest orientation per grasp (lowest WITH median)\n" + "-" * 50)
    orients = [-45.0, -22.5, 0.0, 22.5, 45.0]
    for g in GRASP_ORDER:
        cells = []
        for ori in orients:
            gp = [p["with_force"] for p in pairs
                  if p["grasp"] == g and p["orient"] == ori]
            if gp:
                cells.append((ori, np.median(gp), len(gp)))
        cells.sort(key=lambda x: x[1])  # ascending — hardest first
        hardest = cells[0]
        easiest = cells[-1]
        print(f"  {g:10s} hardest: {hardest[0]:>+6.1f}° (med {hardest[1]:>5.1f} N) | "
              f"easiest: {easiest[0]:>+6.1f}° (med {easiest[1]:>5.1f} N) | "
              f"spread {easiest[1] - hardest[1]:>5.1f} N")

    # ── §6 Wins / losses / ties
    print("\n§6. Pair-wise winner counts (force-based)\n" + "-" * 50)
    print(f"  {'Grasp':10s} {'WITH wins':>10s} {'tie':>5s} {'W/O wins':>10s}")
    for g in GRASP_ORDER:
        gp = [p for p in pairs if p["grasp"] == g]
        ww = sum(1 for p in gp if p["with_force"] >  p["no_force"])
        wo = sum(1 for p in gp if p["with_force"] <  p["no_force"])
        tie= sum(1 for p in gp if p["with_force"] == p["no_force"])
        print(f"  {g:10s} {ww:>10d} {tie:>5d} {wo:>10d}")
    ww = sum(1 for p in pairs if p["with_force"] >  p["no_force"])
    wo = sum(1 for p in pairs if p["with_force"] <  p["no_force"])
    tie= sum(1 for p in pairs if p["with_force"] == p["no_force"])
    print(f"  {'─'*10} {'─'*10} {'─'*5} {'─'*10}")
    print(f"  {'OVERALL':10s} {ww:>10d} {tie:>5d} {wo:>10d}")

    print("\n" + "═" * 78)
    print("  Paste the §1–§6 numbers above into OBSERVATIONS.md as needed.")
    print("═" * 78 + "\n")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--session",
        default="eval_results/paired_session_20260515_074256.json")
    parser.add_argument("--outdir", default="eval_results/figures")
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    script_dir = Path(__file__).resolve().parent.parent
    sp = Path(args.session)
    if not sp.exists():
        sp = script_dir / args.session

    pairs = load_pairs(str(sp))
    print(f"Loaded {len(pairs)} pairs from {sp}")

    fig_failure_modes      (pairs, f"{args.outdir}/fig_obs_1_failure_modes.png")
    fig_variance           (pairs, f"{args.outdir}/fig_obs_2_variance.png")
    fig_order_check        (pairs, f"{args.outdir}/fig_obs_3_order_check.png")
    fig_orientation_dive   (pairs, f"{args.outdir}/fig_obs_4_orientation_dive.png")

    print_observations(pairs)


if __name__ == "__main__":
    main()
