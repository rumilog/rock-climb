#!/usr/bin/env python3
"""
Parse training logs and generate training-curve figures for both models.

Usage:
    python3 eval_results/generate_training_curves.py
    python3 eval_results/generate_training_curves.py --outdir eval_results/figures
"""
import re
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 10, "figure.dpi": 150,
    "savefig.dpi": 200, "savefig.bbox": "tight",
})

WITH_COLOR    = "#2563EB"
WITHOUT_COLOR = "#DC2626"

TELE_ROOT = Path(__file__).resolve().parent.parent

WITH_LOG = TELE_ROOT / "checkpoints" / "pc_with_taxonomy_rig_train.log"
NO_LOG   = TELE_ROOT / "checkpoints" / "pc_no_taxonomy_rig" / "train.log"


def parse_log(path):
    """Return (epochs, losses, lrs) arrays from a training log."""
    epochs, losses, lrs = [], [], []
    pat = re.compile(
        r"Epoch\s+(\d+)/\d+\s+loss=([\d.]+)\s+lr=([\d.e+-]+)")
    with open(path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                epochs.append(int(m.group(1)))
                losses.append(float(m.group(2)))
                lrs.append(float(m.group(3)))
    return np.array(epochs), np.array(losses), np.array(lrs)


def smooth(x, w=50):
    """Simple moving average."""
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode='valid')


def fig_training_curves(outdir):
    e_w, l_w, lr_w = parse_log(WITH_LOG)
    e_n, l_n, lr_n = parse_log(NO_LOG)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    # ── Left: full training loss ──────────────────────────────────────────────
    ax = axes[0]
    ax.plot(e_w, l_w, color=WITH_COLOR,    alpha=0.18, linewidth=0.5)
    ax.plot(e_n, l_n, color=WITHOUT_COLOR,  alpha=0.18, linewidth=0.5)

    w = 50
    ax.plot(e_w[w-1:], smooth(l_w, w), color=WITH_COLOR,    linewidth=2.0,
            label=f"With taxonomy  (best={min(l_w):.5f})")
    ax.plot(e_n[w-1:], smooth(l_n, w), color=WITHOUT_COLOR,  linewidth=2.0,
            label=f"Without taxonomy (best={min(l_n):.5f})")

    # warmup zone
    warmup_ep = 500 * (max(e_w) / (192 * max(e_w) / max(e_w)))
    warmup_ep = min(500 / 192 * max(e_w) / max(e_w), max(e_w) * 0.05)
    ax.axvspan(0, 26, color="#F3F4F6", alpha=0.6, zorder=0, label="LR warmup (~500 steps)")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss (DDPM MSE)")
    ax.set_title("Training curves — both models converge smoothly\n"
                 "(faint = raw per-epoch, solid = 50-epoch moving average)")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.set_xlim(0, max(max(e_w), max(e_n)))
    ax.set_ylim(0, None)
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # ── Right: log-scale last 80% ─────────────────────────────────────────────
    ax2 = axes[1]
    cutoff = int(len(e_w) * 0.20)
    ax2.semilogy(e_w[cutoff:], l_w[cutoff:], color=WITH_COLOR,    alpha=0.18, linewidth=0.5)
    ax2.semilogy(e_n[cutoff:], l_n[cutoff:], color=WITHOUT_COLOR,  alpha=0.18, linewidth=0.5)
    ax2.semilogy(e_w[cutoff + w - 1:], smooth(l_w[cutoff:], w),
                 color=WITH_COLOR,    linewidth=2.0, label="With taxonomy")
    ax2.semilogy(e_n[cutoff + w - 1:], smooth(l_n[cutoff:], w),
                 color=WITHOUT_COLOR,  linewidth=2.0, label="Without taxonomy")

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Training loss (log scale)")
    ax2.set_title("Late-stage convergence (epoch 600–3000, log scale)\n"
                  "Both models reach similar final loss (~0.0017–0.0018)")
    ax2.legend(loc="upper right", framealpha=0.95)
    ax2.yaxis.grid(True, alpha=0.3, linestyle="--", which="both")
    ax2.set_axisbelow(True)

    fig.tight_layout()
    out = f"{outdir}/fig_training_curves.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")

    # Also print summary
    print(f"\nTraining summary:")
    print(f"  With taxonomy:    {len(e_w)} epochs, best loss = {min(l_w):.6f}")
    print(f"  Without taxonomy: {len(e_n)} epochs, best loss = {min(l_n):.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="eval_results/figures")
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    fig_training_curves(args.outdir)


if __name__ == "__main__":
    main()
