"""
Action-distribution comparison: same input, two models.

For a representative set of training inputs (8 episodes per grasp type),
sample K action rollouts from each diffusion policy, then visualise:

  Panel A — All predicted actions in joint-PCA space, coloured by grasp type,
            marker = WITH (filled circle) vs WITHOUT (open triangle).
  Panel B — Within-input action spread (how multimodal are the K samples
            from a single input?). If the WITHOUT-model suffers mode
            collapse / mode averaging, its samples for a single input will
            be either tightly grouped (averaging) OR split across modes
            (ambiguity). WITH-tax samples should be tight around the
            correct mode.
  Panel C — Pairwise L2 distance between WITH-model vs WITHOUT-model action
            predictions on the SAME input, per grasp type. If the models
            disagree most on the types where evaluation also disagrees
            (crimp, sloper, pinch), that's direct mechanistic evidence
            that the conditioning signal is changing the decoder's output.

Run:
    source ~/franka/bin/activate
    python3 eval_results/generate_action_dist_viz.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import zarr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA

# Let us import PointCloudDiffusionPolicy directly.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data_collection"))
from train import PointCloudDiffusionPolicy  # noqa: E402

ZARR_PATH  = Path("/mnt/ssd/rumi_tele_datasets/climbing_holds.zarr")
WITH_CKPT  = Path("/home/rumi/Desktop/tele/checkpoints/pc_with_taxonomy/best.pt")
NO_CKPT    = Path("/home/rumi/Desktop/tele/checkpoints/pc_no_taxonomy/best.pt")
OUT_DIR    = Path("/home/rumi/Desktop/tele/eval_results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_PER_GRASP  = 8              # how many training episodes per grasp type to probe
K_SAMPLES    = 8              # how many action rollouts per input
INFER_STEPS  = 10             # DDIM steps (matches eval)

GRASP_ID_TO_NAME = {0: "Crimp", 1: "Sloper", 2: "Pinch", 3: "Jug"}
GRASP_COLORS     = {0: "#F59E0B", 1: "#10B981", 2: "#8B5CF6", 3: "#3B82F6"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 11, "figure.dpi": 150, "savefig.dpi": 200,
})


# ── load data ────────────────────────────────────────────────────────────────

def load_sample_inputs():
    """Select N_PER_GRASP training episodes per grasp type, return their
    first-timestep obs_state (normalised), PC, and grasp_type_id."""
    root = zarr.open(str(ZARR_PATH), mode="r")
    ep_ends   = root["meta/episode_ends"][:]
    grasp_ids = root["meta/grasp_type_id"][:]
    states    = root["data/state"][:]
    pcs       = root["data/point_cloud"][:]
    starts    = np.concatenate([[0], ep_ends[:-1]])

    with open(WITH_CKPT.parent / "norm_stats.json") as f:
        ns = json.load(f)
    s_min = np.array(ns["state_min"],   dtype=np.float32)
    s_rng = np.array(ns["state_range"], dtype=np.float32)
    a_min = np.array(ns["action_min"],  dtype=np.float32)
    a_rng = np.array(ns["action_range"], dtype=np.float32)

    rng = np.random.default_rng(42)
    sel_state, sel_pc, sel_gid, sel_epidx = [], [], [], []
    for gid in sorted(np.unique(grasp_ids)):
        ep_idx = np.where(grasp_ids == gid)[0]
        chosen = rng.choice(ep_idx, size=N_PER_GRASP, replace=False)
        for ei in chosen:
            s = states[starts[ei]]
            s_norm = 2.0 * (s.astype(np.float32) - s_min) / s_rng - 1.0
            sel_state.append(np.stack([s_norm, s_norm], axis=0))   # (2, 23)
            sel_pc.append(pcs[starts[ei]].astype(np.float32))       # (1024, 3)
            sel_gid.append(int(gid))
            sel_epidx.append(int(ei))
    return {
        "obs_state":     np.stack(sel_state),                 # (N, 2, 23)
        "obs_pc":        np.stack(sel_pc),                    # (N, 1024, 3)
        "grasp_type_id": np.array(sel_gid, dtype=np.int64),
        "episode_index": np.array(sel_epidx, dtype=np.int64),
        "action_min":    a_min,
        "action_range":  a_rng,
    }


# ── load models ──────────────────────────────────────────────────────────────

def load_policy(ckpt_path, use_grasp):
    ck  = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = ck["config"]
    pol = PointCloudDiffusionPolicy(
        state_dim=cfg["state_dim"],
        action_dim=cfg["action_dim"],
        obs_horizon=cfg["obs_horizon"],
        pred_horizon=cfg["pred_horizon"],
        num_diffusion_steps=cfg["diffusion_steps"],
        down_dims=tuple(cfg["down_dims"]),
        n_grasp_types=cfg["n_grasp_types"],
        use_grasp_conditioning=use_grasp,
    ).to(DEVICE)
    pol.load_state_dict(ck["model_state_dict"])
    pol.eval()
    return pol


# ── sample actions ───────────────────────────────────────────────────────────

@torch.no_grad()
def sample_actions(policy, inputs, k_samples=K_SAMPLES):
    """Return (N, K, pred_horizon, action_dim) denormalised action predictions."""
    N = inputs["obs_state"].shape[0]
    obs_state = torch.from_numpy(inputs["obs_state"]).to(DEVICE)
    obs_pc    = torch.from_numpy(inputs["obs_pc"]).to(DEVICE)
    gid       = torch.from_numpy(inputs["grasp_type_id"]).to(DEVICE)

    samples = []
    for k in range(k_samples):
        torch.manual_seed(1000 + k)   # same noise sequence across models ⇒ fair
        a_norm = policy.predict_action(obs_state, obs_pc, gid,
                                       num_inference_steps=INFER_STEPS)
        a_norm = a_norm.cpu().numpy()
        # denormalise to raw joint angles (same 23 dims: arm7 + hand16)
        a = (a_norm + 1.0) / 2.0 * inputs["action_range"] + inputs["action_min"]
        samples.append(a)
    return np.stack(samples, axis=1)          # (N, K, T, 23)


# ── per-grasp-type spread detail ─────────────────────────────────────────────

def _cov_ellipse(ax, pts, color, n_std=1.5, lw=2.5, ls="-", alpha_face=0.10,
                 zorder=2):
    """Draw an n-sigma covariance ellipse around `pts` (M, 2)."""
    if len(pts) < 3:
        return
    mu  = pts.mean(axis=0)
    cov = np.cov(pts, rowvar=False)
    eigval, eigvec = np.linalg.eigh(cov)
    order = np.argsort(eigval)[::-1]
    eigval, eigvec = eigval[order], eigvec[:, order]
    angle = np.degrees(np.arctan2(eigvec[1, 0], eigvec[0, 0]))
    w, h  = 2 * n_std * np.sqrt(np.maximum(eigval, 1e-12))
    e = Ellipse(xy=mu, width=w, height=h, angle=angle,
                edgecolor=color, facecolor=color,
                lw=lw, ls=ls, alpha=alpha_face, zorder=zorder)
    e.set_clip_on(False)
    ax.add_patch(e)
    # extra outline at full alpha for contrast
    e_outline = Ellipse(xy=mu, width=w, height=h, angle=angle,
                        edgecolor=color, facecolor="none",
                        lw=lw, ls=ls, zorder=zorder + 1)
    e_outline.set_clip_on(False)
    ax.add_patch(e_outline)


def fig_spread_detail(acts_w, acts_n, gids, within_w, within_n, out_path):
    """Per-grasp-type zoomed view of within-input sampling spread.

    Rollouts are centred per input (subtract that input's mean), so each cloud
    shows *only* the sampling variability at fixed scene — not the between-
    input variability. A local PCA is fit per panel so axes fill the panel and
    the spread difference is actually visible.
    """
    WITH_COLOR  = "#1E3A8A"
    WITHOUT_COLOR = "#B91C1C"

    fig = plt.figure(figsize=(13.5, 12.5))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.48, wspace=0.32)

    grasps = sorted(np.unique(gids))
    T, D = acts_w.shape[2], acts_w.shape[3]

    for idx, gid in enumerate(grasps):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        mask = gids == gid

        aw = acts_w[mask]            # (n_inp, K, T, 23)
        an = acts_n[mask]
        # centre each input's K rollouts on that input's own mean
        aw_c = aw - aw.mean(axis=1, keepdims=True)
        an_c = an - an.mean(axis=1, keepdims=True)
        fw = aw_c.reshape(-1, T * D)
        fn = an_c.reshape(-1, T * D)

        pca = PCA(n_components=2, random_state=42)
        pca.fit(np.concatenate([fw, fn], axis=0))
        pts_w = pca.transform(fw)
        pts_n = pca.transform(fn)

        # centre axes on (0,0) so the "WITH cloud tight, WITHOUT cloud wider"
        # story reads left-to-right / visually
        lim = 1.2 * max(np.linalg.norm(pts_w, axis=1).max(),
                        np.linalg.norm(pts_n, axis=1).max())
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.axhline(0, color="#999999", lw=0.6, zorder=0)
        ax.axvline(0, color="#999999", lw=0.6, zorder=0)

        ax.scatter(pts_w[:, 0], pts_w[:, 1],
                   s=55, marker="o", facecolor=WITH_COLOR,
                   edgecolor="white", linewidths=0.6, alpha=0.85,
                   label="WITH taxonomy (K=8 rollouts per input)",
                   zorder=5)
        ax.scatter(pts_n[:, 0], pts_n[:, 1],
                   s=95, marker="X", color=WITHOUT_COLOR,
                   linewidths=1.5, alpha=0.85,
                   label="WITHOUT taxonomy",
                   zorder=5)

        _cov_ellipse(ax, pts_w, WITH_COLOR,    n_std=1.5, ls="-")
        _cov_ellipse(ax, pts_n, WITHOUT_COLOR, n_std=1.5, ls="--")

        w_mean = float(within_w[mask].mean())
        n_mean = float(within_n[mask].mean())
        pct    = (n_mean - w_mean) / max(w_mean, 1e-9) * 100.0

        var_pc1 = pca.explained_variance_ratio_[0] * 100
        var_pc2 = pca.explained_variance_ratio_[1] * 100

        delta_color = "#B91C1C" if pct > 5 else "#555555"
        wider = "wider" if pct > 0 else "tighter"
        ax.text(
            0.5, 1.18,
            f"{GRASP_ID_TO_NAME[gid]}",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=13, color=GRASP_COLORS[gid], fontweight="bold",
        )
        ax.text(
            0.5, 1.10,
            f"within-input L2:   WITH = {w_mean:.3f}    "
            f"WITHOUT = {n_mean:.3f}",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=9.5, color="#333333",
        )
        ax.text(
            0.5, 1.02,
            f"WITHOUT is {pct:+.0f}% {wider} than WITH at fixed scene",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=10, color=delta_color, fontweight="bold",
        )
        ax.set_xlabel(f"local PC1 ({var_pc1:.1f}%)", fontsize=9)
        ax.set_ylabel(f"local PC2 ({var_pc2:.1f}%)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_facecolor("#F8F9FA")
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.set_axisbelow(True)
        ax.set_aspect("equal", adjustable="box")

        if idx == 0:
            ax.legend(loc="upper left", fontsize=8.5, framealpha=0.95)

    fig.suptitle(
        "Within-input sampling spread (same scene, 8 rollouts per model)\n"
        "Rollouts are mean-centred per input so each cloud shows ONLY "
        "sampling jitter.  Ellipse = 1.5σ.  Wider red cloud = less precise "
        "at fixed scene.",
        fontsize=12.5, y=0.985,
    )
    fig.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ── figure ───────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")
    inputs = load_sample_inputs()
    N = inputs["obs_state"].shape[0]
    print(f"Selected {N} inputs ({N_PER_GRASP} per grasp type, K={K_SAMPLES} rollouts each)")

    print("Loading WITH-taxonomy policy...")
    pol_w = load_policy(WITH_CKPT, use_grasp=True)
    print("Loading WITHOUT-taxonomy policy...")
    pol_n = load_policy(NO_CKPT,   use_grasp=False)

    print("Sampling actions from WITH-taxonomy policy...")
    acts_w = sample_actions(pol_w, inputs)
    print("Sampling actions from WITHOUT-taxonomy policy...")
    acts_n = sample_actions(pol_n, inputs)

    T, D = acts_w.shape[2], acts_w.shape[3]
    flat_w = acts_w.reshape(N * K_SAMPLES, T * D)
    flat_n = acts_n.reshape(N * K_SAMPLES, T * D)

    # Fit PCA on all samples together so both models share the same projection
    pca = PCA(n_components=2, random_state=42)
    pca.fit(np.concatenate([flat_w, flat_n], axis=0))
    xy_w = pca.transform(flat_w).reshape(N, K_SAMPLES, 2)
    xy_n = pca.transform(flat_n).reshape(N, K_SAMPLES, 2)

    gids = inputs["grasp_type_id"]
    var_exp = pca.explained_variance_ratio_ * 100

    # ── within-input spread + between-model L2 ────────────────────────────────
    within_w = np.zeros(N)
    within_n = np.zeros(N)
    between  = np.zeros(N)
    for i in range(N):
        mu_w = acts_w[i].mean(0)
        mu_n = acts_n[i].mean(0)
        within_w[i] = np.linalg.norm(acts_w[i] - mu_w, axis=(1, 2)).mean()
        within_n[i] = np.linalg.norm(acts_n[i] - mu_n, axis=(1, 2)).mean()
        between[i]  = np.linalg.norm(mu_w - mu_n)

    # ── plot ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 5.8))
    gs  = GridSpec(1, 3, figure=fig, width_ratios=[1.35, 1, 1], wspace=0.30)

    # Panel A — joint-space PCA of predicted actions
    axA = fig.add_subplot(gs[0, 0])
    for gid in sorted(np.unique(gids)):
        mask = gids == gid
        pts_w = xy_w[mask].reshape(-1, 2)
        pts_n = xy_n[mask].reshape(-1, 2)
        axA.scatter(pts_w[:, 0], pts_w[:, 1],
                    c=GRASP_COLORS[gid], marker="o", s=40,
                    alpha=0.85, edgecolors="white", linewidths=0.5,
                    label=f"{GRASP_ID_TO_NAME[gid]} · With" if gid == 0 else None)
        axA.scatter(pts_n[:, 0], pts_n[:, 1],
                    facecolors="none", edgecolors=GRASP_COLORS[gid],
                    marker="^", s=55, linewidths=1.5,
                    label=f"{GRASP_ID_TO_NAME[gid]} · W/o" if gid == 0 else None)

    axA.set_title(f"A. Predicted actions (PCA)\n"
                  f"PC1={var_exp[0]:.1f}%  PC2={var_exp[1]:.1f}%  · "
                  f"filled=WITH, outline=WITHOUT", fontsize=11)
    axA.set_xticks([]); axA.set_yticks([])
    axA.set_facecolor("#F8F9FA")

    # custom legend for panel A
    legend_handles = []
    for gid in sorted(GRASP_ID_TO_NAME):
        legend_handles.append(mpatches.Patch(color=GRASP_COLORS[gid],
                                             label=GRASP_ID_TO_NAME[gid]))
    axA.legend(handles=legend_handles, loc="upper right", fontsize=9,
               framealpha=0.95, title="Grasp type (input)")

    # Panel B — within-input spread per grasp type
    axB = fig.add_subplot(gs[0, 1])
    bar_x   = np.arange(4)
    width   = 0.38
    within_w_by_gid = [within_w[gids == g].mean() for g in sorted(np.unique(gids))]
    within_n_by_gid = [within_n[gids == g].mean() for g in sorted(np.unique(gids))]
    axB.bar(bar_x - width/2, within_w_by_gid, width, color="#2563EB",
            label="With taxonomy")
    axB.bar(bar_x + width/2, within_n_by_gid, width, color="#DC2626",
            label="Without taxonomy")
    axB.set_xticks(bar_x)
    axB.set_xticklabels([GRASP_ID_TO_NAME[g] for g in sorted(np.unique(gids))],
                        fontsize=9)
    axB.set_ylabel("Mean within-input\naction spread  (L2)")
    axB.set_title("B. Sampling variance per input\n"
                  "(higher = more multimodal output)", fontsize=11)
    axB.legend(loc="upper left", fontsize=9)
    axB.yaxis.grid(True, alpha=0.3, linestyle="--"); axB.set_axisbelow(True)

    # Panel C — between-model action disagreement per grasp type
    axC = fig.add_subplot(gs[0, 2])
    bet_by_gid = [between[gids == g].mean()           for g in sorted(np.unique(gids))]
    bet_sem    = [between[gids == g].std() / np.sqrt((gids == g).sum())
                  for g in sorted(np.unique(gids))]
    colors = [GRASP_COLORS[g] for g in sorted(np.unique(gids))]
    axC.bar(bar_x, bet_by_gid, width=0.55, color=colors,
            yerr=bet_sem, capsize=4, error_kw={"linewidth": 1.2})
    axC.set_xticks(bar_x)
    axC.set_xticklabels([GRASP_ID_TO_NAME[g] for g in sorted(np.unique(gids))],
                        fontsize=9)
    axC.set_ylabel("L2 between predicted actions\n(With vs Without, same input)")
    axC.set_title("C. How differently do the two\nmodels act given the same scene?",
                  fontsize=11)
    axC.yaxis.grid(True, alpha=0.3, linestyle="--"); axC.set_axisbelow(True)

    fig.suptitle("Action-distribution analysis — does the grasp label change "
                 "what the diffusion decoder actually outputs?",
                 fontsize=12.5, y=1.02)

    out = OUT_DIR / "fig_action_distribution.png"
    fig.savefig(str(out), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

    # ── standalone spread-detail figure (zoomed per grasp type) ───────────────
    print("Generating per-grasp-type sampling-spread detail figure...")
    fig_spread_detail(
        acts_w, acts_n, gids, within_w, within_n,
        OUT_DIR / "fig_action_spread_detail.png",
    )

    print("\nSummary table:")
    print(f"{'Grasp':8s} {'within WITH':>12s} {'within WITHOUT':>15s} {'between (W↔W/o)':>18s}")
    for g in sorted(np.unique(gids)):
        name = GRASP_ID_TO_NAME[g]
        ww = within_w[gids == g].mean()
        wn = within_n[gids == g].mean()
        bw = between [gids == g].mean()
        print(f"{name:8s} {ww:12.3f} {wn:15.3f} {bw:18.3f}")


if __name__ == "__main__":
    main()
