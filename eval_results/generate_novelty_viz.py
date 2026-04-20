"""
Novelty-showcasing figures.

FIGURE N1 — Benchmark overview
    2 x 4 grid: one column per grasp type (crimp, jug, sloper, pinch)
    Row 1 : side-view scatter of one representative training PC (hold geometry)
    Row 2 : mean final-step LEAP hand joint configuration across all demos
            of that type (the target hand pose the demonstrator closes to).
    Purpose: establish the benchmark in a single figure — four geometrically
    distinct holds → four distinct target hand poses.

FIGURE N2 — Mode-commitment fingerprint
    Panel A : 4 small ribbon plots (one per grasp type). Each shows a
              representative hand-joint trajectory over the 16-step
              prediction horizon for demo-GT, WITH-model, WITHOUT-model.
              Band = ±1 std across demos / rollouts.
    Panel B : four-way radar plot of the final-timestep hand pose per model.
              Visual "does the Without model commit to four different hand
              configurations or collapse to one?"

Run:
    source ~/franka/bin/activate
    source ~/frankapy/catkin_ws/devel/setup.bash
    python3 eval_results/generate_novelty_viz.py
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data_collection"))
from train import PointCloudDiffusionPolicy  # noqa: E402

ZARR_PATH  = Path("/mnt/ssd/rumi_tele_datasets/climbing_holds.zarr")
WITH_CKPT  = Path("/home/rumi/Desktop/tele/checkpoints/pc_with_taxonomy/best.pt")
NO_CKPT    = Path("/home/rumi/Desktop/tele/checkpoints/pc_no_taxonomy/best.pt")
OUT_DIR    = Path("/home/rumi/Desktop/tele/eval_results/figures")

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INFER_STEPS  = 10
N_PER_GRASP  = 12
K_SAMPLES    = 6

GRASP_ID_TO_NAME = {0: "Crimp", 1: "Sloper", 2: "Pinch", 3: "Jug"}
GRASP_ORDER     = [0, 3, 1, 2]            # display order: crimp, jug, sloper, pinch
GRASP_COLORS    = {0: "#F59E0B", 1: "#10B981", 2: "#8B5CF6", 3: "#3B82F6"}

# LEAP hand joint naming (16 dof, order matches state/action indices 7..22)
HAND_JOINT_NAMES = [
    "Idx MCP_abd", "Idx MCP_flex", "Idx PIP", "Idx DIP",
    "Mid MCP_abd", "Mid MCP_flex", "Mid PIP", "Mid DIP",
    "Pky MCP_abd", "Pky MCP_flex", "Pky PIP", "Pky DIP",
    "Thumb CMC",  "Thumb MCP",   "Thumb IP", "Thumb tip",
]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 11, "figure.dpi": 150, "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


# ═══════════════════════════════════════════════════════════════════════════
# data loading
# ═══════════════════════════════════════════════════════════════════════════

def load_all():
    root       = zarr.open(str(ZARR_PATH), mode="r")
    ep_ends    = root["meta/episode_ends"][:]
    grasp_ids  = root["meta/grasp_type_id"][:]
    states     = root["data/state"][:]
    actions    = root["data/action"][:]
    pcs        = root["data/point_cloud"][:]
    starts     = np.concatenate([[0], ep_ends[:-1]])

    with open(WITH_CKPT.parent / "norm_stats.json") as f:
        ns = json.load(f)
    s_min = np.array(ns["state_min"],    dtype=np.float32)
    s_rng = np.array(ns["state_range"],  dtype=np.float32)
    a_min = np.array(ns["action_min"],   dtype=np.float32)
    a_rng = np.array(ns["action_range"], dtype=np.float32)

    # Gather representative PCs and per-type action trajectories.
    rep_pcs          = {}    # one PC per grasp type (first good episode)
    demo_traj_per_g  = {}    # (n_demos, pred_horizon, action_dim)
    demo_final_pose  = {}    # mean final hand pose per grasp type
    pred_horizon     = 16

    for gid in np.unique(grasp_ids):
        ep_idx = np.where(grasp_ids == gid)[0]
        rep_pcs[int(gid)] = pcs[starts[ep_idx[0]]]
        trajs = []
        finals = []
        for ei in ep_idx:
            s, e = starts[ei], ep_ends[ei]
            # take the LAST pred_horizon timesteps → "what the demonstrator
            # was doing right before settling", which is the interesting
            # part for grasp-type discrimination.
            if e - s < pred_horizon:
                continue
            trajs.append(actions[e - pred_horizon : e])
            finals.append(actions[e - 1])
        demo_traj_per_g[int(gid)]   = np.stack(trajs)       # (n_eps, T, 23)
        demo_final_pose[int(gid)]   = np.stack(finals).mean(0)  # (23,)

    return {
        "rep_pcs":         rep_pcs,
        "demo_traj":       demo_traj_per_g,
        "demo_final_pose": demo_final_pose,
        "starts":          starts,
        "ep_ends":         ep_ends,
        "grasp_ids":       grasp_ids,
        "states":          states,
        "pcs":             pcs,
        "norm": {"s_min": s_min, "s_rng": s_rng,
                 "a_min": a_min, "a_rng": a_rng},
    }


def sample_inputs_per_type(data, n_per_type=N_PER_GRASP):
    rng = np.random.default_rng(42)
    s_min, s_rng = data["norm"]["s_min"], data["norm"]["s_rng"]
    grouped = {}
    for gid in np.unique(data["grasp_ids"]):
        ep_idx = np.where(data["grasp_ids"] == gid)[0]
        chosen = rng.choice(ep_idx, size=n_per_type, replace=False)
        batch_state, batch_pc = [], []
        for ei in chosen:
            s = data["states"][data["starts"][ei]]
            s_n = 2.0 * (s.astype(np.float32) - s_min) / s_rng - 1.0
            batch_state.append(np.stack([s_n, s_n], axis=0))
            batch_pc.append(data["pcs"][data["starts"][ei]].astype(np.float32))
        grouped[int(gid)] = {
            "obs_state": np.stack(batch_state),
            "obs_pc":    np.stack(batch_pc),
            "grasp_id":  int(gid),
        }
    return grouped


def load_policy(ckpt, use_grasp):
    ck  = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    cfg = ck["config"]
    pol = PointCloudDiffusionPolicy(
        state_dim=cfg["state_dim"], action_dim=cfg["action_dim"],
        obs_horizon=cfg["obs_horizon"], pred_horizon=cfg["pred_horizon"],
        num_diffusion_steps=cfg["diffusion_steps"],
        down_dims=tuple(cfg["down_dims"]),
        n_grasp_types=cfg["n_grasp_types"],
        use_grasp_conditioning=use_grasp,
    ).to(DEVICE)
    pol.load_state_dict(ck["model_state_dict"])
    pol.eval()
    return pol


@torch.no_grad()
def sample_trajectories(policy, inputs, k=K_SAMPLES):
    """For one grasp type, return (n_inputs * k, pred_horizon, 23) DENORMALISED actions."""
    obs_state = torch.from_numpy(inputs["obs_state"]).to(DEVICE)
    obs_pc    = torch.from_numpy(inputs["obs_pc"]).to(DEVICE)
    gid       = torch.full((obs_state.shape[0],), inputs["grasp_id"],
                           dtype=torch.long, device=DEVICE)
    outs = []
    for ki in range(k):
        torch.manual_seed(7000 + ki)
        a = policy.predict_action(obs_state, obs_pc, gid,
                                  num_inference_steps=INFER_STEPS).cpu().numpy()
        outs.append(a)
    return np.concatenate(outs, axis=0)   # (n_inputs * k, T, 23)


def denormalise(a_norm, a_min, a_rng):
    return (a_norm + 1.0) / 2.0 * a_rng + a_min


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE N1 — benchmark overview
# ═══════════════════════════════════════════════════════════════════════════

def fig_n1_benchmark(data, out_path):
    fig, axes = plt.subplots(
        2, 4, figsize=(15.5, 6.3),
        gridspec_kw={"hspace": 0.55, "wspace": 0.28,
                     "height_ratios": [1.05, 1.0]},
    )
    fig.subplots_adjust(top=0.88, bottom=0.13, left=0.06, right=0.98)

    # compute tight bounds across all representative PCs so they're
    # comparable AND actually zoomed in.
    all_pts = np.concatenate([data["rep_pcs"][g] for g in GRASP_ORDER], axis=0)
    x_mu = all_pts[:, 0].mean()
    y_mu = all_pts[:, 1].mean()
    z_lo, z_hi = 0.0, max(0.09, all_pts[:, 2].max() * 1.10)

    for j, gid in enumerate(GRASP_ORDER):
        ax = axes[0, j]
        pc = data["rep_pcs"][gid]
        if len(pc) > 1400:
            idx = np.random.default_rng(0).choice(len(pc), 1400, replace=False)
            pc = pc[idx]

        # centre each hold at 0 on X to make side-profile comparable.
        x_local = pc[:, 0] - pc[:, 0].mean()
        ax.scatter(x_local, pc[:, 2], s=3.8, color=GRASP_COLORS[gid],
                   alpha=0.65, edgecolors="none")

        # hold silhouette metrics shown in an info-box
        z_cent  = pc[:, 2].mean() * 100
        z_peak  = pc[:, 2].max()  * 100
        width_x = (pc[:, 0].max() - pc[:, 0].min()) * 100
        depth_y = (pc[:, 1].max() - pc[:, 1].min()) * 100
        info = (f"Z centroid = {z_cent:.1f} cm\n"
                f"peak height = {z_peak:.1f} cm\n"
                f"X × Y footprint = {width_x:.1f} × {depth_y:.1f} cm")
        ax.text(0.03, 0.97, info, transform=ax.transAxes, ha="left",
                va="top", fontsize=7.5, color="#333333",
                bbox=dict(facecolor="white", edgecolor="#cccccc",
                          boxstyle="round,pad=0.25"))

        ax.set_xlim(-0.08, 0.08)
        ax.set_ylim(z_lo, z_hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_facecolor("#F8F9FA")
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.set_title(f"{GRASP_ID_TO_NAME[gid]}", fontsize=13, fontweight="bold",
                     color=GRASP_COLORS[gid], pad=6)
        if j == 0:
            ax.set_ylabel("World Z (m)\nside view", fontsize=9)
        ax.set_xlabel("X offset from centroid (m)", fontsize=8)

    for j, gid in enumerate(GRASP_ORDER):
        ax = axes[1, j]
        pose = data["demo_final_pose"][gid][7:]
        xs = np.arange(16)
        ax.bar(xs, pose, color=GRASP_COLORS[gid], alpha=0.9,
               edgecolor="#333333", linewidth=0.4)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_ylim(-0.7, 2.3)
        ax.set_xticks(xs[::2])
        ax.set_xticklabels([HAND_JOINT_NAMES[i] for i in xs[::2]],
                           rotation=50, ha="right", fontsize=6.5)
        ax.yaxis.grid(True, alpha=0.25, linestyle="--"); ax.set_axisbelow(True)
        if j == 0:
            ax.set_ylabel("Joint angle (rad)\nmean demo\nfinal pose", fontsize=9)

    fig.suptitle(
        "The Climbing Holds benchmark — four taxonomically distinct grasp types\n"
        "Top: side view of one hold per type (point cloud, X centred on hold)   "
        "·   Bottom: mean target LEAP hand pose at demo end (16 joints)",
        fontsize=12.5, y=0.99,
    )
    fig.savefig(str(out_path))
    plt.close(fig)
    print(f"Saved {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE N2 — mode-commitment fingerprint
# ═══════════════════════════════════════════════════════════════════════════

# A joint that visibly differs across grasp types from Panel D/hand plot:
# index MCP_flex = hand joint index 1, so action dim 7+1 = 8. We'll
# auto-pick the hand joint with the biggest between-type spread so the
# figure is convincing.
def pick_most_discriminative_hand_joint(demo_final_pose):
    finals = np.stack([demo_final_pose[g] for g in GRASP_ORDER])[:, 7:]  # (4, 16)
    spread = finals.std(axis=0)
    return int(np.argmax(spread)), spread


def fig_n2_mode_commitment(data, samples_w, samples_n, out_path):
    """
    samples_w / samples_n : dict[gid] -> (M, T=16, 23) DENORMALISED
    """
    T = 16
    joint_idx, spread = pick_most_discriminative_hand_joint(data["demo_final_pose"])
    print(f"  most-discriminative hand joint: #{joint_idx} ({HAND_JOINT_NAMES[joint_idx]})  "
          f"std across grasp types = {spread[joint_idx]:.2f} rad")

    fig = plt.figure(figsize=(16, 9.5))
    gs  = GridSpec(2, 12, figure=fig, hspace=0.55, wspace=1.15,
                   height_ratios=[1.0, 1.05])

    # ── top row: time-resolved ribbons for the discriminative joint ──
    time_axis = np.arange(T)
    top_spans = [(0, 3), (3, 6), (6, 9), (9, 12)]
    for j, gid in enumerate(GRASP_ORDER):
        ax = fig.add_subplot(gs[0, top_spans[j][0]:top_spans[j][1]])

        gt = data["demo_traj"][gid][:, :, 7 + joint_idx]   # (n_demos, T)
        gt_mu, gt_sd = gt.mean(0), gt.std(0)
        w = samples_w[gid][:, :, 7 + joint_idx]             # (M, T)
        n = samples_n[gid][:, :, 7 + joint_idx]
        w_mu, w_sd = w.mean(0), w.std(0)
        n_mu, n_sd = n.mean(0), n.std(0)

        ax.fill_between(time_axis, gt_mu - gt_sd, gt_mu + gt_sd,
                        color="#9CA3AF", alpha=0.35, label="Demos ±1σ")
        ax.plot(time_axis, gt_mu, color="#1F2937", linewidth=1.8,
                label="Demo mean")
        ax.fill_between(time_axis, w_mu - w_sd, w_mu + w_sd,
                        color="#2563EB", alpha=0.20)
        ax.plot(time_axis, w_mu, color="#2563EB", linewidth=2.0,
                label="With-tax")
        ax.fill_between(time_axis, n_mu - n_sd, n_mu + n_sd,
                        color="#DC2626", alpha=0.18)
        ax.plot(time_axis, n_mu, color="#DC2626", linewidth=2.0,
                linestyle="--", label="W/o-tax")

        ax.set_title(GRASP_ID_TO_NAME[gid],
                     color=GRASP_COLORS[gid], fontweight="bold", fontsize=12)
        ax.yaxis.grid(True, alpha=0.25, linestyle="--")
        ax.set_axisbelow(True)
        ax.set_xlabel("Prediction step")
        if j == 0:
            ax.set_ylabel(f"{HAND_JOINT_NAMES[joint_idx]}  (rad)")
        if j == len(GRASP_ORDER) - 1:
            ax.legend(loc="best", fontsize=8, framealpha=0.92)

    # ── bottom row: final-step hand-pose fingerprint per model ──
    #   For each of 3 sources (Demo / With / W/o) we plot 4 overlaid bar
    #   traces (one per grasp type). If a source "commits", the 4 traces
    #   differ; if it collapses, the 4 traces look alike.
    source_names  = ["Demonstrations", "With-taxonomy model", "Without-taxonomy model"]
    source_values = {
        "Demonstrations":        {g: data["demo_final_pose"][g][7:] for g in GRASP_ORDER},
        "With-taxonomy model":   {g: samples_w[g][:, -1, 7:].mean(0) for g in GRASP_ORDER},
        "Without-taxonomy model":{g: samples_n[g][:, -1, 7:].mean(0) for g in GRASP_ORDER},
    }

    # quantify between-type spread (rad) and put it in each panel's title
    def spread_rad(src_dict):
        arr = np.stack([src_dict[g] for g in GRASP_ORDER])   # (4, 16)
        # mean over joints of per-joint std across types
        return float(arr.std(axis=0).mean())

    bot_spans = [(0, 4), (4, 8), (8, 12)]
    ax_bot = [fig.add_subplot(gs[1, s:e]) for (s, e) in bot_spans]
    ax_bot_demo, ax_bot_with, ax_bot_no = ax_bot
    xs = np.arange(16)
    for ax, name in zip(ax_bot, source_names):
        width = 0.18
        for k, gid in enumerate(GRASP_ORDER):
            pose = source_values[name][gid]
            ax.bar(xs + (k - 1.5) * width, pose, width,
                   color=GRASP_COLORS[gid],
                   label=GRASP_ID_TO_NAME[gid] if ax is ax_bot_demo else None,
                   alpha=0.88, edgecolor="#333333", linewidth=0.3)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(xs[::2])
        ax.set_xticklabels([HAND_JOINT_NAMES[i] for i in xs[::2]],
                           rotation=55, ha="right", fontsize=6.5)
        ax.set_ylim(-0.7, 2.3)
        spread_val = spread_rad(source_values[name])
        ax.set_title(f"{name}\nbetween-type spread = {spread_val:.2f} rad",
                     fontsize=10, pad=6)
        ax.yaxis.grid(True, alpha=0.2, linestyle="--")
        ax.set_axisbelow(True)
    ax_bot_demo.set_ylabel("Final hand-pose joint angle (rad)")
    ax_bot_demo.legend(loc="upper right", fontsize=8, ncol=2,
                       framealpha=0.92, title="Grasp type")

    # Make the bottom row subplots consume gs[1, 0:2], gs[1, 2], gs[1, 3]
    # (currently each spans a single column; demo panel wider by config).
    # Actually they each occupy one column; demo is first. This is OK.

    fig.suptitle(
        "Mode-commitment fingerprint — do both models actually produce "
        "four distinct grasps?\n"
        "Top: one hand-joint trajectory over 16 prediction steps · "
        "Bottom: mean final hand pose per grasp type (wider 4-coloured "
        "spread = more committed / less collapsed)",
        fontsize=12, y=1.01,
    )
    fig.savefig(str(out_path))
    plt.close(fig)
    print(f"Saved {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print(f"Device: {DEVICE}")
    data = load_all()

    # Figure N1 — no models needed
    print("Generating N1 (benchmark overview)...")
    fig_n1_benchmark(data, OUT_DIR / "fig_novelty_benchmark.png")

    # Figure N2 needs model samples
    print("Loading policies...")
    pol_w = load_policy(WITH_CKPT, use_grasp=True)
    pol_n = load_policy(NO_CKPT,   use_grasp=False)

    print(f"Sampling {N_PER_GRASP}×{K_SAMPLES} trajectories per grasp type "
          f"from each policy ({INFER_STEPS}-step DDIM)...")
    inputs = sample_inputs_per_type(data)
    a_min, a_rng = data["norm"]["a_min"], data["norm"]["a_rng"]
    samples_w, samples_n = {}, {}
    for gid, inp in inputs.items():
        print(f"  grasp={GRASP_ID_TO_NAME[gid]}  "
              f"n_inputs={inp['obs_state'].shape[0]}")
        raw_w = sample_trajectories(pol_w, inp)
        raw_n = sample_trajectories(pol_n, inp)
        samples_w[gid] = denormalise(raw_w, a_min, a_rng)
        samples_n[gid] = denormalise(raw_n, a_min, a_rng)

    print("Generating N2 (mode-commitment fingerprint)...")
    fig_n2_mode_commitment(data, samples_w, samples_n,
                           OUT_DIR / "fig_novelty_mode_commitment.png")

    # ── quantitative summary for the caption ─────────────────────────────────
    def between_type_spread(src):
        arr = np.stack([src[g] for g in GRASP_ORDER])
        return float(arr.std(axis=0).mean())

    demo_spread  = between_type_spread({g: data["demo_final_pose"][g][7:]  for g in GRASP_ORDER})
    with_spread  = between_type_spread({g: samples_w[g][:, -1, 7:].mean(0) for g in GRASP_ORDER})
    no_spread    = between_type_spread({g: samples_n[g][:, -1, 7:].mean(0) for g in GRASP_ORDER})

    print("\nBetween-grasp-type hand-pose spread at final prediction step:")
    print(f"  Demonstrations       : {demo_spread:.3f} rad")
    print(f"  With-taxonomy model  : {with_spread:.3f} rad  "
          f"({with_spread/demo_spread*100:.0f}% of demo spread)")
    print(f"  W/o-taxonomy model   : {no_spread:.3f} rad  "
          f"({no_spread/demo_spread*100:.0f}% of demo spread)")

    print("\nDone. Figures saved to", OUT_DIR)


if __name__ == "__main__":
    main()
