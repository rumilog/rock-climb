"""
Latent space visualizations for the paper.

Extracts embeddings from trained checkpoints and plots t-SNE:
  Panel A — PointNet geometry embeddings (do hold shapes cluster by grasp type?)
  Panel B — Full encoder (WITH taxonomy): PC + state + grasp label fused
  Panel C — Full encoder (WITHOUT taxonomy): PC + state only
  Panel D — Raw robot joint trajectories (PCA): what did the arm/hand actually DO?

Run:
    source ~/franka/bin/activate
    python3 eval_results/generate_latent_viz.py
"""

import sys
import math
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import zarr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold

SESSION_JSON = Path("/home/rumi/Desktop/tele/eval_results/"
                    "paired_session_20260417_131412.json")

# ── paths ─────────────────────────────────────────────────────────────────────
ZARR_PATH   = Path("/mnt/ssd/rumi_tele_datasets/climbing_holds.zarr")
WITH_CKPT   = Path("/home/rumi/Desktop/tele/checkpoints/pc_with_taxonomy/best.pt")
NO_CKPT     = Path("/home/rumi/Desktop/tele/checkpoints/pc_no_taxonomy/best.pt")
OUT_DIR     = Path("/home/rumi/Desktop/tele/eval_results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── inline model defs (copied from train.py, no import needed) ─────────────
N_GRASP_TYPES = 4

class PointNetEncoder(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
        )
        self.proj = nn.Linear(256, out_dim)

    def forward(self, x):
        feat = self.point_mlp(x)
        feat = feat.max(dim=1)[0]
        return self.proj(feat)


class GraspTypeEncoder(nn.Module):
    def __init__(self, n_types=N_GRASP_TYPES, out_dim=64):
        super().__init__()
        self.n_types = n_types
        self.mlp = nn.Sequential(
            nn.Linear(n_types, out_dim), nn.ReLU(),
            nn.Linear(out_dim, out_dim), nn.ReLU(),
        )

    def forward(self, grasp_type_id):
        one_hot = F.one_hot(grasp_type_id, num_classes=self.n_types).float()
        return self.mlp(one_hot)


class PointCloudObservationEncoder(nn.Module):
    def __init__(self, state_dim, obs_horizon, pc_dim=256, state_mlp_dim=128,
                 grasp_dim=64, n_grasp_types=N_GRASP_TYPES,
                 use_grasp_conditioning=True):
        super().__init__()
        self.obs_horizon = obs_horizon
        self.use_grasp_conditioning = use_grasp_conditioning
        self.pc_encoder = PointNetEncoder(out_dim=pc_dim)
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim * obs_horizon, 256), nn.ReLU(),
            nn.Linear(256, state_mlp_dim), nn.ReLU(),
        )
        if use_grasp_conditioning:
            self.grasp_encoder = GraspTypeEncoder(n_grasp_types, grasp_dim)
            fuse_in = pc_dim + state_mlp_dim + grasp_dim
        else:
            self.grasp_encoder = None
            fuse_in = pc_dim + state_mlp_dim
        self.fuse = nn.Sequential(nn.Linear(fuse_in, 512), nn.ReLU())
        self.out_dim = 512

    def forward(self, obs_state, obs_pc, grasp_type_id):
        B = obs_state.shape[0]
        pc_feat    = self.pc_encoder(obs_pc)
        state_flat = obs_state.reshape(B, -1)
        state_feat = self.state_mlp(state_flat)
        if self.use_grasp_conditioning:
            grasp_feat = self.grasp_encoder(grasp_type_id)
            return self.fuse(torch.cat([pc_feat, state_feat, grasp_feat], dim=-1))
        else:
            return self.fuse(torch.cat([pc_feat, state_feat], dim=-1))


# ── colour / label scheme ─────────────────────────────────────────────────────
# grasp_type_id in zarr: 0=crimp, 1=sloper, 2=pinch, 3=jug
GRASP_ID_TO_NAME = {0: "Crimp", 1: "Sloper", 2: "Pinch", 3: "Jug"}
GRASP_COLORS     = {0: "#F59E0B", 1: "#10B981", 2: "#8B5CF6", 3: "#3B82F6"}
GRASP_MARKERS    = {0: "o", 1: "s", 2: "^", 3: "D"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 11, "figure.dpi": 150, "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


# ── data loading ──────────────────────────────────────────────────────────────

def load_episode_data():
    """Return per-episode arrays: point clouds, grasp type ids, and state sequences."""
    root       = zarr.open(str(ZARR_PATH), mode="r")
    ep_ends    = root["meta/episode_ends"][:]
    grasp_ids  = root["meta/grasp_type_id"][:]        # (n_ep,)
    quality    = root["meta/quality"][:] if "quality" in root["meta"] else np.ones(len(ep_ends))

    states     = root["data/state"][:]                 # (N_ts, 23)
    pcs        = root["data/point_cloud"][:]           # (N_ts, 1024, 3)
    actions    = root["data/action"][:]                # (N_ts, 23)

    starts = np.concatenate([[0], ep_ends[:-1]])

    ep_pcs, ep_grasp, ep_states_first, ep_states_mean, ep_actions_mean = [], [], [], [], []
    ep_arm_traj, ep_hand_traj = [], []

    for i, (s, e) in enumerate(zip(starts, ep_ends)):
        if quality[i] != 1:
            continue
        # One PC per episode (captured at episode start, same for all timesteps)
        ep_pcs.append(pcs[s])                           # (1024, 3)
        ep_grasp.append(grasp_ids[i])
        ep_states_first.append(states[s])               # first timestep state
        ep_states_mean.append(states[s:e].mean(0))      # mean state over episode
        ep_actions_mean.append(actions[s:e].mean(0))    # mean action over episode

        # Full arm+hand trajectory flattened → for PCA of "what the robot did"
        traj = actions[s:e]                             # (T, 23)
        # Subsample to 50 timesteps for uniform length
        T = traj.shape[0]
        idx = np.linspace(0, T - 1, min(50, T), dtype=int)
        traj_sub = traj[idx].flatten()                  # (50*23,)
        ep_arm_traj.append(traj_sub)

    return {
        "pcs":          np.array(ep_pcs),           # (N_ep, 1024, 3)
        "grasp_ids":    np.array(ep_grasp),          # (N_ep,)
        "states_first": np.array(ep_states_first),  # (N_ep, 23)
        "states_mean":  np.array(ep_states_mean),   # (N_ep, 23)
        "actions_mean": np.array(ep_actions_mean),  # (N_ep, 23)
        "traj_flat":    np.array(ep_arm_traj),       # (N_ep, 50*23)
        # Per-episode PC centroid (for matching to eval trials, which only save centroids).
        "pc_centroids": np.array([pc.mean(0) for pc in ep_pcs]),  # (N_ep, 3)
    }


def load_eval_session(path):
    """Load paired eval session and return per-pair dicts keyed by grasp_type_id."""
    with open(path) as f:
        session = json.load(f)

    pairs = []
    for p in session["pairs"]:
        if p.get("aborted", False):
            continue
        w_cent = np.array(p["pc_stats"]["WITH_TAXONOMY"]["centroid"])
        n_cent = np.array(p["pc_stats"]["WITHOUT_TAXONOMY"]["centroid"])
        pairs.append({
            "grasp_type_id": p["grasp_type_id"],
            "grasp_type":    p["grasp_type"],
            "hold_id":       p["hold_id"],
            "with_ok":       1 if p["with_rating"] == "good" else 0,
            "no_ok":         1 if p["no_rating"]   == "good" else 0,
            "pc_centroid":   (w_cent + n_cent) / 2.0,   # mean over the two trials
        })
    return pairs


def build_encoder(ckpt_path, use_grasp_cond, state_dim=23, obs_horizon=2):
    ckpt   = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    enc = PointCloudObservationEncoder(
        state_dim=state_dim,
        obs_horizon=obs_horizon,
        use_grasp_conditioning=use_grasp_cond,
    )
    # Load only the obs_encoder weights
    sd = {k.replace("obs_encoder.", ""): v
          for k, v in ckpt["model_state_dict"].items()
          if k.startswith("obs_encoder.")}
    enc.load_state_dict(sd)
    enc.eval()
    return enc


@torch.no_grad()
def extract_embeddings(enc, data, norm_stats_path):
    """Run encoder over all episodes → (N_ep, emb_dim)."""
    with open(norm_stats_path) as f:
        ns = json.load(f)

    state_min   = np.array(ns["state_min"],   dtype=np.float32)
    state_range = np.array(ns["state_range"], dtype=np.float32)

    pcs       = torch.from_numpy(data["pcs"].astype(np.float32))      # (N, 1024, 3)
    # Build (N, 2, 23) obs_state by duplicating (obs_horizon=2 means 2 consecutive states)
    s_norm    = 2.0 * (data["states_first"].astype(np.float32) - state_min) / state_range - 1.0
    obs_state = torch.from_numpy(np.stack([s_norm, s_norm], axis=1))   # (N, 2, 23)
    grasp_ids = torch.from_numpy(data["grasp_ids"].astype(np.int64))

    BATCH = 64
    N     = pcs.shape[0]
    embs  = []
    for start in range(0, N, BATCH):
        e  = min(start + BATCH, N)
        emb = enc(obs_state[start:e], pcs[start:e], grasp_ids[start:e])
        embs.append(emb.numpy())
    return np.concatenate(embs, axis=0)   # (N, 512)


@torch.no_grad()
def extract_pointnet_embeddings(enc, data):
    """Extract ONLY the PointNet PC embedding (256-d), ignoring state/grasp."""
    pcs   = torch.from_numpy(data["pcs"].astype(np.float32))
    BATCH = 64
    N     = pcs.shape[0]
    embs  = []
    for start in range(0, N, BATCH):
        e   = min(start + BATCH, N)
        emb = enc.pc_encoder(pcs[start:e])
        embs.append(emb.numpy())
    return np.concatenate(embs, axis=0)   # (N, 256)


def tsne_2d(X, perplexity=None, seed=42):
    n = X.shape[0]
    perp = perplexity if perplexity is not None else min(30, max(5, n // 5))
    # First reduce with PCA to 50-d for speed
    pca_dim = min(50, X.shape[1], n - 1)
    if X.shape[1] > pca_dim:
        X = PCA(n_components=pca_dim, random_state=seed).fit_transform(X)
    ts = TSNE(n_components=2, perplexity=perp, random_state=seed,
              n_iter=1500, learning_rate="auto", init="pca")
    return ts.fit_transform(X)


def cluster_quality(X_highdim, labels):
    """
    Quantify how well grasp types separate in a latent space.

    Returns:
        sil  : silhouette score on the high-dim embedding (range [-1, 1];
               >0.25 is reasonable, >0.5 is strong, >0.7 is excellent)
        knn  : 5-fold stratified kNN classification accuracy with k=5
               (1.0 = labels are perfectly recoverable from the latent)
    """
    labels = np.asarray(labels)
    sil = float(silhouette_score(X_highdim, labels))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs = []
    for tr, te in skf.split(X_highdim, labels):
        clf = KNeighborsClassifier(n_neighbors=5)
        clf.fit(X_highdim[tr], labels[tr])
        accs.append(clf.score(X_highdim[te], labels[te]))
    return sil, float(np.mean(accs))


def match_eval_to_training(eval_pairs, data):
    """
    For each eval pair, find the nearest training episode of the SAME grasp type
    by PC-centroid distance. Returns a list of records with the matched index.
    """
    matched = []
    for pair in eval_pairs:
        gid    = pair["grasp_type_id"]
        mask   = np.where(data["grasp_ids"] == gid)[0]
        if len(mask) == 0:
            continue
        cents  = data["pc_centroids"][mask]
        dists  = np.linalg.norm(cents - pair["pc_centroid"][None, :], axis=1)
        nearest = int(mask[np.argmin(dists)])
        matched.append({**pair,
                        "matched_ep":   nearest,
                        "match_dist_mm": float(dists.min() * 1000)})
    return matched


def classify_pair(with_ok, no_ok):
    if   with_ok == 1 and no_ok == 0: return "WITH wins"
    elif with_ok == 1 and no_ok == 1: return "Tie — both good"
    elif with_ok == 0 and no_ok == 0: return "Tie — both bad"
    else:                              return "WITHOUT wins"


OUTCOME_STYLE = {
    "WITH wins":        {"color": "#2563EB", "marker": "o", "s": 95, "edge": "white"},
    "WITHOUT wins":     {"color": "#DC2626", "marker": "X", "s": 105, "edge": "white"},
    "Tie — both good":  {"color": "#22C55E", "marker": "s", "s": 70, "edge": "white"},
    "Tie — both bad":   {"color": "#4B5563", "marker": "P", "s": 90, "edge": "white"},
}


def scatter_grasp(ax, xy, grasp_ids, title, alpha=0.75, s=55, add_legend=False):
    for gid in sorted(np.unique(grasp_ids)):
        mask = grasp_ids == gid
        ax.scatter(xy[mask, 0], xy[mask, 1],
                   c=GRASP_COLORS[gid], marker=GRASP_MARKERS[gid],
                   s=s, alpha=alpha, edgecolors="white", linewidths=0.4,
                   label=GRASP_ID_TO_NAME[gid], zorder=3)
    ax.set_title(title, fontsize=10, pad=6)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines[["top","right","bottom","left"]].set_visible(False)
    if add_legend:
        ax.legend(loc="lower right", fontsize=8, markerscale=1.2,
                  framealpha=0.85, edgecolor="#cccccc")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading episode data from zarr...")
    data = load_episode_data()
    N    = len(data["grasp_ids"])
    print(f"  {N} episodes loaded  (grasp type counts: "
          + ", ".join(f"{GRASP_ID_TO_NAME[g]}={np.sum(data['grasp_ids']==g)}"
                      for g in sorted(np.unique(data["grasp_ids"]))) + ")")

    norm_stats = str(Path(WITH_CKPT).parent / "norm_stats.json")

    print("Building WITH-taxonomy encoder...")
    enc_with = build_encoder(WITH_CKPT, use_grasp_cond=True)
    print("Building WITHOUT-taxonomy encoder...")
    enc_no   = build_encoder(NO_CKPT,   use_grasp_cond=False)

    print("Extracting PointNet (geometry) embeddings...")
    pc_emb = extract_pointnet_embeddings(enc_with, data)       # (N, 256)

    print("Extracting WITH-taxonomy fused embeddings...")
    with_emb = extract_embeddings(enc_with, data, norm_stats)  # (N, 512)

    print("Extracting WITHOUT-taxonomy fused embeddings...")
    no_emb   = extract_embeddings(enc_no,   data, norm_stats)  # (N, 512)

    grasp_ids = data["grasp_ids"]

    # ── t-SNE reductions ──────────────────────────────────────────────────────
    print("Running t-SNE on PointNet embeddings...")
    pc_2d   = tsne_2d(pc_emb)
    print("Running t-SNE on WITH-taxonomy embeddings...")
    with_2d = tsne_2d(with_emb)
    print("Running t-SNE on WITHOUT-taxonomy embeddings...")
    no_2d   = tsne_2d(no_emb)

    # PCA on trajectory (robot joint sequence)
    print("Running PCA on robot joint trajectories...")
    traj_flat = data["traj_flat"]
    pca = PCA(n_components=2, random_state=42)
    traj_2d = pca.fit_transform(traj_flat)
    var_explained = pca.explained_variance_ratio_ * 100

    # ── quantify cluster quality on the HIGH-DIM embeddings ───────────────────
    # (silhouette + kNN classification accuracy of grasp type from the latent)
    print("Computing cluster quality metrics...")
    sil_pc,   knn_pc   = cluster_quality(pc_emb,    grasp_ids)
    sil_with, knn_with = cluster_quality(with_emb,  grasp_ids)
    sil_no,   knn_no   = cluster_quality(no_emb,    grasp_ids)
    sil_traj, knn_traj = cluster_quality(traj_flat, grasp_ids)

    def _metric_suffix(sil, knn):
        return f"\nsilhouette={sil:.2f}  ·  5-NN acc={knn*100:.1f}%"

    print(f"  PointNet (256-d):          silhouette={sil_pc:+.3f}  kNN acc={knn_pc*100:5.1f}%")
    print(f"  WITH-tax fused (512-d):    silhouette={sil_with:+.3f}  kNN acc={knn_with*100:5.1f}%")
    print(f"  WITHOUT-tax fused (512-d): silhouette={sil_no:+.3f}  kNN acc={knn_no*100:5.1f}%")
    print(f"  Action trajectory (~1150-d): silhouette={sil_traj:+.3f}  kNN acc={knn_traj*100:5.1f}%")

    # ── Figure A: 2×2 latent space panel ─────────────────────────────────────
    fig = plt.figure(figsize=(13, 11))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    scatter_grasp(ax1, pc_2d, grasp_ids,
                  "A. PointNet geometry embedding\n(t-SNE of 256-d PC features)"
                  + _metric_suffix(sil_pc, knn_pc))
    scatter_grasp(ax2, with_2d, grasp_ids,
                  "B. WITH taxonomy: fused encoder output\n(t-SNE of 512-d: PC + state + grasp label)"
                  + _metric_suffix(sil_with, knn_with))
    scatter_grasp(ax3, no_2d, grasp_ids,
                  "C. WITHOUT taxonomy: fused encoder output\n(t-SNE of 512-d: PC + state, no label)"
                  + _metric_suffix(sil_no, knn_no))
    scatter_grasp(ax4, traj_2d, grasp_ids,
                  f"D. Robot joint trajectories (PCA)\n"
                  f"PC1={var_explained[0]:.1f}%  PC2={var_explained[1]:.1f}% variance"
                  + _metric_suffix(sil_traj, knn_traj),
                  add_legend=True)

    # Shared legend patches
    patches = [mpatches.Patch(color=GRASP_COLORS[g], label=GRASP_ID_TO_NAME[g])
               for g in sorted(GRASP_ID_TO_NAME)]
    fig.legend(handles=patches, loc="lower center", ncol=4,
               fontsize=10, framealpha=0.9, edgecolor="#cccccc",
               bbox_to_anchor=(0.5, -0.01))

    fig.suptitle("Learned representations — taxonomy-conditioned vs unconditioned diffusion policy\n"
                 "(each point = one training episode, coloured by grasp type)",
                 fontsize=12, y=1.01)

    out = OUT_DIR / "fig_latent_4panel.png"
    fig.savefig(str(out), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

    # ── Figure B: side-by-side WITH vs WITHOUT (bigger, for slides) ───────────
    fig2, (axA, axB) = plt.subplots(1, 2, figsize=(12, 5.5))

    scatter_grasp(axA, with_2d, grasp_ids,
                  "With taxonomy conditioning\n(grasp type label passed to encoder)"
                  + _metric_suffix(sil_with, knn_with))
    scatter_grasp(axB, no_2d, grasp_ids,
                  "Without taxonomy conditioning\n(no grasp type label — ablation)"
                  + _metric_suffix(sil_no, knn_no),
                  add_legend=True)

    for ax in (axA, axB):
        ax.set_facecolor("#F8F9FA")

    fig2.suptitle("Fused encoder representation — t-SNE of 512-d conditioning vector\n"
                  "(same PointNet + state encoder; only grasp label branch differs)",
                  fontsize=11, y=1.03)

    patches2 = [mpatches.Patch(color=GRASP_COLORS[g], label=GRASP_ID_TO_NAME[g])
                for g in sorted(GRASP_ID_TO_NAME)]
    fig2.legend(handles=patches2, loc="lower center", ncol=4,
                fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, -0.05))

    out2 = OUT_DIR / "fig_latent_with_vs_without.png"
    fig2.savefig(str(out2), bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved {out2}")

    # ── Figure C: robot trajectory PCA with per-joint importance ─────────────
    fig3, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: PCA scatter
    ax_pca = axes[0]
    scatter_grasp(ax_pca, traj_2d, grasp_ids,
                  f"Robot joint trajectories (PCA)\nPC1={var_explained[0]:.1f}%  "
                  f"PC2={var_explained[1]:.1f}%  total={sum(var_explained):.1f}%"
                  + _metric_suffix(sil_traj, knn_traj),
                  add_legend=True)
    ax_pca.set_facecolor("#F8F9FA")
    ax_pca.set_xlabel(f"PC1 ({var_explained[0]:.1f}% var)", fontsize=9)
    ax_pca.set_ylabel(f"PC2 ({var_explained[1]:.1f}% var)", fontsize=9)
    ax_pca.set_xticks([]); ax_pca.set_yticks([])

    # Right: mean joint configuration per grasp type (arm joints 0-6, hand joints 7-22)
    ax_joints = axes[1]
    joint_names_arm  = [f"arm_{i}" for i in range(7)]
    joint_names_hand = [f"hand_{i}" for i in range(16)]
    joint_names = joint_names_arm + joint_names_hand

    x_pos = np.arange(23)
    for gid in sorted(np.unique(grasp_ids)):
        mask  = grasp_ids == gid
        mean_action = data["actions_mean"][mask].mean(0)
        ax_joints.plot(x_pos, mean_action, color=GRASP_COLORS[gid],
                       marker=GRASP_MARKERS[gid], markersize=4,
                       linewidth=1.5, label=GRASP_ID_TO_NAME[gid], alpha=0.9)

    ax_joints.axvline(6.5, color="#999999", linewidth=1, linestyle="--")
    ax_joints.text(3.0, ax_joints.get_ylim()[0] if ax_joints.get_ylim()[0] != 0.0 else -1.8,
                   "Arm joints", ha="center", fontsize=8, color="#555555")
    ax_joints.text(14.5, ax_joints.get_ylim()[0] if ax_joints.get_ylim()[0] != 0.0 else -1.8,
                   "Hand joints", ha="center", fontsize=8, color="#555555")
    ax_joints.set_xticks(x_pos[::2])
    ax_joints.set_xticklabels(joint_names[::2], rotation=45, ha="right", fontsize=7)
    ax_joints.set_ylabel("Mean joint angle (rad, normalized)")
    ax_joints.set_title("Mean joint configuration per grasp type\n(averaged across all episodes of each type)")
    ax_joints.legend(fontsize=9, loc="upper right")
    ax_joints.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax_joints.set_axisbelow(True)

    out3 = OUT_DIR / "fig_latent_trajectories.png"
    fig3.tight_layout()
    fig3.savefig(str(out3), bbox_inches="tight")
    plt.close(fig3)
    print(f"Saved {out3}")

    # ── Figure D: action-PCA with paired-eval outcomes overlaid ──────────────
    if SESSION_JSON.exists():
        print("Overlaying paired-eval outcomes onto action-trajectory PCA...")
        eval_pairs = load_eval_session(SESSION_JSON)
        print(f"  {len(eval_pairs)} completed pairs loaded from {SESSION_JSON.name}")

        matched = match_eval_to_training(eval_pairs, data)
        if matched:
            avg_md = np.mean([m["match_dist_mm"] for m in matched])
            print(f"  nearest-training match distance: mean={avg_md:.1f} mm "
                  f"(centroid-matched within same grasp type)")

        fig4, (axL, axR) = plt.subplots(1, 2, figsize=(14, 6.2),
                                        gridspec_kw={"wspace": 0.08})

        # Left panel: training trajectories, grey background + coloured by outcome
        # Right panel: same, but showing WITHOUT-model outcomes only (fail-mode map)
        for ax, title, filter_fn in [
            (axL, "All paired outcomes overlaid",             None),
            (axR, "Where the Without-taxonomy policy FAILS",  lambda m: m["no_ok"] == 0),
        ]:
            # Faint grey scatter = all training episodes (context)
            ax.scatter(traj_2d[:, 0], traj_2d[:, 1],
                       color="#CBD5E1", s=25, alpha=0.55, zorder=1,
                       edgecolors="none", label=None)

            # cluster centroids in PCA space, to anchor the eye
            for gid in sorted(np.unique(grasp_ids)):
                mask = grasp_ids == gid
                cx, cy = traj_2d[mask, 0].mean(), traj_2d[mask, 1].mean()
                ax.scatter([cx], [cy], marker="*", s=260,
                           color=GRASP_COLORS[gid], edgecolors="black",
                           linewidths=1.0, zorder=5)
                ax.annotate(GRASP_ID_TO_NAME[gid], xy=(cx, cy),
                            xytext=(8, 8), textcoords="offset points",
                            fontsize=9, color="#222222",
                            fontweight="bold")

            if filter_fn is None:
                items = matched
            else:
                items = [m for m in matched if filter_fn(m)]

            # Jitter eval markers slightly so overlapping trials don't hide each other
            rng = np.random.default_rng(0)
            for m in items:
                ep = m["matched_ep"]
                jitter = rng.uniform(-0.6, 0.6, 2)
                pos = traj_2d[ep] + jitter
                cat = classify_pair(m["with_ok"], m["no_ok"])
                st  = OUTCOME_STYLE[cat]
                ax.scatter(pos[0], pos[1], color=st["color"], marker=st["marker"],
                           s=st["s"], alpha=0.85, edgecolors=st["edge"],
                           linewidths=1.0, zorder=4)

            ax.set_title(title, fontsize=11, pad=6)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_facecolor("#F8F9FA")
            for spine in ax.spines.values():
                spine.set_visible(False)

        # Shared outcome legend
        patches_out = [mpatches.Patch(color=st["color"], label=cat)
                       for cat, st in OUTCOME_STYLE.items()]
        fig4.legend(handles=patches_out, loc="lower center", ncol=4,
                    fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.03))

        fig4.suptitle("Paired-eval outcomes mapped into training action-space (PCA)\n"
                      "Each eval trial's point cloud is matched to the nearest training episode; "
                      "marker colour = outcome of that pair",
                      fontsize=12, y=1.02)

        out4 = OUT_DIR / "fig_latent_eval_overlay.png"
        fig4.savefig(str(out4), bbox_inches="tight")
        plt.close(fig4)
        print(f"Saved {out4}")

    print("\nDone. All latent space figures saved to", OUT_DIR)


if __name__ == "__main__":
    main()
