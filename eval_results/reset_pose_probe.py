"""Does the unlabeled policy pick the right grasp when started from the eval reset pose?

Simulated closed loop matching paired_eval.py (200 steps, execute 8 of each 16-step
chunk, 10 DDIM steps, static PC), assuming perfect tracking (next state = commanded
action). Final hand pose is classified by nearest demonstrator final-hand centroid.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import zarr

ROOT = Path("/home/rumi/Desktop/tele")
sys.path.insert(0, str(ROOT / "eval_results"))
sys.path.insert(0, str(ROOT / "data_collection"))
import generate_action_dist_viz as g  # noqa: E402

g.ZARR_PATH = Path("/mnt/ssd/rumi_tele_datasets/climbing_holds_rig.zarr")
g.WITH_CKPT = ROOT / "checkpoints/pc_with_taxonomy_rig/best.pt"
g.NO_CKPT = ROOT / "checkpoints/pc_no_taxonomy_rig/best.pt"
N_PER_GRASP = int(sys.argv[1]) if len(sys.argv) > 1 else 8
g.N_PER_GRASP = N_PER_GRASP
K, MAX_STEPS, EXEC, INF = 8, 200, 8, 10

RESET_ARM = np.array([-0.1111, -0.1703, -0.0621, -2.3442, 0.0408, 2.1952, 0.1559], np.float32)
RESET_HAND = np.array([0.078, 0.31, 0.41, 0.25, 0.14, 0.46, 0.50, 0.30,
                       0.43, 0.56, 0.55, 0.30, 0.17, -0.47, -0.10, 1.22], np.float32)
NAMES = ["crimp", "sloper", "pinch", "jug"]

inputs = g.load_sample_inputs()
gids = inputs["grasp_type_id"]
N = len(gids)

import json  # noqa: E402
ns = json.load(open(g.WITH_CKPT.parent / "norm_stats.json"))
s_min, s_rng = np.array(ns["state_min"], np.float32), np.array(ns["state_range"], np.float32)
a_min, a_rng = inputs["action_min"], inputs["action_range"]
norm_s = lambda s: 2.0 * (s - s_min) / s_rng - 1.0  # noqa: E731

# demonstrator final hand pose per type (all episodes, last frame)
root = zarr.open(str(g.ZARR_PATH), "r")
ends, egid = root["meta/episode_ends"][:], root["meta/grasp_type_id"][:]
states = root["data/state"]
final_hand = np.stack([states[e - 1][7:] for e in ends])
centroids = np.stack([final_hand[egid == t].mean(0) for t in range(4)])
d_demo = np.linalg.norm(final_hand[:, None] - centroids[None], axis=2)
print(f"demo self-consistency (nearest centroid = own type): {(d_demo.argmin(1) == egid).mean():.0%}")
print("demo final-hand dist to own centroid (mean):",
      np.round([np.linalg.norm(final_hand[egid == t] - centroids[t], axis=1).mean() for t in range(4)], 2))
print("start-hand dist to each type centroid, reset pose:",
      np.round(np.linalg.norm(RESET_HAND - centroids, axis=1), 2))

pol = {"with": g.load_policy(g.WITH_CKPT, True), "without": g.load_policy(g.NO_CKPT, False)}


@torch.no_grad()
def rollout(policy, start_state):
    """start_state (N, 23) raw -> final raw state (N*K, 23)."""
    s0 = np.repeat(start_state, K, axis=0)
    buf = [s0.copy(), s0.copy()]
    pc = torch.from_numpy(np.repeat(inputs["obs_pc"], K, axis=0)).to(g.DEVICE)
    gid = torch.from_numpy(np.repeat(gids, K)).to(g.DEVICE)
    torch.manual_seed(0)
    step = 0
    while step < MAX_STEPS:
        obs = torch.from_numpy(np.stack([norm_s(b) for b in buf[-2:]], 1).astype(np.float32)).to(g.DEVICE)
        a = policy.predict_action(obs, pc, gid, num_inference_steps=INF).cpu().numpy()
        a = (a + 1.0) / 2.0 * a_rng + a_min
        for i in range(min(EXEC, MAX_STEPS - step)):
            buf.append(a[:, i].astype(np.float32))
            step += 1
    return buf[-1]


first = np.stack([(inputs["obs_state"][i, 0] + 1.0) / 2.0 * s_rng + s_min for i in range(N)])
reset = np.tile(np.concatenate([RESET_ARM, RESET_HAND]), (N, 1))
true = np.repeat(gids, K)

for cond, start in [("training first frame", first), ("eval reset pose", reset)]:
    print(f"\n=== start state: {cond} ===")
    for m, p in pol.items():
        fin = rollout(p, start)[:, 7:]
        pred = np.linalg.norm(fin[:, None] - centroids[None], axis=2).argmin(1)
        conf = np.zeros((4, 4), int)
        for t, q in zip(true, pred):
            conf[t, q] += 1
        print(f"{m:8s} correct grasp: {(pred == true).mean():5.0%}   per type: " +
              "  ".join(f"{NAMES[t]} {conf[t, t] / conf[t].sum():.0%}" for t in range(4)))
        print(f"         final hand: dist to reset {np.linalg.norm(fin - RESET_HAND, axis=1).mean():.2f}, "
              f"to jug centroid {np.linalg.norm(fin - centroids[3], axis=1).mean():.2f}, "
              f"spread across inputs {np.linalg.norm(fin - fin.mean(0), axis=1).mean():.2f}")
        print("         confusion (rows = true, cols = predicted c/s/p/j): " +
              " | ".join(" ".join(f"{v:2d}" for v in r) for r in conf))
