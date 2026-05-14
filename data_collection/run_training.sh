#!/bin/bash
# Training launch script — climbing_holds_rig dataset
# Runs WITH-taxonomy then WITHOUT-taxonomy sequentially.
#
# Sequential is intentional: two simultaneous runs on one GPU halve compute
# throughput for each, making both take ~2× longer with no benefit.
#
# ── CLUSTER SETUP (one-time) ───────────────────────────────────────────────
#
#   1. Install deps:
#        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
#        pip install numpy zarr opencv-python-headless huggingface_hub
#
#   2. Download the dataset:
#        huggingface-cli download rlogh/climbing-holds-rig \
#            --repo-type dataset --local-dir ./datasets/climbing_holds_rig.zarr
#
#   3. Run:
#        bash data_collection/run_training.sh
#
# ── OVERRIDES ──────────────────────────────────────────────────────────────
#
#   ZARR_PATH=/custom/path/rig.zarr bash run_training.sh
#   VENV_PATH=/custom/venv        bash run_training.sh
#
# ── MONITOR ────────────────────────────────────────────────────────────────
#
#   tail -f checkpoints/pc_with_taxonomy_rig/train.log
#   tail -f checkpoints/pc_no_taxonomy_rig/train.log
#
# ───────────────────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

ZARR="${ZARR_PATH:-/mnt/ssd/rumi_tele_datasets/climbing_holds_rig.zarr}"
CKPT_ROOT="${REPO_ROOT}/checkpoints"

# Activate venv if present (local robot machine); skip gracefully on cluster
VENV="${VENV_PATH:-${HOME}/franka}"
if [ -f "${VENV}/bin/activate" ]; then
    source "${VENV}/bin/activate"
fi

cd "$SCRIPT_DIR"

echo "Zarr:       $ZARR"
echo "Checkpoints: $CKPT_ROOT"
echo ""

# ── Run 1/2: WITH taxonomy conditioning ────────────────────────────────────
echo "=== [1/2] WITH taxonomy conditioning ==="
python3 train.py \
    --zarr "$ZARR" \
    --ckpt-dir "${CKPT_ROOT}/pc_with_taxonomy_rig" \
    --point-cloud \
    --epochs 3000 \
    --batch 128 \
    --augment \
    --good-only \
    --amp \
    --save-every 200

# ── Run 2/2: WITHOUT taxonomy conditioning (ablation) ──────────────────────
echo "=== [2/2] WITHOUT taxonomy conditioning ==="
python3 train.py \
    --zarr "$ZARR" \
    --ckpt-dir "${CKPT_ROOT}/pc_no_taxonomy_rig" \
    --point-cloud \
    --no-grasp-conditioning \
    --epochs 3000 \
    --batch 128 \
    --augment \
    --good-only \
    --amp \
    --save-every 200

echo ""
echo "=== Both runs complete ==="
echo "Checkpoints: ${CKPT_ROOT}/pc_with_taxonomy_rig/best.pt"
echo "             ${CKPT_ROOT}/pc_no_taxonomy_rig/best.pt"
