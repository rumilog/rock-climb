#!/bin/bash
# Training launch script for diffusion policy
# Run with: bash run_training.sh

source ~/franka/bin/activate
cd /home/rumi/Desktop/tele/data_collection

python3 train.py \
    --zarr ../datasets/climbing_holds_v2.zarr \
    --ckpt-dir ../checkpoints/v2_fixed \
    --epochs 600 \
    --batch 64 \
    --img-size 224 \
    --diffusion-steps 100 \
    --augment \
    --good-only \
    --amp \
    --save-every 50
