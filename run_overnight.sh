#!/bin/bash
# Overnight training experiments — 3 configurations, ~2.5 hours each
# Total: ~7.5 hours (well within 9 hour window)

cd ~/Desktop/tele
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Experiment 2: Larger U-Net (512,1024,2048) ==="
echo "Started at: $(date)"
python3.10 -u data_collection/train.py \
    --epochs 3000 --amp --batch 64 --workers 0 \
    --down-dims "512,1024,2048" \
    --ckpt-dir checkpoints/run2_large_unet \
    2>&1 | tee checkpoints/run2_large_unet/train.log
echo "Finished at: $(date)"
echo ""

echo "=== Experiment 3: Shorter pred_horizon=8 ==="
echo "Started at: $(date)"
python3.10 -u data_collection/train.py \
    --epochs 3000 --amp --batch 64 --workers 0 \
    --pred-horizon 8 \
    --ckpt-dir checkpoints/run3_pred8 \
    2>&1 | tee checkpoints/run3_pred8/train.log
echo "Finished at: $(date)"
echo ""

echo "=== All experiments complete ==="
echo "Finished at: $(date)"
