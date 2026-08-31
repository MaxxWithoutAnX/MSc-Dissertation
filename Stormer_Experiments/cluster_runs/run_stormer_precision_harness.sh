#!/bin/bash

START=$(date +%s)
echo "=== START: $(date) ==="
trap 'echo "=== END: $(date)  ELAPSED $(($(date +%s) - START))s ==="' EXIT

source /vol/bitbucket/mes25/miniconda3/etc/profile.d/conda.sh
conda activate climate                  # torchao 0.9.0 -- REQUIRED for W4's UIntXWeightOnlyConfig
source /vol/cuda/12.4.0/setup.sh
export MPLBACKEND=Agg
export PHYSQ_CORE=/vol/bitbucket/mes25/physq_core

cd /vol/bitbucket/mes25/stormer_pipeline
mkdir -p logs

time python run_stormer_precision_harness.py \
    --manifest stormer_harness_configs.csv \
    --root-dir /vol/bitbucket/mes25/wb2_h5df \
    --data-split test_2021 \
    --checkpoint /vol/bitbucket/mes25/stormer_checkpoints/stormer_1.40625_patch_size_2.ckpt \
    --outdir harness_results_2021_stormer \
    --n-inits 12 \
    --leads 24,72,120,168 --score-lead 120

