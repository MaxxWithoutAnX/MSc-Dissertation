#!/bin/bash

START=$(date +%s)
echo "=== START: $(date) ==="
trap 'echo "=== END: $(date)  ELAPSED $(($(date +%s) - START))s ==="' EXIT

source /vol/bitbucket/mes25/miniconda3/etc/profile.d/conda.sh
conda activate climate                       # torchao 0.9.0 -- REQUIRED for W4 (+W8A8_sq later)
source /vol/cuda/12.4.0/setup.sh
export MPLBACKEND=Agg

cd /vol/bitbucket/mes25/aurora_pipeline

time python run_precision_harness.py \
    --manifest harness_configs.csv \
    --data data/era5_sampled_2021_4pm_aurora_0p25.nc \
    --checkpoint /vol/bitbucket/mes25/aurora_checkpoints/aurora-0.25-pretrained.ckpt \
    --outdir harness_results_2021 \
    --n-inits 12 \
    --steps 28 --leads 24,72,120,168 --score-lead 120

