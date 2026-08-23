#!/bin/bash

START=$(date +%s)
echo "=== START: $(date) ==="
trap 'echo "=== END: $(date)  ELAPSED $(($(date +%s) - START))s ==="' EXIT

source /vol/bitbucket/mes25/miniconda3/etc/profile.d/conda.sh
conda activate climate
source /vol/cuda/12.4.0/setup.sh
export MPLBACKEND=Agg

cd /vol/bitbucket/mes25/aurora_pipeline
time python preflight_mixed_compile.py \
    --data data/era5_sampled_2021_4pm_aurora_0p25.nc \
    --checkpoint /vol/bitbucket/mes25/aurora_checkpoints/aurora-0.25-pretrained.ckpt
