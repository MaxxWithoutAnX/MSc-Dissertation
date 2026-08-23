#!/bin/bash

# --- Time tracking ---
START=$(date +%s)
echo "=== START: $(date) ==="
trap 'echo "=== END:   $(date) ==="; echo "=== ELAPSED: $(($(date +%s) - START)) seconds ===" ' EXIT

# --- Shell setup ---
source /vol/bitbucket/mes25/miniconda3/etc/profile.d/conda.sh
conda activate climate
source /vol/cuda/12.4.0/setup.sh
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

export SAMPLED_YEAR=2020
export SAMPLED_PER_MONTH=4

# --- Run ---
cd /vol/bitbucket/mes25/aurora_pipeline
time python inference_sampled_W8.py
