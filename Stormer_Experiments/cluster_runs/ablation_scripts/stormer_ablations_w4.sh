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

# --- Config ---
export ABLATION_N_INITS=48   # 48 inits (~4/month, seasonally spread); 0 = all 12h-stride inits

# --- Run ---
cd /vol/bitbucket/mes25/stormer_pipeline
time python stormer_ablations_W4.py
