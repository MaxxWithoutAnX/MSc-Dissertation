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
# torch.compile diagnostics: show what Inductor is doing during the W8A8 compile
export TORCH_LOGS=recompiles

# --- Run ---
cd /vol/bitbucket/mes25/aurora_pipeline
time python model_sizes.py