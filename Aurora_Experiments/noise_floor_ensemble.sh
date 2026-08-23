#!/bin/bash

START=$(date +%s)
echo "=== START: $(date)  member=$SLURM_ARRAY_TASK_ID ==="
trap 'echo "=== END: $(date)  ELAPSED $(($(date +%s) - START))s ==="' EXIT

source /vol/bitbucket/mes25/miniconda3/etc/profile.d/conda.sh
conda activate climate
source /vol/cuda/12.4.0/setup.sh
export MPLBACKEND=Agg

export SAMPLED_YEAR=2021          # the harness scoring year (floor must match its leads)
export SAMPLED_PER_MONTH=4
export NF_N_INITS=12              # 1/month seasonal spread; enough for a run-to-run spread
export NF_EPS=1e-6               # fp32 round-off scale
export NF_OUT=noise_floor_ens

cd /vol/bitbucket/mes25/aurora_pipeline
time python noise_floor_ensemble.py

