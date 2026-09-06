#!/bin/bash
#SBATCH --job-name=stormer-ablations-w8a8sq
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=/vol/bitbucket/mes25/stormer_pipeline/logs/%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mes25@ic.ac.uk

# --- Time tracking ---
START=$(date +%s)
echo "=== START: $(date) ==="
trap 'echo "=== END:   $(date) ==="; echo "=== ELAPSED: $(($(date +%s) - START)) seconds ===" ' EXIT

# --- Shell setup ---
source /vol/bitbucket/mes25/miniconda3/etc/profile.d/conda.sh
conda activate climate
source /vol/cuda/12.4.0/setup.sh
export MPLBACKEND=Agg
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Config ---
export STORMER_H5_ROOT=/vol/bitbucket/mes25/wb2_h5df
export STORMER_SPLIT=era5_2020   # 2020 = ablation year; harness uses era5_2021
export ABLATION_N_INITS=48   # 48 inits (~4/month, seasonally spread); 0 = all 12h-stride inits
export SAMPLED_YEAR=2020     # recipe naming; must match the smoothquant inference run
export SAMPLED_PER_MONTH=4

# --- Run ---
cd /vol/bitbucket/mes25/stormer_pipeline

OUT_DIR="ablation_W8A8_smoothquant_medium_fullyear_${ABLATION_N_INITS}i"
SRC_DIR="ablation_W8A8_medium_fullyear_${ABLATION_N_INITS}i"
mkdir -p "$OUT_DIR"
cp -n "$SRC_DIR/fp32_metrics.pt" "$SRC_DIR/fp32_wind_preds.pt" "$OUT_DIR/" 2>/dev/null || true

time python stormer_ablations_W8A8_smoothquant.py
