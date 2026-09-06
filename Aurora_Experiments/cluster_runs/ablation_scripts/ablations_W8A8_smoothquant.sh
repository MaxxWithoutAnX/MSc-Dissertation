#!/bin/bash
#SBATCH --job-name=aurora-ablations-W8A8sq
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=/vol/bitbucket/mes25/aurora_pipeline/logs/%x-%j.out
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

export SAMPLED_YEAR=2020
export SAMPLED_PER_MONTH=4
export ABLATION_N_INITS=12   # 12 inits (one per month, seasonal spread); 0/unset = all 48

# --- Run ---
cd /vol/bitbucket/mes25/aurora_pipeline

OUT_DIR="ablation_W8A8_smoothquant_medium_sampled_${SAMPLED_YEAR}_${SAMPLED_PER_MONTH}pm"
SRC_DIR="ablation_W8_medium_sampled_${SAMPLED_YEAR}_${SAMPLED_PER_MONTH}pm"
mkdir -p "$OUT_DIR"
cp -n "$SRC_DIR/fp32_metrics.pt" "$SRC_DIR/fp32_wind_preds.pt" "$OUT_DIR/" 2>/dev/null || true

time python ablations_W8A8_smoothquant.py
