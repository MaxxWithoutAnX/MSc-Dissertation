#!/bin/bash
#SBATCH --job-name=aurora-ablations-W8
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=10:00:00
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

export SAMPLED_YEAR=2020
export SAMPLED_PER_MONTH=4
export ABLATION_N_INITS=12   # 12 inits (one per month, seasonal spread); 0/unset = all 48

# --- Run ---
cd /vol/bitbucket/mes25/aurora_pipeline
time python ablations_W8.py
