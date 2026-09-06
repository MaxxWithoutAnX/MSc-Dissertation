#!/bin/bash
#SBATCH --job-name=aurora-model-size-check
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:15:00
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
# torch.compile diagnostics: show what Inductor is doing during the W8A8 compile
export TORCH_LOGS=recompiles

# --- Run ---
cd /vol/bitbucket/mes25/aurora_pipeline
time python model_sizes.py