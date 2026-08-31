#!/bin/bash
#SBATCH --job-name=stormer-nf-ensemble
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00
#SBATCH --output=/vol/bitbucket/mes25/stormer_pipeline/logs/%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mes25@ic.ac.uk

START=$(date +%s)
echo "=== START: $(date) ==="
trap 'echo "=== END: $(date)  ELAPSED $(($(date +%s) - START))s ==="' EXIT

source /vol/bitbucket/mes25/miniconda3/etc/profile.d/conda.sh
conda activate climate
source /vol/cuda/12.4.0/setup.sh
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1               # SLURM logs are block-buffered otherwise
export PHYSQ_CORE=/vol/bitbucket/mes25/physq_core

export NF_SPLIT=test_2021               # the harness scoring year (floor must match its inits)
export NF_N_INITS=47                    # == run_stormer_precision_harness_n47.sh
export NF_LEADS=24,72,120,168           # == the harness's leads
export NF_EPS=1e-6                      # fp32 round-off scale
export NF_OUT=noise_floor_ens_n47
export NF_COMPILE=0
MEMBERS=${NF_MEMBERS:-16}               # seeds 0..15, matching Aurora's 1 ref + 15 perturbed

cd /vol/bitbucket/mes25/stormer_pipeline
mkdir -p logs "$NF_OUT"


FAILED=()
for SEED in $(seq 0 $((MEMBERS - 1))); do
    MEMBER=$(printf "%s/member_%03d.pt" "$NF_OUT" "$SEED")
    if [ -f "$MEMBER" ]; then
        echo ">>> seed $SEED: $MEMBER already exists, skipping"
        continue
    fi
    echo
    echo ">>> seed $SEED  ($(date))  elapsed $(($(date +%s) - START))s"
    export SLURM_ARRAY_TASK_ID=$SEED       # the .py reads the seed from here
    if time python noise_floor_ens_n47emble.py; then
        echo ">>> seed $SEED: OK"
    else
        echo ">>> seed $SEED: FAILED (continuing with the remaining members)"
        FAILED+=("$SEED")
    fi
done

echo
echo "=== ENSEMBLE SUMMARY ==="
ls -1 "$NF_OUT"/member_*.pt 2>/dev/null | wc -l | xargs echo "  members present:"
if [ ${#FAILED[@]} -ne 0 ]; then
    echo "  FAILED seeds: ${FAILED[*]}"
    echo "  resubmit this script to retry them (completed members are skipped)"
    exit 1
fi
echo "  all $MEMBERS members complete"

