#!/bin/bash
#SBATCH --job-name=nf-ensemble
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --output=/vol/bitbucket/mes25/aurora_pipeline/logs/%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mes25@ic.ac.uk

START=$(date +%s)
echo "=== START: $(date) ==="
trap 'echo "=== END: $(date)  ELAPSED $(($(date +%s) - START))s ==="' EXIT

source /vol/bitbucket/mes25/miniconda3/etc/profile.d/conda.sh
conda activate climate
source /vol/cuda/12.4.0/setup.sh
export MPLBACKEND=Agg

export SAMPLED_YEAR=2021      
export SAMPLED_PER_MONTH=4       
export NF_N_INITS=48              
export NF_EPS=1e-6                
export NF_OUT=noise_floor_ens

cd /vol/bitbucket/mes25/aurora_pipeline
mkdir -p "$NF_OUT"


EXISTING=$(ls -1 "$NF_OUT"/member_*.pt 2>/dev/null | head -1)
if [ -n "$EXISTING" ]; then
    python - "$EXISTING" "$NF_N_INITS" <<'PY' || exit 1
import sys, torch
from plot_common import MetricsRun
path, want = sys.argv[1], int(sys.argv[2])
d = torch.load(path, map_location="cpu", weights_only=False)
n = len(MetricsRun("m", d["metrics"]).dates)
print(f"existing-member check: {path} was built with {n} inits, want {want}", flush=True)
if n != want:
    sys.exit(f"ABORT: {path} has {n} inits but NF_N_INITS={want}. The loop skips by FILENAME,"
             f" so these would be silently reused. Delete {path.rsplit('/',1)[0]}/member_*.pt"
             f" (or point NF_OUT elsewhere) and resubmit.")
PY
fi

python - "$SAMPLED_YEAR" "$SAMPLED_PER_MONTH" "$NF_N_INITS" <<'PY' || exit 1
import json, sys
import numpy as np, xarray as xr
year, pm, want = sys.argv[1], sys.argv[2], int(sys.argv[3])
path = f"data/era5_sampled_{year}_{pm}pm_aurora_0p25.nc"
ds = xr.open_dataset(path)
times = ds.time.values
inits = [np.datetime64(s) for s in json.loads(ds.attrs["init_times"])]
found = [t for t in inits if len(np.where(times == t)[0])]
print(f"data check: {path}", flush=True)
print(f"  {len(times)} timesteps, {len(inits)} inits declared, {len(found)} resolvable",
      flush=True)
if len(found) < want:
    sys.exit(f"ABORT: only {len(found)} resolvable inits, need {want}.")
PY

FAILED=()
for SEED in $(seq 0 15); do
    MEMBER=$(printf "%s/member_%03d.pt" "$NF_OUT" "$SEED")
    if [ -f "$MEMBER" ]; then
        echo ">>> seed $SEED: $MEMBER already exists, skipping"
        continue
    fi
    echo
    echo ">>> seed $SEED  ($(date))  elapsed $(($(date +%s) - START))s"
    export SLURM_ARRAY_TASK_ID=$SEED       # the .py reads the seed from here
    if time python noise_floor_ensemble.py; then
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
echo "  all 16 members complete"

# Then aggregate on CPU (no GPU job needed). PASS --data EXPLICITLY: both scripts still
# default to the 1pm/12-init file, deliberately, so the n=12 dirs (harness_results_2021,
# harness_results_corrected) stay re-gateable against the floor they were built with.
#     python build_ensemble_floor.py --ens noise_floor_ens \
#            --data data/era5_sampled_2021_4pm_aurora_0p25.nc
# -> validation/noise_floor_detailed.pt. The n=48 harness ran null-floored, so re-gate it in
# place with no inference:
#     python rescore_with_floor.py --outdir harness_results_n48 \
#            --data data/era5_sampled_2021_4pm_aurora_0p25.nc
#
# ORDER MATTERS AGAINST run_precision_harness_n48_fill.sh. That script's guard 1 ABORTS if
# validation/noise_floor_detailed.pt exists, because its rows must be measured ungated to
# match the 17 ungated rows already in harness_results_n48. So: finish the fill run FIRST,
# then build the floor, then rescore all 28 rows in one pass. Building the floor while the
# fill job is still queued will block the resubmit.
