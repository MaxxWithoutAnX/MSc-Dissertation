#!/bin/bash

set -euo pipefail

START=$(date +%s)
echo "=== START: $(date) ==="
trap 'echo "=== END:   $(date) ==="; echo "=== ELAPSED: $(($(date +%s) - START)) seconds ==="' EXIT

# --- Shell setup ---
source /vol/bitbucket/mes25/miniconda3/etc/profile.d/conda.sh
conda activate climate
export MPLBACKEND=Agg

export PHYSQ_CORE=/vol/bitbucket/mes25/physq_core
export STORMER_CKPT=/vol/bitbucket/mes25/stormer_checkpoints/stormer_1.40625_patch_size_2.ckpt

cd /vol/bitbucket/mes25/stormer_pipeline

# --- Preflight: fail early and legibly on anything missing ---
python - <<'PY'
import os
import sys
missing = []
for p in (os.environ["PHYSQ_CORE"],
          "ablation_analysis_ablation_W4/sensitivity.csv",
          "ablation_analysis_ablation_W8/sensitivity.csv",
          "ablation_analysis_ablation_W8A8/sensitivity.csv",
          "ablation_analysis_ablations_W8A8_sq/sensitivity.csv",
          "ablation_comp.py", "plot_common.py", "stormer_groups.py", "physq_path.py"):
    if not os.path.exists(p):
        missing.append(p)
if missing:
    sys.exit("MISSING (stage these first):\n  " + "\n  ".join(missing))
print("preflight OK", flush=True)
PY

echo "--- build_cost_tables ---"
time python build_cost_tables.py

echo "--- run_stormer_frontiers ---"
time python run_stormer_frontiers.py

echo "--- select_stormer_configs ---"
time python select_stormer_configs.py

echo "--- artifacts ---"
ls -la cost_tables.pt stormer_harness_configs.csv
ls -la frontier_stormer/
