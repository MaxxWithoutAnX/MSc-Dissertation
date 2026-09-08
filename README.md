# Physics-Aware Evaluation of Post Training Quantisation in AI Weather Prediction Models

This repo contains the code for my Imperial College London MSc Artificial Intelligence dissertation titled: Physics-Aware Evaluation of Post Training Quantisation in AI Weather Prediction Models.


## Overview

This project looks into the effects of quantisation on two AI Weather models, Stormer and Aurora, and how RMSE might silently understate degradation errors. The project aimed to answer three research questions: 
1. Do Physics-Aware diagnostics detect quantisation damage that aggregate skill scores under report, and by how much?
2. How is quantisation damage in AI weather prediction models structured, and is it concentrated enough that selective protection is worthwhile?
3. Do Physics and RMSE sensitivity signals carry differing information and does the damage structure replicate across architecture?

The first question is answered through quantising each model at differing precisions (weight only 4- and 8- bit, 8-bit weight and dynamic activation, and 8-bit weight and activation with smoothquant), and comparing RMSE and physics based metrics to the full precision runs.

The second is answered through an ablation analysis, quantising one group at a time and comparing the damage between groups, relative to a full precision baseline.

The third question is answered through an allocator, developing a Pareto curve of predicted damage against quantisation cost for mixed precision models, assuming additivity between ablation results. This curve is then tested by running a select few mixed precision models and evaluating the differences between the real and predicted runs.


## Repository layout

```
dissertation-experiments/
├── requirements.txt            Combined cluster env (both models) - the env results were produced in
├── Aurora_Experiments/
│   ├── physq_core/             Shared analysis core: allocator, frontiers, stats, figure specs
│   ├── cluster_runs/           Job drivers + shell/SLURM scripts actually submitted on the cluster
│   ├── data_prep/              WeatherBench2 download / sampling
│   ├── precision_harness.py    Per-config quantised rollout -> metrics
│   ├── eval_metrics.py         RMSE, spectra, balance and conservation diagnostics
│   ├── make_figures.py         Dissertation-figure driver (registry in physq_core/figure_specs.py)
│   ├── harness_configs*.csv    Config manifests consumed by the harness. Should also contains *.pt from cluster runs but only CSV files pushed due to size
│   ├── frontiers/, plots/, figures/, ablation_analysis/   Derived outputs
│   └── harness_results_n48/    Measured harness runs (n=48 ensemble)
└── Stormer_Experiments/
    ├── (mirrors the Aurora layout; physq_core is shared via PHYSQ_CORE)
    ├── stormer_precision_harness.py
    ├── data_prep/              WB2 download, regrid, per-year HDF5 prep
    ├── harness_results_n47/, harness_results_adaln/
    └── stormer
```

Stormer's scripts are deliberate ports of Aurora's, not accidental duplication. The analysis core physq_core/ exists once, under Aurora_Experiments, and Stormer reaches it through the PHYSQ_CORE. The allocator, frontiers, statistics and figure registry are therefore shared code, not copies. eval_metrics.py, ablation_comp.py, build_cost_tables.py and plot_common.py are ports that differ where the architectures do, mainly in group definitions, variable names and grid shape. stormer/ is an external clone of https://github.com/tung-nd/stormer.git, excluded by .gitignore because it carries its own 34 MB git history; recreate it with the clone in the installation section above.


## Installation

```bash
conda create -n climate python=3.11.5
conda activate climate
pip install -r requirements.txt

# Stormer installs from a clone which should land in Stormer_Experiments/ specifically
git clone https://github.com/tung-nd/stormer.git Stormer_Experiments/stormer
cd Stormer_Experiments/stormer
git checkout 58dfee5
pip install -e .
cd ../..
```

Stormer (patch 2) requires an A30 GPU with a peak VRAM of 18.22 GB when running the full precision model whilst Aurora requires an A40 GPU with a peak VRAM of 24.57 GB when running full precision Autocast model. Aurora-small and Stormer (patch 4) can be run on consumer GPUs with a 12GB Geforce RTX 2060 tested although these models give smoother outputs and for that reason are not used in this dissertation, but can easily replace the larger models for local testing. A single one week rollout takes ~1.6 minutes for Aurora and ~43 seconds for Stormer on their respective hardware. Over 48 initialisation dates this becomes ~1.28 hours and ~34.4 minutes for Aurora and Stormer respectively.


## Data

```bash
# --- Aurora: sampled ERA5 at 0.25 deg, 48 inits/year (4/month, cycling 00Z/12Z) ---
# 2021 = mixed-precision harness; 2020 = ablations (which subsample to 12 inits).
for Y in 2021 2020; do
  python Aurora_Experiments/data_prep/wb2_download_samples.py \
      --year $Y --per-month 4 --hours 0 12 --leads 24 72 120 168 \
      --out-dir Aurora_Experiments/data
done
# Produces Aurora_Experiments/data/era5_sampled_{2021,2020}_4pm_aurora_0p25.nc

# --- Stormer: full-year ERA5 -> regrid to 1.40625 deg -> one HDF5 per 6 h step ---
python Stormer_Experiments/data_prep/download_wb2.py \
    --file 1959-2022-6h-512x256_equiangular_conservative.zarr \
    --save_dir data/wb2_nc --start_year 2020 --end_year 2021
python Stormer_Experiments/data_prep/regrid_wb2.py \
    --root_dir data/wb2_nc --save_dir data/wb2_nc_1p40625 \
    --ddeg_out 1.40625 --start_year 2020 --end_year 2021
python Stormer_Experiments/data_prep/prepare_year_h5.py \
    --root_dir data/wb2_nc_1p40625 --save_dir data/wb2_h5df \
    --year 2021 --split era5_2021
# After this regridding, you may want to delete the original files to save storage.

python Stormer_Experiments/data_prep/process_one_step_data.py \
    --root_dir data/wb2_nc_1p40625 --save_dir data/wb2_h5df \
    --start_year 2020 --end_year 2021 --split era5_2020
# Produces data/wb2_h5df/{era5_2021/2021_*.h5, era5_2020/2020_*.h5}

```

One date of ERA5 takes 171 MB at 0.25 degree resolution and 82 MB 1.40625 degrees. For one initialisation, Aurora requires 2 input dates 6 hours apart whilst Stormer requires one. 4 dates are also needed for scoring against ground truth at 24, 72, 120, and 168 hours lead time. This costs 1.03 GB on Aurora and 82 MB on Stormer. For a whole year at 4/month this becomes 49.3 GB and 3.9 GB for Aurora and Stormer respectively.

Expected on-disk layout once the commands above have run:

```
Aurora_Experiments/data/                     one self-contained NetCDF per year
├── era5_sampled_2021_4pm_aurora_0p25.nc     harness
└── era5_sampled_2020_4pm_aurora_0p25.nc     ablations
      each: 288 timesteps (48 inits x 6), 721 x 1440, 13 levels, 12 variables

data/wb2_nc/                                 raw WeatherBench2 download, 512 x 256
├── land_sea_mask.nc, geopotential_at_surface.nc, ...       13 constants, one file each
└── 2m_temperature/{2020,2021}.nc, ...                      28 variables, one dir each

data/wb2_nc_1p40625/                         same tree after regrid_wb2.py, 256 x 128

data/wb2_h5df/                               what the Stormer runs actually read
├── lat.npy, lon.npy                         128- and 256-element grid vectors
├── era5_2020/    2020_0000.h5 .. 2020_1463.h5    ablations (1464 files, leap year)
└── era5_2021/    2021_0000.h5 .. 2021_1459.h5    harness   (1460 files)
      one file per 6-hourly step; each field 128 x 256, float32
```

The index in `{year}_{idx}.h5` is the 6-hourly step since 1 Jan 00Z of that year, not a
counter, so a sparse store keeps the same filenames as a dense one.

All data is ERA5, taken from the public WeatherBench 2 buckets on Google Cloud. Aurora reads the native 0.25 degree store and Stormer the 512x256 conservative regrid, downsampled again to 1.40625 degrees to match its training grid.


## Reproducing the results

### 1. Cost tables

```bash
cd Aurora_Experiments   
python build_cost_tables.py                       # Produces cost_tables.pt
```

Generates "cost_tables.pt". This contains the cost to quantise a group, which is a set of Linear layers in the model
sharing a similar role and depth. This cost table is used later for allocatin and for comparisons between allocation metric axes.

### 2. OAT ablations (cluster)

```bash
# 2020 data. Aurora takes 12 inits, Stormer 48
sbatch cluster_runs/ablation_scripts/ablations.sh          # W4
sbatch cluster_runs/ablation_scripts/ablations_W8.sh
sbatch cluster_runs/ablation_scripts/ablations_W8A8.sh
sbatch cluster_runs/ablation_scripts/ablations_W8A8_smoothquant.sh 
# -> ablation_*/oat_results.pt (one group quantised at a time)
```

Ablations are run on 12 initialisation dates (1 per month) for Aurora and 48 initialisation dates (4 per month) for Stormer due to runtime. Aurora contains 19 groups whilst Stormer contains 11 as a result of the differing model sizes and architetures.

### 3. Noise floor (cluster, then aggregate)

```bash
sbatch cluster_runs/noise_floor_ens_n48/noise_floor_ensemble.sh   # Produces noise_floor_ens/member_*.pt
python build_ensemble_floor.py --ens noise_floor_ens     --out noise_floor_detailed.pt --quantile 0.9 --leads 24,72,120,168
```

Every ensemble member reruns the unquantised model on the same initialisations with inputs perturbed by a relative 1e-6, so members differ only by roundoff noise. The spread across them is the smallest change the pipeline can resolve, and a quantised run that moves a metric by less than that cannot be said to have moved it. The floor is the 0.9 quantile of that spread per metric, built on the same dates the harness scores: 48 for Aurora and 47 for Stormer, which drops the December init whose 168 hour lead falls past the end of 2021.

### 4. Ablation analysis

```bash
python ablation_comp.py --noise-floor noise_floor_detailed.pt
# reads ablation_*/oat_results.pt and produces ablation_analysis/ablations_*/sensitivity.csv
```

Creates sensitivity.csv scripts which later scripts build off. CSV file contains for every (group, metric, lead) the mean, SVR, and distortion values.

### 5. Frontiers and config selection

```bash
python regen_all_frontiers.py                     # Produces frontiers/corrected/*.csv
python physq_core/select_harness_configs.py       # Produces harness_configs*.csv
```

Generates a predicted frontier of model skill against cost, given the ablation results. Assumes additivity of distortion Generates frontier CSVs which are then used to pick models to run on the cluster (select_harness_configs.py) and compare results. Points are chosen by at equally spread points throughout the frontiers at matched costs.

### 6. Precision harness (cluster)

```bash
sbatch cluster_runs/run_precision_harness.sh
# or directly 
python cluster_runs/run_precision_harness.py \
    --manifest harness_configs.csv \
    --data data/era5_sampled_2021_4pm_aurora_0p25.nc \
    --checkpoint <aurora-0.25-pretrained.ckpt> \
    --outdir harness_results_n48 \
    --n-inits 48 --steps 28 --leads 24,72,120,168 --score-lead 120
```

Runs the chosen configurations on the cluster. One long script as cannot submit more than 3 at a time to Imperial DoC SLURM. --resume is used so restarting continues from the last configuration. Results written to harness_results.csv and more detailed in harness_results.pt holding the full per initialisation metrics.

### 7. Derived statistics

```bash
python regen_derived_stats.py --results harness_results_n48
cd ../Stormer_Experiments && python run_stormer_stats.py --outdir harness_results_n47 --lead 120
```

### 8. Figures

```bash
cd ../Aurora_Experiments && python make_figures.py    # Creates figures for both models in figures/
```



## Results

The experiemnts found that quantisation damage was highly localised in AIWP models (See Figs 03/A03). Selective protecion works as a result of this and a large protection of quantisation damage can be negated by just protecting a few layers for balance and RMSE of a model. The different metric axes (balance, RMSE, conservation) rank different groups damage differently and their rankings change under different quantisation schemes. However, a common heuristic of just protecting the input/output layers of the model, where it converts the physical variables into a latent space, achieves an extremely high reduction in damage. For full results, see the dissertation. (To be attatched after work has been graded to prevent any turnitin conflicts/issues).


## Notes and caveats

Both models use 2021 for the mixed-precision harness and 2020 for the ablations. The years are matched within each experiment, so the cross-model comparisons are like for like, but the ablation sensitivities and the harness measurements come from different weather and should not be pooled.

Aurora's checkpoint is aurora-0.25-pretrained, not the rollout-finetuned one, and the scoring is ERA5 against ERA5. Its short-lead RMSE is therefore not comparable to the published WeatherBench 2 numbers and the 24 hour column in particular should not be read as a skill result. Stormer replicates its published baseline within 5 percent.

FLOP counts in cost_tables.pt come from a shape-capture pass with a stubbed attention kernel (see build_cost_tables.py). They count Linear layers only, so attention's own matmuls, normalisation, and elementwise work are excluded. The tables are a relative cost proxy for allocation, not an absolute FLOP budget.

stormer_harness_configs.csv declares 38 configurations but harness_results_n47 holds 28. The 10 missing are all W8A8_sq_*, so the selector emits those rows but were never run. Aurora measured 38, recorded in frontiers/provenance.csv.

Generative AI was used to help debug scripts, create graphs, and for assisting in checking correct functionality of the written code. Generative AI was not used to develop the methodology or come up with the design itself but to support and stress test work done by the author. Generative AI was also used in helping write .sh scripts.


## Citation and acknowledgements

This work evaluates two published models, Aurora and Stormer, using their published checkpoints. No weights were retrained or fine-tuned. Aurora is used through the microsoft-aurora package at version 1.8.0 with the aurora-0.25-pretrained checkpoint. Stormer is used from https://github.com/tung-nd/stormer at commit 58dfee5 with the 1.40625 degree patch-size-2 checkpoint. Cluster compute was provided by the Department of Computing at Imperial College London.
