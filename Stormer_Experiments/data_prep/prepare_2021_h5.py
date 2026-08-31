"""Materialise one year into the wb2_h5df h5 layout, trimmed to Stormer's 69 variables."""
import argparse
import os

import numpy as np

CONSTANT_VARS = ["land_sea_mask"]          # only needed so create_one_step_dataset can
                                           # source lat/lon; never fed to the model
SINGLE_VARS = ["2m_temperature", "10m_u_component_of_wind",
               "10m_v_component_of_wind", "mean_sea_level_pressure"]
PRESSURE_VARS = ["geopotential", "u_component_of_wind", "v_component_of_wind",
                 "temperature", "specific_humidity"]

VARS_69 = CONSTANT_VARS + SINGLE_VARS + PRESSURE_VARS

_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

MODEL_VARIABLES = SINGLE_VARS + [
    f"{v}_{lev}" for v in PRESSURE_VARS for lev in _LEVELS]


def year_list(year):
    return [year]


def build_arg_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root_dir", required=True, help="regridded 1.40625 NetCDF root")
    ap.add_argument("--save_dir", required=True, help="wb2_h5df root")
    ap.add_argument("--year", type=int, default=2021)
    ap.add_argument("--split", default="test_2021")
    ap.add_argument("--chunk_size", type=int, default=10)
    return ap


def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    from process_one_step_data import create_one_step_dataset

    if args.split == "test":
        raise SystemExit(
            "--split test would write into the 2020 OAT directory. "
            "ERA5MultiLeadtimeDataset globs its whole root, so mixing years changes "
            "len(dataset) and every index. Use --split test_2021.")

    lat_path = os.path.join(args.save_dir, "lat.npy")
    lat_before = np.load(lat_path) if os.path.exists(lat_path) else None

    create_one_step_dataset(
        root_dir=args.root_dir,
        save_dir=args.save_dir,
        split=args.split,
        years=year_list(args.year),
        list_vars=VARS_69,
        chunk_size=args.chunk_size,
    )

    if lat_before is not None:
        lat_after = np.load(lat_path)
        if not np.array_equal(lat_before, lat_after):
            raise RuntimeError(
                f"{lat_path} CHANGED. Every stored metric was computed against the old "
                f"latitude vector; lat-weighted RMSE and the balance metrics are not "
                f"comparable across this change. Restore lat.npy and investigate.")

    out = os.path.join(args.save_dir, args.split)
    n = len([f for f in os.listdir(out) if f.endswith(".h5")])
    print(f"wrote {n} files to {out}", flush=True)


if __name__ == "__main__":
    main()
