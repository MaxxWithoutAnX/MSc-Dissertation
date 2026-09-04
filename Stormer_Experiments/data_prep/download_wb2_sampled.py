"""Download only the ERA5 timesteps a sampled Stormer run actually reads. 

    save_dir/
      land_sea_mask.nc, geopotential_at_surface.nc, ...     (constants, full field)
      2m_temperature/2021.nc                                (235 timesteps, not 1460)
      geopotential/2021.nc
      ...

    python download_wb2_sampled.py --save_dir /vol/bitbucket/mes25/wb2_nc_sampled

Inits whose longest lead would fall past the end of the year are DROPPED, not silently
truncated. get_out_path's cross-year fallback infers the year boundary by walking forward
until a file is missing, which returns the wrong file on a sparse store.

REQUIRES the timestamp-derived index in process_one_step_data.create_one_step_dataset. Its
original `idx_in_year += 1` counter numbers output files sequentially, which on sparse input
produces filenames whose index no longer encodes the time. file_to_datetime and get_out_path
both decode time FROM the filename, so that silently shifts every init and every lead target.
"""
import argparse
import calendar
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

ZARR = ("gs://weatherbench2/datasets/era5/"
        "1959-2022-6h-512x256_equiangular_conservative.zarr")

# The 9 source variables that expand to Stormer's 69 fields: 4 surface + 5 pressure-level
# across the 13 levels in DEFAULT_PRESSURE_LEVELS.
KEEP = ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind",
        "mean_sea_level_pressure", "geopotential", "u_component_of_wind",
        "v_component_of_wind", "temperature", "specific_humidity"]

DATA_FREQ = 6


def sample_init_times(year, per_month, hours):
    """Identical to sample_init_times in the inference scripts. Keep them in sync: the
    downloaded set and the set the run asks for must agree exactly, or inits go missing
    at run time with only a WARN line."""
    inits = []
    for month in range(1, 13):
        ndays = calendar.monthrange(year, month)[1]
        days = sorted({int(round((k + 0.5) * ndays / per_month)) for k in range(per_month)})
        for k, day in enumerate(days):
            inits.append(datetime(year, month, day, hours[k % len(hours)]))
    return sorted(inits)


def select(year, per_month, hours, leads):
    """-> (kept_inits, dropped_inits, sorted_needed_indices).

    Index is 6-hourly steps since Jan 1 00Z of `year`, which is exactly what the
    {year}_{idx:04}.h5 filename encodes."""
    n_steps = (366 if calendar.isleap(year) else 365) * 24 // DATA_FREQ
    def idx(t):
        return int((t - datetime(year, 1, 1)).total_seconds() // (DATA_FREQ * 3600))

    kept, dropped = [], []
    for t in sample_init_times(year, per_month, hours):
        (kept if idx(t) + max(leads) // DATA_FREQ <= n_steps - 1 else dropped).append(t)
    need = sorted({idx(t) + lead // DATA_FREQ for t in kept for lead in [0] + leads})
    return kept, dropped, need


def build_arg_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--save_dir", default="/vol/bitbucket/mes25/wb2_nc_sampled")
    ap.add_argument("--file", default=ZARR.rsplit("/", 1)[1])
    ap.add_argument("--year", type=int, default=2021)
    ap.add_argument("--per-month", type=int, default=4)
    ap.add_argument("--hours", type=int, nargs="+", default=[0, 12])
    ap.add_argument("--leads", type=int, nargs="+", default=[24, 72, 120, 168])
    ap.add_argument("--dry-run", action="store_true",
                    help="print the selection and its size, download nothing")
    return ap


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    kept, dropped, need = select(args.year, args.per_month, args.hours, args.leads)
    times = [datetime(args.year, 1, 1) + timedelta(hours=DATA_FREQ * i) for i in need]

    print(f"year {args.year}  per_month {args.per_month}  hours {args.hours}  "
          f"leads {args.leads}", flush=True)
    print(f"inits kept    : {len(kept)}", flush=True)
    for t in dropped:
        print(f"  DROPPED init {t}  (longest lead falls past {args.year}-12-31)", flush=True)
    print(f"distinct steps: {len(need)}  (index {need[0]}..{need[-1]} of "
          f"{(366 if calendar.isleap(args.year) else 365)*4 - 1})", flush=True)
    n_fields = 4 + 5 * 13
    print(f"download      : {len(need)*n_fields*512*256*4/1e9:.1f} GB", flush=True)
    if args.dry_run:
        return

    os.makedirs(args.save_dir, exist_ok=True)
    ds = xr.open_zarr("gs://weatherbench2/datasets/era5/" + args.file,
                      storage_options={"token": "anon"})
    absent = [v for v in KEEP if v not in ds]
    if absent:
        raise SystemExit(f"variables absent from {args.file}: {absent}")


    want = pd.DatetimeIndex(times)
    missing = want.difference(pd.DatetimeIndex(ds.time.values))
    if len(missing):
        raise SystemExit(f"{len(missing)} required timesteps absent from the store, "
                         f"earliest {missing[0]}")
    sel = want.values


    consts = [v for v in ds.data_vars if len(ds[v].dims) < 3]
    for v in tqdm(consts, desc="constants"):
        out = os.path.join(args.save_dir, f"{v}.nc")
        if not os.path.exists(out):
            ds[[v]].to_netcdf(out)

    for v in tqdm(KEEP, desc="variables"):
        vdir = os.path.join(args.save_dir, v)
        os.makedirs(vdir, exist_ok=True)
        out = os.path.join(vdir, f"{args.year}.nc")
        if os.path.exists(out):      # resumable; delete a truncated file before rerunning
            continue
        tmp = out + ".part"
        ds[[v]].sel(time=sel).to_netcdf(tmp)
        os.replace(tmp, out)         # rename only after a complete write

    print(f"wrote {len(KEEP)} variables x {len(need)} steps + {len(consts)} constants "
          f"to {args.save_dir}", flush=True)


if __name__ == "__main__":
    main()
