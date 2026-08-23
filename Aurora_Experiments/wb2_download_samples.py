"""Download a sampled ERA5 (0.25 deg) subset for Aurora from WeatherBench2."""
from __future__ import annotations

import argparse
import asyncio
import calendar
import json
import sys
from pathlib import Path

import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar

# Fix zarr v3 + gcsfs deadlock/slowdown on Windows ProactorEventLoop.
if sys.platform == "win32":
    asyncio.set_event_loop(asyncio.SelectorEventLoop())

# Public WeatherBench2 ERA5 store: 0.25 deg, hourly, 37 pressure levels.
WB2_ERA5_URL = (
    "gs://weatherbench2/datasets/era5/"
    "1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr"
)

# Variables required to build an Aurora `Batch`.
SURFACE_VARS = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
]
STATIC_VARS = [
    "land_sea_mask",
    "soil_type",
    "geopotential_at_surface",
]
ATMOS_VARS = [
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
    "geopotential",
]

# The 13 pressure levels (hPa) Aurora is trained on.
PRESSURE_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

# Default forecast lead times (hours) to score.
DEFAULT_LEADS = [24, 72, 120, 168]


def open_era5() -> xr.Dataset:
    """Open the WeatherBench2 ERA5 zarr lazily (no data transferred yet)."""
    return xr.open_zarr(
        WB2_ERA5_URL,
        storage_options={"token": "anon", "timeout": 60},
        chunks={},  # keep native chunking -> dask-backed, streams on write
        decode_timedelta=True,
    )


def sample_init_times(year: int, per_month: int, hours: list[int]) -> pd.DatetimeIndex:
    inits = []
    for month in range(1, 13):
        ndays = calendar.monthrange(year, month)[1]
        days = sorted({int(round((k + 0.5) * ndays / per_month)) for k in range(per_month)})
        for k, day in enumerate(days):
            hour = hours[k % len(hours)]
            inits.append(pd.Timestamp(year=year, month=month, day=day, hour=hour))
    return pd.DatetimeIndex(sorted(inits))


def required_times(inits: pd.DatetimeIndex, leads: list[int]) -> pd.DatetimeIndex:
    times: set[pd.Timestamp] = set()
    for t0 in inits:
        times.add(t0 - pd.Timedelta(hours=6))
        times.add(t0)
        for lead in leads:
            times.add(t0 + pd.Timedelta(hours=lead))
    return pd.DatetimeIndex(sorted(times))


def select_sampled(
    ds: xr.Dataset, inits: pd.DatetimeIndex, leads: list[int]
) -> xr.Dataset:
    """Subset the store to exactly the vars, levels and (sparse) times needed."""
    keep = [v for v in SURFACE_VARS + STATIC_VARS + ATMOS_VARS if v in ds]
    missing = set(SURFACE_VARS + STATIC_VARS + ATMOS_VARS) - set(keep)
    if missing:
        raise KeyError(f"Variables not found in store: {sorted(missing)}")

    ds = ds[keep].sel(level=PRESSURE_LEVELS)
    ds = ds.sel(time=required_times(inits, leads))

    ds.attrs["init_times"] = json.dumps([t.isoformat() for t in inits])
    ds.attrs["leads_hours"] = json.dumps(list(leads))
    return ds


def estimate_size_gb(ds: xr.Dataset) -> float:
    """Rough uncompressed size of the selection in GB."""
    return ds.nbytes / 1024**3


def time_chunked_encoding(ds: xr.Dataset) -> dict:
    """Per-variable encoding: chunk `time` at 1, span every other axis fully."""
    encoding = {}
    for var in ds.data_vars:
        chunks = tuple(1 if d == "time" else ds.sizes[d] for d in ds[var].dims)
        encoding[var] = {"zlib": True, "complevel": 1, "chunksizes": chunks}
    return encoding


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2020)
    parser.add_argument("--per-month", type=int, default=4)
    parser.add_argument("--hours", type=int, nargs="+", default=[0, 12],
                        help="Init hours to cycle through across each month's picks.")
    parser.add_argument("--leads", type=int, nargs="+", default=DEFAULT_LEADS,
                        help="Forecast lead times (h) whose valid times must be present.")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selection and size estimate without downloading.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    inits = sample_init_times(args.year, args.per_month, args.hours)
    times = required_times(inits, args.leads)

    print(f"Opening {WB2_ERA5_URL} ...")
    ds = open_era5()
    subset = select_sampled(ds, inits, args.leads)

    n_inits = len(inits)
    n_times = subset.sizes["time"]
    size_gb = estimate_size_gb(subset)
    out_path = args.out_dir / f"era5_sampled_{args.year}_{args.per_month}pm_aurora_0p25.nc"

    print(
        f"\n{args.year}: {n_inits} inits ({args.per_month}/month, hours {args.hours}), "
        f"leads {args.leads}\n"
        f"  -> {n_times} unique timesteps, ~{size_gb:.1f} GB uncompressed -> {out_path}"
    )
    print(f"  inits[:6]: {[t.isoformat() for t in inits[:6]]}")

    if args.dry_run:
        # Sanity-check that every init's history and truth valid-times are present.
        tset = set(times)
        ok = all(
            (t0 - pd.Timedelta(hours=6)) in tset
            and t0 in tset
            and all((t0 + pd.Timedelta(hours=L)) in tset for L in args.leads)
            for t0 in inits
        )
        print(f"  all init history+truth valid-times present in selection: {ok}")
        print("\nDry run complete. Re-run without --dry-run to download.")
        return

    with ProgressBar():
        subset.to_netcdf(out_path, engine="netcdf4", encoding=time_chunked_encoding(subset))
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
