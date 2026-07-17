import argparse
import asyncio
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError

import xarray as xr

# Fix zarr v3 + gcsfs deadlock on Windows ProactorEventLoop
if sys.platform == "win32":
    asyncio.set_event_loop(asyncio.SelectorEventLoop())

DOWNLOAD_TIMEOUT = 1800
MAX_RETRIES = 3

CLIM_ZARR = (
    "gs://weatherbench2/datasets/era5-hourly-climatology/"
    "1990-2019_6h_240x121_equiangular_with_poles_conservative.zarr"
)

SURFACE_VARS = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
]

# atmospheric_var -> levels we need for key_vars
LEVEL_VARS = {
    "geopotential":          [500],
    "temperature":           [850],
    "specific_humidity":     [700],
    "u_component_of_wind":   [500],
    "v_component_of_wind":   [500],
}


def write_netcdf(ds, out_path):
    if os.path.exists(out_path):
        os.remove(out_path)
    enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(out_path, encoding=enc)


def write_with_retry(ds, out_path):
    for attempt in range(1, MAX_RETRIES + 1):
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(write_netcdf, ds, out_path)
            try:
                future.result(timeout=DOWNLOAD_TIMEOUT)
                return
            except TimeoutError:
                print(f"  Timeout on {out_path} (attempt {attempt}/{MAX_RETRIES}), retrying...")
            except Exception as e:
                print(f"  Error on {out_path} (attempt {attempt}/{MAX_RETRIES}): {e}")
            if os.path.exists(out_path):
                os.remove(out_path)
            time.sleep(5)
    raise RuntimeError(f"Failed to download climatology after {MAX_RETRIES} attempts.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, required=True,
                        help="Output netCDF file path, e.g. clim_raw.nc")
    args = parser.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.save_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"Opening {CLIM_ZARR}", flush=True)
    ds = xr.open_zarr(CLIM_ZARR, storage_options={"token": "anon", "timeout": 60})
    print(f"Source dims: {dict(ds.sizes)}", flush=True)

    # Validate availability
    missing = [v for v in (SURFACE_VARS + list(LEVEL_VARS)) if v not in ds.data_vars]
    if missing:
        raise KeyError(f"Variables not found in zarr: {missing}")

    keep_vars = SURFACE_VARS + list(LEVEL_VARS.keys())
    sub = ds[keep_vars]

    # Slice the level dim to only the levels we actually need
    needed_levels = sorted({lvl for lvls in LEVEL_VARS.values() for lvl in lvls})
    if "level" in sub.dims:
        sub = sub.sel(level=needed_levels)
    elif "pressure_level" in sub.dims:
        sub = sub.sel(pressure_level=needed_levels)

    print(f"Subset dims: {dict(sub.sizes)}", flush=True)
    print(f"Subset vars: {list(sub.data_vars)}", flush=True)
    print(f"Writing compressed netCDF to {args.save_path}", flush=True)
    write_with_retry(sub, args.save_path)
    size_mb = os.path.getsize(args.save_path) / 1e6
    print(f"Done. File size: {size_mb:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
