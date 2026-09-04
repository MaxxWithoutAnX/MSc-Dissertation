""" Download WB2 data. Adapted from Stormer Github
"""
import argparse
import asyncio
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError

import xarray as xr
from tqdm import tqdm

# Fix zarr v3 + gcsfs deadlock on Windows ProactorEventLoop
if sys.platform == "win32":
    asyncio.set_event_loop(asyncio.SelectorEventLoop())

DOWNLOAD_TIMEOUT = 300  # seconds before a stalled download is cancelled
MAX_RETRIES = 3
MIN_FILE_BYTES = 1_000_000  # files under 1MB are treated as corrupt/incomplete


def download_year(ds_var, year, out_path):
    if os.path.exists(out_path):
        os.remove(out_path)
    ds_var.sel(time=str(year)).to_netcdf(out_path)


def download_with_retry(ds_var, year, out_path):
    for attempt in range(1, MAX_RETRIES + 1):
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(download_year, ds_var, year, out_path)
            try:
                future.result(timeout=DOWNLOAD_TIMEOUT)
                return
            except TimeoutError:
                tqdm.write(f"  Timeout on {os.path.basename(out_path)} (attempt {attempt}/{MAX_RETRIES}), retrying...")
            except Exception as e:
                tqdm.write(f"  Error on {os.path.basename(out_path)} (attempt {attempt}/{MAX_RETRIES}): {e}")
            if os.path.exists(out_path):
                os.remove(out_path)
            time.sleep(5)
    tqdm.write(f"  Skipping {out_path} after {MAX_RETRIES} failed attempts.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--start_year", type=int, default=2020,
                        help="First year to download (inclusive).")
    parser.add_argument("--end_year", type=int, default=2021,
                        help="Last year to download (INCLUSIVE, as in regrid_wb2.py).")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    ds = xr.open_zarr(
        'gs://weatherbench2/datasets/era5/' + args.file,
        storage_options={"token": "anon", "timeout": 60},
    )

    years = list(range(args.start_year, args.end_year + 1))
    variables = list(ds.keys())

    for var in tqdm(variables, desc="variables", position=0):
        ds_var = ds[[var]]
        if len(ds_var.dims) < 3:  # constant variables
            out_path = os.path.join(args.save_dir, f'{var}.nc')
            if os.path.exists(out_path):
                continue
            ds_var.to_netcdf(out_path)
        else:
            save_dir_var = os.path.join(args.save_dir, var)
            os.makedirs(save_dir_var, exist_ok=True)
            for year in tqdm(years, desc="years", position=1, leave=False):
                out_path = os.path.join(save_dir_var, f'{year}.nc')
                if os.path.exists(out_path) and os.path.getsize(out_path) >= MIN_FILE_BYTES:
                    continue
                download_with_retry(ds_var, year, out_path)


if __name__ == "__main__":
    main()
