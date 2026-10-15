"""USGS data release for Toney et al. (2026): doi:10.5066/P13WIS6X (ScienceBase item
6927572fd4be02716f722db3). Files are discovered from the ScienceBase JSON, so renamed
download URLs do not break the script.

Harmonized outputs (DataTree groups):
- ``force``: force-time function (component: up/north/east, bound: best/lower/upper),
  ``time`` in UTC = inversion zero time 2023-09-14T02:06:27Z + offset (release metadata)
- ``shaltop``: SHALTOP run 0520 thickness/speed, ``time`` in UTC assuming simulation t=0 at
  the inversion zero time (NOT stated in the release; flag ``time_reference`` records this)
- ``dem_post``: post-event SETSM DEM, UTM 5N
- ``outlines``: source and total (source+deposit) polygons
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import rioxarray
import xarray as xr
from cyclopts import Parameter

from . import common

SB_ITEM = "https://www.sciencebase.gov/catalog/item/{}?format=json"
FORCE_T0 = "2023-09-14T02:06:27"  # release metadata, attribute "time (s)"


def download(raw_dir: str = "data/raw/usgs_release") -> Path:
    out = Path(raw_dir)
    out.mkdir(parents=True, exist_ok=True)
    item = common.event()["references"]["sciencebase_item"]
    meta = requests.get(SB_ITEM.format(item), timeout=60).json()
    for f in meta.get("files", []):
        dest = out / f["name"]
        if dest.exists() and dest.stat().st_size == f["size"]:
            continue
        print(f"  downloading {f['name']} ({f['size']/1e6:.1f} MB)")
        with requests.get(f["url"], stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(dest, "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
    return out


def load_force(path: Path) -> xr.Dataset:
    df = pd.read_csv(path)
    t = pd.Timestamp(FORCE_T0) + pd.to_timedelta(df["time (s)"], unit="s")
    comps, bounds = ["up", "north", "east"], ["best", "lower", "upper"]
    arr = np.empty((len(comps), len(bounds), len(df)))
    for i, c in enumerate(comps):
        arr[i, 0] = df[f"{c} force (N)"]
        arr[i, 1] = df[f"{c} force, lower bound (N)"]
        arr[i, 2] = df[f"{c} force, upper bound (N)"]
    da = xr.DataArray(arr, dims=("component", "bound", "time"),
                      coords={"component": comps, "bound": bounds, "time": t.values,
                              "time_offset_s": ("time", df["time (s)"].to_numpy())},
                      name="force", attrs={"units": "N", "zero_time_utc": FORCE_T0,
                                           "bounds": "jackknife (Toney et al. 2026, sec 3.4)",
                                           "band": "15-200 s"})
    return da.to_dataset()


def load_shaltop(path: Path) -> xr.Dataset:
    ds = xr.open_dataset(path).load()
    ds = ds.rename({"time": "sim_time"})
    ds = ds.assign_coords(time=("sim_time", (pd.Timestamp(FORCE_T0) + pd.to_timedelta(ds.sim_time.values, unit="s")).values))
    ds = ds.rio.write_crs("EPSG:32605")
    ds.attrs["time_reference"] = "ASSUMED: simulation t=0 == force inversion zero time; not stated in release"
    return ds.drop_vars("spatial_ref", errors="ignore").rio.write_crs("EPSG:32605")


def load_dem_post(path: Path) -> xr.DataArray:
    da = rioxarray.open_rasterio(path).squeeze("band", drop=True)
    da = da.where(da != da.rio.nodata).rename("elevation")
    da.attrs.update(units="m", source="SETSM post-event DEM (USGS release), aligned, UTM 5N")
    return da


def load_outlines(zip_path: Path) -> gpd.GeoDataFrame:
    import zipfile

    shp = [n for n in zipfile.ZipFile(zip_path).namelist() if n.endswith(".shp")][0]
    gdf = gpd.read_file(f"/vsizip/{zip_path.resolve()}/{shp}")
    gdf["area_m2"] = gdf.to_crs("EPSG:32605").area
    return gdf


def load_all(raw_dir: str = "data/raw/usgs_release") -> dict[str, xr.Dataset]:
    d = download(raw_dir)
    find = lambda pat: next(d.glob(pat))  # noqa: E731
    outlines = load_outlines(find("dem_diff*.zip"))
    Path("data/staged").mkdir(parents=True, exist_ok=True)
    outlines.to_file("data/staged/release_outlines.geojson", driver="GeoJSON")
    return {
        "force": load_force(find("*jackknife*.csv")),
        "shaltop": load_shaltop(find("shaltop*.nc")),
        "dem_post": load_dem_post(find("setsm*.tif")).to_dataset(),
        "outlines": common.gdf_to_xvec(outlines),
    }


def stage_release(output_path: Annotated[str, Parameter(name=["--output", "-o"])] = "data/staged/usgs_release.zarr",
                  raw_dir: Annotated[str, Parameter(name=["--raw-dir"])] = "data/raw/usgs_release"):
    """Download the USGS data release (force, SHALTOP, post-event DEM, outlines) and stage it."""
    common.check_s3_env(output_path)
    tree = load_all(raw_dir)
    for ds in tree.values():
        ds.attrs.update(common.provenance(source="USGS data release doi:10.5066/P13WIS6X"))
    common.save_datatree(xr.DataTree.from_dict(tree), output_path)
