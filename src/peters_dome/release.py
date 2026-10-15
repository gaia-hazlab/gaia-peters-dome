"""USGS data release for Toney et al. (2026): doi:10.5066/P13WIS6X (ScienceBase item
6927572fd4be02716f722db3). Files are discovered from the ScienceBase JSON and read in
place over HTTP; nothing is written outside the container store.

These are PUBLISHED results, staged unmodified (no inversion or simulation is rerun here):
- ``model/force_published``: force-time function (component: up/north/east,
  bound: best/lower/upper); ``time`` in UTC = inversion zero time 2023-09-14T02:06:27Z
  + offset (release metadata)
- ``model/shaltop``: SHALTOP run 0520 thickness/speed on its native UTM 5N grid; ``time`` in
  UTC assuming simulation t=0 at the inversion zero time (NOT stated in the release;
  ``time_reference`` attr records this)
- ``rasters/topography/setsm_post_event``: post-event SETSM DEM, native UTM 5N
- ``vectors/landslide_outlines``: source and total (source + deposit) polygons
"""

from __future__ import annotations

import io as _io
import logging
import zipfile

import fsspec
import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import rioxarray
import xarray as xr

from . import common
from .params import STORE, Output, Overwrite, SkipExisting

log = logging.getLogger(__name__)

SB_ITEM = "https://www.sciencebase.gov/catalog/item/{}?format=json"
FORCE_T0 = "2023-09-14T02:06:27"  # release metadata, attribute "time (s)"
PUBLISHED = "published result from the USGS data release, staged unmodified (not recomputed here)"

GROUPS = {
    "force": "model/force_published",
    "shaltop": "model/shaltop",
    "dem_post": "rasters/topography/setsm_post_event",
    "outlines": "vectors/landslide_outlines",
}


def files() -> dict[str, str]:
    item = common.event()["references"]["sciencebase_item"]
    meta = requests.get(SB_ITEM.format(item), timeout=60).json()
    return {f["name"]: f["url"] for f in meta.get("files", [])}


def _pick(fs: dict[str, str], key: str) -> str:
    return next(url for name, url in fs.items() if key in name)


def load_force(src) -> xr.Dataset:
    df = pd.read_csv(src)
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
                                           "band": "15-200 s", "status": PUBLISHED})
    return da.to_dataset()


def load_shaltop(url: str) -> xr.Dataset:
    with fsspec.open(url, "rb") as f:
        ds = xr.open_dataset(_io.BytesIO(f.read()), engine="h5netcdf").load()
    ds = ds.rename({"time": "sim_time"})
    ds = ds.assign_coords(time=("sim_time", (pd.Timestamp(FORCE_T0) + pd.to_timedelta(ds.sim_time.values, unit="s")).values))
    ds = ds.drop_vars("spatial_ref", errors="ignore").rio.write_crs("EPSG:32605")
    ds.attrs.update(time_reference="ASSUMED: simulation t=0 == force inversion zero time; not stated in release",
                    status=PUBLISHED)
    return ds


def load_dem_post(url: str) -> xr.Dataset:
    # ScienceBase does not serve HTTP range requests, so /vsicurl fails: read into memory
    from rasterio.io import MemoryFile

    raw = requests.get(url, timeout=600).content
    with MemoryFile(raw) as mem, mem.open() as src:
        da = rioxarray.open_rasterio(src).load().squeeze("band", drop=True)
    crs = da.rio.crs
    da = da.where(da != da.rio.nodata).rename("elevation").rio.write_crs(crs)
    da.attrs.update(units="m", source="SETSM post-event DEM (USGS release), aligned, UTM 5N", status=PUBLISHED)
    return da.to_dataset()


def load_outlines(url: str) -> gpd.GeoDataFrame:
    raw = requests.get(url, timeout=120).content
    shp = next(n for n in zipfile.ZipFile(_io.BytesIO(raw)).namelist() if n.endswith(".shp"))
    gdf = _read_zip_shp(raw, shp)
    gdf["area_m2"] = gdf.to_crs("EPSG:32605").area
    return gdf


def _read_zip_shp(raw: bytes, shp: str) -> gpd.GeoDataFrame:
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        zipfile.ZipFile(_io.BytesIO(raw)).extractall(d)
        return gpd.read_file(f"{d}/{shp}")


def stage_release(output_path: Output = STORE, overwrite: Overwrite = False,
                  skip_existing: SkipExisting = False):
    """Stage the USGS data release (published force-time function, SHALTOP run,
    post-event DEM, outlines), unmodified, into four groups of the store."""
    todo = {k: g for k, g in GROUPS.items() if common.check_group(output_path, g, overwrite, skip_existing)}
    if not todo:
        return
    fs = files()
    prov = common.provenance(source="USGS data release doi:10.5066/P13WIS6X")
    loaders = {
        "force": lambda: load_force(_pick(fs, "jackknife")),
        "shaltop": lambda: load_shaltop(_pick(fs, "shaltop")),
        "dem_post": lambda: load_dem_post(_pick(fs, "setsm")),
        "outlines": lambda: common.gdf_to_xvec(load_outlines(_pick(fs, "dem_diff"))),
    }
    for key, group in todo.items():
        log.info("release: %s", key)
        ds = loaders[key]()
        ds.attrs.update(prov)
        common.write_group(ds, output_path, group, overwrite)
