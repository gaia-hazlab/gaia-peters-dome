"""SAR, optical and DEM rasters from public STAC APIs.

High-resolution rasters (<= 30 m) are staged in UTM 5N (EPSG:32605) rather than the
gaia-cli default EPSG:4326: at 63 N a degree of longitude is ~2.2x shorter than a degree
of latitude, and the USGS release DEM is already UTM 5N. ``build_datacube.py`` puts
everything on one grid.
"""

from __future__ import annotations

from typing import Annotated

import geopandas as gpd
import numpy as np
import odc.stac
import planetary_computer
import pystac_client
import xarray as xr
from cyclopts import Parameter

from . import common

odc.stac.configure_rio(cloud_defaults=True)

PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
EARTH_SEARCH = "https://earth-search.aws.element84.com/v1"
PGC_STAC = "https://stac.pgc.umn.edu/api/v1"


def _search(api, collections, aoi, start, end, query=None, sign=False):
    client = pystac_client.Client.open(
        api, modifier=planetary_computer.sign_inplace if sign else None
    )
    search = client.search(
        collections=collections,
        intersects=aoi.to_crs("EPSG:4326").geometry.union_all(),
        datetime=f"{start}/{end}",
        query=query,
    )
    items = list(search.items())
    print(f"  {collections}: {len(items)} items {start}..{end}")
    return items


def _load(items, bands, aoi, resolution, crs=None, groupby="solar_day", **kw):
    crs = crs or common.utm_crs()
    ds = odc.stac.load(
        items,
        bands=bands,
        crs=crs,
        resolution=resolution,
        bbox=tuple(aoi.to_crs("EPSG:4326").total_bounds),
        groupby=groupby,
        chunks={},
        **kw,
    )
    return common.clip_projected(ds, aoi)


# ---- SAR ---------------------------------------------------------------------------------

def load_sar(aoi: gpd.GeoDataFrame, start_date: str, end_date: str, resolution: float = 10.0) -> xr.Dataset:
    """Sentinel-1 radiometrically terrain-corrected gamma0 (VV, VH), linear power.

    Source: Planetary Computer ``sentinel-1-rtc``. In Aug-Oct 2023 only Sentinel-1A flew
    (12-day repeat); the event lies between two passes of relative orbit 131 descending
    (2023-09-09 and 2023-09-21, checked 2026-10-14)."""
    items = _search(PC_STAC, ["sentinel-1-rtc"], aoi, start_date, end_date, sign=True)
    ds = _load(items, ["vv", "vh"], aoi, resolution, groupby="id")
    # attach orbit metadata by matching acquisition time (odc sorts slices by time)
    stamps = np.array([np.datetime64(it.datetime.replace(tzinfo=None), "ns") for it in items])
    idx = [int(np.argmin(np.abs(stamps - t))) for t in ds.time.values.astype("datetime64[ns]")]
    ds = ds.assign_coords(
        orbit_state=("time", [items[i].properties.get("sat:orbit_state", "") for i in idx]),
        relative_orbit=("time", [int(items[i].properties.get("sat:relative_orbit", -1)) for i in idx]),
        platform=("time", [items[i].properties.get("platform", "") for i in idx]),
    )
    ds = ds.where(ds > 0)
    for v in ("vv", "vh"):
        ds[v].attrs.update(units="1", long_name=f"gamma0 {v.upper()} (linear power)")
    ds = ds.sortby("time")
    ds.attrs = common.provenance(source="Planetary Computer sentinel-1-rtc", resolution_m=resolution)
    return ds


# ---- Optical -----------------------------------------------------------------------------

S2_BANDS = ["blue", "green", "red", "nir", "swir16", "scl"]
LANDSAT_BANDS = ["blue", "green", "red", "nir08", "swir16", "qa_pixel"]


def load_sentinel2(aoi, start_date, end_date, max_cloud=80, resolution=10.0) -> xr.Dataset:
    """Sentinel-2 L2A collection 1 (Earth Search, no auth). Reflectance scaled to 0-1;
    ``scl`` kept for cloud/snow masking (8, 9, 10 cloud; 11 snow/ice)."""
    items = _search(EARTH_SEARCH, ["sentinel-2-c1-l2a"], aoi, start_date, end_date,
                    query={"eo:cloud_cover": {"lt": max_cloud}})
    ds = _load(items, S2_BANDS, aoi, resolution)
    refl = [b for b in S2_BANDS if b != "scl"]
    # Earth Search C1 L2A carries offset -0.1 / scale 1e-4 in raster:bands; apply explicitly
    ds[refl] = (ds[refl].where(ds[refl] > 0) * 1e-4 - 0.1).astype("float32")
    for b in refl:
        ds[b].attrs["units"] = "1"
    ds.attrs = common.provenance(source="Earth Search sentinel-2-c1-l2a", max_cloud=max_cloud)
    return ds


def load_landsat(aoi, start_date, end_date, max_cloud=80, resolution=30.0) -> xr.Dataset:
    """Landsat 8/9 Collection 2 Level-2 surface reflectance (Planetary Computer)."""
    items = _search(PC_STAC, ["landsat-c2-l2"], aoi, start_date, end_date, sign=True,
                    query={"eo:cloud_cover": {"lt": max_cloud},
                           "platform": {"in": ["landsat-8", "landsat-9"]}})
    ds = _load(items, LANDSAT_BANDS, aoi, resolution)
    refl = [b for b in LANDSAT_BANDS if b != "qa_pixel"]
    ds[refl] = (ds[refl].where(ds[refl] > 0) * 2.75e-5 - 0.2).astype("float32")
    ds.attrs = common.provenance(source="Planetary Computer landsat-c2-l2", max_cloud=max_cloud)
    return ds


# ---- DEMs --------------------------------------------------------------------------------

def load_copdem(aoi, resolution=30.0) -> xr.DataArray:
    items = _search(PC_STAC, ["cop-dem-glo-30"], aoi, "2010-01-01", "2030-01-01", sign=True)
    da = _load(items, ["data"], aoi, resolution, groupby=None)["data"].max("time")
    da = da.where(da > -100).rename("elevation")
    da.attrs = common.provenance(units="m", vertical_datum="EGM2008", source="Copernicus GLO-30")
    return da


def load_arcticdem_strips(aoi, start_date, end_date, resolution=10.0) -> xr.DataArray:
    """Multitemporal ArcticDEM strips (PGC STAC, s2s041 2 m, loaded at ``resolution`` via
    COG overviews). One time slice per acquisition date; heights are WGS84 ellipsoidal and
    NOT co-registered (co-register before differencing, e.g. against stable terrain)."""
    items = _search(PGC_STAC, ["arcticdem-strips-s2s041-2m"], aoi, start_date, end_date)
    da = _load(items, ["dem"], aoi, resolution)["dem"]
    da = da.where(da > -9999).rename("elevation").dropna("time", how="all")
    da.attrs = common.provenance(units="m", vertical_datum="WGS84 ellipsoid",
                                 source="PGC ArcticDEM strips s2s041 2m", coregistered="no")
    return da


def load_arcticdem_mosaic(aoi, resolution=10.0) -> xr.DataArray:
    items = _search(PGC_STAC, ["arcticdem-mosaics-v4.1-2m"], aoi, "2000-01-01", "2030-01-01")
    da = _load(items, ["dem"], aoi, resolution, groupby=None)["dem"].max("time")
    da = da.where(da > -9999).rename("elevation")
    da.attrs = common.provenance(units="m", vertical_datum="WGS84 ellipsoid",
                                 source="PGC ArcticDEM mosaic v4.1 2m")
    return da


# ---- CLI ---------------------------------------------------------------------------------

_I = Annotated[str, Parameter(name=["--input", "-i"])]
_S = Annotated[str, Parameter(name=["--start", "-s"])]
_E = Annotated[str, Parameter(name=["--end", "-e"])]
_O = Annotated[str, Parameter(name=["--output", "-o"])]


def stage_sar(vectorPath: _I = "aoi/AOI.geojson", start_date: _S = "2023-08-01",
              end_date: _E = "2023-10-31", output_path: _O = "data/staged/sar_s1_rtc.zarr",
              resolution: Annotated[float, Parameter(name=["--resolution"])] = 10.0):
    """Stage Sentinel-1 RTC gamma0 (VV, VH) clipped to the AOI."""
    common.check_s3_env(output_path)
    common.save_zarr(load_sar(common.load_aoi(vectorPath), start_date, end_date, resolution), output_path)


def stage_optical(vectorPath: _I = "aoi/AOI.geojson", start_date: _S = "2023-08-01",
                  end_date: _E = "2023-10-31", output_path: _O = "data/staged/optical_s2.zarr",
                  sensor: Annotated[str, Parameter(name=["--sensor"])] = "sentinel2",
                  max_cloud: Annotated[float, Parameter(name=["--max-cloud"])] = 80):
    """Stage optical surface reflectance (sentinel2 | landsat) clipped to the AOI."""
    common.check_s3_env(output_path)
    aoi = common.load_aoi(vectorPath)
    fn = {"sentinel2": load_sentinel2, "landsat": load_landsat}[sensor]
    common.save_zarr(fn(aoi, start_date, end_date, max_cloud), output_path)


def stage_dem(vectorPath: _I = "aoi/AOI.geojson", start_date: _S = "2008-01-01",
              end_date: _E = "2025-12-31", output_path: _O = "data/staged/dem.zarr",
              product: Annotated[str, Parameter(name=["--product"])] = "arcticdem-strips",
              resolution: Annotated[float, Parameter(name=["--resolution"])] = 10.0):
    """Stage a DEM: arcticdem-strips (multitemporal) | arcticdem-mosaic | copdem."""
    common.check_s3_env(output_path)
    aoi = common.load_aoi(vectorPath)
    if product == "arcticdem-strips":
        da = load_arcticdem_strips(aoi, start_date, end_date, resolution)
    elif product == "arcticdem-mosaic":
        da = load_arcticdem_mosaic(aoi, resolution)
    elif product == "copdem":
        da = load_copdem(aoi, max(resolution, 30.0))
    else:
        raise ValueError(product)
    common.save_zarr(da, output_path)


def nearest_pair(da: xr.DataArray | xr.Dataset, t_event: np.datetime64):
    """Last scene before and first scene after the event."""
    before = da.sel(time=da.time < t_event).isel(time=-1)
    after = da.sel(time=da.time > t_event).isel(time=0)
    return before, after
