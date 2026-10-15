"""Shared helpers. Writers and AOI loading come from gaia-cli (``gaia_cli.io``) so that
stores produced here are interchangeable with ``gaia stage ...`` output."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import numpy as np
import rioxarray  # noqa: F401  (registers .rio)
import xarray as xr
import xvec  # noqa: F401  (registers .xvec; gaia_cli.io uses it without importing)
import yaml
from gaia_cli import io as gio

ROOT = Path(__file__).resolve().parents[2]

load_aoi = gio.load_aoi
check_s3_env = gio.check_s3_env
save_zarr = gio.save_zarr
save_xvec_zarr = gio.save_xvec_zarr
save_datatree = gio.save_datatree


@lru_cache
def event() -> dict:
    with open(ROOT / "config" / "event.yml") as f:
        return yaml.safe_load(f)


def utm_crs() -> str:
    return event()["event"]["utm_crs"]


def provenance(**extra) -> dict:
    ev = event()
    attrs = {
        "event": ev["event"]["name"],
        "event_origin_time_utc": ev["event"]["origin_time_utc"],
        "paper_doi": ev["references"]["paper_doi"],
        "data_release_doi": ev["references"]["data_release_doi"],
        "staged_by": "peters_dome (gaia-cli conventions)",
    }
    attrs.update({k: str(v) for k, v in extra.items()})
    return attrs


def clip_projected(da: xr.DataArray | xr.Dataset, aoi: gpd.GeoDataFrame):
    """Clip a raster in any CRS to the AOI polygon without reprojecting it."""
    return da.rio.clip(aoi.to_crs(da.rio.crs).geometry, all_touched=True)


def to_projected_bbox(aoi: gpd.GeoDataFrame, crs: str) -> tuple[float, float, float, float]:
    return tuple(aoi.to_crs(crs).total_bounds)


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(np.asarray(lon2) - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def gdf_to_xvec(gdf: gpd.GeoDataFrame) -> xr.Dataset:
    """Vector features (polygons/points with attribute columns) -> CF-encoded xvec Dataset,
    ready for Zarr. Mirrors the encoding gaia-cli uses for station data."""
    gdf = gdf.to_crs("EPSG:4326").reset_index(drop=True)
    cols = [c for c in gdf.columns if c != gdf.geometry.name]
    ds = xr.Dataset(
        {c: ("geometry", _zarr_safe(gdf[c].to_numpy())) for c in cols},
        coords={"geometry": gdf.geometry.to_numpy()},
    )
    ds = ds.xvec.set_geom_indexes("geometry", crs="EPSG:4326")
    return ds.xvec.encode_cf()


def _zarr_safe(values: np.ndarray) -> np.ndarray:
    if values.dtype == object:
        import pandas as pd

        return np.array(["" if v is None or (np.ndim(v) == 0 and pd.isna(v)) else str(v) for v in values], dtype=str)
    if np.issubdtype(values.dtype, np.datetime64):
        return values.astype("datetime64[ns]")
    return values
