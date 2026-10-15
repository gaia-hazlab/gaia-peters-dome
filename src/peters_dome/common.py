"""Shared helpers. AOI loading, clipping, Zarr preparation and store resolution come from
gaia-cli (``gaia_cli.io``), so the container follows the ``gaia stage`` conventions.

Everything is written into ONE Zarr v3 store (default ``data/peters_dome.zarr``), one
group per dataset, laid out like ``gaia stage all``: ``stations/<name>`` for xvec-encoded
point time series, ``rasters/<name>`` for gridded data on native grids, plus
``vectors/<name>`` (xvec-encoded polygons/lines) and ``model/<name>`` (published model
output and non-gridded derived products). Open it with
``gaia_cli.io.open_datatree(path)``.
"""

from __future__ import annotations

import logging
import sys
from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray  # noqa: F401  (registers .rio)
import xarray as xr
import xvec  # noqa: F401  (registers .xvec)
import yaml
from gaia_cli import io as gio

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
STORE = "data/peters_dome.zarr"
CACHE = Path("data/cache")  # download cache for sources that cannot be read in place

load_aoi = gio.load_aoi
check_s3_env = gio.check_s3_env


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


def clip(ds: xr.DataArray | xr.Dataset, aoi: gpd.GeoDataFrame):
    """gaia-cli clip: AOI reprojected to the raster's CRS, raster kept on its native grid."""
    return gio.clip_to_aoi(ds, aoi)


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(np.asarray(lon2) - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


# ---- the single store ------------------------------------------------------------------

def _group_path(store: str, group: str) -> str:
    """Path of a group inside the store; the root group ("") is the store itself
    (no trailing slash, so S3 keys never get a double slash)."""
    store, group = store.rstrip("/"), group.strip("/")
    return f"{store}/{group}" if group else store


def group_exists(store: str, group: str) -> bool:
    if store.startswith("s3://"):
        return gio.output_exists(_group_path(store, group) + "/zarr.json")
    return (Path(store) / group.strip("/") / "zarr.json").exists()


def check_group(store: str, group: str, overwrite: bool = False, skip_existing: bool = False) -> bool:
    """Return True if the caller should proceed. Mirrors ``gaia_cli.io.check_output``
    at group level: an existing group is an error unless --overwrite (or --skip-existing,
    which returns False so batch runs can resume)."""
    check_s3_env(store)
    if not group_exists(store, group):
        return True
    if overwrite:
        return True
    if skip_existing:
        log.info("skip %s:%s (exists)", store, group)
        return False
    sys.exit(f"Error: group '{group}' already exists in '{store}'. Use --overwrite to replace it.")


def _init_root(store: str):
    if group_exists(store, ""):
        return
    root = xr.Dataset(attrs={
        **provenance(),
        **{f"event_{k}": str(v) for k, v in event()["event"].items()},
        "layout": "stations/ (xvec), rasters/ (native grids), vectors/ (xvec), model/, datacube/",
        "open_with": "gaia_cli.io.open_datatree(path); xvec groups: ds.xvec.decode_cf()",
    })
    root.to_zarr(gio._zarr_store_for_path(store), mode="w-", zarr_format=3, consolidated=False)


def write_group(ds: xr.Dataset | xr.DataArray, store: str, group: str, overwrite: bool = False):
    """Write one dataset as a group of the store, using gaia-cli's Zarr preparation
    (string re-encoding, grid_mapping attrs, single-chunk small datasets)."""
    if isinstance(ds, xr.DataArray):
        ds = ds.to_dataset(name=ds.name or "data")
    if "spatial_ref" in ds.data_vars:  # keep the grid mapping a coordinate (gaia align needs it)
        ds = ds.set_coords("spatial_ref")
    _init_root(store)
    ds, encoding = gio._prepare_for_zarr(ds)
    log.info("writing %s:%s", store, group)
    ds.to_zarr(
        gio._zarr_store_for_path(store),
        group=group,
        mode="w" if overwrite else "w-",
        zarr_format=3,
        consolidated=False,
        write_empty_chunks=True,
        encoding=encoding,
    )


def stations_to_dataset(da: xr.DataArray) -> xr.Dataset:
    """Same encoding as ``gaia_cli.io.save_xvec_zarr``."""
    da = da.swap_dims({"station": "geometry"})
    da = da.xvec.set_geom_indexes("geometry", crs="EPSG:4326")
    return da.to_dataset().xvec.encode_cf()


_CF_GEOMETRY_NAMES = {"x", "y", "lat", "lon", "node_count", "part_node_count", "interior_ring",
                      "geometry_container", "spatial_ref", "crs"}


def gdf_to_xvec(gdf: gpd.GeoDataFrame) -> xr.Dataset:
    """Vector features (polygons/lines/points with attribute columns) -> CF-encoded xvec
    Dataset, the encoding gaia-cli uses for stations."""
    gdf = gdf.to_crs("EPSG:4326").reset_index(drop=True)
    # attribute columns must not shadow the names CF geometry encoding uses. Case-insensitive:
    # on macOS/Windows file systems, Zarr arrays "LON" and "lon" are the same directory
    # (GlaThiDa has LAT/LON columns)
    gdf = gdf.rename(columns={c: f"{c}_attr" for c in gdf.columns
                              if c != gdf.geometry.name and c.lower() in _CF_GEOMETRY_NAMES})
    cols = [c for c in gdf.columns if c != gdf.geometry.name]
    ds = xr.Dataset(
        {c: ("geometry", _zarr_safe(gdf[c].to_numpy())) for c in cols},
        coords={"geometry": gdf.geometry.to_numpy()},
    )
    ds = ds.xvec.set_geom_indexes("geometry", crs="EPSG:4326")
    return ds.xvec.encode_cf()


def _zarr_safe(values: np.ndarray) -> np.ndarray:
    if values.dtype == object:
        return np.array(["" if v is None or (np.ndim(v) == 0 and pd.isna(v)) else str(v) for v in values], dtype=str)
    if np.issubdtype(values.dtype, np.datetime64):
        return values.astype("datetime64[ns]")
    return values


def open_group(store: str, group: str, decode_xvec: bool = False) -> xr.Dataset:
    ds = gio.open_datatree(store)[group].to_dataset()
    return ds.xvec.decode_cf() if decode_xvec else ds


def open_vectors(store: str, group: str) -> gpd.GeoDataFrame:
    ds = open_group(store, group, decode_xvec=True)
    return gpd.GeoDataFrame(
        {v: ds[v].values for v in ds.data_vars},
        geometry=list(ds.geometry.values), crs="EPSG:4326",
    )


def with_crs(ds):
    """Restore the CRS rioxarray stores in ``spatial_ref`` after a Zarr round trip."""
    if "spatial_ref" in ds.coords or "spatial_ref" in ds:
        wkt = ds["spatial_ref"].attrs.get("crs_wkt") or ds["spatial_ref"].attrs.get("spatial_ref")
        if wkt:
            ds = ds.rio.write_crs(wkt)
    return ds
