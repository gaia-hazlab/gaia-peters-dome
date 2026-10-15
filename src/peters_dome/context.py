"""Glaciological, geological and soil context layers.

Vectors (outlines, map units, point measurements) are written as CF-encoded xvec groups
(``vectors/...``) of the container store, the encoding gaia-cli uses for stations.
Rasters (ITS_LIVE, SoilGrids) stay on their native grids (EPSG:3413 and Homolosine).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import geopandas as gpd
import requests
import rioxarray
import xarray as xr
from cyclopts import Parameter

from . import common
from .params import STORE, Group, Input, Output, Overwrite, SkipExisting

log = logging.getLogger(__name__)

GLIMS_WFS = "https://www.glims.org/geoserver/GLIMS/wfs"
ITSLIVE = "https://its-live-data.s3.amazonaws.com/velocity_mosaic/v2"
SIM3340_ZIP = "https://pubs.usgs.gov/sim/3340/sim3340_shp.zip"
SIM3340_DIR = "sim3340_shp/AKgeol_web_shp"
MACROSTRAT = "https://macrostrat.org/api/v2/geologic_units/map"
SOILGRIDS = "https://files.isric.org/soilgrids/latest/data"
SOIL_PROPS = {  # property: (units after scaling, scale factor) per ISRIC SoilGrids 2.0 docs
    "bdod": ("kg/dm3", 0.01),
    "cfvo": ("cm3/100cm3", 0.1),
    "clay": ("g/100g", 0.1),
    "sand": ("g/100g", 0.1),
    "silt": ("g/100g", 0.1),
    "soc": ("g/kg", 0.1),
}
SOIL_DEPTHS = ["0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm", "100-200cm"]


# ---- Glaciers ----------------------------------------------------------------------------

def _wfs(layer: str, aoi: gpd.GeoDataFrame, max_features: int = 5000) -> gpd.GeoDataFrame:
    minx, miny, maxx, maxy = aoi.to_crs("EPSG:4326").total_bounds
    r = requests.get(GLIMS_WFS, params={
        "service": "WFS", "version": "1.1.0", "request": "GetFeature", "typeName": layer,
        "bbox": f"{minx},{miny},{maxx},{maxy},EPSG:4326", "outputFormat": "application/json",
        "maxFeatures": max_features, "srsName": "EPSG:4326"}, timeout=120)
    r.raise_for_status()
    feats = r.json()["features"]
    gdf = gpd.GeoDataFrame.from_features(feats, crs="EPSG:4326") if feats else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    # some GLIMS layers (RGI_Alaska) come back in EPSG:3857 metres unless srsName is honoured
    if len(gdf) and (abs(gdf.total_bounds[[0, 2]]).max() > 180 or abs(gdf.total_bounds[[1, 3]]).max() > 90):
        raise ValueError(f"{layer}: coordinates are not lon/lat (bounds {gdf.total_bounds})")
    log.info("%s: %d features", layer, len(gdf))
    return gdf


def load_glacier_outlines(aoi) -> dict[str, gpd.GeoDataFrame]:
    """RGI (Alaska layer, served by GLIMS), all GLIMS outlines, and GlaThiDa ice-thickness
    points, intersecting the AOI bounding box."""
    return {
        "rgi": _wfs("GLIMS:RGI_Alaska", aoi),
        "glims": _wfs("GLIMS:GLIMS_Glacier_Outlines", aoi),
        "glathida": _wfs("GLIMS:glathida_points", aoi),
    }


def load_itslive(aoi, year: str = "0000") -> xr.Dataset:
    """ITS_LIVE v2 surface velocity mosaic, RGI region 01, native 120 m EPSG:3413 grid.
    ``year='0000'`` is the multi-year composite; annual mosaics exist for 2014-2022
    (checked 2026-10-14)."""
    kind = "static" if year == "0000" else "annual"
    out = {}
    for v in ("v", "vx", "vy", "v_error"):
        url = f"{ITSLIVE}/{kind}/cog/ITS_LIVE_velocity_120m_RGI01A_{year}_v02_{v}.tif"
        da = rioxarray.open_rasterio(url).squeeze("band", drop=True)  # lazy: clip reads only the window
        crs = da.rio.crs
        # the COG carries a second grid-mapping coord ('mapping'); where() drops grid_mapping
        da = common.clip(da.drop_vars("mapping", errors="ignore").rio.write_crs(crs), aoi)
        out[v] = da.where(da != da.rio.nodata).rio.write_crs(crs)
    ds = xr.Dataset(out)
    for v in ds:
        ds[v].attrs["units"] = "m/yr"
    ds.attrs = common.provenance(source="ITS_LIVE v2 velocity mosaic RGI01A", year=year)
    return ds


# ---- Geology -----------------------------------------------------------------------------

def _download(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading %s -> %s (one time)", url, dest)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(tmp, "wb") as fh:
            fh.writelines(r.iter_content(1 << 22))
    tmp.rename(dest)
    return dest


def load_geology(aoi, layers=("AKStategeol_poly", "AKStategeol_arc")) -> dict[str, gpd.GeoDataFrame]:
    """Geologic Map of Alaska (Wilson et al., 2015, USGS SIM 3340). The 710 MB shapefile
    zip goes to the download cache (data/cache/, not part of the container): reading it
    remotely is slow because GDAL cannot use the .sbn spatial index over HTTP."""
    import pyogrio

    legacy = Path("data/raw/geology/sim3340_shp.zip")  # earlier cache location
    dest = common.CACHE / "sim3340_shp.zip"
    if legacy.exists() and not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        legacy.rename(dest)
    zpath = _download(SIM3340_ZIP, dest)
    out = {}
    for layer in layers:
        path = f"/vsizip/{zpath.resolve()}/{SIM3340_DIR}/{layer}.shp"
        info = pyogrio.read_info(path)
        bbox = tuple(aoi.to_crs(info["crs"]).total_bounds)
        gdf = pyogrio.read_dataframe(path, bbox=bbox)
        gdf = gpd.clip(gdf, aoi.to_crs(gdf.crs), keep_geom_type=True)
        log.info("SIM3340 %s: %d features", layer, len(gdf))
        out[layer] = gdf
    return out


def macrostrat_at(lat: float, lon: float) -> list[dict]:
    """Quick lithology check at a point (Macrostrat map units, all scales)."""
    r = requests.get(MACROSTRAT, params={"lat": lat, "lng": lon}, timeout=60)
    r.raise_for_status()
    return r.json()["success"]["data"]


# ---- Soil --------------------------------------------------------------------------------

def load_soilgrids(aoi, props=tuple(SOIL_PROPS), depths=tuple(SOIL_DEPTHS)) -> xr.Dataset:
    """SoilGrids 2.0 mean predictions on the native 250 m Interrupted Goode Homolosine
    grid, windowed read from ISRIC VRTs. Dims: (depth, y, x). Expect NaN over glacier ice
    and bare rock, which covers most of the source area."""
    out = {}
    for p in props:
        layers = []
        for d in depths:
            url = f"/vsicurl/{SOILGRIDS}/{p}/{p}_{d}_mean.vrt"
            da = rioxarray.open_rasterio(url).squeeze("band", drop=True)  # lazy: clip reads only the window
            crs = da.rio.crs
            da = common.clip(da, aoi)
            da = (da.where(da != da.rio.nodata) * SOIL_PROPS[p][1]).rio.write_crs(crs)
            layers.append(da.expand_dims(depth=[d]))
        out[p] = xr.concat(layers, "depth")
        out[p].attrs["units"] = SOIL_PROPS[p][0]
    ds = xr.Dataset(out)
    top = [float(d.split("-")[0]) for d in depths]
    bot = [float(d.split("-")[1].rstrip("cm")) for d in depths]
    ds = ds.assign_coords(depth_top_cm=("depth", top), depth_bottom_cm=("depth", bot))
    ds.attrs = common.provenance(source="ISRIC SoilGrids 2.0 (mean)", resolution_m=250)
    return ds


# ---- CLI ---------------------------------------------------------------------------------

def stage_glacier(vectorPath: Input = "aoi/peters_dome_regional.geojson",
                  output_path: Output = STORE,
                  itslive_year: Annotated[str, Parameter(name=["--itslive-year"])] = "0000",
                  overwrite: Overwrite = False, skip_existing: SkipExisting = False):
    """Stage RGI and GLIMS outlines, GlaThiDa points (vectors/glacier/*) and ITS_LIVE
    velocity (rasters/glacier/itslive_velocity)."""
    aoi = common.load_aoi(vectorPath)
    groups = {f"vectors/glacier/{k}": None for k in ("rgi", "glims", "glathida")}
    todo = {g for g in groups if common.check_group(output_path, g, overwrite, skip_existing)}
    if todo:
        for k, gdf in load_glacier_outlines(aoi).items():
            g = f"vectors/glacier/{k}"
            if g in todo and len(gdf):
                common.write_group(common.gdf_to_xvec(gdf), output_path, g, overwrite)
    g = "rasters/glacier/itslive_velocity"
    if common.check_group(output_path, g, overwrite, skip_existing):
        common.write_group(load_itslive(aoi, itslive_year), output_path, g, overwrite)


def stage_geology(vectorPath: Input = "aoi/peters_dome_regional.geojson",
                  output_path: Output = STORE,
                  overwrite: Overwrite = False, skip_existing: SkipExisting = False):
    """Stage Geologic Map of Alaska units and contacts/faults as vectors/geology/*."""
    groups = {"AKStategeol_poly": "vectors/geology/units", "AKStategeol_arc": "vectors/geology/contacts_faults"}
    todo = [layer for layer, g in groups.items() if common.check_group(output_path, g, overwrite, skip_existing)]
    if not todo:
        return
    for layer, gdf in load_geology(common.load_aoi(vectorPath), layers=tuple(todo)).items():
        common.write_group(common.gdf_to_xvec(gdf), output_path, groups[layer], overwrite)


def stage_soil(vectorPath: Input = "aoi/AOI.geojson", output_path: Output = STORE,
               group: Group = "rasters/soil/soilgrids",
               overwrite: Overwrite = False, skip_existing: SkipExisting = False):
    """Stage SoilGrids 2.0 layered soil properties (6 depths, 0-200 cm), native grid."""
    if common.check_group(output_path, group, overwrite, skip_existing):
        common.write_group(load_soilgrids(common.load_aoi(vectorPath)), output_path, group, overwrite)
