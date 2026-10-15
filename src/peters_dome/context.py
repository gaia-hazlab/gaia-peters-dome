"""Glaciological, geological and soil context layers.

Vectors (outlines, map units, point measurements) are stored as CF-encoded xvec Datasets
with a ``geometry`` dimension (same encoding gaia-cli uses for stations), and also written
as GeoJSON next to the Zarr store for GIS use.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import geopandas as gpd
import requests
import rioxarray
import xarray as xr
from cyclopts import Parameter

from . import common

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
    r = requests.get(GLIMS_WFS, params=dict(
        service="WFS", version="1.1.0", request="GetFeature", typeName=layer,
        bbox=f"{minx},{miny},{maxx},{maxy},EPSG:4326", outputFormat="application/json",
        maxFeatures=max_features), timeout=120)
    r.raise_for_status()
    feats = r.json()["features"]
    gdf = gpd.GeoDataFrame.from_features(feats, crs="EPSG:4326") if feats else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    print(f"  {layer}: {len(gdf)} features")
    return gdf


def load_glacier_outlines(aoi) -> dict[str, gpd.GeoDataFrame]:
    """RGI (Alaska layer, served by GLIMS), all GLIMS outlines, and GlaThiDa ice-thickness
    points, intersecting the AOI bounding box."""
    return {
        "rgi": _wfs("GLIMS:RGI_Alaska", aoi),
        "glims": _wfs("GLIMS:GLIMS_Glacier_Outlines", aoi),
        "glathida": _wfs("GLIMS:glathida_points", aoi),
    }


def load_itslive(aoi, year: str = "0000", resolution: float = 120.0) -> xr.Dataset:
    """ITS_LIVE v2 surface velocity mosaic, RGI region 01. ``year='0000'`` is the
    multi-year composite; annual mosaics exist for 2014-2022 (checked 2026-10-14)."""
    kind = "static" if year == "0000" else "annual"
    out = {}
    for v in ("v", "vx", "vy", "v_error"):
        url = f"{ITSLIVE}/{kind}/cog/ITS_LIVE_velocity_120m_RGI01A_{year}_v02_{v}.tif"
        da = rioxarray.open_rasterio(url, chunks={}).squeeze("band", drop=True)
        da = common.clip_projected(da.rio.clip_box(*aoi.to_crs(da.rio.crs).total_bounds), aoi)
        crs = da.rio.crs
        # where() drops the grid_mapping attr; the COG also carries a 'mapping' coord
        da = da.where(da != da.rio.nodata).drop_vars("mapping", errors="ignore").rio.write_crs(crs)
        out[v] = da.rio.reproject(common.utm_crs(), resolution=resolution)
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
    print(f"  downloading {url} -> {dest} (one time)")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(1 << 22):
                fh.write(chunk)
    tmp.rename(dest)
    return dest


def load_geology(aoi, layers=("AKStategeol_poly", "AKStategeol_arc"),
                 raw_dir: str = "data/raw/geology") -> dict[str, gpd.GeoDataFrame]:
    """Geologic Map of Alaska (Wilson et al., 2015, USGS SIM 3340). The 710 MB shapefile
    zip is downloaded once to ``raw_dir`` (reading it remotely is slow: GDAL cannot use the
    .sbn spatial index over HTTP), then read with a bbox filter."""
    import pyogrio

    zpath = _download(SIM3340_ZIP, Path(raw_dir) / "sim3340_shp.zip")
    out = {}
    for layer in layers:
        path = f"/vsizip/{zpath.resolve()}/{SIM3340_DIR}/{layer}.shp"
        info = pyogrio.read_info(path)
        bbox = tuple(aoi.to_crs(info["crs"]).total_bounds)
        gdf = pyogrio.read_dataframe(path, bbox=bbox)
        gdf = gpd.clip(gdf, aoi.to_crs(gdf.crs), keep_geom_type=True)
        print(f"  SIM3340 {layer}: {len(gdf)} features")
        out[layer] = gdf
    return out


def macrostrat_at(lat: float, lon: float) -> list[dict]:
    """Quick lithology check at a point (Macrostrat map units, all scales)."""
    r = requests.get(MACROSTRAT, params={"lat": lat, "lng": lon}, timeout=60)
    r.raise_for_status()
    return r.json()["success"]["data"]


# ---- Soil --------------------------------------------------------------------------------

def load_soilgrids(aoi, props=tuple(SOIL_PROPS), depths=tuple(SOIL_DEPTHS), resolution=250.0) -> xr.Dataset:
    """SoilGrids 2.0 mean predictions (250 m, Interrupted Goode Homolosine), windowed read
    from ISRIC VRTs, reprojected to UTM 5N. Dims: (depth, y, x). Expect NaN over glacier
    ice and bare rock, which covers most of the source area."""
    out = {}
    for p in props:
        layers = []
        for d in depths:
            url = f"/vsicurl/{SOILGRIDS}/{p}/{p}_{d}_mean.vrt"
            da = rioxarray.open_rasterio(url, chunks={}).squeeze("band", drop=True)
            da = da.rio.clip_box(*aoi.to_crs(da.rio.crs).total_bounds)
            da = da.where(da != da.rio.nodata) * SOIL_PROPS[p][1]
            da = da.rio.write_crs(da.rio.crs).rio.reproject(common.utm_crs(), resolution=resolution)
            layers.append(common.clip_projected(da, aoi).expand_dims(depth=[d]))
        out[p] = xr.concat(layers, "depth")
        out[p].attrs["units"] = SOIL_PROPS[p][0]
    ds = xr.Dataset(out)
    top = [float(d.split("-")[0]) for d in depths]
    bot = [float(d.split("-")[1].rstrip("cm")) for d in depths]
    ds = ds.assign_coords(depth_top_cm=("depth", top), depth_bottom_cm=("depth", bot))
    ds.attrs = common.provenance(source="ISRIC SoilGrids 2.0 (mean)", resolution_m=resolution)
    return ds


# ---- CLI ---------------------------------------------------------------------------------

_I = Annotated[str, Parameter(name=["--input", "-i"])]
_O = Annotated[str, Parameter(name=["--output", "-o"])]


def _write_vectors(vectors: dict[str, gpd.GeoDataFrame], output_path: str):
    """One xvec Zarr group per layer under ``output_path``, plus GeoJSON siblings."""
    stem = Path(output_path)
    stem.parent.mkdir(parents=True, exist_ok=True)
    tree = {}
    for name, gdf in vectors.items():
        if len(gdf) == 0:
            continue
        gdf.to_file(stem.with_name(f"{stem.stem}_{name}.geojson"), driver="GeoJSON")
        tree[name] = common.gdf_to_xvec(gdf)
    common.save_datatree(xr.DataTree.from_dict(tree), output_path)


def stage_glacier(vectorPath: _I = "aoi/peters_dome_regional.geojson",
                  output_path: _O = "data/staged/glacier.zarr",
                  itslive_year: Annotated[str, Parameter(name=["--itslive-year"])] = "0000"):
    """Stage glacier outlines (RGI, GLIMS), GlaThiDa points and ITS_LIVE velocity."""
    common.check_s3_env(output_path)
    aoi = common.load_aoi(vectorPath)
    vec = load_glacier_outlines(aoi)
    vel = load_itslive(aoi, itslive_year)
    tree = {k: common.gdf_to_xvec(v) for k, v in vec.items() if len(v)}
    tree["itslive_velocity"] = vel
    for k, v in vec.items():
        if len(v):
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            v.to_file(Path(output_path).with_name(f"glacier_{k}.geojson"), driver="GeoJSON")
    common.save_datatree(xr.DataTree.from_dict(tree), output_path)


def stage_geology(vectorPath: _I = "aoi/peters_dome_regional.geojson",
                  output_path: _O = "data/staged/geology.zarr"):
    """Stage Geologic Map of Alaska units and contacts/faults clipped to the AOI."""
    common.check_s3_env(output_path)
    aoi = common.load_aoi(vectorPath)
    _write_vectors(load_geology(aoi), output_path)


def stage_soil(vectorPath: _I = "aoi/AOI.geojson", output_path: _O = "data/staged/soil.zarr",
               resolution: Annotated[float, Parameter(name=["--resolution"])] = 250.0):
    """Stage SoilGrids 2.0 layered soil properties (6 depths, 0-200 cm)."""
    common.check_s3_env(output_path)
    common.save_zarr(load_soilgrids(common.load_aoi(vectorPath), resolution=resolution), output_path)

