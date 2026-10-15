"""Put the local-AOI layers on one grid: UTM 5N (EPSG:32605), 10 m, extent of
aoi/peters_dome_local.geojson. Output: data/peters_dome_datacube.zarr, dims (y, x) plus
``depth`` for soil and ``epoch`` (before/after) for imagery.

Reads from data/staged/ (run stage_all.sh first). Layers whose store is missing are
skipped. Before/after imagery = last acquisition before and first after the ComCat
origin time in config/event.yml; Sentinel-2 scenes are ranked by clear-sky fraction
from the SCL band (classes 4 vegetation, 5 bare, 6 water, 11 snow/ice count as clear).

Caveats recorded as attributes on the output:
- dh_post_minus_mosaic mixes the release SETSM DEM (aligned, vertical reference per
  release metadata) with the ArcticDEM v4.1 mosaic (WGS84 ellipsoid); co-register
  before quantitative use.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray  # noqa: F401
import xarray as xr
from rasterio.enums import Resampling
from rasterio.features import rasterize

from peters_dome import common

CLEAR_SCL = [4, 5, 6, 11]


def template(aoi: gpd.GeoDataFrame, res: float) -> xr.DataArray:
    minx, miny, maxx, maxy = aoi.to_crs(common.utm_crs()).total_bounds
    x = np.arange(np.floor(minx / res) * res + res / 2, maxx, res)
    y = np.arange(np.ceil(maxy / res) * res - res / 2, miny, -res)
    da = xr.DataArray(np.zeros((len(y), len(x)), "float32"), dims=("y", "x"), coords={"y": y, "x": x})
    return da.rio.write_crs(common.utm_crs()).rio.write_transform()


def match(da, tmpl, resampling=Resampling.bilinear):
    if "spatial_ref" in da.coords:
        da = da.rio.write_crs(da.rio.crs)
    return da.rio.reproject_match(tmpl, resampling=resampling).where(lambda v: np.isfinite(v))


def _open(staged: Path, name: str, group: str | None = None):
    p = staged / name
    if not p.exists():
        print(f"  missing {name}, skipped")
        return None
    if group:
        return xr.open_datatree(p, engine="zarr", consolidated=False)[group].to_dataset()
    return xr.open_zarr(p, consolidated=False)


def _crs(ds):
    """Restore CRS that rioxarray stores in spatial_ref after a Zarr round trip."""
    if "spatial_ref" in ds.coords or "spatial_ref" in ds:
        wkt = ds["spatial_ref"].attrs.get("crs_wkt") or ds["spatial_ref"].attrs.get("spatial_ref")
        if wkt:
            ds = ds.rio.write_crs(wkt)
    return ds


def pre_post(ds, t_event, score=None):
    t = ds.time.values
    before, after = np.where(t < t_event)[0], np.where(t > t_event)[0]
    if score is not None:
        s = score.values
        before = before[np.argsort(s[before])[::-1]][:1] if len(before) else before
        after = after[np.argsort(s[after])[::-1]][:1] if len(after) else after
    else:
        before, after = before[-1:], after[:1]
    if not len(before) or not len(after):
        return None
    out = xr.concat([ds.isel(time=before[0]), ds.isel(time=after[0])], dim="epoch")
    return out.assign_coords(epoch=["before", "after"])


def build(staged: Path, res: float) -> xr.Dataset:
    ev = common.event()["event"]
    t_event = np.datetime64(pd.Timestamp(ev["origin_time_utc"]).tz_localize(None), "ns")
    aoi = common.load_aoi(str(common.ROOT / "aoi/peters_dome_local.geojson"))
    tmpl = template(aoi, res)
    layers: dict[str, xr.DataArray] = {}

    # topography
    for name, key in [("dem_copdem.zarr", "elevation_copdem"),
                      ("dem_arcticdem_mosaic.zarr", "elevation_arcticdem_mosaic")]:
        ds = _open(staged, name)
        if ds is not None:
            layers[key] = match(_crs(ds)["elevation"], tmpl)
    rel = _open(staged, "usgs_release.zarr", "dem_post")
    if rel is not None:
        layers["elevation_post_setsm"] = match(_crs(rel)["elevation"], tmpl)
        if "elevation_arcticdem_mosaic" in layers:
            dh = layers["elevation_post_setsm"] - layers["elevation_arcticdem_mosaic"]
            dh.attrs["caveat"] = "not co-registered; vertical references may differ"
            layers["dh_post_minus_mosaic"] = dh

    # SAR before/after
    sar = _open(staged, "sar_s1_rtc.zarr")
    if sar is not None:
        pp = pre_post(_crs(sar)[["vv", "vh"]], t_event)
        if pp is not None:
            for v in ("vv", "vh"):
                layers[f"s1_{v}"] = match(pp[v], tmpl)
            layers["s1_vv_log_ratio_db"] = 10 * np.log10(layers["s1_vv"].sel(epoch="after") / layers["s1_vv"].sel(epoch="before"))
            layers["s1_time"] = pp.time

    # Sentinel-2 before/after, best clear-sky scene
    s2 = _open(staged, "optical_s2.zarr")
    if s2 is not None:
        s2 = _crs(s2)
        clear = s2["scl"].isin(CLEAR_SCL).mean(("x", "y")).compute()
        pp = pre_post(s2, t_event, score=clear)
        if pp is not None:
            for b in ("blue", "green", "red", "nir", "swir16"):
                layers[f"s2_{b}"] = match(pp[b], tmpl)
            layers["s2_ndsi"] = (layers["s2_green"] - layers["s2_swir16"]) / (layers["s2_green"] + layers["s2_swir16"])
            layers["s2_time"] = pp.time

    # glacier velocity + glacier mask
    gl = _open(staged, "glacier.zarr", "itslive_velocity")
    if gl is not None:
        layers["itslive_v"] = match(_crs(gl)["v"], tmpl)
    rgi_path = staged / "glacier_rgi.geojson"
    if rgi_path.exists():
        rgi = gpd.read_file(rgi_path).to_crs(common.utm_crs())
        layers["glacier_mask"] = _burn(rgi.geometry, tmpl, 1).astype("uint8")

    # landslide outlines (1 = total footprint, 2 = source)
    ol = staged / "release_outlines.geojson"
    if ol.exists():
        g = gpd.read_file(ol).to_crs(common.utm_crs())
        m = _burn(g[g["name"] == "total"].geometry, tmpl, 1) + _burn(g[g["name"] == "source"].geometry, tmpl, 1)
        layers["landslide_mask"] = m.astype("uint8").assign_attrs(flag_values=[0, 1, 2],
                                                                  flag_meanings="outside total_footprint source")

    # SHALTOP maxima
    sh = _open(staged, "usgs_release.zarr", "shaltop")
    if sh is not None:
        sh = _crs(sh)
        layers["shaltop_max_thickness"] = match(sh["vertical_thickness"].max("sim_time"), tmpl)
        layers["shaltop_max_speed"] = match(sh["flow_speed"].max("sim_time"), tmpl)

    # geology units
    geo = staged / "geology_AKStategeol_poly.geojson"
    if geo.exists():
        g = gpd.read_file(geo).to_crs(common.utm_crs())
        field = next(f for f in ("STATE_LABEL", "STATE_UNIT", "UNIT_NAME", "LABEL") if f in g.columns)
        labels = sorted(g[field].astype(str).unique())
        code = {lab: i + 1 for i, lab in enumerate(labels)}
        shapes = [(geom, code[str(v)]) for geom, v in zip(g.geometry, g[field])]
        arr = rasterize(shapes, out_shape=tmpl.shape, transform=tmpl.rio.transform(), fill=0, dtype="int32")
        layers["geology_unit"] = tmpl.copy(data=arr).astype("int32").assign_attrs(
            flag_values=list(code.values()), flag_meanings=" ".join(s.replace(" ", "_") for s in labels),
            source_field=field, source="USGS SIM 3340")

    # soil (keeps depth)
    soil = _open(staged, "soil.zarr")
    if soil is not None:
        soil = _crs(soil)
        for v in ("clay", "sand", "silt", "bdod", "cfvo", "soc"):
            if v in soil:
                layers[f"soil_{v}"] = match(soil[v], tmpl, Resampling.nearest)

    ds = xr.Dataset({k: v.drop_vars("spatial_ref", errors="ignore") for k, v in layers.items()
                     if k not in ("s1_time", "s2_time")})
    for k in ("s1_time", "s2_time"):
        if k in layers:
            ds = ds.assign_coords({k: ("epoch", layers[k].values)})
    ds = ds.rio.write_crs(common.utm_crs())
    ds.attrs = common.provenance(grid=f"{common.utm_crs()} {res} m, aoi/peters_dome_local.geojson")
    return ds


def _burn(geoms, tmpl, value):
    geoms = [g for g in geoms if g is not None and not g.is_empty]
    if not geoms:
        return tmpl.copy(data=np.zeros(tmpl.shape, "uint8"))
    arr = rasterize([(g, value) for g in geoms], out_shape=tmpl.shape,
                    transform=tmpl.rio.transform(), fill=0, dtype="uint8")
    return tmpl.copy(data=arr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--staged", default="data/staged")
    ap.add_argument("--output", default="data/peters_dome_datacube.zarr")
    ap.add_argument("--resolution", type=float, default=10.0)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    if a.overwrite and not a.output.startswith("s3://"):
        shutil.rmtree(a.output, ignore_errors=True)
    common.check_s3_env(a.output)
    ds = build(Path(a.staged), a.resolution)
    print(ds)
    common.save_zarr(ds, a.output)


if __name__ == "__main__":
    main()
