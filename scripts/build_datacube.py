"""Event datacube: put the local-AOI layers on one grid (UTM 5N, EPSG:32605, 10 m, extent
of aoi/peters_dome_local.geojson) and write it as the ``datacube`` group of the SAME
container store. Dims (y, x), plus ``depth`` for soil and ``epoch`` (before/after) for
imagery.

Why not ``gaia align``: align aggregates every timed raster to the coarsest product's time
step (here hourly ERA5 would set the step for 12-day SAR) and writes a separate store. The
cube needs event-specific before/after scene selection instead. ``gaia align`` still works
on the container for the met rasters, e.g.
``gaia align -i data/peters_dome.zarr -o met_aligned.zarr --like rasters/met/era5``.

Before/after imagery = last acquisition before and first after the ComCat origin time in
config/event.yml; Sentinel-2 scenes are ranked by clear-sky fraction from the SCL band
(classes 4 vegetation, 5 bare, 6 water, 11 snow/ice count as clear). Categorical layers
carry ``flag_values`` (gaia-cli convention: resampled with mode, never averaged).

Caveat recorded on the layer: dh_post_minus_mosaic mixes the release SETSM DEM with the
ArcticDEM v4.1 mosaic (WGS84 ellipsoid) without co-registration.
"""

from __future__ import annotations

import argparse
import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray  # noqa: F401
import xarray as xr
from rasterio.enums import Resampling
from rasterio.features import rasterize

from peters_dome import common

log = logging.getLogger("peters_dome.cube")
CLEAR_SCL = [4, 5, 6, 11]


def template(aoi: gpd.GeoDataFrame, res: float) -> xr.DataArray:
    minx, miny, maxx, maxy = aoi.to_crs(common.utm_crs()).total_bounds
    x = np.arange(np.floor(minx / res) * res + res / 2, maxx, res)
    y = np.arange(np.ceil(maxy / res) * res - res / 2, miny, -res)
    da = xr.DataArray(np.zeros((len(y), len(x)), "float32"), dims=("y", "x"), coords={"y": y, "x": x})
    return da.rio.write_crs(common.utm_crs()).rio.write_transform()


def match(da, tmpl, resampling=Resampling.bilinear):
    return da.rio.reproject_match(tmpl, resampling=resampling).where(np.isfinite)


def pre_post(ds, t_event, score=None):
    t = ds.time.values
    before, after = np.where(t < t_event)[0], np.where(t > t_event)[0]
    if not len(before) or not len(after):
        return None
    if score is not None:
        s = score.values
        b, a = before[np.argmax(s[before])], after[np.argmax(s[after])]
    else:
        b, a = before[-1], after[0]
    out = xr.concat([ds.isel(time=b), ds.isel(time=a)], dim="epoch")
    return out.assign_coords(epoch=["before", "after"])


def _burn(geoms, tmpl, value=1, dtype="uint8"):
    shapes = [(g, value) for g in geoms if g is not None and not g.is_empty]
    if not shapes:
        return tmpl.copy(data=np.zeros(tmpl.shape, dtype))
    return tmpl.copy(data=rasterize(shapes, out_shape=tmpl.shape, transform=tmpl.rio.transform(),
                                    fill=0, dtype=dtype))


class Store:
    def __init__(self, path):
        self.path = path

    def has(self, group):
        return common.group_exists(self.path, group)

    def raster(self, group):
        return common.with_crs(common.open_group(self.path, group))

    def vectors(self, group):
        return common.open_vectors(self.path, group).to_crs(common.utm_crs())


def build(store: Store, res: float) -> xr.Dataset:
    ev = common.event()["event"]
    t_event = np.datetime64(pd.Timestamp(ev["origin_time_utc"]).tz_localize(None), "ns")
    aoi = common.load_aoi(str(common.ROOT / "aoi/peters_dome_local.geojson"))
    tmpl = template(aoi, res)
    layers: dict[str, xr.DataArray] = {}
    coords = {}

    def have(group):
        ok = store.has(group)
        if not ok:
            log.info("missing %s, skipped", group)
        return ok

    # topography
    for group, key in [("rasters/topography/copdem", "elevation_copdem"),
                       ("rasters/topography/arcticdem_mosaic", "elevation_arcticdem_mosaic"),
                       ("rasters/topography/setsm_post_event", "elevation_post_setsm")]:
        if have(group):
            layers[key] = match(store.raster(group)["elevation"], tmpl)
    if {"elevation_post_setsm", "elevation_arcticdem_mosaic"} <= set(layers):
        dh = layers["elevation_post_setsm"] - layers["elevation_arcticdem_mosaic"]
        layers["dh_post_minus_mosaic"] = dh.assign_attrs(caveat="not co-registered; vertical references may differ")

    # SAR before/after
    if have("rasters/sar/sentinel1_rtc"):
        pp = pre_post(store.raster("rasters/sar/sentinel1_rtc")[["vv", "vh"]], t_event)
        if pp is not None:
            for v in ("vv", "vh"):
                layers[f"s1_{v}"] = match(pp[v], tmpl)
            layers["s1_vv_log_ratio_db"] = 10 * np.log10(
                layers["s1_vv"].sel(epoch="after") / layers["s1_vv"].sel(epoch="before"))
            coords["s1_time"] = ("epoch", pp.time.values)

    # Sentinel-2 before/after, best clear-sky scene
    if have("rasters/optical/sentinel2"):
        s2 = store.raster("rasters/optical/sentinel2")
        clear = s2["scl"].isin(CLEAR_SCL).mean(("x", "y")).compute()
        pp = pre_post(s2, t_event, score=clear)
        if pp is not None:
            for b in ("blue", "green", "red", "nir", "swir16"):
                layers[f"s2_{b}"] = match(pp[b], tmpl)
            layers["s2_ndsi"] = (layers["s2_green"] - layers["s2_swir16"]) / (layers["s2_green"] + layers["s2_swir16"])
            coords["s2_time"] = ("epoch", pp.time.values)

    # glacier velocity and mask
    if have("rasters/glacier/itslive_velocity"):
        layers["itslive_v"] = match(store.raster("rasters/glacier/itslive_velocity")["v"], tmpl)
    if have("vectors/glacier/rgi"):
        layers["glacier_mask"] = _burn(store.vectors("vectors/glacier/rgi").geometry, tmpl).assign_attrs(
            flag_values=[0, 1], flag_meanings="no_glacier glacier")

    # landslide outlines (published): 1 = source + deposit, 2 = source
    if have("vectors/landslide_outlines"):
        g = store.vectors("vectors/landslide_outlines")
        m = _burn(g[g["name"] == "total"].geometry, tmpl) + _burn(g[g["name"] == "source"].geometry, tmpl)
        layers["landslide_mask"] = m.astype("uint8").assign_attrs(
            flag_values=[0, 1, 2], flag_meanings="outside total_footprint source")

    # SHALTOP maxima (published run)
    if have("model/shaltop"):
        sh = store.raster("model/shaltop")
        layers["shaltop_max_thickness"] = match(sh["vertical_thickness"].max("sim_time"), tmpl)
        speed = sh["flow_speed"].where(sh["vertical_thickness"] >= 1.0)
        layers["shaltop_max_speed"] = match(speed.max("sim_time"), tmpl).assign_attrs(
            note="cells thinner than 1 m masked (spurious margin speeds)")

    # geology units
    if have("vectors/geology/units"):
        g = store.vectors("vectors/geology/units")
        field = next(f for f in ("STATE_LABEL", "STATE_UNIT", "UNIT_NAME", "LABEL") if f in g.columns)
        labels = sorted(g[field].astype(str).unique())
        code = {lab: i + 1 for i, lab in enumerate(labels)}
        arr = rasterize([(geom, code[str(v)]) for geom, v in zip(g.geometry, g[field], strict=True)],
                        out_shape=tmpl.shape, transform=tmpl.rio.transform(), fill=0, dtype="int32")
        layers["geology_unit"] = tmpl.copy(data=arr).assign_attrs(
            flag_values=list(code.values()), flag_meanings=" ".join(s.replace(" ", "_") for s in labels),
            source_field=field, source="USGS SIM 3340")

    # soil (keeps depth)
    if have("rasters/soil/soilgrids"):
        soil = store.raster("rasters/soil/soilgrids")
        for v in ("clay", "sand", "silt", "bdod", "cfvo", "soc"):
            if v in soil:
                layers[f"soil_{v}"] = match(soil[v], tmpl, Resampling.nearest)

    # acquisition times live in s1_time / s2_time; drop each layer's own time coord
    ds = xr.Dataset({k: v.drop_vars(["spatial_ref", "time"], errors="ignore") for k, v in layers.items()})
    ds = ds.assign_coords(coords).rio.write_crs(common.utm_crs())
    ds.attrs = common.provenance(grid=f"{common.utm_crs()} {res} m, aoi/peters_dome_local.geojson",
                                 built_by="scripts/build_datacube.py")
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=common.STORE)
    ap.add_argument("--group", default="datacube")
    ap.add_argument("--resolution", type=float, default=10.0)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    common.check_group(a.store, a.group, a.overwrite)
    ds = build(Store(a.store), a.resolution)
    log.info("datacube variables: %s", list(ds.data_vars))
    common.write_group(ds, a.store, a.group, a.overwrite)


if __name__ == "__main__":
    main()
