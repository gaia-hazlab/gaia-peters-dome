"""Meteorology.

- ``era5``: hourly ERA5 (0.25 deg) or ERA5-Land (0.1 deg) surface fields on the native
  reanalysis nodes inside the AOI, via the Open-Meteo archive API (no key). Gridded
  Dataset (time, y, x), EPSG:4326, gaia-cli names/units (temperature degC,
  precipitation mm, wind_speed m/s).
  Why not the ARCO-ERA5 Zarr on GCS: it is chunked one hour per global field, so a
  3-month series at a few nodes reads tens of GB.
- ``profile``: hourly pressure-level profiles (1000-30 hPa) at named points (source,
  infrasound arrays) for propagation/freezing-level context. Open-Meteo historical
  forecast API, best-match NWP model, NOT ERA5. 30 hPa (~24 km) is the top level, so
  stratospheric ducting above that needs ERA5 model levels or G2S (not staged here).
- in-situ stations: ``gaia stage synoptic`` (gaia-cli), see scripts/stage_all.sh.
"""

from __future__ import annotations

from typing import Annotated

import numpy as np
import pandas as pd
import requests
import xarray as xr
from cyclopts import Parameter

from . import common

ARCHIVE_API = "https://archive-api.open-meteo.com/v1/archive"
HISTFC_API = "https://historical-forecast-api.open-meteo.com/v1/forecast"

SURFACE_VARS = {  # open-meteo name: (gaia name, scale to target units, units)
    "temperature_2m": ("temperature", 1.0, "degC"),
    "precipitation": ("precipitation", 1.0, "mm"),
    "rain": ("rain", 1.0, "mm"),
    "snowfall": ("snowfall", 10.0, "mm"),  # cm of snow -> mm (snow depth, not SWE)
    "snow_depth": ("snow_depth", 1.0, "m"),
    "wind_speed_10m": ("wind_speed", 1 / 3.6, "m/s"),
    "wind_direction_10m": ("wind_direction", 1.0, "degree"),
    "surface_pressure": ("surface_pressure", 1.0, "hPa"),
    "shortwave_radiation": ("shortwave_radiation", 1.0, "W/m2"),
    "soil_temperature_0_to_7cm": ("soil_temperature_0_7cm", 1.0, "degC"),
    "soil_moisture_0_to_7cm": ("soil_moisture_0_7cm", 1.0, "m3/m3"),
}
LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30]
LEVEL_VARS = {
    "temperature": ("temperature", 1.0, "degC"),
    "wind_speed": ("wind_speed", 1 / 3.6, "m/s"),
    "wind_direction": ("wind_direction", 1.0, "degree"),
    "geopotential_height": ("geopotential_height", 1.0, "m"),
    "relative_humidity": ("relative_humidity", 1.0, "%"),
}


def _nodes(aoi, step):
    minx, miny, maxx, maxy = aoi.to_crs("EPSG:4326").total_bounds
    xs = np.arange(np.floor(minx / step) * step, maxx + step / 2, step)
    ys = np.arange(np.floor(miny / step) * step, maxy + step / 2, step)
    return np.round(xs, 4), np.round(ys, 4)


def _get(url, params, retries=3):
    for i in range(retries):
        r = requests.get(url, params=params, timeout=180)
        if r.status_code == 200:
            out = r.json()
            return out if isinstance(out, list) else [out]
        if r.status_code == 429 and i < retries - 1:
            import time

            time.sleep(60)
            continue
        raise RuntimeError(f"{url} {r.status_code}: {r.text[:300]}")


def load_era5(aoi, start_date, end_date, model: str = "era5") -> xr.Dataset:
    step = {"era5": 0.25, "era5_land": 0.1}[model]
    xs, ys = _nodes(aoi, step)
    pts = [(y, x) for y in ys for x in xs]
    vars_ = list(SURFACE_VARS)
    records = []
    for k in range(0, len(pts), 50):  # API accepts many locations per call
        chunk = pts[k:k + 50]
        records += _get(ARCHIVE_API, dict(
            latitude=",".join(f"{p[0]}" for p in chunk),
            longitude=",".join(f"{p[1]}" for p in chunk),
            start_date=pd.Timestamp(start_date).date(), end_date=pd.Timestamp(end_date).date(),
            hourly=",".join(vars_), models=model, timezone="GMT"))
    time = pd.to_datetime(records[0]["hourly"]["time"])
    data = {v: np.full((len(time), len(ys), len(xs)), np.nan, dtype="float32") for v in vars_}
    elev = np.full((len(ys), len(xs)), np.nan, dtype="float32")
    for (y, x), rec in zip(pts, records):
        j, i = int(np.argmin(abs(ys - y))), int(np.argmin(abs(xs - x)))
        elev[j, i] = rec.get("elevation", np.nan)
        for v in vars_:
            vals = rec["hourly"].get(v)
            if vals is not None:
                data[v][:, j, i] = np.array(vals, dtype="float64") * SURFACE_VARS[v][1]
    ds = xr.Dataset(
        {SURFACE_VARS[v][0]: (("time", "y", "x"), data[v], {"units": SURFACE_VARS[v][2]}) for v in vars_},
        coords={"time": time, "y": ys, "x": xs, "node_elevation": (("y", "x"), elev, {"units": "m"})},
    )
    ds = ds.dropna("x", how="all", subset=["temperature"]).dropna("y", how="all", subset=["temperature"])
    ds = ds.rio.write_crs("EPSG:4326")
    ds.attrs = common.provenance(source=f"Open-Meteo archive API, model={model}",
                                 note="node values; Open-Meteo snaps request points to the model grid")
    return ds


def load_profiles(points: dict[str, tuple[float, float]], start_date, end_date) -> xr.Dataset:
    hourly = [f"{v}_{lev}hPa" for v in LEVEL_VARS for lev in LEVELS] + ["freezing_level_height"]
    names = list(points)
    recs = _get(HISTFC_API, dict(
        latitude=",".join(str(points[n][0]) for n in names),
        longitude=",".join(str(points[n][1]) for n in names),
        start_date=pd.Timestamp(start_date).date(), end_date=pd.Timestamp(end_date).date(),
        hourly=",".join(hourly), timezone="GMT"))
    time = pd.to_datetime(recs[0]["hourly"]["time"])
    out = {}
    for v, (name, scale, units) in LEVEL_VARS.items():
        arr = np.array([[np.array(r["hourly"][f"{v}_{lev}hPa"], dtype="float64") for lev in LEVELS] for r in recs])
        out[name] = (("point", "level", "time"), (arr * scale).astype("float32"), {"units": units})
    fl = np.array([np.array(r["hourly"]["freezing_level_height"], dtype="float64") for r in recs])
    out["freezing_level_height"] = (("point", "time"), fl.astype("float32"), {"units": "m"})
    ds = xr.Dataset(out, coords={
        "point": names, "level": ("level", LEVELS, {"units": "hPa"}), "time": time,
        "latitude": ("point", [points[n][0] for n in names]),
        "longitude": ("point", [points[n][1] for n in names]),
    })
    ds = ds.sel(time=slice(start_date, end_date)).transpose("point", "time", "level")
    ds.attrs = common.provenance(source="Open-Meteo historical-forecast API (best_match NWP, not ERA5)")
    return ds


def default_points() -> dict[str, tuple[float, float]]:
    ev = common.event()["event"]
    return {
        "source": (ev["source_area_lat"], ev["source_area_lon"]),
        # IMS array I53US (Fairbanks), centroid of IM.I53H1-H8 from EarthScope station service
        "I53US": (64.8669, -147.8581),
        # great-circle midpoint of the source -> I53US path
        "path_midpoint": (64.014, -149.574),
    }


_I = Annotated[str, Parameter(name=["--input", "-i"])]
_S = Annotated[str, Parameter(name=["--start", "-s"])]
_E = Annotated[str, Parameter(name=["--end", "-e"])]
_O = Annotated[str, Parameter(name=["--output", "-o"])]


def stage_era5(vectorPath: _I = "aoi/peters_dome_regional.geojson", start_date: _S = "2023-06-01",
               end_date: _E = "2023-09-21", output_path: _O = "data/staged/met_era5.zarr",
               model: Annotated[str, Parameter(name=["--model"])] = "era5"):
    """Stage hourly ERA5 / ERA5-Land surface meteorology on reanalysis nodes in the AOI."""
    common.check_s3_env(output_path)
    common.save_zarr(load_era5(common.load_aoi(vectorPath), start_date, end_date, model), output_path)


def stage_profile(start_date: _S = "2023-09-13T00:00", end_date: _E = "2023-09-14T12:00",
                  output_path: _O = "data/staged/met_profiles.zarr"):
    """Stage hourly pressure-level atmospheric profiles at the source and infrasound path."""
    common.check_s3_env(output_path)
    common.save_zarr(load_profiles(default_points(), start_date, end_date), output_path)
