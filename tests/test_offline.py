import numpy as np
import pandas as pd
import pytest
import xarray as xr
from obspy import Trace, UTCDateTime
from shapely.geometry import Point

from peters_dome import common, release, waveforms

ROOT = common.ROOT


@pytest.mark.parametrize("name", ["AOI", "peters_dome_local", "peters_dome_regional", "peters_dome_waveforms"])
def test_aoi_contains_source(name):
    aoi = common.load_aoi(str(ROOT / f"aoi/{name}.geojson"))
    ev = common.event()["event"]
    assert len(aoi) == 1 and aoi.crs.to_epsg() == 4326
    assert aoi.geometry.iloc[0].contains(Point(ev["source_area_lon"], ev["source_area_lat"]))


def test_local_aoi_covers_release_bbox():
    aoi = common.load_aoi(str(ROOT / "aoi/peters_dome_local.geojson"))
    minx, miny, maxx, maxy = common.event()["event"]["release_bbox"]
    b = aoi.total_bounds
    assert b[0] < minx and b[1] < miny and b[2] > maxx and b[3] > maxy


def test_waveform_aoi_radius():
    aoi = common.load_aoi(str(ROOT / "aoi/peters_dome_waveforms.geojson"))
    ev = common.event()["event"]
    ring = np.array(aoi.geometry.iloc[0].exterior.coords)
    d = common.haversine_km(ev["latitude"], ev["longitude"], ring[:, 1], ring[:, 0])
    assert np.allclose(d, 400, atol=1)


def test_to_grid_fills_gap_with_nan():
    fs, t0 = 10.0, UTCDateTime("2023-09-14T02:00:00")
    a = Trace(np.ones(100), header={"sampling_rate": 40.0, "starttime": t0})
    b = Trace(np.ones(100), header={"sampling_rate": 40.0, "starttime": t0 + 5})
    out = waveforms._to_grid([a, b], t0, 100, fs)
    assert np.isnan(out[30:50]).all() and np.isfinite(out[:20]).all() and np.isfinite(out[55:70]).all()


def test_force_loader(tmp_path):
    cols = ["time (s)"] + [f"{c} force{s} (N)" for s in ["", ", lower bound", ", upper bound"] for c in ["up", "north", "east"]]
    df = pd.DataFrame(np.arange(3 * len(cols)).reshape(3, -1), columns=cols)
    df["time (s)"] = [-1.0, 0.0, 1.0]
    df.to_csv(tmp_path / "f.csv", index=False)
    ds = release.load_force(tmp_path / "f.csv")
    assert ds.force.dims == ("component", "bound", "time")
    assert ds.time.values[1] == np.datetime64("2023-09-14T02:06:27")


def test_vector_roundtrip(tmp_path):
    import geopandas as gpd

    gdf = gpd.GeoDataFrame({"name": ["a", None], "v": [1.0, 2.0]},
                           geometry=[Point(0, 0).buffer(1), Point(2, 2).buffer(1)], crs="EPSG:4326")
    common.save_zarr(common.gdf_to_xvec(gdf), str(tmp_path / "v.zarr"))
    back = xr.open_zarr(tmp_path / "v.zarr", consolidated=False).xvec.decode_cf()
    assert back.sizes["geometry"] == 2 and list(back["name"].values) == ["a", ""]
