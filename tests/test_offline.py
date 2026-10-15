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

    gdf = gpd.GeoDataFrame({"name": ["a", None], "v": [1.0, 2.0], "LON": [0.0, 1.0]},
                           geometry=[Point(0, 0).buffer(1), Point(2, 2).buffer(1)], crs="EPSG:4326")
    store = str(tmp_path / "c.zarr")
    common.write_group(common.gdf_to_xvec(gdf), store, "vectors/test")
    back = common.open_vectors(store, "vectors/test")
    assert len(back) == 2 and list(back["name"]) == ["a", ""]
    assert "LON_attr" in back.columns  # column renamed so CF geometry decoding works


def test_single_store_groups(tmp_path):
    store = str(tmp_path / "c.zarr")
    ras = xr.Dataset({"v": (("y", "x"), np.ones((2, 3)))}, coords={"y": [1.0, 0.0], "x": [0.0, 1.0, 2.0]})
    ras = ras.rio.write_crs("EPSG:32605")
    sta = xr.DataArray(np.zeros((2, 4)), dims=("station", "time"),
                       coords={"station": ["A.B", "A.C"], "time": pd.date_range("2023-09-14", periods=4, freq="s"),
                               "geometry": ("station", [Point(0, 0), Point(1, 1)])}, name="pressure")
    common.write_group(ras, store, "rasters/test")
    common.write_group(common.stations_to_dataset(sta), store, "stations/test")
    tree = common.gio.open_datatree(store)
    assert {"/rasters/test", "/stations/test"} <= set(tree.groups)
    assert tree.attrs["event_origin_time_utc"].startswith("2023-09-14T02:06:49")
    assert common.open_group(store, "rasters/test").rio.crs is not None or "spatial_ref" in tree["rasters/test"].coords
    assert common.check_group(store, "rasters/test", skip_existing=True) is False
    with pytest.raises(SystemExit):
        common.check_group(store, "rasters/test")
    common.write_group(ras * 2, store, "rasters/test", overwrite=True)
    assert float(common.open_group(store, "rasters/test").v.max()) == 2.0
    assert "/stations/test" in common.gio.open_datatree(store).groups  # sibling survives overwrite
