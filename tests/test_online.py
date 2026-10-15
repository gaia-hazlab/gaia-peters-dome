"""Smoke tests against live services. Run with ``pixi run test-online``."""

import pytest

from peters_dome import common, met, waveforms

pytestmark = pytest.mark.online


def test_seismic_near_field():
    aoi = common.load_aoi(str(common.ROOT / "aoi/peters_dome_waveforms.geojson"))
    da = waveforms.load_seismic(aoi, "2023-09-14T02:06:00", "2023-09-14T02:08:00", max_distance_km=60)
    assert da.sizes["station"] >= 2 and "Z" in da.component.values


def test_profiles():
    ds = met.load_profiles(met.default_points(), "2023-09-14T00:00", "2023-09-14T03:00")
    assert set(ds.point.values) == {"source", "I53US", "path_midpoint"}
    assert ds.temperature.notnull().any()
