"""config/release.yml must name a source, licence and attribution for every published group,
and must not cover Synoptic, whose terms were not reviewed for redistribution."""

import sys

import pytest
import yaml

from peters_dome import common

sys.path.insert(0, str(common.ROOT / "scripts"))
from hf_metadata import source_for

CFG = yaml.safe_load((common.ROOT / "config/release.yml").read_text())

PUBLISHED = [
    "stations/seismic", "stations/seismic_lp", "stations/seismic_continuous", "stations/infrasound",
    "rasters/sar/sentinel1_rtc", "rasters/optical/sentinel2", "rasters/optical/landsat",
    "rasters/topography/copdem", "rasters/topography/arcticdem_mosaic", "rasters/topography/arcticdem_strips",
    "rasters/topography/setsm_post_event", "rasters/glacier/itslive_velocity", "rasters/soil/soilgrids",
    "rasters/met/era5", "rasters/met/era5_land", "vectors/glacier/rgi", "vectors/glacier/glims",
    "vectors/glacier/glathida", "vectors/geology/units", "vectors/geology/contacts_faults",
    "vectors/landslide_outlines", "model/force_published", "model/shaltop", "model/atmosphere_profiles", "datacube",
]


@pytest.mark.parametrize("group", PUBLISHED)
def test_every_group_has_terms(group):
    src = source_for(group, CFG["sources"])
    assert all(src.get(k) for k in ("source", "license", "attribution", "checked"))


def test_longest_prefix_wins():
    assert "seismic_lp" not in source_for("stations/seismic", CFG["sources"])["source"]
    assert "IU" in source_for("stations/seismic_lp", CFG["sources"])["source"]


def test_synoptic_is_not_publishable():
    with pytest.raises(KeyError):
        source_for("stations/synoptic", CFG["sources"])
