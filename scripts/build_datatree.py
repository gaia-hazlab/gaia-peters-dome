"""Assemble every staged store in data/staged/ into one xr.DataTree Zarr store.

Group layout follows gaia-cli ``stage all`` (stations/, rasters/) plus vectors/ and model/.
Missing stores are skipped with a message, so a partial staging run still builds.

    pixi run build-tree                      # -> data/peters_dome.zarr
    pixi run python scripts/build_datatree.py --staged data/staged --output s3://.../peters_dome.zarr
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import xarray as xr

from peters_dome import common

# store name -> group path. For stores that are already DataTrees, children are grafted
# under the given prefix.
FLAT = {
    "seismic.zarr": "stations/seismic",
    "seismic_lp.zarr": "stations/seismic_lp",
    "infrasound.zarr": "stations/infrasound",
    "met_synoptic.zarr": "stations/synoptic",
    "sar_s1_rtc.zarr": "rasters/sar/sentinel1_rtc",
    "optical_s2.zarr": "rasters/optical/sentinel2",
    "optical_landsat.zarr": "rasters/optical/landsat",
    "dem_copdem.zarr": "rasters/topography/copdem",
    "dem_arcticdem_mosaic.zarr": "rasters/topography/arcticdem_mosaic",
    "dem_arcticdem_strips.zarr": "rasters/topography/arcticdem_strips",
    "soil.zarr": "rasters/soil/soilgrids",
    "met_era5.zarr": "rasters/met/era5",
    "met_era5_land.zarr": "rasters/met/era5_land",
    "met_profiles.zarr": "model/atmosphere_profiles",
}
TREES = {
    "glacier.zarr": {"itslive_velocity": "rasters/glacier/itslive_velocity",
                     "*": "vectors/glacier/{}"},
    "geology.zarr": {"*": "vectors/geology/{}"},
    "usgs_release.zarr": {"force": "model/force_inversion",
                          "shaltop": "model/shaltop",
                          "dem_post": "rasters/topography/setsm_post_event",
                          "outlines": "vectors/landslide_outlines"},
}


def _open(path: Path) -> xr.Dataset:
    return xr.open_zarr(path, consolidated=False).drop_encoding()


def build(staged: Path) -> xr.DataTree:
    nodes: dict[str, xr.Dataset] = {}
    for name, group in FLAT.items():
        p = staged / name
        if not p.exists():
            print(f"  missing {name}, skipped")
            continue
        nodes[group] = _open(p)
    for name, mapping in TREES.items():
        p = staged / name
        if not p.exists():
            print(f"  missing {name}, skipped")
            continue
        tree = xr.open_datatree(p, engine="zarr", consolidated=False)
        for child in tree.children:
            target = mapping.get(child, mapping.get("*", "").format(child))
            if target:
                nodes[target] = tree[child].to_dataset().drop_encoding()
    ev = common.event()
    root = xr.Dataset(attrs={
        **common.provenance(),
        **{f"event_{k}": str(v) for k, v in ev["event"].items()},
        "aoi_local": "aoi/peters_dome_local.geojson",
        "aoi_regional": "aoi/peters_dome_regional.geojson",
        "aoi_waveforms": "aoi/peters_dome_waveforms.geojson",
        "vector_encoding": "xvec CF (decode with ds.xvec.decode_cf())",
    })
    nodes["/"] = root
    return xr.DataTree.from_dict(nodes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--staged", default="data/staged")
    ap.add_argument("--output", default="data/peters_dome.zarr")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    if a.overwrite and not a.output.startswith("s3://"):
        shutil.rmtree(a.output, ignore_errors=True)
    common.check_s3_env(a.output)
    dt = build(Path(a.staged))
    print(dt.groups)
    common.save_datatree(dt, a.output)


if __name__ == "__main__":
    main()
