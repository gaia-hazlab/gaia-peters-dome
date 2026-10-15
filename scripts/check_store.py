"""Validate the container store against the gaia-cli conventions.

- rasters/*  : dims include x and y, spatial_ref is a coordinate, CRS readable; if a
               time_bounds variable exists, time.attrs["bounds"] points at it and timed
               variables carry cell_methods (what `gaia align` relies on)
- stations/* and vectors/* : decode with xvec (CF geometries round-trip)
- model/*    : opens
Exit code 1 if any group fails.

    pixi run check
"""

from __future__ import annotations

import argparse
import sys

from gaia_cli import io as gio

from peters_dome import common


def check_raster(ds) -> list[str]:
    errs = []
    if not {"x", "y"} <= set(ds.dims):
        errs.append(f"dims {tuple(ds.dims)} lack x/y")
    if "spatial_ref" not in ds.coords:
        errs.append("spatial_ref is not a coordinate")
    elif common.with_crs(ds).rio.crs is None:
        errs.append("CRS unreadable")
    if "time_bounds" in ds:
        if ds.time.attrs.get("bounds") != "time_bounds":
            errs.append("time.attrs['bounds'] missing")
        missing = [v for v in ds.data_vars if "time" in ds[v].dims and v != "time_bounds"
                   and "cell_methods" not in ds[v].attrs]
        if missing:
            errs.append(f"no cell_methods on {missing}")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=common.STORE)
    a = ap.parse_args()
    tree = gio.open_datatree(a.store)
    failed = 0
    for path, node in tree.subtree_with_keys:
        ds = node.to_dataset()
        if not ds.data_vars:
            continue
        top = path.split("/")[0]
        try:
            if top in ("stations", "vectors"):
                ds.xvec.decode_cf()
                errs = []
            elif top in ("rasters", "datacube"):
                errs = check_raster(ds)
            else:
                errs = []
        except Exception as e:  # noqa: BLE001  report every group, then fail
            errs = [f"{type(e).__name__}: {e}"]
        status = "ok  " if not errs else "FAIL"
        failed += bool(errs)
        print(f"{status} {path:42s} {dict(ds.sizes)}" + ("" if not errs else f"  {'; '.join(errs)}"))
    print(f"{failed} group(s) failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
