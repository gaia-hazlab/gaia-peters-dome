"""Export the data bundle for the 3D atlas prototype: docs/atlas/data/.

Reads the container store (default data/peters_dome.zarr) and writes small browser-ready files.
Nothing is recomputed: every layer is a group of the store, resampled for the web.

  meta.json            grid, event, time bases, layer list, provenance (store, git commit, build date)
  terrain.bin          uint16 elevation, ArcticDEM v4.1 mosaic (pre-event), 40 m, value = offset + 0.1 m * n
  tex_*.png            drapes on the datacube grid at 20 m (Sentinel-2 before/after, elevation change,
                       SHALTOP max thickness and speed, hillshade)
  shaltop_frames.png   SHALTOP run 0520 vertical thickness, 81 frames of 151 x 251 (20 m) in a 9 x 9 sprite
  outlines.json        landslide source and total footprint, EPSG:32605 metres and lon/lat
  force.json           published force-time function (up, north, east; best, lower, upper)
  seismic.bin          vertical-component envelopes, 1-5 Hz, 1 sample/s, 48 stations, uint8 per-trace normalised
  stations.json        seismic and infrasound station list (distance, back azimuth)
  stations.geojson, outlines.geojson   the same in lon/lat, for 2D maps (the GAIA dashboard reads these)

Usage:  pixi run atlas            (python scripts/build_atlas.py --store data/peters_dome.zarr)
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import LightSource
from PIL import Image
from scipy.ndimage import distance_transform_edt, uniform_filter1d
from scipy.signal import butter, hilbert, sosfiltfilt

from peters_dome import common

OUT = Path(__file__).resolve().parents[1] / "docs" / "atlas" / "data"
EV = common.event()["event"]
T0 = np.datetime64(EV["origin_time_utc"].rstrip("Z"), "ns")
TERRAIN_STEP = 4      # datacube 10 m -> 40 m mesh
TEX_STEP = 2          # datacube 10 m -> 20 m drapes


def rel_s(t):
    return (np.asarray(t, dtype="datetime64[ns]") - T0) / np.timedelta64(1, "s")


def fill_nan(a):
    """Nearest-neighbour fill, so the mesh has no holes (5 % of the AOI edge has no DEM)."""
    a = np.asarray(a, dtype="float64")
    m = np.isnan(a)
    if not m.any():
        return a
    idx = distance_transform_edt(m, return_distances=False, return_indices=True)
    return a[tuple(idx)]


def png(path, rgba):
    Image.fromarray(np.ascontiguousarray(rgba)).save(path, optimize=True)
    return path.name


def colorize(a, cmap, vmin, vmax, base=None, alpha_mask=None):
    """RGBA uint8 of `a` through `cmap`; NaN (or alpha_mask False) shows `base` (a hillshade) instead."""
    x = np.clip((a - vmin) / (vmax - vmin), 0, 1)
    rgb = colormaps[cmap](np.nan_to_num(x))[..., :3]
    show = ~np.isnan(a) if alpha_mask is None else (alpha_mask & ~np.isnan(a))
    if base is not None:
        rgb = np.where(show[..., None], 0.8 * rgb + 0.2 * base[..., None], base[..., None])
    out = np.empty(a.shape + (4,), np.uint8)
    out[..., :3] = (rgb * 255).astype(np.uint8)
    out[..., 3] = 255
    return out


def truecolor(dc, epoch, base):
    r, g, b = (dc[f"s2_{c}"].sel(epoch=epoch).values for c in ("red", "green", "blue"))
    rgb = np.stack([r, g, b], -1)
    lo, hi = np.nanpercentile(rgb, [1, 98])                     # one stretch for all bands keeps the colour balance
    rgb = np.clip((rgb - lo) / (hi - lo), 0, 1) ** 0.8
    nan = np.isnan(rgb).any(-1)
    rgb = np.where(nan[..., None], base[..., None], np.nan_to_num(rgb))
    out = np.full(rgb.shape[:2] + (4,), 255, np.uint8)
    out[..., :3] = (rgb * 255).astype(np.uint8)
    return out


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", default="data/peters_dome.zarr")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    store = args.store
    dc = common.open_group(store, "datacube")

    # ---- terrain (pre-event ArcticDEM mosaic) and hillshade
    z10 = fill_nan(dc["elevation_arcticdem_mosaic"].values)
    zt = z10[::TERRAIN_STEP, ::TERRAIN_STEP]
    zmin = float(np.floor(zt.min()))
    (OUT / "terrain.bin").write_bytes(np.round((zt - zmin) / 0.1).astype("<u2").tobytes())
    z20 = z10[::TEX_STEP, ::TEX_STEP]
    shade = LightSource(azdeg=315, altdeg=40).hillshade(z20, vert_exag=1.5, dx=10 * TEX_STEP, dy=10 * TEX_STEP)
    shade = 0.35 + 0.6 * shade

    sub = dc.isel(x=slice(None, None, TEX_STEP), y=slice(None, None, TEX_STEP))
    tex = {}
    tex["s2_before"] = png(OUT / "tex_s2_before.png", truecolor(sub, "before", shade))
    tex["s2_after"] = png(OUT / "tex_s2_after.png", truecolor(sub, "after", shade))
    tex["dh"] = png(OUT / "tex_dh.png", colorize(sub["dh_post_minus_mosaic"].values, "RdBu_r", -40, 40, shade))
    tex["thickness"] = png(OUT / "tex_thickness.png", colorize(
        sub["shaltop_max_thickness"].values, "magma_r", 0, 60, shade, sub["shaltop_max_thickness"].values > 0.5))
    tex["speed"] = png(OUT / "tex_speed.png", colorize(
        sub["shaltop_max_speed"].values, "viridis", 0, 70, shade, sub["shaltop_max_speed"].values > 0.5))
    hs = np.full(shade.shape + (4,), 255, np.uint8)
    hs[..., :3] = (shade[..., None] * np.array([1.0, 0.99, 0.96]) * 255).clip(0, 255).astype(np.uint8)
    tex["hillshade"] = png(OUT / "tex_hillshade.png", hs)

    x, y = dc.x.values, dc.y.values
    grid = {"crs": "EPSG:32605", "x0": float(x[0]), "y0": float(y[0]), "dx": float(x[1] - x[0]),
            "dy": float(y[1] - y[0]), "nx": int(x.size), "ny": int(y.size)}

    # ---- SHALTOP frames (thickness, 0-60 m to 0-255) in a 9 x 9 sprite
    sh = common.open_group(store, "model/shaltop")
    th = np.nan_to_num(sh["vertical_thickness"].values)          # (sim_time, y, x)
    nt, ny_s, nx_s = th.shape
    cols = int(np.ceil(np.sqrt(nt)))
    sprite = np.zeros((cols * ny_s, cols * nx_s), np.uint8)
    for k in range(nt):
        r, c = divmod(k, cols)
        sprite[r * ny_s:(r + 1) * ny_s, c * nx_s:(c + 1) * nx_s] = np.clip(th[k] / 60 * 255, 0, 255)
    Image.fromarray(sprite, "L").save(OUT / "shaltop_frames.png", optimize=True)
    sx, sy = sh.x.values, sh.y.values
    shaltop = {"file": "shaltop_frames.png", "frames": nt, "cols": cols, "nx": nx_s, "ny": ny_s,
               "x0": float(sx[0]), "y0": float(sy[0]), "dx": float(sx[1] - sx[0]), "dy": float(sy[1] - sy[0]),
               "t_rel_s": [round(float(v), 3) for v in rel_s(sh.time.values)], "max_m": 60,
               "note": sh.attrs.get("time_reference", "")}

    # ---- outlines
    ol_utm = common.open_vectors(store, "vectors/landslide_outlines").to_crs("EPSG:32605")
    ol_ll = ol_utm.to_crs("EPSG:4326")
    outlines = [{"name": n, "area_km2": round(float(a) / 1e6, 3),
                 "utm": [[round(px, 1), round(py, 1)] for px, py in g.exterior.coords]}
                for n, a, g in zip(ol_utm["name"], ol_utm["area_m2"], ol_utm.geometry)]
    (OUT / "outlines.json").write_text(json.dumps(outlines))
    (OUT / "outlines.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"name": f"landslide {n}", "area_km2": round(float(a) / 1e6, 3),
                                           "source": "USGS data release doi:10.5066/P13WIS6X"},
         "geometry": {"type": "Polygon", "coordinates": [[[round(px, 5), round(py, 5)] for px, py in g.exterior.coords]]}}
        for n, a, g in zip(ol_ll["name"], ol_ll["area_m2"], ol_ll.geometry)]}))

    # ---- force-time function
    f = common.open_group(store, "model/force_published")
    force = {"t_rel_s": [round(float(v), 3) for v in rel_s(f.time.values)], "units": "N",
             "components": {str(c): {str(b): [float(v) for v in f.force.sel(component=c, bound=b).values]
                                     for b in f.bound.values} for c in f.component.values}}
    (OUT / "force.json").write_text(json.dumps(force))

    # ---- seismic envelopes (Z, 1-5 Hz) and station lists
    seis = common.open_group(store, "stations/seismic")
    fs = 1 / float((seis.time.values[1] - seis.time.values[0]) / np.timedelta64(1, "s"))
    sos = butter(4, [1, 5], btype="band", fs=fs, output="sos")
    zc = seis["velocity"].sel(component="Z").values
    order = np.argsort(seis.distance_km.values)
    step = round(fs)
    env = []
    for i in order:
        tr = np.nan_to_num(zc[i] - np.nanmean(zc[i]))
        e = np.abs(hilbert(sosfiltfilt(sos, tr)))
        e = uniform_filter1d(e, step)[::step]
        env.append(np.clip(e / (np.percentile(e, 99.5) or 1), 0, 1))
    env = np.asarray(env)
    (OUT / "seismic.bin").write_bytes((env * 255).astype(np.uint8).tobytes())
    t_seis = rel_s(seis.time.values[::step])

    def stlist(ds, kind):
        return [{"kind": kind, "net": str(ds.network.values[i]), "sta": str(ds.station.values[i]),
                 "lat": round(float(ds.lat.values[i]), 4), "lon": round(float(ds.lon.values[i]), 4),
                 "dist_km": round(float(ds.distance_km.values[i]), 1),
                 "baz_deg": round(float(ds.back_azimuth_deg.values[i]), 1)} for i in range(ds.sizes["geometry"])]

    infra = common.open_group(store, "stations/infrasound")
    seis_list = [stlist(seis, "seismic")[i] for i in order]
    stations = {"seismic": seis_list, "infrasound": stlist(infra, "infrasound")}
    (OUT / "stations.json").write_text(json.dumps(stations))
    (OUT / "stations.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [s["lon"], s["lat"]]},
         "properties": {"network": s["net"], "station": s["sta"], "kind": s["kind"], "distance_km": s["dist_km"]}}
        for s in seis_list + stations["infrasound"]]}))

    meta = {
        "event": EV, "grid": grid,
        "terrain": {"file": "terrain.bin", "step": TERRAIN_STEP, "nx": int(zt.shape[1]), "ny": int(zt.shape[0]),
                    "offset_m": zmin, "scale_m": 0.1, "source": "ArcticDEM v4.1 mosaic (pre-event), datacube grid"},
        "textures": {"step": TEX_STEP, "nx": int(z20.shape[1]), "ny": int(z20.shape[0]), "files": tex,
                     "s2_dates": [str(v)[:10] for v in dc.s2_time.values]},
        "shaltop": shaltop,
        "seismic": {"file": "seismic.bin", "n": int(env.shape[0]), "nt": int(env.shape[1]),
                    "t0_rel_s": round(float(t_seis[0]), 3), "dt_s": 1.0, "band_hz": [1, 5], "component": "Z"},
        "provenance": {"store": store, "repo_commit": git_commit(), "built_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                       "built_by": "scripts/build_atlas.py", "data_release_doi": "10.5066/P13WIS6X",
                       "paper_doi": "10.1029/2025JF008577"},
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    total = sum(p.stat().st_size for p in OUT.iterdir())
    print(f"wrote {len(list(OUT.iterdir()))} files, {total / 1e6:.1f} MB to {OUT}")


if __name__ == "__main__":
    main()
