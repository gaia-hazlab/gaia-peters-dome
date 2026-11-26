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
  infrasound.bin       pressure envelopes, 0.5-5 Hz, 1 sample/s, 31 sensors, same encoding
  stations.json        seismic and infrasound station list (distance, back azimuth, x/y km from the event)
  stations.geojson, outlines.geojson   the same in lon/lat, for 2D maps (the GAIA dashboard reads these)
  region_terrain.bin, region_tex.png   relief of the station network, +-420 km around the event, for the
                       network view. The only input not in the store: AWS Terrain Tiles (terrarium, zoom 6,
                       16 tiles, cached in data/cache/terrarium/), resampled to an azimuthal-equidistant grid.

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
REGION_HALF_KM = 420  # network view: farthest sensor is 390 km away
REGION_STEP_KM = 2    # mesh; the drape is at half this step
TERRARIUM = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
TERRARIUM_Z = 6       # ~1.1 km per pixel at 63 N
CACHE = Path(__file__).resolve().parents[1] / "data" / "cache" / "terrarium"


def aeqd():
    """Azimuthal-equidistant projection centred on the event: x east, y north, metres."""
    from pyproj import Transformer
    crs = f"+proj=aeqd +lat_0={EV['latitude']} +lon_0={EV['longitude']} +datum=WGS84 +units=m"
    return Transformer.from_crs("EPSG:4326", crs, always_xy=True), Transformer.from_crs(crs, "EPSG:4326", always_xy=True)


def regional_relief(step_km):
    """Elevation (m, bathymetry negative) on a +-REGION_HALF_KM grid around the event, from Terrain Tiles."""
    import math
    from urllib.request import urlopen
    _, inv = aeqd()
    n = int(2 * REGION_HALF_KM / step_km) + 1
    ax = np.linspace(-REGION_HALF_KM, REGION_HALF_KM, n) * 1e3
    X, Y = np.meshgrid(ax, ax[::-1])                       # row 0 = north
    lon, lat = inv.transform(X, Y)
    z2 = 2 ** TERRARIUM_Z
    px = (lon + 180) / 360 * z2 * 256                      # global Web Mercator pixel coordinates
    py = (1 - np.arcsinh(np.tan(np.radians(lat))) / math.pi) / 2 * z2 * 256
    tx0, tx1 = int(px.min() // 256), int(px.max() // 256)
    ty0, ty1 = int(py.min() // 256), int(py.max() // 256)
    CACHE.mkdir(parents=True, exist_ok=True)
    mos = np.zeros(((ty1 - ty0 + 1) * 256, (tx1 - tx0 + 1) * 256), np.float32)
    for ty in range(ty0, ty1 + 1):
        for tx in range(tx0, tx1 + 1):
            f = CACHE / f"{TERRARIUM_Z}_{tx}_{ty}.png"
            if not f.exists():
                f.write_bytes(urlopen(TERRARIUM.format(z=TERRARIUM_Z, x=tx, y=ty), timeout=60).read())
            rgb = np.asarray(Image.open(f).convert("RGB"), np.float32)
            mos[(ty - ty0) * 256:(ty - ty0 + 1) * 256, (tx - tx0) * 256:(tx - tx0 + 1) * 256] = \
                rgb[..., 0] * 256 + rgb[..., 1] + rgb[..., 2] / 256 - 32768
    from scipy.ndimage import map_coordinates
    return map_coordinates(mos, [py - ty0 * 256, px - tx0 * 256], order=1, mode="nearest"), n


def relief_texture(z, step_km):
    """Hillshaded hypsometric tint on land, shaded blue at sea (the network view's drape)."""
    ls = LightSource(azdeg=315, altdeg=35)
    shade = ls.hillshade(z, vert_exag=6, dx=step_km * 1e3, dy=step_km * 1e3)
    land = colormaps["gist_earth"](np.clip(0.38 + z / 6000 * 0.6, 0.38, 0.98))[..., :3]
    sea = colormaps["Blues"](np.clip(0.35 - z / 6000, 0.35, 0.9))[..., :3]
    land = 0.55 * land + 0.45 * np.array([0.62, 0.6, 0.55])           # muted, closer to the atlas palette
    rgb = np.where((z > 0)[..., None], land * (0.45 + 0.6 * shade[..., None]), sea * (0.85 + 0.15 * shade[..., None]))
    out = np.full(z.shape + (4,), 255, np.uint8)
    out[..., :3] = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    return out


def envelopes(x, fs, band, t_rel, noise_window=(-900, -600)):
    """Band-pass, Hilbert envelope, 1 sample/s.

    Each trace is scaled by max(its 99.9th percentile, 10 x its median pre-event envelope), so background
    noise sits near 0.1 and arrivals stand out; a trace with no clear arrival stays low.
    The noise window ends before the ComCat precursor (-475 s).
    """
    sos = butter(4, band, btype="band", fs=fs, output="sos")
    step = round(fs)
    t1 = t_rel[::step]
    pre = (t1 >= noise_window[0]) & (t1 < noise_window[1])
    out = []
    for tr in x:
        tr = np.nan_to_num(tr - np.nanmean(tr))
        e = uniform_filter1d(np.abs(hilbert(sosfiltfilt(sos, tr))), step)[::step]
        scale = max(np.percentile(e, 99.9), 10 * np.median(e[pre[:e.size]])) or 1
        out.append(np.clip(e / scale, 0, 1))
    return np.asarray(out), step


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
    order = np.argsort(seis.distance_km.values)
    env, step = envelopes(seis["velocity"].sel(component="Z").values[order], fs, [1, 5], rel_s(seis.time.values))
    (OUT / "seismic.bin").write_bytes((env * 255).astype(np.uint8).tobytes())
    t_seis = rel_s(seis.time.values[::step])

    infra = common.open_group(store, "stations/infrasound")
    fsi = 1 / float((infra.time.values[1] - infra.time.values[0]) / np.timedelta64(1, "s"))
    iorder = np.argsort(infra.distance_km.values)
    ienv, istep = envelopes(infra["pressure"].values[iorder], fsi, [0.5, 5], rel_s(infra.time.values))
    (OUT / "infrasound.bin").write_bytes((ienv * 255).astype(np.uint8).tobytes())
    t_infra = rel_s(infra.time.values[::istep])

    fwd, _ = aeqd()

    def stlist(ds, kind):
        x, y = fwd.transform(ds.lon.values, ds.lat.values)
        return [{"kind": kind, "net": str(ds.network.values[i]), "sta": str(ds.station.values[i]),
                 "lat": round(float(ds.lat.values[i]), 4), "lon": round(float(ds.lon.values[i]), 4),
                 "elev_m": round(float(ds.elevation.values[i]), 1),
                 "x_km": round(float(x[i]) / 1e3, 2), "y_km": round(float(y[i]) / 1e3, 2),
                 "dist_km": round(float(ds.distance_km.values[i]), 1),
                 "baz_deg": round(float(ds.back_azimuth_deg.values[i]), 1)} for i in range(ds.sizes["geometry"])]

    seis_list = [stlist(seis, "seismic")[i] for i in order]
    stations = {"seismic": seis_list, "infrasound": [stlist(infra, "infrasound")[i] for i in iorder]}

    # ---- network view: relief around the event (Terrain Tiles; not in the store)
    zr, nr = regional_relief(REGION_STEP_KM)
    (OUT / "region_terrain.bin").write_bytes(np.round(zr).astype("<i2").tobytes())
    zt2, _ = regional_relief(REGION_STEP_KM / 2)
    png(OUT / "region_tex.png", relief_texture(zt2, REGION_STEP_KM / 2))
    from pyproj import Transformer
    ex, ey = Transformer.from_crs("EPSG:4326", "EPSG:32605", always_xy=True).transform(EV["longitude"], EV["latitude"])
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
        "infrasound": {"file": "infrasound.bin", "n": int(ienv.shape[0]), "nt": int(ienv.shape[1]),
                       "t0_rel_s": round(float(t_infra[0]), 3), "dt_s": 1.0, "band_hz": [0.5, 5]},
        "event_utm": [round(ex, 1), round(ey, 1)],
        "region": {"file": "region_terrain.bin", "texture": "region_tex.png", "n": nr, "half_km": REGION_HALF_KM,
                   "step_km": REGION_STEP_KM, "units": "m, int16, bathymetry negative",
                   "projection": "azimuthal equidistant centred on the event (ComCat location)",
                   "source": "AWS Terrain Tiles (terrarium, zoom 6), s3://elevation-tiles-prod"},
        "provenance": {"store": store, "repo_commit": git_commit(), "built_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                       "built_by": "scripts/build_atlas.py", "data_release_doi": "10.5066/P13WIS6X",
                       "paper_doi": "10.1029/2025JF008577"},
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    total = sum(p.stat().st_size for p in OUT.iterdir())
    print(f"wrote {len(list(OUT.iterdir()))} files, {total / 1e6:.1f} MB to {OUT}")


if __name__ == "__main__":
    main()
