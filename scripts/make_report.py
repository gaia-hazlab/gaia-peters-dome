"""Render the GitHub Pages report: docs/index.html + docs/figures/*.png.

Reads the stores in data/staged/ (run scripts/stage_all.sh first). A panel whose store is
missing is listed as "not staged" in the page instead of failing the build.

    pixi run report
"""

from __future__ import annotations

import argparse
import html
import json
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
import xvec  # noqa: E402, F401
from matplotlib.colors import LightSource, TwoSlopeNorm  # noqa: E402
from scipy.signal import butter, sosfiltfilt  # noqa: E402

from peters_dome import common  # noqa: E402

# Reference palette (dataviz skill, references/palette.md), light mode.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
COAST = "https://naciscdn.org/naturalearth/50m/physical/ne_50m_coastline.zip"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "font.size": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.titlesize": 10, "axes.titleweight": "bold",
    "legend.frameon": False,
})

EV = common.event()["event"]
T0 = np.datetime64(EV["origin_time_utc"].rstrip("Z"), "ns")
TPRE = np.datetime64(EV["precursor_time_utc"].rstrip("Z"), "ns")


def _open(staged: Path, name: str, group: str | None = None, xvec_decode=False):
    p = staged / name
    if not p.exists():
        raise FileNotFoundError(f"{name} not staged")
    if group:
        ds = xr.open_datatree(p, engine="zarr", consolidated=False)[group].to_dataset()
    else:
        ds = xr.open_zarr(p, consolidated=False)
    return ds.xvec.decode_cf() if xvec_decode else ds


def _bandpass(x, fs, lo, hi):
    x = np.nan_to_num(np.asarray(x, dtype="float64") - np.nanmean(x))
    sos = butter(4, [lo, min(hi, 0.45 * fs)], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def _outlines(staged: Path, crs="EPSG:32605"):
    p = staged / "release_outlines.geojson"
    return gpd.read_file(p).to_crs(crs) if p.exists() else None


def _draw_outlines(ax, ol):
    if ol is None:
        return
    for name, ls in (("total", "-"), ("source", "--")):
        ol[ol["name"] == name].boundary.plot(ax=ax, color=INK, lw=1.0, ls=ls)


def _rel_s(t):
    return (np.asarray(t, dtype="datetime64[ns]") - T0) / np.timedelta64(1, "s")


# ---- figures -----------------------------------------------------------------------------

def fig_map(staged, out):
    seis = _open(staged, "seismic.zarr", xvec_decode=True)
    infra = _open(staged, "infrasound.zarr", xvec_decode=True)
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    try:
        coast = gpd.read_file(COAST, bbox=(-160, 58, -140, 68))
        coast.plot(ax=ax, color=MUTED, lw=0.6)
    except Exception:
        pass
    lat0, lon0 = EV["latitude"], EV["longitude"]
    for r in (100, 200, 300, 400):
        az = np.radians(np.arange(0, 361, 2))
        d = r / 6371
        la = np.arcsin(np.sin(np.radians(lat0)) * np.cos(d) + np.cos(np.radians(lat0)) * np.sin(d) * np.cos(az))
        lo = np.radians(lon0) + np.arctan2(np.sin(az) * np.sin(d) * np.cos(np.radians(lat0)),
                                           np.cos(d) - np.sin(np.radians(lat0)) * np.sin(la))
        ax.plot(np.degrees(lo), np.degrees(la), color=GRID, lw=0.8, zorder=1)
        ax.text(np.degrees(lo[45]) + 0.1, np.degrees(la[45]), f"{r} km", color=MUTED, fontsize=7, va="bottom")
    sx = [g.x for g in seis.geometry.values]
    sy = [g.y for g in seis.geometry.values]
    ax.scatter(sx, sy, marker="^", s=42, color=BLUE, edgecolor="white", lw=0.8, zorder=3,
               label=f"seismic, {len(sx)} stations (HH/BH)")
    ix = [g.x for g in infra.geometry.values]
    iy = [g.y for g in infra.geometry.values]
    ax.scatter(ix, iy, marker="o", s=34, color=ORANGE, edgecolor="white", lw=0.8, zorder=4,
               label=f"infrasound, {len(ix)} sensors (BDF/HDF)")
    i53 = [s for s in infra.station.values if "I53" in str(s)]
    if i53:
        g = infra.sel(geometry=infra.station.isin(i53).values).geometry.values
        cx, cy = np.mean([p.x for p in g]), np.mean([p.y for p in g])
        ax.annotate("I53US array", (cx, cy), xytext=(cx + 0.6, cy + 0.35), color=INK2, fontsize=8,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.scatter([lon0], [lat0], marker="*", s=260, color=INK, edgecolor="white", lw=1.0, zorder=5,
               label="Peters Dome landslide")
    ax.set_xlim(lon0 - 9.5, lon0 + 9.5)
    ax.set_ylim(lat0 - 3.8, lat0 + 3.8)
    ax.set_aspect(1 / np.cos(np.radians(lat0)))
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    rows = []
    for ds, kind in ((seis, "seismic"), (infra, "infrasound")):
        for k in range(ds.sizes["geometry"]):
            rows.append(dict(type=kind, station=str(ds.station.values[k]),
                             channel=str(ds.channel.values[k]),
                             distance_km=round(float(ds.distance_km.values[k]), 1),
                             back_azimuth_deg=round(float(ds.back_azimuth_deg.values[k]), 1)))
    return pd.DataFrame(rows).sort_values(["type", "distance_km"])


def _section(ax, da, fs, lo, hi, tmin, tmax, nmax=30, scale=0.9, color=INK, label_every=1):
    dist = da.distance_km.values
    good = [k for k in np.argsort(dist) if not np.isnan(da.isel(geometry=k).values).all()]
    min_sep = (dist[good[-1]] - dist[good[0]]) / nmax
    keep = []
    for k in good:  # thin to a minimum distance spacing; also keeps one element per array
        if not keep or dist[k] - dist[keep[-1]] >= min_sep:
            keep.append(k)
    t = _rel_s(da.time.values)
    sel = (t >= tmin) & (t <= tmax)
    spacing = min_sep
    for n, k in enumerate(keep):
        x = da.isel(geometry=k).values
        y = _bandpass(x, fs, lo, hi)[sel]
        y = y / (np.abs(y).max() or 1) * spacing * scale
        ax.plot(t[sel], dist[k] + y, color=color, lw=0.5)
        if n % label_every == 0:
            ax.text(tmax, dist[k], f" {da.station.values[k]}", fontsize=6, color=MUTED, va="center")
    ax.set_xlim(tmin, tmax)
    ax.set_xlabel(f"time after {str(T0)[:19]} UTC (s)")
    ax.set_ylabel("distance from source (km)")


def fig_seismic(staged, out):
    ds = _open(staged, "seismic.zarr", xvec_decode=True)
    fs = float(ds.velocity.attrs.get("sampling_rate_hz", 20))
    da = ds.velocity.sel(component="Z")
    fig, axes = plt.subplots(1, 2, figsize=(10, 6.2), sharey=True)
    _section(axes[0], da, fs, 1.0, 5.0, -60, 240, nmax=30)
    axes[0].set_title("Vertical velocity, 1–5 Hz")
    _section(axes[1], da, fs, 1 / 100, 1 / 15, -60, 240, nmax=30, color=BLUE)
    axes[1].set_title("Vertical velocity, 15–100 s")
    axes[1].set_ylabel("")
    for ax in axes:
        ax.axvline(0, color=ORANGE, lw=1)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return ds.sizes["geometry"]


def fig_infrasound(staged, out):
    ds = _open(staged, "infrasound.zarr", xvec_decode=True)
    fs = float(ds.pressure.attrs.get("sampling_rate_hz", 20))
    da = ds.pressure
    fig, ax = plt.subplots(figsize=(10, 6.4))
    _section(ax, da, fs, 0.5, 5.0, -600, 1800, nmax=40, color=INK)
    d = np.array([0, 420])
    for c, ls in ((0.34, "-"), (0.30, "--"), (0.25, ":")):
        ax.plot(d / c, d, color=ORANGE, lw=1, ls=ls, label=f"{c:.2f} km/s")
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.axvline(_rel_s(TPRE), color=MUTED, lw=0.8, ls=":")
    ax.set_ylim(da.distance_km.min() - 15, da.distance_km.max() + 15)
    ax.set_title("Infrasound pressure, 0.5–5 Hz, each trace normalized")
    ax.legend(title="celerity", loc="upper left", fontsize=8, framealpha=0.9, frameon=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return ds.sizes["geometry"]


def fig_i53(staged, out):
    ds = _open(staged, "infrasound.zarr", xvec_decode=True)
    fs = float(ds.pressure.attrs.get("sampling_rate_hz", 20))
    m = np.array(["I53" in str(s) for s in ds.station.values])
    if not m.any():
        raise FileNotFoundError("no I53US elements in infrasound store")
    da = ds.pressure.isel(geometry=np.where(m)[0])
    t = _rel_s(da.time.values)
    dist = float(da.distance_km.mean())
    tmin, tmax = 300, 1100
    sel = (t >= tmin) & (t <= tmax)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for k in range(da.sizes["geometry"]):
        y = _bandpass(da.isel(geometry=k).values, fs, 0.5, 5.0)[sel]
        ax.plot(t[sel], y + k * 0.6 * np.nanmax(np.abs(y)) * 2, color=INK, lw=0.5)
        ax.text(tmax, k * 0.6 * np.nanmax(np.abs(y)) * 2, f" {da.station.values[k]}", fontsize=7, color=MUTED, va="center")
    c = 0.29
    ax.axvline(dist / c, color=ORANGE, lw=1.2, label=f"ComCat landslide origin + {dist:.0f} km / {c} km/s")
    ax.axvline(_rel_s(TPRE) + dist / c, color=ORANGE, lw=1.2, ls="--",
               label=f"ComCat precursor ({str(TPRE)[11:19]}) + same travel time")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9, frameon=True)
    ax.set_xlim(tmin, tmax)
    ax.set_yticks([])
    ax.set_xlabel(f"time after {str(T0)[:19]} UTC (s)")
    ax.set_title(f"IMS array I53US, 8 elements at {dist:.0f} km, 0.5–5 Hz (Pa, offset)")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_force(staged, out):
    ds = _open(staged, "usgs_release.zarr", "force")
    t = ds.time_offset_s.values
    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    for ax, comp in zip(axes, ["up", "north", "east"]):
        f = ds.force.sel(component=comp) / 1e9
        ax.fill_between(t, f.sel(bound="lower"), f.sel(bound="upper"), color=BLUE, alpha=0.2, lw=0)
        ax.plot(t, f.sel(bound="best"), color=BLUE, lw=1.6)
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_ylabel(f"{comp} (GN)")
    axes[-1].set_xlabel(f"time after inversion zero time {ds.force.attrs.get('zero_time_utc', '')} UTC (s)")
    axes[0].set_title("Force on the Earth, 15–200 s; band = jackknife bounds")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_sar(staged, out):
    ds = common_crs(_open(staged, "sar_s1_rtc.zarr"))
    t = ds.time.values
    b, a = np.where(t < T0)[0][-1], np.where(t > T0)[0][0]
    vb, va = ds.vv.isel(time=b).load(), ds.vv.isel(time=a).load()
    db = lambda x: 10 * np.log10(x)  # noqa: E731
    lr = db(va) - db(vb)
    ol = _outlines(staged)
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.2), sharex=True, sharey=True, layout="constrained")
    for ax, x, ttl in ((axes[0], db(vb), f"VV before, {str(t[b])[:10]}"),
                       (axes[1], db(va), f"VV after, {str(t[a])[:10]}")):
        im = x.plot.imshow(ax=ax, cmap="gray", vmin=-25, vmax=5, add_colorbar=False)
        ax.set_title(ttl)
    fig.colorbar(im, ax=axes[:2], shrink=0.7, label="γ⁰ VV (dB)")
    im2 = lr.plot.imshow(ax=axes[2], cmap="RdBu_r", norm=TwoSlopeNorm(0, -6, 6), add_colorbar=False)
    fig.colorbar(im2, ax=axes[2], shrink=0.7, label="after − before (dB)")
    axes[2].set_title("Change")
    for ax in axes:
        _draw_outlines(ax, ol)
        ax.set_aspect("equal")
        ax.set_ylabel("")
    _km_axes(*axes)
    orb = f"{str(ds.orbit_state.values[a])} track {int(ds.relative_orbit.values[a])}" if "orbit_state" in ds.coords else ""
    fig.suptitle(f"Sentinel-1 RTC {orb}", fontsize=10, fontweight="bold")
    fig.savefig(out)
    plt.close(fig)


def fig_optical(staged, out):
    ds = common_crs(_open(staged, "optical_s2.zarr"))
    clear = ds.scl.isin([4, 5, 6, 11]).mean(("x", "y")).compute().values
    t = ds.time.values
    bi, ai = np.where(t < T0)[0], np.where(t > T0)[0]
    b, a = bi[np.argmax(clear[bi])], ai[np.argmax(clear[ai])]
    ol = _outlines(staged)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.2), sharex=True, sharey=True)
    for ax, k, lab in ((axes[0], b, "before"), (axes[1], a, "after")):
        rgb = ds[["red", "green", "blue"]].isel(time=k).to_array("band").load()
        rgb = (rgb / 1.0).clip(0, 1) ** (1 / 1.6)
        rgb.plot.imshow(ax=ax, rgb="band")
        _draw_outlines(ax, ol)
        ax.set_title(f"Sentinel-2 {lab}, {str(t[k])[:10]} (clear {clear[k]:.0%})")
        ax.set_aspect("equal")
        ax.set_ylabel("")
    _km_axes(*axes)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_shaltop(staged, out):
    sh = _open(staged, "usgs_release.zarr", "shaltop")
    topo = sh.topography.load()
    dx = float(abs(sh.x[1] - sh.x[0]))
    hs = LightSource(azdeg=315, altdeg=35).hillshade(topo.values[::-1] if sh.y[0] < sh.y[-1] else topo.values, dx=dx, dy=dx)
    ext = [float(sh.x.min()), float(sh.x.max()), float(sh.y.min()), float(sh.y.max())]
    hmax = sh.vertical_thickness.max("sim_time").load().where(lambda v: v > 0.5)
    # thin-film margin cells (< 1 m) carry spurious speeds; mask them before the maximum
    speed = sh.flow_speed.where(sh.vertical_thickness >= 1.0)
    vmax = speed.max("sim_time").load().where(lambda v: v > 1)
    ol = _outlines(staged)
    fig, axes = plt.subplots(1, 2, figsize=(10, 6.4), sharex=True, sharey=True)
    for ax, da, lab, cmap in ((axes[0], hmax, "max flow thickness (m)", "Blues"),
                              (axes[1], vmax, "max flow speed (m/s)", "Oranges")):
        ax.imshow(hs, extent=ext, cmap="gray", origin="upper")
        im = da.plot.imshow(ax=ax, cmap=cmap, alpha=0.85, add_colorbar=False)
        fig.colorbar(im, ax=ax, shrink=0.6, label=lab)
        _draw_outlines(ax, ol)
        ax.set_aspect("equal")
        ax.set_title(lab.split(" (")[0].capitalize())
        ax.set_ylabel("")
    _km_axes(*axes)
    fig.suptitle("SHALTOP run 0520 (USGS release); solid = source + deposit, dashed = source",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return dict(p99=float(np.nanpercentile(vmax, 99)), raw=float(sh.flow_speed.max()))


def fig_met(staged, out):
    ds = _open(staged, "met_era5.zarr")
    node = ds.sel(x=EV["source_area_lon"], y=EV["source_area_lat"], method="nearest").load()
    daily_t = node.temperature.resample(time="1D").mean()
    tmin = node.temperature.resample(time="1D").min()
    tmax = node.temperature.resample(time="1D").max()
    daily_p = node.precipitation.resample(time="1D").sum()
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    axes[0].fill_between(daily_t.time, tmin, tmax, color=BLUE, alpha=0.18, lw=0)
    axes[0].plot(daily_t.time, daily_t, color=BLUE, lw=1.6)
    axes[0].axhline(0, color=MUTED, lw=0.8)
    axes[0].set_ylabel("2 m air temperature (°C)")
    axes[0].set_title(f"ERA5 node {float(node.y):.2f} N, {abs(float(node.x)):.2f} W "
                      f"(model elevation {float(node.node_elevation):.0f} m); daily mean, band = daily min–max")
    axes[1].bar(daily_p.time, daily_p, width=0.8, color=BLUE)
    axes[1].set_ylabel("precipitation (mm/day)")
    for ax in axes:
        ax.axvline(T0, color=ORANGE, lw=1.2)
    axes[0].text(T0, axes[0].get_ylim()[1], " landslide", color=ORANGE, fontsize=8, va="top")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    pre = node.sel(time=slice(T0 - np.timedelta64(14, "D"), T0))
    return dict(t_mean_14d=float(pre.temperature.mean()), p_sum_14d=float(pre.precipitation.sum()),
                t_max_14d=float(pre.temperature.max()), node_elev=float(node.node_elevation))


def fig_profile(staged, out):
    ds = _open(staged, "met_profiles.zarr").load()
    k = int(np.argmin(np.abs(ds.time.values - T0)))
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 5.2), sharey=True)
    for pt, color in (("source", BLUE), ("path_midpoint", AQUA), ("I53US", ORANGE)):
        p = ds.sel(point=pt).isel(time=k)
        z = p.geopotential_height / 1000
        axes[0].plot(p.temperature, z, color=color, lw=1.8, marker="o", ms=3, label=pt.replace("_", " "))
        axes[1].plot(p.wind_speed, z, color=color, lw=1.8, marker="o", ms=3)
    axes[0].axvline(0, color=MUTED, lw=0.8)
    axes[0].set_xlabel("temperature (°C)")
    axes[1].set_xlabel("wind speed (m/s)")
    axes[0].set_ylabel("geopotential height (km)")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Atmospheric profiles at {str(ds.time.values[k])[:16]} UTC, 1000–30 hPa "
                 "(Open-Meteo historical forecast)", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return float(ds.freezing_level_height.sel(point="source").isel(time=k))


def fig_glacier(staged, out):
    vel = common_crs(_open(staged, "glacier.zarr", "itslive_velocity"))
    v = vel.v.load()
    fig, ax = plt.subplots(figsize=(8, 5.6))
    im = v.plot.imshow(ax=ax, cmap="Blues", vmin=0, vmax=float(np.nanpercentile(v, 99)), add_colorbar=False)
    fig.colorbar(im, ax=ax, shrink=0.7, label="surface speed (m/yr)")
    rgi = staged / "glacier_rgi.geojson"
    if rgi.exists():
        gpd.read_file(rgi).to_crs(common.utm_crs()).boundary.plot(ax=ax, color=INK2, lw=0.4)
    ol = _outlines(staged)
    _draw_outlines(ax, ol)
    src = gpd.GeoSeries.from_xy([EV["source_area_lon"]], [EV["source_area_lat"]], crs=4326).to_crs(common.utm_crs())
    ax.scatter(src.x, src.y, marker="*", s=180, color=ORANGE, edgecolor="white", zorder=5)
    ax.set_aspect("equal")
    ax.set_title("ITS_LIVE v2 multi-year speed with RGI outlines (regional AOI)")
    _km_axes(ax)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def _km_axes(*axes):
    from matplotlib.ticker import FuncFormatter

    f = FuncFormatter(lambda v, _: f"{v / 1000:.1f}".rstrip("0").rstrip("."))
    for ax in axes:
        ax.grid(False)
        ax.xaxis.set_major_formatter(f)
        ax.yaxis.set_major_formatter(f)
        ax.set_xlabel("UTM 5N easting (km)")
    axes[0].set_ylabel("northing (km)")


def common_crs(ds):
    if "spatial_ref" in ds.coords or "spatial_ref" in ds:
        wkt = ds["spatial_ref"].attrs.get("crs_wkt") or ds["spatial_ref"].attrs.get("spatial_ref")
        if wkt:
            ds = ds.rio.write_crs(wkt)
    return ds


# ---- page --------------------------------------------------------------------------------

FIGS = [
    # id, function, title, caption template ({result} filled from function return)
    ("map", fig_map, "Stations", "Seismic stations within 250 km (HH/BH, response removed) and infrasound sensors within 400 km (BDF/HDF, sensitivity removed), from the EarthScope FDSN service. Rings every 100 km around the ComCat location. Store: <code>stations/seismic</code>, <code>stations/infrasound</code>."),
    ("seismic", fig_seismic, "Seismic record section", "Vertical velocity at {result} stations on the common 20 Hz grid, zero-phase Butterworth filtered and each trace normalized; traces thinned to a minimum distance spacing so they do not overprint. Orange line: ComCat landslide origin ({t0}). The ComCat precursor ({tpre}) is 475 s earlier, outside this window."),
    ("infrasound", fig_infrasound, "Infrasound record section", "Pressure from {result} sensors, 0.5–5 Hz, each trace normalized; thinned to a minimum distance spacing, so each array shows one element. Orange lines: arrival times for constant celerities from the ComCat origin. Dotted vertical line: ComCat precursor time."),
    ("i53", fig_i53, "I53US array", "The eight elements of IMS array I53US (Fairbanks), ~252 km from the source, 0.5–5 Hz. The main pulse arrives about 870 s after the ComCat origin, an apparent celerity of ~0.29 km/s. Shifting the ComCat precursor time by the same travel time (dashed) lands on the arrival near 400 s; the arrival near 720 s has no ComCat entry. This is consistent with the two precursory events the paper reports, but here it is a visual alignment only: no array processing (back azimuth, trace velocity) has been run on these data."),
    ("force", fig_force, "Force-time function", "Force exerted by the landslide on the Earth, inverted from 15–200 s displacement seismograms (USGS data release, file <code>peters_dome_15-200_s_jackknife.csv</code>). Shaded band: jackknife lower/upper bounds. Zero time 2023-09-14 02:06:27 UTC (release metadata)."),
    ("sar", fig_sar, "Sentinel-1 before / after", "Radiometrically terrain-corrected γ⁰ VV from Planetary Computer <code>sentinel-1-rtc</code>, last scene before and first after the event (12-day repeat, Sentinel-1A only in 2023). Outlines from the USGS release: solid = source + deposit, dashed = source."),
    ("optical", fig_optical, "Sentinel-2 before / after", "Sentinel-2 L2A true colour (Earth Search, collection 1), clearest scene on each side of the event by SCL clear-sky fraction; gamma 1.6 for display. Same outlines as above."),
    ("shaltop", fig_shaltop, "Flow model", "SHALTOP run 0520 from the USGS release over its own topography (hillshade). Speed is shown only where flow thickness is at least 1 m: unmasked, thin margin cells reach {result[raw]:.0f} m/s, which is numerical, not physical. With the mask, the 99th percentile of the per-cell maximum speed is {result[p99]:.0f} m/s; the paper reports speeds exceeding 60 m/s."),
    ("glacier", fig_glacier, "Glacier context", "ITS_LIVE v2 composite surface speed (120 m), reprojected to UTM 5N, with RGI v6 Alaska outlines served by GLIMS. Star: source area."),
    ("met", fig_met, "Antecedent weather", "Hourly ERA5 (0.25°) via the Open-Meteo archive API at the node nearest the source, aggregated to days. The node's model elevation is far below the ~3,000 m source, so absolute temperatures are warm-biased for the release zone; use the series for timing of warm spells and storms, not for the source temperature. 14 days before the event: mean {result[t_mean_14d]:.1f} °C, max {result[t_max_14d]:.1f} °C, precipitation {result[p_sum_14d]:.0f} mm."),
    ("profile", fig_profile, "Atmosphere at event time", "Hourly pressure-level profiles at the source, the great-circle midpoint, and I53US, for the hour nearest the origin. Freezing level at the source: {result:.0f} m. Top level 30 hPa (~24 km), so stratospheric ducting is not resolved here."),
]

CSS = """
:root{--surface:#fcfcfb;--card:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--muted:#8a8984;--rule:#e4e3df;--accent:#2a78d6}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--surface:#1a1a19;--card:#242422;--ink:#ffffff;--ink2:#c3c2b7;--muted:#8f8e86;--rule:#383835;--accent:#3987e5}}
:root[data-theme="dark"]{--surface:#1a1a19;--card:#242422;--ink:#ffffff;--ink2:#c3c2b7;--muted:#8f8e86;--rule:#383835;--accent:#3987e5}
*{box-sizing:border-box}
body{margin:0;background:var(--surface);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
main{max-width:1080px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:28px;line-height:1.2;margin:0 0 6px}
h2{font-size:19px;margin:40px 0 8px;padding-top:8px;border-top:1px solid var(--rule)}
.sub{color:var(--ink2);margin:0 0 20px}
a{color:var(--accent)}
nav{display:flex;flex-wrap:wrap;gap:6px 14px;font-size:13px;margin:16px 0 8px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:20px 0}
.tile{background:var(--card);border:1px solid var(--rule);border-radius:8px;padding:12px 14px}
.tile .v{font-size:22px;font-weight:600}
.tile .k{font-size:12px;color:var(--ink2)}
figure{margin:12px 0 0;background:#fff;border:1px solid var(--rule);border-radius:8px;padding:10px}
figure img{width:100%;height:auto;display:block}
figcaption{color:var(--ink2);font-size:13.5px;margin-top:8px}
.missing{color:var(--muted);font-style:italic;border:1px dashed var(--rule);border-radius:8px;padding:14px}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:4px 8px;border-bottom:1px solid var(--rule)}
th{color:var(--ink2);font-weight:600}
details{margin-top:10px}
code{font-size:12.5px}
footer{margin-top:48px;color:var(--muted);font-size:12.5px}
"""


def build_page(results, station_table, staged_list, sha):
    ev = EV
    tiles = [
        ("origin (UTC)", "2023-09-14 02:06:49"),
        ("volume", "6.1 × 10⁶ m³"),
        ("source / footprint", "0.143 / 1.520 km²"),
        ("peak speed", "> 60 m/s"),
        ("seismic stations", str(results.get("seismic", "–"))),
        ("infrasound sensors", str(results.get("infrasound", "–"))),
    ]
    sections = []
    for fid, _, title, cap in FIGS:
        sections.append(f'<h2 id="{fid}">{html.escape(title)}</h2>')
        r = results.get(fid)
        if isinstance(r, Exception) or fid not in results:
            sections.append(f'<p class="missing">Not available in this build: {html.escape(str(r))}</p>')
            continue
        caption = cap.format(result=r, t0=ev["origin_time_utc"], tpre=ev["precursor_time_utc"])
        sections.append(f'<figure><img src="figures/{fid}.png" alt="{html.escape(title)}" loading="lazy">'
                        f"<figcaption>{caption}</figcaption></figure>")
        if fid == "map" and station_table is not None:
            rows = "".join(f"<tr><td>{r.type}</td><td>{r.station}</td><td>{r.channel}</td>"
                           f"<td>{r.distance_km}</td><td>{r.back_azimuth_deg}</td></tr>"
                           for r in station_table.itertuples())
            sections.append("<details><summary>Station table</summary><table><thead><tr><th>type</th>"
                            "<th>station</th><th>channel</th><th>distance (km)</th><th>back azimuth (°)</th>"
                            f"</tr></thead><tbody>{rows}</tbody></table></details>")
    nav = "".join(f'<a href="#{f[0]}">{f[2]}</a>' for f in FIGS)
    staged_rows = "".join(f"<tr><td><code>{n}</code></td></tr>" for n in staged_list)
    built = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Peters Dome Landslide</title>
<meta name="description" content="Staged seismic, infrasound, SAR, optical, model and weather data for the 2023 Peters Dome ice-rock avalanche, Denali.">
<style>{CSS}</style></head>
<body><main>
<h1>Peters Dome ice–rock avalanche, 13 September 2023</h1>
<p class="sub">Denali National Park and Preserve, Alaska. Data container built with
<a href="https://github.com/gaia-hazlab/gaia-cli">gaia-cli</a> conventions for the event studied in
Toney et al. (2026), <em>JGR Earth Surface</em>, <a href="https://doi.org/{common.event()['references']['paper_doi']}">doi:{common.event()['references']['paper_doi']}</a>.
Model output, force inversion and outlines come from the USGS data release
<a href="https://doi.org/{common.event()['references']['data_release_doi']}">doi:{common.event()['references']['data_release_doi']}</a>.</p>
<div class="tiles">{''.join(f'<div class="tile"><div class="v">{v}</div><div class="k">{k}</div></div>' for k, v in tiles)}</div>
<nav>{nav}</nav>
{''.join(sections)}
<h2 id="data">Staged stores in this build</h2>
<table><tbody>{staged_rows}</tbody></table>
<p>Reproduce: <code>pixi run stage-all &amp;&amp; pixi run report</code> in
<a href="https://github.com/gaia-hazlab/gaia-peters-dome">gaia-hazlab/gaia-peters-dome</a>.</p>
<footer>Built {built} from commit <code>{sha}</code>. Origin time and location: USGS ComCat; volume, outlines, force and SHALTOP: USGS data release; peak speed: paper abstract.</footer>
</main></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--staged", default="data/staged")
    ap.add_argument("--out", default="docs")
    ap.add_argument("--skip", default="", help="comma-separated panel ids to leave out (e.g. stores still being written)")
    a = ap.parse_args()
    skip = set(filter(None, a.skip.split(",")))
    staged, out = Path(a.staged), Path(a.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    results, table = {}, None
    for fid, fn, _, _ in FIGS:
        if fid in skip:
            results[fid] = FileNotFoundError("not staged yet in this build")
            (out / "figures" / f"{fid}.png").unlink(missing_ok=True)
            continue
        try:
            r = fn(staged, out / "figures" / f"{fid}.png")
            if fid == "map":
                table, r = r, len(r)
            results[fid] = r
            print(f"  ok   {fid}")
        except Exception as e:
            results[fid] = e
            print(f"  skip {fid}: {type(e).__name__}: {e}")
            if not isinstance(e, FileNotFoundError):
                traceback.print_exc()
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        sha = "unknown"
    staged_list = sorted(p.name for p in staged.glob("*.zarr"))
    (out / "index.html").write_text(build_page(results, table, staged_list, sha))
    (out / ".nojekyll").write_text("")
    (out / "build.json").write_text(json.dumps({k: (str(v) if isinstance(v, Exception) else "ok") for k, v in results.items()}, indent=1))
    print(f"wrote {out / 'index.html'}")


if __name__ == "__main__":
    main()
