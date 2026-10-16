"""Seismic and infrasound waveforms from FDSN (EarthScope DMC by default).

Harmonization:
- instrument response removed (seismic -> m/s or m; infrasound -> Pa)
- anti-alias lowpass at 0.4 * target rate, then linear interpolation onto one common
  UTC time grid, so every trace shares ``time`` and gaps become NaN
- seismic: DataArray (station, component, time), components Z/N/E (1/2 mapped to N/E
  only when the channel azimuth is within 5 degrees of 0/90; otherwise kept as 1/2)
- infrasound: DataArray (station, time)
- per-station coordinates: network, location, channel, elevation, distance_km,
  back_azimuth_deg, and a shapely Point ``geometry`` (gaia-cli xvec convention)
"""

from __future__ import annotations

import logging
from typing import Annotated

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from cyclopts import Parameter
from obspy import Stream, UTCDateTime
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNException
from obspy.core.util.obspy_types import ObsPyException
from obspy.geodetics import gps2dist_azimuth
from shapely.geometry import Point

from . import common
from .params import STORE, End, Group, Input, Output, Overwrite, SkipExisting, Start

log = logging.getLogger(__name__)

FDSN_URL = "https://service.earthscope.org"
SEISMIC_CHANNELS = "HH?,BH?"
INFRASOUND_CHANNELS = "BDF,HDF"


def _event_center():
    ev = common.event()["event"]
    return ev["latitude"], ev["longitude"]


def find_stations(aoi, start, end, channels, client, max_distance_km=None):
    """Stations whose location falls inside the AOI polygon (and optional radius)."""
    lat0, lon0 = _event_center()
    minx, miny, maxx, maxy = aoi.to_crs("EPSG:4326").total_bounds
    inv = client.get_stations(
        minlatitude=miny, maxlatitude=maxy, minlongitude=minx, maxlongitude=maxx,
        starttime=UTCDateTime(start), endtime=UTCDateTime(end),
        channel=channels, level="response",
    )
    poly = aoi.to_crs("EPSG:4326").geometry.union_all()

    def keep(sta):
        if not poly.contains(Point(sta.longitude, sta.latitude)):
            return False
        dist = common.haversine_km(lat0, lon0, sta.latitude, sta.longitude)
        return max_distance_km is None or dist <= max_distance_km

    inv = inv.copy()
    for net in inv:
        net.stations = [s for s in net if keep(s)]
    inv.networks = [n for n in inv if n.stations]
    return inv


def _pick_channels(inv, channels: str):
    """One location/band per station: prefer the first band in ``channels`` (HH before BH)."""
    prefs = [c[:2] for c in channels.split(",")]
    picks = []
    for net in inv:
        for sta in net:
            chans = [c for c in sta.channels if c.code[:2] in prefs]
            if not chans:
                continue
            best = min(chans, key=lambda c: (prefs.index(c.code[:2]), c.location_code))
            group = [c for c in chans if c.code[:2] == best.code[:2] and c.location_code == best.location_code]
            picks.append((net, sta, best.location_code, best.code[:2], group))
    return picks


def _to_grid(st: Stream, t0: UTCDateTime, npts: int, fs: float) -> np.ndarray:
    """Lowpass each contiguous segment and interpolate onto the common grid; NaN elsewhere."""
    out = np.full(npts, np.nan)
    grid = np.arange(npts) / fs
    for tr in st:
        tr = tr.copy()
        if tr.stats.sampling_rate > fs:
            tr.filter("lowpass", freq=0.4 * fs, corners=4, zerophase=True)
        tt = tr.times() + (tr.stats.starttime - t0)
        inside = (grid >= tt[0]) & (grid <= tt[-1])
        out[inside] = np.interp(grid[inside], tt, tr.data)
    return out


def _component(cha) -> str:
    c = cha.code[-1]
    if c in "ZNE":
        return c
    az = cha.azimuth % 180 if cha.azimuth is not None else None
    if az is not None and min(az, 180 - az) <= 5:
        return "N"
    if az is not None and abs(az - 90) <= 5:
        return "E"
    return c


def _bulk(client, nslc: list[tuple[str, str, str, str]], t0, t1, chunk: int = 50) -> Stream:
    """FDSN bulk dataselect in chunks; a failed chunk is logged and skipped."""
    out = Stream()
    for k in range(0, len(nslc), chunk):
        try:
            out += client.get_waveforms_bulk([(*r, t0, t1) for r in nslc[k:k + chunk]])
        except (FDSNException, OSError) as e:  # no data, timeouts
            log.info(f"bulk chunk {k // chunk}: {type(e).__name__}")
    return out


def _grid_station(allst, inv, net, sta, loc, band, group, t0, npts, fs, output, pre_filt) -> dict:
    """One station's channels: merge, demean, remove response, grid. {component: array}."""
    st = allst.select(network=net.code, station=sta.code, location=loc, channel=band + "?").copy()
    if not len(st):
        log.info(f"skip {net.code}.{sta.code}: no data")
        return {}
    st = st.merge(method=1, fill_value=None).split()
    st.detrend("demean")
    try:
        st.remove_response(inventory=inv, output=output, pre_filt=pre_filt, water_level=None)
    except (ValueError, ObsPyException) as e:
        log.info(f"skip {net.code}.{sta.code}: response {e}")
        return {}
    comps = {}
    for cha in group:
        sub = st.select(channel=cha.code)
        if len(sub):
            comps[_component(cha)] = _to_grid(sub, t0, npts, fs)
    return comps


def load_seismic(
    aoi: gpd.GeoDataFrame,
    start_date: str,
    end_date: str,
    channels: str = SEISMIC_CHANNELS,
    sampling_rate: float = 20.0,
    output: str = "VEL",
    max_distance_km: float | None = 250.0,
    fdsn_url: str = FDSN_URL,
) -> xr.DataArray:
    client = Client(base_url=fdsn_url)
    t0, t1 = UTCDateTime(start_date), UTCDateTime(end_date)
    inv = find_stations(aoi, start_date, end_date, channels, client, max_distance_km)
    picks = _pick_channels(inv, channels)
    npts = round((t1 - t0) * sampling_rate)
    time = pd.date_range(t0.datetime, periods=npts, freq=pd.Timedelta(seconds=1 / sampling_rate))
    lat0, lon0 = _event_center()
    pre_filt = (0.004, 0.005, 0.4 * sampling_rate, 0.45 * sampling_rate)

    log.info(f"requesting {len(picks)} stations")
    allst = _bulk(client, [(n.code, s.code, loc, band + "?") for n, s, loc, band, _ in picks], t0 - 120, t1 + 120)
    arrays = []
    for net, sta, loc, band, group in picks:
        comps = _grid_station(allst, inv, net, sta, loc, band, group, t0, npts, sampling_rate, output, pre_filt)
        if not comps:
            continue
        dist_m, _az, baz = gps2dist_azimuth(lat0, lon0, sta.latitude, sta.longitude)
        da = xr.DataArray(
            np.stack(list(comps.values())),
            dims=("component", "time"),
            coords={"component": list(comps), "time": time},
        ).expand_dims(station=[f"{net.code}.{sta.code}"])
        da = da.assign_coords(
            network=("station", [net.code]),
            location=("station", [loc]),
            channel=("station", [band + "?"]),
            elevation=("station", [sta.elevation]),
            distance_km=("station", [dist_m / 1e3]),
            back_azimuth_deg=("station", [baz]),
            geometry=("station", [Point(sta.longitude, sta.latitude)]),
        )
        arrays.append(da)
        log.info(f"{net.code}.{sta.code}.{loc}.{band}? {dist_m/1e3:6.1f} km  {list(comps)}")

    if not arrays:
        raise RuntimeError("No waveforms retrieved")
    da = xr.concat(arrays, dim="station", join="outer").reindex(component=["Z", "N", "E", "1", "2"])
    da = da.dropna("component", how="all").sortby("distance_km")
    da.name = "velocity" if output == "VEL" else output.lower()
    da.attrs = common.provenance(
        units={"VEL": "m/s", "DISP": "m", "ACC": "m/s**2"}[output],
        sampling_rate_hz=sampling_rate, fdsn=fdsn_url, pre_filt=pre_filt,
        processing="merge, demean, remove_response, lowpass 0.4*fs, linear interp to common grid",
    )
    return da


def load_infrasound(
    aoi: gpd.GeoDataFrame,
    start_date: str,
    end_date: str,
    channels: str = INFRASOUND_CHANNELS,
    sampling_rate: float = 20.0,
    max_distance_km: float | None = None,
    fdsn_url: str = FDSN_URL,
) -> xr.DataArray:
    """Pressure in Pa. Array elements (e.g. IM.I53H1..8, AV arrays) are separate stations so
    array processing can be run on the staged store directly."""
    client = Client(base_url=fdsn_url)
    t0, t1 = UTCDateTime(start_date), UTCDateTime(end_date)
    inv = find_stations(aoi, start_date, end_date, channels, client, max_distance_km)
    npts = round((t1 - t0) * sampling_rate)
    time = pd.date_range(t0.datetime, periods=npts, freq=pd.Timedelta(seconds=1 / sampling_rate))
    lat0, lon0 = _event_center()

    order = channels.split(",")
    picks = []
    for net in inv:
        for sta in net:
            chans = sorted(sta.channels, key=lambda c: (order.index(c.code) if c.code in order else 9, c.location_code))
            if chans:
                picks.append((net, sta, chans[0]))
    log.info(f"requesting {len(picks)} stations")
    allst = _bulk(client, [(n.code, s.code, c.location_code, c.code) for n, s, c in picks], t0 - 60, t1 + 60)

    arrays = []
    for net, sta, cha in picks:
        st = allst.select(network=net.code, station=sta.code, location=cha.location_code, channel=cha.code).copy()
        try:
            if not len(st):
                raise ValueError("no data")
            st = st.merge(method=1, fill_value=None).split()
            st.detrend("demean")
            st.remove_sensitivity(inventory=inv)
        except (ValueError, ObsPyException) as e:
            log.info(f"skip {net.code}.{sta.code}: {e}")
            continue
        dist_m, _az, baz = gps2dist_azimuth(lat0, lon0, sta.latitude, sta.longitude)
        da = xr.DataArray(_to_grid(st, t0, npts, sampling_rate), dims="time", coords={"time": time})
        da = da.expand_dims(station=[".".join(filter(None, [net.code, sta.code, cha.location_code]))]).assign_coords(
            network=("station", [net.code]),
            channel=("station", [cha.code]),
            elevation=("station", [sta.elevation]),
            distance_km=("station", [dist_m / 1e3]),
            back_azimuth_deg=("station", [baz]),
            geometry=("station", [Point(sta.longitude, sta.latitude)]),
        )
        arrays.append(da)
        log.info(f"{net.code}.{sta.code}.{cha.location_code}.{cha.code} {dist_m/1e3:6.1f} km")

    if not arrays:
        raise RuntimeError("No infrasound retrieved")
    da = xr.concat(arrays, dim="station", join="outer").sortby("distance_km")
    da.name = "pressure"
    da.attrs = common.provenance(
        units="Pa", sampling_rate_hz=sampling_rate, fdsn=fdsn_url,
        processing="merge, demean, remove_sensitivity, lowpass 0.4*fs, linear interp to common grid",
    )
    return da


# ---- CLI entrypoints (gaia-cli parameter conventions) ----------------------------------

def stage_seismic(
    vectorPath: Input = "aoi/peters_dome_waveforms.geojson",
    start_date: Start = "2023-09-14T01:50:00",
    end_date: End = "2023-09-14T02:30:00",
    output_path: Output = STORE,
    group: Group = "stations/seismic",
    channels: Annotated[str, Parameter(name=["--channels"])] = SEISMIC_CHANNELS,
    sampling_rate: Annotated[float, Parameter(name=["--sampling-rate"])] = 20.0,
    units: Annotated[str, Parameter(name=["--units"])] = "VEL",
    max_distance_km: Annotated[float, Parameter(name=["--max-distance-km"])] = 250.0,
    overwrite: Overwrite = False,
    skip_existing: SkipExisting = False,
):
    """Stage seismic waveforms (response removed) on a common time grid."""
    if not common.check_group(output_path, group, overwrite, skip_existing):
        return
    da = load_seismic(common.load_aoi(vectorPath), start_date, end_date, channels,
                      sampling_rate, units, max_distance_km)
    common.write_group(common.stations_to_dataset(da), output_path, group, overwrite)


def stage_infrasound(
    vectorPath: Input = "aoi/peters_dome_waveforms.geojson",
    start_date: Start = "2023-09-14T01:50:00",
    end_date: End = "2023-09-14T03:00:00",
    output_path: Output = STORE,
    group: Group = "stations/infrasound",
    channels: Annotated[str, Parameter(name=["--channels"])] = INFRASOUND_CHANNELS,
    sampling_rate: Annotated[float, Parameter(name=["--sampling-rate"])] = 20.0,
    overwrite: Overwrite = False,
    skip_existing: SkipExisting = False,
):
    """Stage infrasound pressure (Pa) on a common time grid."""
    if not common.check_group(output_path, group, overwrite, skip_existing):
        return
    da = load_infrasound(common.load_aoi(vectorPath), start_date, end_date, channels, sampling_rate)
    common.write_group(common.stations_to_dataset(da), output_path, group, overwrite)


# ---- continuous waveforms (weeks), written day by day -----------------------------------

CONTINUOUS_STATIONS = "AK.KTH,AK.CAST,AK.PPLA"   # nearest three broadbands, 47-58 km


def _continuous_skeleton(inv, picks, t0, npts, fs, chunk) -> xr.Dataset:
    """Lazy all-NaN (station, component, time) float32 array with the station coordinates,
    for a compute=False write; days are filled afterwards with region writes."""
    import dask.array as dsa

    lat0, lon0 = _event_center()
    time = pd.date_range(t0.datetime, periods=npts, freq=pd.Timedelta(seconds=1 / fs))
    comps = ["Z", "N", "E"]
    rows = [gps2dist_azimuth(lat0, lon0, s.latitude, s.longitude) for _, s, *_ in picks]
    da = xr.DataArray(
        dsa.full((len(picks), len(comps), npts), np.nan, dtype="float32", chunks=(len(picks), len(comps), chunk)),
        dims=("station", "component", "time"),
        coords={
            "station": [f"{n.code}.{s.code}" for n, s, *_ in picks],
            "component": comps,
            "time": time,
            "network": ("station", [n.code for n, *_ in picks]),
            "location": ("station", [p[2] for p in picks]),
            "channel": ("station", [p[3] + "?" for p in picks]),
            "elevation": ("station", [s.elevation for _, s, *_ in picks]),
            "distance_km": ("station", [r[0] / 1e3 for r in rows]),
            "back_azimuth_deg": ("station", [r[2] for r in rows]),
            "geometry": ("station", [Point(s.longitude, s.latitude) for _, s, *_ in picks]),
        },
        name="velocity",
    )
    return da


def stage_seismic_continuous(
    vectorPath: Input = "aoi/peters_dome_waveforms.geojson",
    start_date: Start = "2023-08-01T00:00:00",
    end_date: End = "2023-09-14T02:10:00",
    output_path: Output = STORE,
    group: Group = "stations/seismic_continuous",
    stations: Annotated[str, Parameter(name=["--stations"])] = CONTINUOUS_STATIONS,
    channels: Annotated[str, Parameter(name=["--channels"])] = "BH?",
    sampling_rate: Annotated[float, Parameter(name=["--sampling-rate"])] = 20.0,
    units: Annotated[str, Parameter(name=["--units"])] = "VEL",
    chunk_seconds: Annotated[int, Parameter(name=["--chunk-seconds"])] = 3600,
    dry_run: Annotated[bool, Parameter(name=["--dry-run"], negative="")] = False,
    resume: Annotated[bool, Parameter(name=["--resume"], negative="")] = False,
    overwrite: Overwrite = False,
    skip_existing: SkipExisting = False,
):
    """Stage CONTINUOUS seismic velocity (response removed, float32) for a few stations over
    weeks. Downloads and writes one UTC day at a time into a time-chunked group, so memory
    stays at one day. --dry-run prints the size estimate only; --resume fills only the days
    that are still all-NaN in an existing group."""
    client = Client(base_url=FDSN_URL)
    t0, t1 = UTCDateTime(start_date), UTCDateTime(end_date)
    want = set(stations.split(","))
    inv = find_stations(common.load_aoi(vectorPath), start_date, end_date, channels, client)
    lat0, lon0 = _event_center()
    picks = sorted((p for p in _pick_channels(inv, channels) if f"{p[0].code}.{p[1].code}" in want),
                   key=lambda p: common.haversine_km(lat0, lon0, p[1].latitude, p[1].longitude))
    if missing := want - {f"{p[0].code}.{p[1].code}" for p in picks}:
        log.warning(f"no {channels} metadata for {sorted(missing)}")
    npts = round((t1 - t0) * sampling_rate)
    chunk = round(chunk_seconds * sampling_rate)
    nchan = sum(len(p[4]) for p in picks)
    raw_fs = max(c.sample_rate for p in picks for c in p[4])
    # Steim2 MiniSEED of broadband noise is ~1.5 bytes/sample (approximate, for planning)
    log.info(f"{len(picks)} stations, {nchan} channels, {(t1 - t0) / 86400:.1f} days; "
             f"download ~{nchan * raw_fs * (t1 - t0) * 1.5 / 1e9:.1f} GB MiniSEED at {raw_fs:g} Hz; "
             f"store {len(picks) * 3 * npts * 4 / 1e9:.2f} GB float32 before compression, "
             f"time chunk {chunk_seconds} s")
    if dry_run:
        return

    if resume and common.group_exists(output_path, group):
        done = common.open_group(output_path, group).velocity
    else:
        if not common.check_group(output_path, group, overwrite, skip_existing):
            return
        da = _continuous_skeleton(inv, picks, t0, npts, sampling_rate, chunk)
        pre_filt = (0.004, 0.005, 0.4 * sampling_rate, 0.45 * sampling_rate)
        da.attrs = common.provenance(
            units={"VEL": "m/s", "DISP": "m", "ACC": "m/s**2"}[units], sampling_rate_hz=sampling_rate,
            fdsn=FDSN_URL, pre_filt=pre_filt, dtype="float32",
            processing="per UTC day (+-120 s margin): merge, demean, remove_response, lowpass 0.4*fs, "
                       "linear interp to common grid; gaps and missing days are NaN",
        )
        common.write_group(common.stations_to_dataset(da), output_path, group, overwrite, compute=False)
        done = None

    pre_filt = (0.004, 0.005, 0.4 * sampling_rate, 0.45 * sampling_rate)
    nslc = [(n.code, s.code, loc, band + "?") for n, s, loc, band, _ in picks]
    day = round(86400 * sampling_rate)
    for i0 in range(0, npts, day):
        i1 = min(i0 + day, npts)
        d0 = t0 + i0 / sampling_rate
        if done is not None and np.isfinite(done.isel(time=slice(i0, i1)).values).any():
            continue
        for attempt in range(3):  # an empty day is usually a timeout; retry before writing NaN
            allst = _bulk(client, nslc, d0 - 120, d0 + (i1 - i0) / sampling_rate + 120)
            if len(allst):
                break
            log.info(f"{d0.date}: no data returned (attempt {attempt + 1}/3)")
        arr = np.full((len(picks), 3, i1 - i0), np.nan, dtype="float32")
        for k, (net, sta, loc, band, grp) in enumerate(picks):
            comps = _grid_station(allst, inv, net, sta, loc, band, grp, d0, i1 - i0, sampling_rate, units, pre_filt)
            for j, c in enumerate("ZNE"):
                if c in comps:
                    arr[k, j] = comps[c]
        common.write_region(xr.Dataset({"velocity": (("geometry", "component", "time"), arr)}),
                            output_path, group, {"time": slice(i0, i1)})
        log.info(f"{d0.date} written, finite fraction {np.isfinite(arr).mean():.3f}")
