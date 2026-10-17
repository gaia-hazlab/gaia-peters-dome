"""Matched-filter search for precursory seismicity before the 2023-09-14 Peters Dome
ice-rock avalanche (ComCat origin 2023-09-14T02:06:49Z).

Templates are cut from ``stations/seismic`` (event window, 20 Hz velocity) and scanned
over ``stations/seismic_continuous`` (2023-08-01 -> 2023-09-14T02:10Z, same processing,
float32) at the three nearest broadbands AK.KTH, AK.CAST, AK.PPLA, all three components.

Method
- 1-5 Hz zero-phase Butterworth (4 poles), the same filter on templates and data.
- One template = the same absolute UTC window on every channel, so the station moveout
  is kept and the per-channel normalized cross-correlations (CC) stack without shifts.
- Network CC = mean over channels whose data window is gap-free; a sample counts only
  if >= MIN_CHANNELS channels are valid.
- Threshold = median + K * MAD of the network CC, computed per UTC day (MAD unscaled).
  K = 8 by default; the control run with time-reversed templates (same length and
  spectrum, no physical counterpart) measures how many detections chance alone gives.
- Declustering: one detection per template length (highest CC kept).
- Amplitude ratio = median over valid channels of RMS(data window) / RMS(template window).

Run (from the repo root):
    pixi run python analysis/precursors/template_matching.py            # scan + figures
    pixi run python analysis/precursors/template_matching.py --plots    # figures from CSVs
Outputs land next to this file.
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, oaconvolve, sosfiltfilt

from peters_dome import common

log = logging.getLogger("precursors")
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="All-NaN slice")
HERE = Path(__file__).resolve().parent
RUNS = {  # station sets; KTH has no data 2023-08-13 to 2023-09-12 21:58 (EarthScope), so rates use CAST+PPLA
    "cast_ppla": ["AK.CAST", "AK.PPLA"],
    "kth_cast_ppla": ["AK.KTH", "AK.CAST", "AK.PPLA"],
}
OUT = HERE / "cast_ppla"
STORE = str(common.ROOT / common.STORE)
CONTINUOUS_STORE = STORE   # --continuous-store overrides (tests on a short scratch store)
STATIONS = RUNS["cast_ppla"]
COMPONENTS = ["Z", "N", "E"]
FS = 20.0
BAND = (1.0, 5.0)
MIN_CHANNELS = 6          # at least two stations' worth of channels
K_MAD = 8.0               # detection threshold, multiples of the daily MAD
K_REPORT = (8, 10, 12, 15, 20)
GAP_PAD_S = 10.0          # samples this close to a gap are masked (filter edge effects)

DAY = "2023-09-14T"
ORIGIN = np.datetime64("2023-09-14T02:06:49")       # ComCat landslide origin
P1_ORIGIN = np.datetime64("2023-09-14T01:58:54")    # ComCat "other event", M1.4
# windows at the stations (UTC). P1/P2 long windows as read off the 1-5 Hz envelopes;
# the short windows are 30 s around the onset or the peak.
TEMPLATES = {
    "P1_long": ("01:58:40", "02:00:30"),
    "P1_onset": ("01:58:58", "01:59:28"),
    "P2_long": ("02:03:10", "02:05:40"),
    "P2_peak": ("02:04:40", "02:05:10"),
    "main_onset": ("02:06:55", "02:07:25"),
}
SEQUENCE = (np.datetime64("2023-09-14T01:57:00"), np.datetime64("2023-09-14T02:10:00"))
OTHER_EVENT = np.datetime64("2023-09-13T13:57:32")  # ComCat "other event", M1.6, ~13 km S

SOS = butter(4, BAND, "bandpass", fs=FS, output="sos")


# ---- signal helpers ----------------------------------------------------------------------

def bandpass(x: np.ndarray) -> np.ndarray:
    """Filter a 1-D trace with NaN gaps: zero-fill, filter, re-mask gaps padded by GAP_PAD_S."""
    bad = ~np.isfinite(x)
    if bad.all():
        return x.copy()
    y = sosfiltfilt(SOS, np.where(bad, 0.0, x - np.nanmean(x)))
    if bad.any():
        pad = int(GAP_PAD_S * FS)
        bad = np.convolve(bad.astype(float), np.ones(2 * pad + 1), "same") > 0
        y[bad] = np.nan
    return y


def sliding_cc(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Normalized CC of template w against every window of x ('valid' length). NaN where
    the window touches a gap or the data are flat."""
    m = len(w)
    w = w - w.mean()
    w = w / np.linalg.norm(w)
    bad = ~np.isfinite(x)
    scale = np.nanstd(x) or 1.0
    x0 = np.where(bad, 0.0, x / scale)
    num = oaconvolve(x0, w[::-1], "valid")
    c1 = np.concatenate([[0.0], np.cumsum(x0)])
    c2 = np.concatenate([[0.0], np.cumsum(x0 * x0)])
    s1 = c1[m:] - c1[:-m]
    s2 = c2[m:] - c2[:-m]
    var = s2 - s1 * s1 / m
    cb = np.concatenate([[0], np.cumsum(bad)])
    ok = ((cb[m:] - cb[:-m]) == 0) & (var > 1e-6 * m)
    cc = np.full(len(num), np.nan)
    cc[ok] = num[ok] / np.sqrt(var[ok])
    return cc


def rms_windows(x: np.ndarray, m: int, idx: np.ndarray) -> np.ndarray:
    return np.array([np.sqrt(np.nanmean(x[i:i + m] ** 2)) for i in idx])


def declump(cc: np.ndarray, thr: np.ndarray | float, sep: int) -> np.ndarray:
    """Indices of local maxima above thr, at least sep samples apart (greedy, highest first)."""
    above = np.flatnonzero(np.nan_to_num(cc, nan=-1) > thr)
    if not len(above):
        return above
    order = above[np.argsort(cc[above])[::-1]]
    taken = np.zeros(len(cc), bool)
    keep = []
    for i in order:
        if taken[i]:
            continue
        keep.append(i)
        taken[max(0, i - sep):i + sep] = True
    return np.sort(np.array(keep))


# ---- data -----------------------------------------------------------------------------------

def nch() -> int:
    return 3 * len(STATIONS)


def _select(ds, stations=None):
    ds = ds.assign_coords(sta=("geometry", [str(s) for s in ds.station.values]),
                          component=[str(c) for c in ds.component.values])
    ds = ds.swap_dims({"geometry": "sta"}).sel(sta=stations or STATIONS, component=COMPONENTS)
    return ds.velocity.transpose("sta", "component", "time")


def load_templates(reverse: bool = False) -> dict:
    """{name: dict(data=(channels, m) array, t0=datetime64, m=int)} from stations/seismic."""
    v = _select(common.open_group(STORE, "stations/seismic"))
    t = v.time.values
    x = np.stack([bandpass(tr) for tr in v.values.reshape(nch(), -1).astype("float64")])
    out = {}
    for name, (a, b) in TEMPLATES.items():
        i0 = np.searchsorted(t, np.datetime64(DAY + a))
        i1 = np.searchsorted(t, np.datetime64(DAY + b))
        d = x[:, i0:i1].copy()
        if reverse:
            d = d[:, ::-1].copy()
        out[name] = {"data": d, "t0": t[i0], "m": i1 - i0}
    return out


def open_continuous():
    return _select(common.open_group(CONTINUOUS_STORE, "stations/seismic_continuous"))


# ---- scan --------------------------------------------------------------------------------

def scan(templates: dict, keep_event_block: bool = False):
    v = open_continuous()
    t = v.time.values
    n = len(t)
    day = int(86400 * FS)
    mmax = max(tp["m"] for tp in templates.values())
    dets, stats, hourly, keep = [], [], [], {}
    for i0 in range(0, n, day):
        i1 = min(i0 + day + mmax, n)
        raw = v.isel(time=slice(i0, i1)).values.reshape(nch(), -1).astype("float64")
        x = np.stack([bandpass(tr) for tr in raw])
        tb = t[i0:i1]
        label = str(tb[0])[:10]
        # hourly background: median of 1-min RMS on Z, per station
        zz = x[::3][:, : min(day, i1 - i0)]
        nmin = zz.shape[1] // int(60 * FS)
        r1 = np.sqrt(np.nanmean(zz[:, : nmin * int(60 * FS)].reshape(len(zz), nmin, -1) ** 2, axis=2))
        for h in range(0, nmin, 60):
            row = {"hour": tb[h * int(60 * FS)]}
            for k, s in enumerate(STATIONS):
                seg = r1[k, h:h + 60]
                row[f"{s}_rms_ms"] = np.nanmedian(seg) if np.isfinite(seg).any() else np.nan
            hourly.append(row)
        for name, tp in templates.items():
            m = tp["m"]
            ccs = np.stack([sliding_cc(x[c], tp["data"][c]) for c in range(nch())])
            nval = np.isfinite(ccs).sum(0)
            with np.errstate(invalid="ignore"):
                stack = np.nansum(ccs, 0) / np.maximum(nval, 1)
            stack[nval < MIN_CHANNELS] = np.nan
            body = stack[: min(day, len(stack))]                  # this day only (overlap is next day's)
            med = np.nanmedian(body) if np.isfinite(body).any() else np.nan
            mad = np.nanmedian(np.abs(body - med)) if np.isfinite(body).any() else np.nan
            stats.append({"day": label, "template": name, "median": med, "mad": mad,
                          "max": np.nanmax(body) if np.isfinite(body).any() else np.nan,
                          "valid_fraction": float(np.isfinite(body).mean())})
            if keep_event_block and tb[0] <= SEQUENCE[0] <= tb[len(body) - 1]:
                keep[name] = (tb[: len(body)], body, med, mad)
            if not np.isfinite(mad) or mad == 0:
                continue
            idx = declump(body, med + min(K_REPORT) * mad, m)
            if not len(idx):
                continue
            tpl_rms = np.sqrt(np.mean(tp["data"] ** 2, axis=1))
            ratios = np.stack([rms_windows(x[c], m, idx) / tpl_rms[c] for c in range(nch())])
            for j, i in enumerate(idx):
                row = {
                    "template": name, "time": tb[i], "cc": body[i], "mad_multiple": (body[i] - med) / mad,
                    "n_channels": int(nval[i]), "amp_ratio": float(np.nanmedian(ratios[:, j])),
                    "day_median": med, "day_mad": mad,
                }
                for k, sname in enumerate(STATIONS):  # per-station CC, mean over Z/N/E
                    row[f"cc_{sname.split('.')[1]}"] = float(np.nanmean(ccs[3 * k:3 * k + 3, i])) \
                        if np.isfinite(ccs[3 * k:3 * k + 3, i]).any() else np.nan
                dets.append(row)
        log.info(f"{label} done")
    return pd.DataFrame(dets), pd.DataFrame(stats), pd.DataFrame(hourly), keep


# ---- main -----------------------------------------------------------------------------

def run_scan():
    tpl = load_templates()
    rev = load_templates(reverse=True)
    dets, stats, hourly, keep = scan(tpl, keep_event_block=True)
    cdets, cstats, _, _ = scan(rev)
    dets["in_sequence"] = (dets.time >= SEQUENCE[0]) & (dets.time <= SEQUENCE[1])
    cdets["in_sequence"] = (cdets.time >= SEQUENCE[0]) & (cdets.time <= SEQUENCE[1])
    dets.to_csv(OUT / "detections_all.csv", index=False, float_format="%.4g")
    cdets.to_csv(OUT / "control_reversed_detections.csv", index=False, float_format="%.4g")
    stats.to_csv(OUT / "cc_daily_stats.csv", index=False, float_format="%.4g")
    cstats.to_csv(OUT / "control_reversed_daily_stats.csv", index=False, float_format="%.4g")
    hourly.to_csv(OUT / "hourly_background_rms.csv", index=False, float_format="%.4g")
    np.savez_compressed(OUT / "event_day_cc.npz", **{f"{k}_{f}": val for k, (tt, cc, med, mad) in keep.items()
                                                       for f, val in [("t0", tt[0].astype("int64")),
                                                                      ("cc", cc.astype("float32")),
                                                                      ("med", med), ("mad", mad)]})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--plots", action="store_true", help="only rebuild figures and tables from the CSVs")
    ap.add_argument("--continuous-store", default=STORE)
    ap.add_argument("--run", choices=list(RUNS), default="cast_ppla",
                    help="station set; outputs go to the subfolder of that name")
    a = ap.parse_args()
    # report.py imports this file as `template_matching`; configure that one module object
    import template_matching as tm

    tm.CONTINUOUS_STORE = a.continuous_store
    tm.STATIONS, tm.OUT = RUNS[a.run], HERE / a.run
    tm.OUT.mkdir(exist_ok=True)
    from report import make_all

    if not a.plots:
        tm.run_scan()
    make_all()
