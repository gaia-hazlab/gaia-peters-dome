"""Figures, rate statistics and summary numbers from the template-matching outputs.
Called by template_matching.py; on its own: ``pixi run python analysis/precursors/report.py``.
Writes catalog.csv, daily_rates.csv, hourly_rates_last72h.csv, threshold_control.csv,
inter_precursor.csv, results.json and fig_*.png into this folder."""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import hilbert
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import template_matching as tm

DAY_NS = np.timedelta64(86400, "s")
MERGE_S = 60          # detections from different templates within 60 s = one event
COLORS = {"P1_long": "#1f77b4", "P1_onset": "#6baed6", "P2_long": "#d62728", "P2_peak": "#fc9272",
          "main_onset": "#2ca02c", "union": "black"}


def _read():
    d = pd.read_csv(tm.OUT / "detections_all.csv", parse_dates=["time"])
    c = pd.read_csv(tm.OUT / "control_reversed_detections.csv", parse_dates=["time"])
    s = pd.read_csv(tm.OUT / "cc_daily_stats.csv")
    h = pd.read_csv(tm.OUT / "hourly_background_rms.csv", parse_dates=["hour"])
    return d, c, s, h


def threshold_control(d, c) -> pd.DataFrame:
    rows = []
    for name in tm.TEMPLATES:
        for k in tm.K_REPORT:
            dd = d[(d.template == name) & ~d.in_sequence & (d.mad_multiple >= k)]
            cc = c[(c.template == name) & ~c.in_sequence & (c.mad_multiple >= k)]
            rows.append({"template": name, "k_mad": k, "detections": len(dd), "control_reversed": len(cc),
                         "max_cc": dd.cc.max() if len(dd) else np.nan})
    return pd.DataFrame(rows)


# named catalogs: (template or None for all, K). P1_onset is the only template whose
# detections exceed the reversed control (threshold_control.csv); the union of all
# templates at 8 MAD is dominated by chance and is kept only for comparison.
CATALOGS = {
    "P1_onset >= 8 MAD": ("P1_onset", 8),
    "P1_onset >= 10 MAD": ("P1_onset", 10),
    "all templates >= 12 MAD": (None, 12),
    "all templates >= 8 MAD": (None, 8),
}
P1_TIME = np.datetime64("2023-09-14T01:58:54")


def union_catalog(d: pd.DataFrame, k: float, template: str | None = None) -> pd.DataFrame:
    """Detections at >= k MAD outside the 01:57-02:10 sequence, merged across templates."""
    x = d[~d.in_sequence & (d.mad_multiple >= k)]
    if template:
        x = x[x.template == template]
    x = x.sort_values("time").reset_index(drop=True)
    if x.empty:
        return x.assign(event=pd.Series(dtype=int), templates=pd.Series(dtype=str))
    ev = (x.time.diff().dt.total_seconds().fillna(1e9) > MERGE_S).cumsum()
    x["event"] = ev
    best = x.loc[x.groupby("event").mad_multiple.idxmax()].copy()
    best["templates"] = x.groupby("event").template.agg(lambda s: "+".join(sorted(set(s)))).values
    best["hours_before_failure"] = (tm.ORIGIN - best.time.values.astype("datetime64[s]")) / np.timedelta64(3600, "s")
    return best.reset_index(drop=True)


def valid_hours_per_day(s: pd.DataFrame) -> pd.Series:
    v = s.groupby("day").valid_fraction.min()
    last = v.index.max()
    hours = pd.Series(24.0, index=v.index)
    hours[last] = 2 + 10 / 60  # 2023-09-14 00:00-02:10
    return v * hours


def rate_tests(cat: pd.DataFrame, vh: pd.Series) -> dict:
    """Binomial tests: share of events in the last 7 d / last 24 h vs the share of valid time."""
    out = {"n_events": len(cat), "valid_hours": float(vh.sum())}
    days = pd.to_datetime(vh.index)
    for label, span in [("last_7d", 7 * 24), ("last_24h", 24)]:
        t_split = pd.Timestamp(tm.ORIGIN) - pd.Timedelta(hours=span)
        # valid hours after t_split: whole days after, plus a fraction of the split day
        frac = []
        for dname, hrs in zip(days, vh.values, strict=True):
            d0 = dname
            d1 = d0 + pd.Timedelta(days=1)
            overlap = max(0.0, (min(d1, pd.Timestamp(tm.ORIGIN)) - max(d0, t_split)).total_seconds() / 3600)
            full = (min(d1, pd.Timestamp(tm.ORIGIN)) - d0).total_seconds() / 3600
            frac.append(hrs * overlap / full if full > 0 else 0)
        p = sum(frac) / vh.sum()
        n_in = int((cat.time >= t_split).sum()) if len(cat) else 0
        res = {"events_in_window": n_in, "expected_if_constant": float(p * len(cat)),
               "valid_time_share": float(p)}
        if len(cat):
            res["p_value_excess"] = float(binomtest(n_in, len(cat), p, alternative="greater").pvalue)
        out[label] = res
    return out


# ---- inter-precursor interval -------------------------------------------------------------

def network_envelope():
    sta = tm.RUNS["kth_cast_ppla"]   # the event window has all three stations
    v = tm._select(tm.common.open_group(tm.STORE, "stations/seismic"), sta)
    t = v.time.values
    x = np.stack([tm.bandpass(tr) for tr in v.values.reshape(3 * len(sta), -1).astype("float64")])
    env = np.abs(hilbert(x, axis=1))
    k = int(5 * tm.FS)
    env = np.stack([np.convolve(e, np.ones(k) / k, "same") for e in env])
    noise = np.median(env[:, (t >= np.datetime64(tm.DAY + "01:50:00")) & (t < np.datetime64(tm.DAY + "01:55:00"))], 1)
    return t, np.median(env / noise[:, None], 0)


def _fit_decay(t, env, a, b):
    """log10 of the noise-corrected envelope, linear in time: returns slope (decades/min), fn."""
    m = (t >= np.datetime64(tm.DAY + a)) & (t < np.datetime64(tm.DAY + b))
    s = (t[m] - t[m][0]) / np.timedelta64(1, "s") / 60
    y = 0.5 * np.log10(np.clip(env[m] ** 2 - 1, 1e-3, None))
    p = np.polyfit(s, y, 1)
    t0 = t[m][0]

    def fn(tt):
        ss = (tt - t0) / np.timedelta64(1, "s") / 60
        return np.sqrt(1 + 10 ** (2 * np.polyval(p, ss)))
    return p[0], fn


def inter_precursor():
    t, env = network_envelope()
    slope_p1, f_p1 = _fit_decay(t, env, "01:59:50", "02:01:00")
    slope_main, _ = _fit_decay(t, env, "02:08:00", "02:10:30")
    rows = []
    edges = pd.date_range(tm.DAY + "02:00:30", tm.DAY + "02:03:30", freq="20s")
    for a, b in itertools.pairwise(edges):
        m = (t >= np.datetime64(a)) & (t < np.datetime64(b))
        rows.append({"start": a.strftime("%H:%M:%S"), "end": b.strftime("%H:%M:%S"),
                     "observed_env_over_noise": float(np.median(env[m])),
                     "p1_coda_model": float(np.median(f_p1(t[m]))),
                     "excess_obs_over_model": float(np.median(env[m]) / np.median(f_p1(t[m])))})
    tab = pd.DataFrame(rows)
    tab.to_csv(tm.OUT / "inter_precursor.csv", index=False, float_format="%.3g")
    # reference: the same 20-s medians over quiet pre-event noise, 01:50:00-01:57:40
    qedges = pd.date_range(tm.DAY + "01:50:00", tm.DAY + "01:57:40", freq="20s")
    quiet = np.array([np.median(env[(t >= np.datetime64(a)) & (t < np.datetime64(b))])
                      for a, b in itertools.pairwise(qedges)])
    pm = (t >= np.datetime64(tm.DAY + "02:01:50")) & (t < np.datetime64(tm.DAY + "02:03:10"))
    plateau_slope = np.polyfit((t[pm] - t[pm][0]) / np.timedelta64(60, "s"), np.log10(env[pm]), 1)[0]

    fig, ax = plt.subplots(figsize=(11, 4.2))
    tw = (t >= np.datetime64(tm.DAY + "01:56")) & (t < np.datetime64(tm.DAY + "02:12"))
    ax.semilogy(t[tw], env[tw], color="0.2", lw=0.8, label="median of 9 channels (KTH, CAST, PPLA; Z/N/E)")
    mm = (t >= np.datetime64(tm.DAY + "01:59:50")) & (t < np.datetime64(tm.DAY + "02:04:00"))
    ax.semilogy(t[mm], f_p1(t[mm]), color="#1f77b4", lw=1.6, ls="--",
                label=f"P1 coda fit 01:59:50-02:01:00, extrapolated ({slope_p1:.2f} decades/min)")
    ax.axvspan(np.datetime64(tm.DAY + "02:00:30"), np.datetime64(tm.DAY + "02:03:10"), color="#ffd92f", alpha=0.3,
               label="inter-precursor interval 02:00:30-02:03:10")
    ax.axhline(1, color="0.5", lw=0.6)
    ax.set_ylabel("1-5 Hz envelope / 01:50-01:55 noise")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlabel("2023-09-14 UTC (time at the stations)")
    ax.set_title(f"Inter-precursor interval vs P1 coda decay (main-event coda: {slope_main:.2f} decades/min)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(tm.OUT / "fig_inter_precursor.png", dpi=130)
    plt.close(fig)
    return {"p1_coda_decay_decades_per_min": float(slope_p1), "main_coda_decay_decades_per_min": float(slope_main),
            "plateau_02:01:50-02:03:10_median": float(np.median(env[pm])),
            "plateau_slope_decades_per_min": float(plateau_slope),
            "quiet_01:50-01:57:40_20s_bins": {"n": len(quiet), "median": float(np.median(quiet)),
                                              "p95": float(np.percentile(quiet, 95)), "max": float(quiet.max())},
            "bins": rows}


# ---- figures ------------------------------------------------------------------------------

def fig_templates():
    tpl = tm.load_templates()
    fig, axs = plt.subplots(1, len(tpl), figsize=(16, 5), sharey=True)
    for ax, (name, tp) in zip(axs, tpl.items(), strict=True):
        tt = np.arange(tp["m"]) / tm.FS
        for c in range(tm.nch()):
            y = tp["data"][c]
            ax.plot(tt, y / np.abs(y).max() * 0.45 + c, lw=0.4, color=COLORS[name])
        ax.set_title(f"{name}\n{tm.TEMPLATES[name][0]}-{tm.TEMPLATES[name][1]}", fontsize=9)
        ax.set_xlabel("s after window start")
    axs[0].set_yticks(range(tm.nch()), [f"{s.split('.')[1]}.{c}" for s in tm.STATIONS for c in tm.COMPONENTS])
    fig.suptitle("Templates, 1-5 Hz velocity, each trace normalized (stations/seismic, 2023-09-14)")
    fig.tight_layout()
    fig.savefig(tm.OUT / "fig_templates.png", dpi=120)
    plt.close(fig)


def fig_event_day():
    z = np.load(tm.OUT / "event_day_cc.npz")
    fig, axs = plt.subplots(len(tm.TEMPLATES), 2, figsize=(15, 11), sharex="col",
                            gridspec_kw={"width_ratios": [2, 1]})
    for r, name in enumerate(tm.TEMPLATES):
        cc, med, mad = z[f"{name}_cc"], float(z[f"{name}_med"]), float(z[f"{name}_mad"])
        t = z[f"{name}_t0"].astype("datetime64[ns]") + (np.arange(len(cc)) * 1e9 / tm.FS).astype("timedelta64[ns]")
        for ax, (a, b) in zip(axs[r], [("00:00", "02:10"), ("01:55", "02:10")], strict=True):
            m = (t >= np.datetime64(tm.DAY + a)) & (t < np.datetime64(tm.DAY + b))
            ax.plot(t[m], cc[m], lw=0.4, color=COLORS[name])
            ax.axhline(med + tm.K_MAD * mad, color="k", lw=0.7, ls="--")
            ax.set_ylabel(name, fontsize=8)
            ax.grid(alpha=0.3)
        axs[r, 1].axvspan(np.datetime64(tm.DAY + "02:00:30"), np.datetime64(tm.DAY + "02:03:10"),
                          color="#ffd92f", alpha=0.3)
    for ax in axs[-1]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axs[0, 0].set_title("network CC, 2023-09-14 00:00-02:10 (dashed: median + 8 MAD of the day)")
    axs[0, 1].set_title("01:55-02:10; yellow = inter-precursor interval")
    fig.tight_layout()
    fig.savefig(tm.OUT / "fig_event_day_cc.png", dpi=120)
    plt.close(fig)


def _daily(cat, days):
    if cat.empty:
        return pd.Series(0, index=days)
    return cat.groupby(cat.time.dt.floor("D")).size().reindex(days, fill_value=0)


def fig_rates(cats: dict, ctrls: dict, d: pd.DataFrame, vh: pd.Series):
    days = pd.to_datetime(vh.index)
    fig, axs = plt.subplots(4, 1, figsize=(12, 14))
    # (a) daily counts per template at 8 MAD
    w = 0.8 / len(tm.TEMPLATES)
    for j, name in enumerate(tm.TEMPLATES):
        dd = d[(d.template == name) & ~d.in_sequence & (d.mad_multiple >= tm.K_MAD)]
        axs[0].bar(days + pd.Timedelta(days=j * w), _daily(dd, days).values, width=w, color=COLORS[name], label=name)
    axs[0].set_ylabel("detections / day (>= 8 MAD)")
    axs[0].legend(fontsize=8, ncol=5)
    axs[0].set_title("(a) detections per template and day, outside the 01:57-02:10 failure sequence")
    # (b) P1_onset >= 8 MAD: real vs reversed control, per valid day
    lab = "P1_onset >= 8 MAD"
    for cat, style, name in [(cats[lab], "o-", "P1_onset"), (ctrls[lab], "s--", "reversed P1_onset (control)")]:
        axs[1].plot(days + pd.Timedelta(hours=12), _daily(cat, days).values / np.maximum(vh.values, 1e-9) * 24,
                    style, ms=3, lw=0.8, label=f"{name}, n={len(cat)}")
    axs[1].set_yscale("symlog", linthresh=1)
    axs[1].set_ylabel("detections per valid day")
    axs[1].legend(fontsize=8)
    axs[1].set_title("(b) P1_onset >= 8 MAD, normalized by valid hours (2023-09-14 has 2.2 h)")
    # (c) cumulative counts, real (solid) and control (dashed)
    for j, (labc, cat) in enumerate(cats.items()):
        col = f"C{j}"
        for x, ls, suffix in [(cat, "-", ""), (ctrls[labc], "--", " control")]:
            if len(x):
                axs[2].step(np.r_[days[0], x.time.values, tm.ORIGIN], np.r_[0, np.arange(1, len(x) + 1), len(x)],
                            where="post", color=col, ls=ls, label=f"{labc}{suffix} (n={len(x)})")
    axs[2].set_yscale("symlog", linthresh=10)
    axs[2].axvline(tm.ORIGIN, color="r", lw=0.8)
    axs[2].set_ylabel("cumulative events")
    axs[2].legend(fontsize=7, ncol=2)
    axs[2].set_title("(c) cumulative counts; dashed = same definition with time-reversed templates")
    # (d) last 6 h, P1_onset detections
    t0 = pd.Timestamp(tm.ORIGIN) - pd.Timedelta(hours=6)
    for labc, mk in [("P1_onset >= 8 MAD", "o"), ("P1_onset >= 10 MAD", "*")]:
        x = cats[labc][cats[labc].time >= t0]
        axs[3].plot(x.time, x.mad_multiple, mk, ms=6 if mk == "o" else 10, label=labc)
    xc = ctrls["P1_onset >= 8 MAD"]
    xc = xc[xc.time >= t0]
    axs[3].plot(xc.time, xc.mad_multiple, "kx", label=f"control, n={len(xc)}")
    for tt, lbl in [(P1_TIME, "P1"), (np.datetime64("2023-09-14T02:03:10"), "P2"), (tm.ORIGIN, "failure")]:
        axs[3].axvline(tt, color="r", lw=0.8)
        axs[3].text(tt, axs[3].get_ylim()[1], lbl, color="r", fontsize=8, va="top")
    axs[3].set_xlim(t0, pd.Timestamp(tm.ORIGIN) + pd.Timedelta(minutes=10))
    axs[3].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    axs[3].set_ylabel("MAD multiple")
    axs[3].legend(fontsize=8)
    axs[3].set_title("(d) last 6 h: P1_onset detections")
    for ax in axs:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(tm.OUT / "fig_rates.png", dpi=120)
    plt.close(fig)


def swarm_stats(cat: pd.DataFrame, hours: float = 6) -> dict:
    """Inter-event times of the detections in the last `hours` before P1, and a linear fit
    of inter-event time vs time (inverse-rate extrapolation to zero)."""
    x = cat[(cat.time >= pd.Timestamp(P1_TIME) - pd.Timedelta(hours=hours)) & (cat.time < pd.Timestamp(P1_TIME))]
    tt = x.time.values.astype("datetime64[s]")
    if len(tt) < 4:
        return {"n": len(tt)}
    dt_min = np.diff(tt) / np.timedelta64(60, "s")
    tmid = tt[1:]
    s_min = (tmid - P1_TIME) / np.timedelta64(60, "s")          # minutes relative to P1 (negative)
    p = np.polyfit(s_min, dt_min, 1)
    return {
        "n": len(tt), "first": str(tt[0]), "last": str(tt[-1]),
        "inter_event_minutes": [round(float(v), 2) for v in dt_min],
        "quiet_minutes_before_P1": float((P1_TIME - tt[-1]) / np.timedelta64(60, "s")),
        "linear_fit_dt_vs_time": {"slope_min_per_min": float(p[0]), "intercept_at_P1_min": float(p[1]),
                                  "zero_crossing_minutes_rel_P1": float(-p[1] / p[0]) if p[0] else None},
        "median_dt_first_half_min": float(np.median(dt_min[: len(dt_min) // 2])),
        "median_dt_second_half_min": float(np.median(dt_min[len(dt_min) // 2:])),
    }


def fig_inverse_rate(cats: dict, swarm: dict, nwin: int = 5):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    for j, (lab, cat) in enumerate(cats.items()):
        if len(cat) < nwin:
            continue
        tt = cat.time.values.astype("datetime64[s]")
        inv = (tt[nwin - 1:] - tt[: len(tt) - nwin + 1]) / np.timedelta64(3600, "s") / (nwin - 1)
        tf = (tm.ORIGIN - tt[nwin - 1:]) / np.timedelta64(3600, "s") / 24
        axs[0].loglog(np.maximum(tf, 1e-4), inv, "o", ms=3, color=f"C{j}", label=lab)
    axs[0].invert_xaxis()
    axs[0].set_xlabel("time to failure (days, window end)")
    axs[0].set_ylabel(f"inverse rate (h per event, {nwin}-event window)")
    axs[0].set_title("(a) 2023-08-01 to failure")
    x = cats["P1_onset >= 8 MAD"]
    x = x[(x.time >= pd.Timestamp(P1_TIME) - pd.Timedelta(hours=6)) & (x.time < pd.Timestamp(P1_TIME))]
    if len(x) >= 4:
        tt = x.time.values.astype("datetime64[s]")
        s_min = (tt[1:] - P1_TIME) / np.timedelta64(60, "s")
        dt_min = np.diff(tt) / np.timedelta64(60, "s")
        axs[1].plot(s_min, dt_min, "o-", color=COLORS["P1_onset"], label="inter-event time, P1_onset >= 8 MAD")
        f = swarm["linear_fit_dt_vs_time"]
        ss = np.linspace(s_min.min(), 0, 50)
        axs[1].plot(ss, f["slope_min_per_min"] * ss + f["intercept_at_P1_min"], "k--", lw=0.8,
                    label=f"linear fit, zero at {f['zero_crossing_minutes_rel_P1']:.0f} min rel. P1")
        last = (tt[-1] - P1_TIME) / np.timedelta64(60, "s")
        axs[1].axvspan(last, 0, color="0.85", label=f"no detection for {-last:.0f} min before P1")
    for tt_, lbl in [(0, "P1 01:58:54"), ((tm.ORIGIN - P1_TIME) / np.timedelta64(60, "s"), "failure")]:
        axs[1].axvline(tt_, color="r", lw=0.8)
        axs[1].text(tt_, 0, lbl, color="r", fontsize=8, rotation=90, va="bottom")
    axs[1].set_ylim(bottom=0)
    axs[1].set_xlabel("minutes relative to P1 (ComCat 2023-09-14 01:58:54)")
    axs[1].set_ylabel("minutes since previous detection")
    axs[1].set_title("(b) the final swarm")
    for ax in axs:
        ax.grid(alpha=0.3, which="both")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
    fig.suptitle("Inverse rate: a decrease towards zero at failure indicates acceleration")
    fig.tight_layout()
    fig.savefig(tm.OUT / "fig_inverse_rate.png", dpi=120)
    plt.close(fig)


def background_stats(h: pd.DataFrame) -> dict:
    """Percentile of the 2023-09-14 00:00-02:00 hourly levels within all hours, per station."""
    out = {}
    last = h.hour >= pd.Timestamp("2023-09-14T00:00")
    for s_ in tm.STATIONS:
        col = h[f"{s_}_rms_ms"]
        ref = col[~last].dropna()
        out[s_] = {"median_all_hours": float(ref.median()),
                   "final_hours": {str(hh)[11:16]: {"rms": float(v), "percentile": float((ref < v).mean() * 100)}
                                   for hh, v in zip(h.hour[last], col[last], strict=True) if np.isfinite(v)}}
    return out


def fig_background(h: pd.DataFrame):
    fig, axs = plt.subplots(2, 1, figsize=(12, 7))
    for s in tm.STATIONS:
        axs[0].semilogy(h.hour, h[f"{s}_rms_ms"], lw=0.6, label=s)
        last = h[h.hour >= pd.Timestamp(tm.ORIGIN) - pd.Timedelta(days=3)]
        axs[1].semilogy(last.hour, last[f"{s}_rms_ms"], ".-", lw=0.6, ms=3, label=s)
    for ax in axs:
        ax.axvline(tm.ORIGIN, color="r", lw=0.8)
        ax.set_ylabel("hourly median of 1-min RMS, Z, 1-5 Hz (m/s)")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    axs[0].set_title("Background 1-5 Hz level (catches emergent signals that do not match a template)")
    axs[1].set_title("last 3 days")
    fig.tight_layout()
    fig.savefig(tm.OUT / "fig_background_rms.png", dpi=120)
    plt.close(fig)


def fig_detection_examples(cat: pd.DataFrame, n: int = 8):
    """Top-n events (by MAD multiple) against their best template, all channels."""
    if cat.empty:
        return
    top = cat.sort_values("mad_multiple", ascending=False).head(n).sort_values("time")
    tpl = tm.load_templates()
    v = tm.open_continuous()
    t = v.time.values
    fig, axs = plt.subplots(1, len(top), figsize=(2.6 * len(top), 7), sharey=True, squeeze=False)
    for ax, (_, ev) in zip(axs[0], top.iterrows(), strict=True):
        tp = tpl[ev.template]
        i = np.searchsorted(t, np.datetime64(ev.time))
        pad = int(300 * tm.FS)
        raw = v.isel(time=slice(max(0, i - pad), i + tp["m"] + pad)).values.reshape(tm.nch(), -1).astype("float64")
        x = np.stack([tm.bandpass(tr) for tr in raw])[:, min(pad, i): min(pad, i) + tp["m"]]
        tt = np.arange(tp["m"]) / tm.FS
        for c in range(tm.nch()):
            y, w = x[c], tp["data"][c]
            sc = np.nanmax(np.abs(y)) or 1
            ax.plot(tt, w / np.abs(w).max() * 0.4 + c, color="0.6", lw=0.5)
            ax.plot(tt, y / sc * 0.4 + c, color=COLORS[ev.template], lw=0.5)
        ax.set_title(f"{str(ev.time)[:19]}\n{ev.template} CC {ev.cc:.2f}\n{ev.mad_multiple:.0f} MAD, amp x{ev.amp_ratio:.2f}",
                     fontsize=7)
        ax.set_xlabel("s")
    axs[0, 0].set_yticks(range(tm.nch()), [f"{s.split('.')[1]}.{c}" for s in tm.STATIONS for c in tm.COMPONENTS])
    fig.suptitle("Strongest detections outside the failure sequence (colour) over their template (grey), 1-5 Hz")
    fig.tight_layout()
    fig.savefig(tm.OUT / "fig_detection_examples.png", dpi=120)
    plt.close(fig)


def fig_detection_stack(d: pd.DataFrame, k: float = tm.K_MAD, reverse: bool = False):
    """Linear stack, per channel, of every detection >= k MAD outside the sequence (each
    window normalized by its RMS). Repeats of a real source converge to the template's
    arrivals and moveout; chance matches stack to incoherent noise. Selection alone biases
    the stack towards the template, so the same stack of the reversed-template control
    (reverse=True, no figure) is the reference."""
    tpl = tm.load_templates(reverse=reverse)
    v = tm.open_continuous()
    t = v.time.values
    names = [n for n in tm.TEMPLATES if ((d.template == n) & ~d.in_sequence & (d.mad_multiple >= k)).sum() >= 3]
    if not names:
        return {}
    fig, axs = plt.subplots(1, len(names), figsize=(3.2 * len(names), 7), sharey=True, squeeze=False)
    out = {}
    for ax, name in zip(axs[0], names, strict=True):
        tp = tpl[name]
        sel = d[(d.template == name) & ~d.in_sequence & (d.mad_multiple >= k)]
        acc = np.zeros((tm.nch(), tp["m"]))
        cnt = np.zeros(tm.nch())
        pad = int(120 * tm.FS)
        for tt in sel.time.values:
            i = np.searchsorted(t, np.datetime64(tt))
            raw = v.isel(time=slice(max(0, i - pad), i + tp["m"] + pad)).values.reshape(tm.nch(), -1).astype("float64")
            x = np.stack([tm.bandpass(tr) for tr in raw])[:, min(pad, i): min(pad, i) + tp["m"]]
            for c in range(tm.nch()):
                if x.shape[1] == tp["m"] and np.isfinite(x[c]).all():
                    acc[c] += x[c] / np.sqrt(np.mean(x[c] ** 2))
                    cnt[c] += 1
        stack = acc / np.maximum(cnt, 1)[:, None]
        cc = [float(np.corrcoef(stack[c], tp["data"][c])[0, 1]) for c in range(tm.nch())]
        out[name] = {"n_detections": len(sel), "stack_vs_template_cc_per_channel": cc,
                     "stack_vs_template_cc_mean": float(np.mean(cc))}
        if reverse:
            continue
        tx = np.arange(tp["m"]) / tm.FS
        for c in range(tm.nch()):
            w = tp["data"][c]
            ax.plot(tx, w / np.abs(w).max() * 0.4 + c, color="0.6", lw=0.5)
            ax.plot(tx, stack[c] / (np.abs(stack[c]).max() or 1) * 0.4 + c, color=COLORS[name], lw=0.6)
        ax.set_title(f"{name}: stack of {len(sel)}\nmean CC with template {np.mean(cc):.2f}", fontsize=8)
        ax.set_xlabel("s after window start")
    if reverse:
        plt.close(fig)
        return out
    axs[0, 0].set_yticks(range(tm.nch()), [f"{s.split('.')[1]}.{c}" for s in tm.STATIONS for c in tm.COMPONENTS])
    fig.suptitle(f"Linear stack of all detections >= {k:g} MAD outside the failure sequence (colour) vs template (grey)")
    fig.tight_layout()
    fig.savefig(tm.OUT / "fig_detection_stack.png", dpi=120)
    plt.close(fig)
    return out


def make_all():
    d, c, s, h = _read()
    vh = valid_hours_per_day(s)
    ctrl = threshold_control(d, c)
    ctrl.to_csv(tm.OUT / "threshold_control.csv", index=False, float_format="%.3g")
    cats = {lab: union_catalog(d, k, t) for lab, (t, k) in CATALOGS.items()}
    ctrls = {lab: union_catalog(c, k, t) for lab, (t, k) in CATALOGS.items()}
    cols = ["event", "time", "hours_before_failure", "templates", "template", "cc", "mad_multiple",
            "amp_ratio", "n_channels"] + [f"cc_{s_.split('.')[1]}" for s_ in tm.STATIONS]
    cats["P1_onset >= 8 MAD"][cols].to_csv(tm.OUT / "catalog.csv", index=False, float_format="%.4g")
    cats["all templates >= 8 MAD"][cols].to_csv(tm.OUT / "catalog_all_templates_ge8mad.csv", index=False,
                                                float_format="%.4g")
    days = pd.to_datetime(vh.index)
    rates = pd.DataFrame({"day": vh.index, "valid_hours": vh.values})
    for lab in CATALOGS:
        key = lab.replace(" >= ", "_ge").replace(" MAD", "mad").replace(" ", "_")
        rates[key] = _daily(cats[lab], days).values
        rates[key + "_control"] = _daily(ctrls[lab], days).values
    rates.to_csv(tm.OUT / "daily_rates.csv", index=False, float_format="%.3g")
    t0 = pd.Timestamp(tm.ORIGIN) - pd.Timedelta(hours=72)
    hrs = pd.date_range(t0.floor("h"), pd.Timestamp(tm.ORIGIN).floor("h"), freq="h")
    hr = pd.DataFrame({"hour": hrs})
    for lab in CATALOGS:
        key = lab.replace(" >= ", "_ge").replace(" MAD", "mad").replace(" ", "_")
        for x, suf in [(cats[lab], ""), (ctrls[lab], "_control")]:
            x = x[x.time >= t0]
            hr[key + suf] = x.groupby(x.time.dt.floor("h")).size().reindex(hrs, fill_value=0).values if len(x) else 0
    hr.to_csv(tm.OUT / "hourly_rates_last72h.csv", index=False)

    seq = d[d.in_sequence & (d.mad_multiple >= tm.K_MAD)].sort_values("time")
    other = d[(d.time - pd.Timestamp(tm.OTHER_EVENT)).abs() < pd.Timedelta(minutes=3)]
    swarm = swarm_stats(cats["P1_onset >= 8 MAD"])
    res = {
        "stations": tm.STATIONS,
        "threshold_k_mad": tm.K_MAD,
        "daily_mad_by_template": s.groupby("template").mad.median().to_dict(),
        "valid_hours_total": float(vh.sum()),
        "catalog_sizes": {lab: {"real": len(cats[lab]), "control": len(ctrls[lab])} for lab in CATALOGS},
        "rate_tests": {lab: rate_tests(cats[lab], vh) for lab in CATALOGS},
        "rate_tests_control": {lab: rate_tests(ctrls[lab], vh) for lab in CATALOGS},
        "swarm_P1_onset_ge8": swarm,
        "swarm_P1_onset_ge8_control": swarm_stats(ctrls["P1_onset >= 8 MAD"]),
        "background_rms": background_stats(h),
        "sequence_detections": seq[["template", "time", "cc", "mad_multiple", "amp_ratio"]].astype(
            {"time": str}).to_dict("records"),
        "comcat_other_event_2023-09-13T13:57:32": other[["template", "time", "cc", "mad_multiple"]].astype(
            {"time": str}).to_dict("records"),
        "inter_precursor": inter_precursor(),
    }
    fig_templates()
    fig_event_day()
    fig_rates(cats, ctrls, d, vh)
    fig_inverse_rate(cats, swarm)
    fig_background(h)
    fig_detection_examples(cats["P1_onset >= 8 MAD"])
    res["detection_stacks"] = fig_detection_stack(d)
    res["detection_stacks_reversed_control"] = fig_detection_stack(c, reverse=True)
    (tm.OUT / "results.json").write_text(json.dumps(res, indent=1, default=float))
    print(ctrl.to_string(index=False))
    print(json.dumps({k: res[k] for k in ["catalog_sizes", "swarm_P1_onset_ge8", "background_rms"]},
                     indent=1, default=float))


if __name__ == "__main__":
    make_all()
