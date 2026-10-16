# Precursory seismicity before the 2023-09-14 Peters Dome failure

## Verdict

- **Weeks to days before failure (2023-08-01 to 2023-09-13 ~23:00 UTC): no precursory
  seismicity detectable with these templates.** At the three nearest broadbands, detection
  counts for every template stay at the level of the time-reversed (chance) control, with no
  trend.
- **Last ~2 h: a swarm of small repeating events that match the onset of precursor P1.**
  It starts at 2023-09-13 23:52 UTC, 2 h 14 min before the failure. Detections come faster
  until about 01:30, then stop: **16 min with no detection before P1** (ComCat 01:58:54).
  The template-matched rate accelerates over ~1.5 h, but inter-event times reach their
  minimum (~1-2 min) about 20-30 min before P1, not at failure.
- **Inter-precursor interval 02:00:30-02:03:10: not quiet.** The 1-5 Hz envelope decays as
  P1 coda until ~02:01:30. It then holds flat at 1.4x noise until P2 starts, above every
  20-s bin of the pre-P1 noise (max 1.27x). No discrete template detection falls in it.

As far as can be checked against the paper's abstract (the full text was not
accessible), the swarm and the inter-precursor plateau are not reported there. The abstract
reports two infrasound precursors minutes before failure; the template
`P1_onset` is cut from the first of these on the seismometers.

## Results

All numbers come from the CAST+PPLA run (`cast_ppla/`) unless the row says otherwise.
KTH has no data at EarthScope from 2023-08-13 to 2023-09-12 21:58 UTC (FDSN dataselect
returns no data on BH, BN and LH). Rates are therefore compared on the two stations present
throughout. The run with KTH (`kth_cast_ppla/`) gives the same swarm (last column).

**Detections vs chance** (`threshold_control.csv`; command: `template_matching.py --run cast_ppla`)

| template | K (MAD) | detections, 44 days, outside 01:57-02:10 | reversed-template control |
|---|---|---|---|
| P1_onset | 8 | 39 | 18 |
| P1_onset | 10 | 10 | 0 |
| P1_onset | 12 | 3 | 0 |
| P1_long | 8 / 10 | 261 / 26 | 177 / 11 |
| P2_long | 8 / 10 / 12 | 313 / 26 / 3 | 244 / 13 / 0 |
| P2_peak | 8 | 9 | 6 |
| main_onset | 8 / 10 | 170 / 7 | 143 / 2 |

At 8 MAD most detections are chance: the reversed templates, which share the real
templates' length, spectrum and autocorrelation, return 70-85% as many. So 8 MAD,
chosen beforehand from the Gaussian argument below, is too low for these data; the control,
not the Gaussian estimate, sets the usable threshold. Only `P1_onset` beats the control.

**When the excess falls** (`catalog.csv`, `daily_rates.csv`, `results.json` key `rate_tests`)

| catalog | all 44 days | before the last 24 h | last 24 h before failure | control, last 48 h | binomial p, last 24 h vs constant rate |
|---|---|---|---|---|---|
| P1_onset >= 8 MAD | 39 | 20 (control 18) | 19 (0.9 expected) | 0 | 7e-21 |
| P1_onset >= 10 MAD | 10 | 0 | 10 (0.24 expected) | 0 | 6e-17 |
| all templates >= 12 MAD | 7 | 4 | 3 (0.17 expected) | 0 | 4e-4 |
| all templates >= 8 MAD (mostly chance) | 707 | 665 | 42 (16.9 expected) | n/a | 1e-7 |

Of the 19 P1_onset detections in the last 24 h, 18 fall between 2023-09-13 23:52:26 and
2023-09-14 01:42:51. The 19th is at 09-13 09:45, within the chance rate. The 20
detections before the last 24 h, against 18 for the control over all 44 days, mean no
excess over the preceding 43 days, and
`fig_rates.png` (b, c) shows no trend in them.

**The swarm** (`results.json` key `swarm_P1_onset_ge8`; `fig_inverse_rate.png` b; `fig_rates.png` d)

| quantity | CAST+PPLA | KTH+CAST+PPLA |
|---|---|---|
| detections, 23:52-01:43 UTC | 18 | 19 |
| median inter-event time, first / second half | 8.1 / 1.6 min | 4.9 / 1.75 min |
| shortest inter-event times | 1.1-1.7 min, 01:20-01:30 | 1.1-1.9 min, same period |
| linear fit of inter-event time vs time, reaches zero at | 22 min before P1 (~01:37) | 22 min before P1 |
| last detection before P1 / quiet gap | 01:42:51 / 16.0 min | same |
| reversed-control detections in the same 6 h | 0 | 0 |
| network CC / MAD multiple | 0.18-0.32 / 8.1-14.1 | 0.16-0.29 / 8.1-14.5 |
| per-station CC (KTH, CAST, PPLA) | n/a | 0.06-0.27, 0.14-0.39, 0.05-0.27; positive at all three stations for all 18 detections with KTH data |
| amplitude ratio to the P1_onset window | median 0.087 | median 0.096 |

The amplitude ratio is RMS over the 30-s window, and the swarm windows sit at the noise RMS.
So the events are at most ~0.1 of P1's onset amplitude, about M <= 0.4 if scaled from
P1's ComCat M1.4. That is an upper bound, not a magnitude. The swarm CCs also drop over
time (≥10 MAD early, 8-9 MAD after 01:20) while events become more frequent.

One more `P1_onset` match falls inside the excluded sequence window, at **01:58:17**
(9.2 MAD, amplitude 0.14), 37 s before P1's ComCat origin. It coincides with the small
envelope bump visible just before P1 (`fig_inter_precursor.png`).

**Background level** (`hourly_background_rms.csv`; `results.json` key `background_rms`;
`fig_background_rms.png`). Hourly median of 1-min RMS, Z, 1-5 Hz, as a percentile of all
hours 2023-08-01 to 09-13. CAST is the only station without daily on/off steps; PPLA has a
step from about 12 to 23 UTC that looks like site noise.

| hour (2023-09-14 UTC) | KTH (3-station run) | CAST | PPLA |
|---|---|---|---|
| 00:00-01:00 | 87th | 40th | 42nd |
| 01:00-02:00 | 75th | 62nd | 49th |

The swarm does not raise the broadband 1-5 Hz level; it is found only by matched filtering.

**Inter-precursor interval** (`inter_precursor.csv`, `fig_inter_precursor.png`; event
group `stations/seismic`, median envelope of the 9 channels of KTH, CAST, PPLA, 5-s
smoothing, normalized by 01:50-01:55)

| window (UTC) | observed / noise | P1 coda model / noise | observed / model |
|---|---|---|---|
| 02:00:30-02:00:50 | 3.48 | 3.65 | 0.95 |
| 02:01:10-02:01:30 | 1.98 | 1.30 | 1.52 |
| 02:01:50-02:02:10 | 1.49 | 1.02 | 1.46 |
| 02:02:30-02:02:50 | 1.37 | 1.00 | 1.37 |
| 02:02:50-02:03:10 | 1.42 | 1.00 | 1.42 |

| reference | value |
|---|---|
| P1 coda decay, fit 01:59:50-02:01:00 | -0.94 decades/min |
| main-event coda decay, fit 02:08:00-02:10:30 | -0.61 decades/min |
| plateau 02:01:50-02:03:10: median, slope | 1.41x, -0.03 decades/min |
| quiet noise 01:50:00-01:57:40, 23 bins of 20 s: median / 95th pct / max | 0.99 / 1.22 / 1.27x |

In 02:00:30-02:03:10 the network CC peaks at 4.9-6.3 MAD for P1_long, P1_onset, P2_peak
and main_onset (`fig_event_day_cc.png`, right column). P2_long is excluded because its
150-s windows starting there overlap P2 itself. So the plateau is low-level emergent signal, not
discrete events similar to P1, P2 or the main onset. Even with the slower main-event coda
decay, P1's coda would be within 10% of noise by about 02:02:30.

**ComCat cross-check.** In a 30 km radius, 2023-07-01 to 09-15, ComCat lists slab
earthquakes at 100-130 km depth, shallow crustal events, and one "other event" before
failure besides P1: 2023-09-13 13:57:32, M1.6, ~13 km south of the source. It matches the templates
only at chance level (P1_long 8.0 MAD, P2_long 8.7 MAD), so it is not a repeat of P1.

**Checks that passed**
- Each template self-detects in the continuous group at CC = 1.000 at its own start time.
- The daily MAD of the P1_onset network CC is stable: 0.0214-0.0236 over all 44 days,
  0.0228 on 2023-09-14. The short last day (2 h 10 min) does not inflate the MAD multiples.
- Linear stacks of the detections (`fig_detection_stack.png`) correlate with the templates
  (P1_onset 0.64 vs 0.53 for the control stack). Detections are selected for correlating
  with the template, which biases any stack towards it. The control comparison is the
  meaningful one, and it separates the two only weakly. The rate contrast is the evidence;
  the stack is not.

**What this does not show**
- Nothing earlier than 2023-08-01, and nothing with waveforms unlike these five templates.
  An emergent or tremor-like precursor that does not raise the hourly 1-5 Hz level would be
  missed.
- Detection power is lower from 2023-08-13 to 09-12, when KTH is missing, and on
  08-13/08-14, when CAST is missing too (`daily_rates.csv`, `valid_hours`).
- No location. The swarm matches the P1 onset at three stations 47-58 km away with the
  same moveout, so it comes from near P1's source, but with only three stations the
  resolution of that statement is unknown.

## Data

| item | value |
|---|---|
| search data | group `stations/seismic_continuous` of `data/peters_dome.zarr`: AK.KTH (47.3 km), AK.CAST (54.2 km), AK.PPLA (57.7 km), BH Z/N/E, response removed, velocity, 20 Hz, float32, 1.7 GB on disk |
| coverage | CAST and PPLA ~100% except CAST 08-13 to 08-15; KTH 5-65% per day 08-01 to 08-12, none 08-13 to 09-12 21:58, then 68-82% (no data at EarthScope) |
| staged with | `pixi run peters-dome stage seismic-continuous` (defaults: 2023-08-01T00:00 to 2023-09-14T02:10 UTC, EarthScope FDSN) |
| template data | group `stations/seismic` (event window 01:50-02:30 UTC), same three stations |
| processing parity | each template self-detects in the continuous group at CC = 1.000 at its own start time (`results.json`, `sequence_detections`), so the day-by-day staging reproduces the event-window processing |

## Method

| step | choice |
|---|---|
| filter | 1-5 Hz, 4-pole Butterworth, zero phase; samples within 10 s of a gap masked |
| templates | same absolute UTC window on every channel (moveout kept), see table below and `fig_templates.png` |
| per-channel CC | normalized sliding cross-correlation (Pearson) against every window; NaN if the window touches a gap |
| network CC | mean over valid channels; kept only where >= 6 channels are valid (both CAST and PPLA in the main run) |
| threshold | median + K * MAD of the network CC, MAD unscaled, computed per UTC day; K = 8 for the catalog, counts also at K = 10, 12, 15, 20 |
| control | the same scan with each template reversed in time (same length, spectrum and autocorrelation; no physical source), giving the chance detection count at each K |
| declustering | one detection per template length per template; across templates, detections within 60 s merge into one event (highest MAD multiple kept) |
| amplitude ratio | median over valid channels of RMS(data window) / RMS(template window) |
| excluded | 2023-09-14 01:57-02:10 (the failure sequence itself); listed separately in `results.json` |

| template | window at the stations (UTC, 2023-09-14) | length |
|---|---|---|
| `P1_long` | 01:58:40-02:00:30 | 110 s |
| `P1_onset` | 01:58:58-01:59:28 | 30 s |
| `P2_long` | 02:03:10-02:05:40 | 150 s |
| `P2_peak` | 02:04:40-02:05:10 | 30 s |
| `main_onset` | 02:06:55-02:07:25 | 30 s |

Why 8 MAD: for Gaussian CC, MAD = 0.674 sigma, so 8 MAD is 5.4 sigma. With about
7 x 10^5 independent samples per day in a 4 Hz band, that gives ~0.02 chance detections per
day per template. In practice the reversed templates give 516 detections at 8 MAD over 44
days for all five templates (~12 per day), most from the long templates (P1_long 177,
P2_long 244). The CC tails are much heavier than Gaussian, so the control, not this
estimate, sets the usable threshold: 10 MAD for P1_onset (0 control detections in 44 days).
`threshold_control.csv` gives both counts at each K.

## Reproduce

```bash
pixi run peters-dome stage seismic-continuous --dry-run   # size estimate only
pixi run peters-dome stage seismic-continuous             # ~2.6 GB download, ~1 h; --resume after an interruption
pixi run python analysis/precursors/template_matching.py --run cast_ppla       # ~7 min: scan, control, tables, figures
pixi run python analysis/precursors/template_matching.py --run kth_cast_ppla
pixi run python analysis/precursors/template_matching.py --run cast_ppla --plots   # tables/figures from the CSVs
```

## Files

| file | content |
|---|---|
| `template_matching.py` | templates, scan, control scan; `--run cast_ppla` (default) or `--run kth_cast_ppla` |
| `report.py` | tables, rate tests, swarm statistics, inter-precursor analysis, figures |
| `cast_ppla/`, `kth_cast_ppla/` | outputs of each run, same file names (below) |
| `fig_rates.png` | (a) daily detections per template, (b) P1_onset vs control per valid day, (c) cumulative counts with controls, (d) last 6 h |
| `fig_inverse_rate.png` | (a) inverse rate vs time to failure, 44 days, (b) inter-event times of the final swarm |
| `fig_event_day_cc.png`, `fig_inter_precursor.png` | network CC on 2023-09-14; envelope vs P1 coda model |
| `fig_templates.png`, `fig_detection_examples.png`, `fig_detection_stack.png`, `fig_background_rms.png` | templates; strongest P1_onset detections; stacks; hourly 1-5 Hz level |
| `catalog.csv` | P1_onset detections >= 8 MAD outside the failure sequence (the catalog that beats the control) |
| `catalog_all_templates_ge8mad.csv` | all templates >= 8 MAD merged within 60 s (mostly chance; for comparison) |
| `detections_all.csv` | every detection >= 8 MAD per template (including the sequence, flag `in_sequence`), with per-station CC |
| `control_reversed_detections.csv` | the same for the reversed templates |
| `threshold_control.csv` | detections vs control at K = 8, 10, 12, 15, 20 |
| `daily_rates.csv`, `hourly_rates_last72h.csv` | counts per catalog definition and its control, valid hours |
| `cc_daily_stats.csv` | per day and template: median, MAD, max of the network CC, valid fraction |
| `hourly_background_rms.csv` | hourly median of 1-min RMS, Z, 1-5 Hz, per station |
| `inter_precursor.csv` | 20-s envelope bins 02:00:30-02:03:30 vs the P1 coda model |
| `results.json` | all numbers quoted above |
| `event_day_cc.npz` | network CC traces on 2023-09-14 00:00-02:10 |
