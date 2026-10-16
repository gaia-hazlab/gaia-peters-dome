# gaia-peters-dome

Data container for the **2023 Peters Dome ice–rock avalanche**, Denali National Park and
Preserve, Alaska, studied in Toney et al. (2026), *Rapid Seismic and Infrasound Assessment
of Large Landslides: A Case Study From Denali National Park and Preserve (Alaska)*, JGR
Earth Surface 131(3), e2025JF008577, [doi:10.1029/2025JF008577](https://doi.org/10.1029/2025JF008577).

The container is **one Zarr v3 store**, `data/peters_dome.zarr`, with one group per
dataset in the [gaia-cli](https://github.com/gaia-hazlab/gaia-cli) `stage all` layout
(`stations/`, `rasters/`, plus `vectors/` and `model/`). Staging follows the gaia-cli
contract as of commit `896b628`:
- each dataset has a `load(aoi, start, end) -> xarray` function and a `stage` command
  with `-i AOI -s START -e END -o STORE --overwrite`;
- rasters stay on their native grid, with only the AOI reprojected (`io.clip_to_aoi`);
- period data carries CF `time_bounds` and `cell_methods`, so `gaia align` can aggregate it;
- Zarr preparation, string encoding, xvec encoding of stations and S3/Kopah store
  handling come from `gaia_cli.io`.

Open the container with `gaia_cli.io.open_datatree("data/peters_dome.zarr")`.

**Report:** <https://gaia-hazlab.github.io/gaia-peters-dome/>, figures built from the
container store by `pixi run report` (writes `docs/`, served by GitHub Pages from `main:/docs`).

## People

| role | GitHub |
|---|---|
| PI, GAIA | @mdenolle |
| seismic and infrasound, Alaska Earthquake Center (co-author, Toney et al. 2026) | @mewest |
| gaia-cli staging design | @scottyhq |

## Event

| quantity | value | source |
|---|---|---|
| origin time | 2023-09-14 02:06:49 UTC (2023-09-13 18:06 AKDT) | USGS ComCat, event type "landslide", M2.6 |
| precursor | 2023-09-14 01:58:54 UTC | ComCat "other event", M1.4 |
| force-inversion zero time | 2023-09-14 02:06:27 UTC | release metadata XML |
| source area | 63.1423 N, 151.1871 W | data release |
| volume | 6.1 × 10⁶ m³ | data release / paper abstract |
| source / total footprint | 0.143 km² / 1.520 km² | release outline polygons, areas in EPSG:32605 |
| peak speed | > 60 m/s | paper abstract |

All of this lives in [config/event.yml](config/event.yml); code reads it from there.

## AOIs

| file | extent | used for |
|---|---|---|
| `aoi/AOI.geojson` (= `peters_dome_local.geojson`) | 151.33–151.09 W, 63.10–63.24 N | DEM, SAR, optical, soil, datacube. USGS release bbox padded by ~3 km |
| `aoi/peters_dome_regional.geojson` | 152.2–150.2 W, 62.75–63.55 N | ERA5, Synoptic stations, glacier outlines, geology |
| `aoi/peters_dome_waveforms.geojson` | 400 km geodesic circle around the ComCat location | seismic and infrasound station selection |

Each file holds one polygon (gaia-cli uses the first geometry), CRS84.

## Datasets

| `peters-dome stage …` | group in `peters_dome.zarr` | source | harmonized form (native grid) | window (stage_all.sh) |
|---|---|---|---|---|
| `seismic` | `stations/seismic` | FDSN, EarthScope DMC; HH?/BH? within 250 km | velocity m/s, (geometry, component, time), 20 Hz common time grid | 01:50–02:30 UTC |
| `seismic --units DISP --sampling-rate 1 -g stations/seismic_lp` | `stations/seismic_lp` | BH? within 400 km | displacement m, 1 Hz | same |
| `seismic-continuous` | `stations/seismic_continuous` | FDSN, EarthScope DMC; BH? at AK.KTH, AK.CAST, AK.PPLA (47-58 km) | velocity m/s, float32, (geometry, component, time), 20 Hz, 1-h time chunks; written one UTC day at a time (`--dry-run` prints the size, `--resume` fills missing days) | 2023-08-01 → 09-14 02:10 UTC |
| `infrasound` | `stations/infrasound` | BDF/HDF within 400 km (AK, AV arrays, IMS I53US at ~252 km) | pressure Pa, (geometry, time), 20 Hz | 01:50–03:00 UTC |
| `sar` | `rasters/sar/sentinel1_rtc` | Sentinel-1 RTC, Planetary Computer | γ⁰ VV/VH linear, EPSG:32605 10 m, orbit metadata per time | 2023-08-01 → 10-31 |
| `optical --sensor sentinel2` | `rasters/optical/sentinel2` | S2 L2A C1, Earth Search | reflectance 0–1 + SCL, EPSG:32605 10 m | same |
| `optical --sensor landsat` | `rasters/optical/landsat` | Landsat 8/9 C2 L2, Planetary Computer | reflectance + QA, EPSG:32605 30 m (majority CRS; some rows are 32606) | same |
| `dem --product copdem` | `rasters/topography/copdem` | Copernicus GLO-30 | m (EGM2008), EPSG:4326 1″ | static |
| `dem --product arcticdem-mosaic` | `rasters/topography/arcticdem_mosaic` | PGC ArcticDEM v4.1 mosaic | m (WGS84 ellipsoid), EPSG:3413 at 10 m (COG overview) | static |
| `dem --product arcticdem-strips` | `rasters/topography/arcticdem_strips` | PGC ArcticDEM strips s2s041 | multitemporal, EPSG:3413 at 10 m, not co-registered | 2008–2025 (strips found: 2010–2020) |
| `release` | `model/force_published`, `model/shaltop`, `rasters/topography/setsm_post_event`, `vectors/landslide_outlines` | USGS release doi:10.5066/P13WIS6X | **published results, staged unmodified**: force-time function, SHALTOP run 0520, post-event SETSM DEM (EPSG:32605), outlines | event |
| `glacier` | `vectors/glacier/{rgi,glims,glathida}`, `rasters/glacier/itslive_velocity` | GLIMS WFS, ITS_LIVE v2 | xvec vectors; velocity m/yr, EPSG:3413 120 m | composite mosaic |
| `geology` | `vectors/geology/{units,contacts_faults}` | USGS SIM 3340 Geologic Map of Alaska (Wilson et al. 2015) | xvec polygons and lines | static |
| `soil` | `rasters/soil/soilgrids` | ISRIC SoilGrids 2.0 | bdod, cfvo, clay, sand, silt, soc at 6 depths 0–200 cm, Homolosine 250 m | static |
| `era5 --model era5 / era5_land` | `rasters/met/era5`, `rasters/met/era5_land` | Open-Meteo archive (ERA5 0.25°, ERA5-Land 0.1°) | hourly, EPSG:4326 nodes, `time_bounds` (t−1 h, t], `cell_methods` sum/point | 2023-06-01 → 09-21 |
| `profile` | `model/atmosphere_profiles` | Open-Meteo historical-forecast (best-match NWP) | T, wind, Z, RH at 15 levels 1000–30 hPa; freezing level | 09-13 00:00 → 09-14 12:00 |
| `synoptic` | `stations/synoptic` | Synoptic API through `gaia_cli.synoptic.load` | hourly precip at stations | needs `SYNOPTIC_TOKEN` |
| `scripts/build_datacube.py` | `datacube` | groups above | common EPSG:32605 10 m grid, before/after on `epoch` | event |

## Run

```bash
pixi install
pixi run stage-all       # every dataset -> its group of data/peters_dome.zarr
pixi run build-cube      # common UTM 5N 10 m grid -> group "datacube" of the same store
pixi run report          # figures + docs/index.html for GitHub Pages
pixi run test            # offline tests; `pixi run test-online` hits live services
```

Every command writes one group; an existing group is an error unless `--overwrite`
(replaces only that group) or `--skip-existing` (what `stage_all.sh` uses, so reruns
resume). `STORE=s3://bucket/peters_dome.zarr pixi run stage-all` writes to S3 or Kopah
(needs `AWS_PROFILE`, as in gaia-cli). The only other files written are the download cache
`data/cache/` (the 710 MB geologic map zip), which is not part of the container.

### Store layout (`data/peters_dome.zarr`)

```
/                       event metadata (attrs)
stations/  seismic, seismic_lp, seismic_continuous, infrasound, synoptic
rasters/   sar/sentinel1_rtc  optical/{sentinel2,landsat}
           topography/{copdem,arcticdem_mosaic,arcticdem_strips,setsm_post_event}
           glacier/itslive_velocity  soil/soilgrids  met/{era5,era5_land}
vectors/   glacier/{rgi,glims,glathida}  geology/{units,contacts_faults}
           landslide_outlines
model/     force_published  shaltop  atmosphere_profiles
datacube   event cube on one grid (scripts/build_datacube.py)
```

Station and vector groups are xvec CF-encoded: `ds.xvec.decode_cf()` after opening.
`model/force_published` and `model/shaltop` are the authors' published results from the
USGS release; no force inversion or flow simulation is rerun in this repository.

`gaia align` runs on the container for the met rasters, e.g.
`gaia align -i data/peters_dome.zarr -o met_aligned.zarr --like rasters/met/era5`
(align writes a separate store by design). It is not used for the event cube because it
aggregates every timed raster to the coarsest time step, which for hourly ERA5 next to
12-day SAR is not meaningful.

### Datacube (group `datacube`)

Before/after pairs on dim `epoch`: Sentinel-1 VV/VH and VV log-ratio (dB), best clear-sky
Sentinel-2 bands and NDSI. Plus elevations (COP-30, ArcticDEM mosaic, post-event SETSM)
and their difference, ITS_LIVE speed, RGI glacier mask, landslide source/footprint mask,
SHALTOP maximum thickness and speed, rasterized geology units (code table in attrs), and
SoilGrids properties with a `depth` dimension.

## Design choices that depart from gaia-cli defaults

- Groups are written one dataset at a time into the existing store (`common.write_group`,
  built on `gaia_cli.io._prepare_for_zarr`), rather than as one `save_datatree` call, so a
  failed or slow dataset does not lose the others. `--skip-existing` is our addition.
- ArcticDEM is read at 10 m from COG overviews in its native EPSG:3413, not at native 2 m
  (~45 M cells per strip over the AOI).
- Station selection uses AOI polygon containment plus an optional distance cap
  (`--max-distance-km`), because the relevant infrasound arrays sit 200–310 km away.
- Meteorology uses Open-Meteo rather than ARCO-ERA5 on GCS: that store is chunked one hour
  per global field, so a 3-month series at a dozen nodes reads tens of GB.
- The geologic map zip (710 MB) is downloaded once to `data/cache/`; reading it in
  place over HTTP was too slow because GDAL cannot use the shapefile `.sbn` index remotely.
  The USGS server delivered ~300 kB/s on 2026-10-14, so the first run takes ~40 min.

## Known gaps

- No post-event ArcticDEM strip over the AOI (latest 2020-11-03); the post-event surface is
  the release SETSM DEM. Pre-event reference = ArcticDEM mosaic or the 2020 strips.
- `dh_post_minus_mosaic` in the datacube is not co-registered.
- No force inversion is run here: `model/force_published` is the published result. The
  long-period displacement records in `stations/seismic_lp` are the input one would use
  to redo it.
- SHALTOP time axis assumes simulation t = 0 at the force-inversion zero time; the release
  does not state this (recorded in the group attrs).
- Atmospheric profiles stop at 30 hPa (~24 km). Stratospheric ducting for the I53US path
  needs ERA5 model levels (CDS key) or G2S.
- Not staged: ice thickness (Farinotti 2019 / Millan 2022 need large downloads or Earthdata
  login), permafrost extent, Daymet (Earthdata login), HRRR-Alaska, oblique aerial photos
  (NPS, not public), Sentinel-1 SLC for InSAR coherence (use `asf_search`).
- Soil grids are mostly NaN over ice and bare rock, which covers the source area.
