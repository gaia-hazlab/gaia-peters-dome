# gaia-peters-dome

Data container for the **2023 Peters Dome ice–rock avalanche**, Denali National Park and
Preserve, Alaska, studied in Toney et al. (2026), *Rapid Seismic and Infrasound Assessment
of Large Landslides: A Case Study From Denali National Park and Preserve (Alaska)*, JGR
Earth Surface 131(3), e2025JF008577, [doi:10.1029/2025JF008577](https://doi.org/10.1029/2025JF008577).

Staging follows the [gaia-cli](https://github.com/gaia-hazlab/gaia-cli) contract: each
dataset has a `load(aoi, start, end) -> xarray` function and a `stage` CLI entrypoint with
`-i AOI -s START -e END -o OUTPUT.zarr`; Zarr v3 writing, xvec encoding of station and
vector data, and DataTree assembly reuse `gaia_cli.io` directly. Every module here can be
moved into `gaia_cli/` as a new `gaia stage <name>` subcommand without changes to its I/O.

**Report:** <https://gaia-hazlab.github.io/gaia-peters-dome/>, figures built from the
staged stores by `pixi run report` (writes `docs/`, served by GitHub Pages from `main:/docs`).

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

```bash
gaia stage synoptic -i aoi/AOI.geojson -s 2023-06-01 -e 2023-09-21 -o met_synoptic.zarr
```

## Datasets

| `peters-dome stage …` | source | harmonized form | window (stage_all.sh) |
|---|---|---|---|
| `seismic` | FDSN, EarthScope DMC; HH?/BH? within 250 km | velocity m/s, (station, component, time), 20 Hz common grid | 01:50–02:30 UTC |
| `seismic --units DISP --sampling-rate 1` | BH? within 400 km | displacement m, 1 Hz, for long-period force work | same |
| `infrasound` | BDF/HDF within 400 km (AK, AV arrays, IMS I53US at ~252 km) | pressure Pa, (station, time), 20 Hz | 01:50–03:00 UTC |
| `sar` | Sentinel-1 RTC, Planetary Computer | γ⁰ VV/VH linear, UTM 5N 10 m, orbit metadata per time | 2023-08-01 → 10-31 |
| `optical --sensor sentinel2` | S2 L2A C1, Earth Search | reflectance 0–1 + SCL, UTM 5N 10 m | same |
| `optical --sensor landsat` | Landsat 8/9 C2 L2, Planetary Computer | reflectance + QA, 30 m | same |
| `dem --product copdem` | Copernicus GLO-30 | m (EGM2008) | static |
| `dem --product arcticdem-mosaic` | PGC ArcticDEM v4.1 mosaic | m (WGS84 ellipsoid), 10 m | static |
| `dem --product arcticdem-strips` | PGC ArcticDEM strips s2s041 | multitemporal stack, not co-registered | 2008–2025 (strips found: 2010–2020) |
| `release` | USGS release doi:10.5066/P13WIS6X | force-time function, SHALTOP run, post-event SETSM DEM, source/deposit polygons | event |
| `glacier` | GLIMS WFS (RGI Alaska, GLIMS outlines, GlaThiDa points), ITS_LIVE v2 | xvec vectors + velocity m/yr | composite mosaic |
| `geology` | USGS SIM 3340 Geologic Map of Alaska (Wilson et al. 2015) | units + contacts/faults, xvec + GeoJSON | static |
| `soil` | ISRIC SoilGrids 2.0 | bdod, cfvo, clay, sand, silt, soc at 6 depths 0–200 cm, 250 m | static |
| `era5 --model era5 / era5_land` | Open-Meteo archive (ERA5 0.25°, ERA5-Land 0.1°) | hourly T, P, snow, wind, radiation, soil T/θ on reanalysis nodes | 2023-06-01 → 09-21 |
| `profile` | Open-Meteo historical-forecast (best-match NWP) | T, wind, Z, RH at 15 levels 1000–30 hPa; freezing level | 09-13 00:00 → 09-14 12:00 |
| `gaia stage synoptic` | Synoptic API via gaia-cli | hourly precip at stations | needs `SYNOPTIC_TOKEN` |

## Run

```bash
pixi install
pixi run stage-all       # every dataset -> data/staged/*.zarr (+ GeoJSON for vectors)
pixi run build-tree      # one DataTree -> data/peters_dome.zarr
pixi run build-cube      # common UTM 5N 10 m grid -> data/peters_dome_datacube.zarr
pixi run report          # figures + docs/index.html for GitHub Pages
pixi run test            # offline tests; `pixi run test-online` hits live services
```

`stage_all.sh` skips stores that already exist and continues past failures, so rerunning
it fills gaps. `OUT=s3://bucket/prefix pixi run stage-all` writes to S3 (needs `AWS_PROFILE`).

### DataTree layout (`data/peters_dome.zarr`)

```
/                       event metadata (attrs)
stations/  seismic, seismic_lp, infrasound, synoptic
rasters/   sar/sentinel1_rtc  optical/{sentinel2,landsat}
           topography/{copdem,arcticdem_mosaic,arcticdem_strips,setsm_post_event}
           glacier/itslive_velocity  soil/soilgrids  met/{era5,era5_land}
vectors/   glacier/{rgi,glims,glathida}  geology/{AKStategeol_poly,AKStategeol_arc}
           landslide_outlines
model/     force_inversion  shaltop  atmosphere_profiles
```

Vector and station groups are xvec CF-encoded: `ds.xvec.decode_cf()` after opening.

### Datacube (`data/peters_dome_datacube.zarr`)

Before/after pairs on dim `epoch`: Sentinel-1 VV/VH and VV log-ratio (dB), best clear-sky
Sentinel-2 bands and NDSI. Plus elevations (COP-30, ArcticDEM mosaic, post-event SETSM)
and their difference, ITS_LIVE speed, RGI glacier mask, landslide source/footprint mask,
SHALTOP maximum thickness and speed, rasterized geology units (code table in attrs), and
SoilGrids properties with a `depth` dimension.

## Design choices that depart from gaia-cli defaults

- Rasters at ≤ 30 m are staged in UTM 5N (EPSG:32605), not EPSG:4326: at 63° N a degree of
  longitude is ~2.2 times shorter than a degree of latitude, and the release DEM is already
  in UTM 5N. Coarse met grids stay in EPSG:4326.
- Station selection uses AOI polygon containment plus an optional distance cap
  (`--max-distance-km`), because the relevant infrasound arrays sit 200–310 km away.
- Meteorology uses Open-Meteo rather than ARCO-ERA5 on GCS: that store is chunked one hour
  per global field, so a 3-month series at a dozen nodes reads tens of GB.
- The geologic map zip (710 MB) is downloaded once to `data/raw/geology/`; reading it in
  place over HTTP was too slow because GDAL cannot use the shapefile `.sbn` index remotely.
  The USGS server delivered ~300 kB/s on 2026-10-14, so the first run takes ~40 min.

## Known gaps

- No post-event ArcticDEM strip over the AOI (latest 2020-11-03); the post-event surface is
  the release SETSM DEM. Pre-event reference = ArcticDEM mosaic or the 2020 strips.
- `dh_post_minus_mosaic` in the datacube is not co-registered.
- SHALTOP time axis assumes simulation t = 0 at the force-inversion zero time; the release
  does not state this (recorded in the group attrs).
- Atmospheric profiles stop at 30 hPa (~24 km). Stratospheric ducting for the I53US path
  needs ERA5 model levels (CDS key) or G2S.
- Not staged: ice thickness (Farinotti 2019 / Millan 2022 need large downloads or Earthdata
  login), permafrost extent, Daymet (Earthdata login), HRRR-Alaska, oblique aerial photos
  (NPS, not public), Sentinel-1 SLC for InSAR coherence (use `asf_search`).
- Soil grids are mostly NaN over ice and bare rock, which covers the source area.
