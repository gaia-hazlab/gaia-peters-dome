#!/usr/bin/env bash
# Stage every modality for the 2023 Peters Dome ice-rock avalanche into data/staged/.
# Usage:  pixi run stage-all            (skips stores that already exist)
#         OUT=s3://bucket/prefix pixi run stage-all   (needs AWS_PROFILE)
# Time windows mirror config/event.yml.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=${OUT:-data/staged}
LOCAL=aoi/AOI.geojson                       # = aoi/peters_dome_local.geojson
REGIONAL=aoi/peters_dome_regional.geojson
WAVE=aoi/peters_dome_waveforms.geojson
mkdir -p data/staged

run() {  # run <store> <cmd...>; skip if a local store already exists
  local store=$1; shift
  if [[ $store != s3://* && -e $store ]]; then echo "== skip $store (exists)"; return; fi
  echo "== $store"; "$@" -o "$store" || echo "!! FAILED: $store (continuing)"
}

# --- event: seismic + infrasound (FDSN, EarthScope) ---------------------------------
run $OUT/seismic.zarr     peters-dome stage seismic    -i $WAVE -s 2023-09-14T01:50:00 -e 2023-09-14T02:30:00 --max-distance-km 250
run $OUT/seismic_lp.zarr  peters-dome stage seismic    -i $WAVE -s 2023-09-14T01:50:00 -e 2023-09-14T02:30:00 --max-distance-km 400 --channels BH? --sampling-rate 1 --units DISP
run $OUT/infrasound.zarr  peters-dome stage infrasound -i $WAVE -s 2023-09-14T01:50:00 -e 2023-09-14T03:00:00

# --- remote sensing ---------------------------------------------------------------------
run $OUT/sar_s1_rtc.zarr      peters-dome stage sar     -i $LOCAL -s 2023-08-01 -e 2023-10-31
run $OUT/optical_s2.zarr      peters-dome stage optical -i $LOCAL -s 2023-08-01 -e 2023-10-31 --sensor sentinel2
run $OUT/optical_landsat.zarr peters-dome stage optical -i $LOCAL -s 2023-08-01 -e 2023-10-31 --sensor landsat
run $OUT/dem_copdem.zarr      peters-dome stage dem     -i $LOCAL --product copdem
run $OUT/dem_arcticdem_mosaic.zarr peters-dome stage dem -i $LOCAL --product arcticdem-mosaic
run $OUT/dem_arcticdem_strips.zarr peters-dome stage dem -i $LOCAL --product arcticdem-strips -s 2008-01-01 -e 2025-12-31
run $OUT/usgs_release.zarr    peters-dome stage release

# --- context ----------------------------------------------------------------------------
run $OUT/glacier.zarr  peters-dome stage glacier -i $REGIONAL
run $OUT/geology.zarr  peters-dome stage geology -i $REGIONAL
run $OUT/soil.zarr     peters-dome stage soil    -i $LOCAL

# --- meteorology ------------------------------------------------------------------------
run $OUT/met_era5.zarr      peters-dome stage era5    -i $REGIONAL -s 2023-06-01 -e 2023-09-21 --model era5
run $OUT/met_era5_land.zarr peters-dome stage era5    -i $REGIONAL -s 2023-06-01 -e 2023-09-21 --model era5_land
run $OUT/met_profiles.zarr  peters-dome stage profile -s 2023-09-13T00:00 -e 2023-09-14T12:00
# In-situ precipitation through gaia-cli itself (needs SYNOPTIC_TOKEN)
if [[ -n ${SYNOPTIC_TOKEN:-} ]]; then
  run $OUT/met_synoptic.zarr gaia stage synoptic -i $REGIONAL -s 2023-06-01 -e 2023-09-21
else
  echo "== skip met_synoptic (SYNOPTIC_TOKEN not set)"
fi

echo "Staged stores in $OUT. Next: pixi run build-tree && pixi run build-cube"
