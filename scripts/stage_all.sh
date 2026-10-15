#!/usr/bin/env bash
# Stage every dataset of the 2023 Peters Dome ice-rock avalanche into ONE Zarr store,
# one group per dataset (gaia-cli layout: stations/, rasters/, vectors/, model/).
# Usage:  pixi run stage-all                       -> data/peters_dome.zarr
#         STORE=s3://bucket/peters_dome.zarr pixi run stage-all   (needs AWS_PROFILE)
# Existing groups are skipped (--skip-existing), so rerunning resumes; a failed dataset
# is reported and the run continues. Time windows mirror config/event.yml.
set -uo pipefail
cd "$(dirname "$0")/.."

STORE=${STORE:-data/peters_dome.zarr}
LOCAL=aoi/AOI.geojson                       # = aoi/peters_dome_local.geojson
REGIONAL=aoi/peters_dome_regional.geojson
WAVE=aoi/peters_dome_waveforms.geojson
mkdir -p data

run() { echo "== $*"; "$@" -o "$STORE" --skip-existing || echo "!! FAILED: $*"; }
pd() { run peters-dome stage "$@"; }

# --- event: seismic + infrasound (FDSN, EarthScope) ---------------------------------
pd seismic    -i $WAVE -s 2023-09-14T01:50:00 -e 2023-09-14T02:30:00 --max-distance-km 250
pd seismic    -i $WAVE -s 2023-09-14T01:50:00 -e 2023-09-14T02:30:00 --max-distance-km 400 \
              --channels 'BH?' --sampling-rate 1 --units DISP --group stations/seismic_lp
pd infrasound -i $WAVE -s 2023-09-14T01:50:00 -e 2023-09-14T03:00:00

# --- remote sensing and published results -------------------------------------------
pd sar     -i $LOCAL -s 2023-08-01 -e 2023-10-31
pd optical -i $LOCAL -s 2023-08-01 -e 2023-10-31 --sensor sentinel2
pd optical -i $LOCAL -s 2023-08-01 -e 2023-10-31 --sensor landsat
pd dem     -i $LOCAL --product copdem
pd dem     -i $LOCAL --product arcticdem-mosaic
pd dem     -i $LOCAL --product arcticdem-strips -s 2008-01-01 -e 2025-12-31
pd release

# --- context ----------------------------------------------------------------------------
pd glacier -i $REGIONAL
pd geology -i $REGIONAL
pd soil    -i $LOCAL

# --- meteorology ------------------------------------------------------------------------
pd era5    -i $REGIONAL -s 2023-06-01 -e 2023-09-21 --model era5
pd era5    -i $REGIONAL -s 2023-06-01 -e 2023-09-21 --model era5_land
pd profile -s 2023-09-13T00:00 -e 2023-09-14T12:00
if [[ -n ${SYNOPTIC_TOKEN:-} ]]; then
  pd synoptic -i $REGIONAL -s 2023-06-01 -e 2023-09-21    # gaia-cli Synoptic loader
else
  echo "== skip synoptic (SYNOPTIC_TOKEN not set)"
fi

echo "Staged into $STORE. Next: pixi run build-cube && pixi run report"
