"""``peters-dome stage <dataset>``: same parameter conventions as ``gaia stage``
(-i AOI, -s start, -e end, --overwrite; s3:// outputs need AWS_PROFILE). Every command
writes one group of the single container store given by -o (default data/peters_dome.zarr)."""

import logging

import cyclopts
from gaia_cli.cli import configure_logging

from .context import stage_geology, stage_glacier, stage_soil
from .imagery import stage_dem, stage_optical, stage_sar
from .met import stage_era5, stage_profile, stage_synoptic
from .release import stage_release
from .waveforms import stage_infrasound, stage_seismic, stage_seismic_continuous

app = cyclopts.App(help="Peters Dome (Denali, 2023-09-14) landslide data container.")
stage_app = cyclopts.App(name="stage", help="Stage one dataset into its group of the container store.")
app.command(stage_app)

for fn, name in [
    (stage_seismic, "seismic"),
    (stage_seismic_continuous, "seismic-continuous"),
    (stage_infrasound, "infrasound"),
    (stage_sar, "sar"),
    (stage_optical, "optical"),
    (stage_dem, "dem"),
    (stage_glacier, "glacier"),
    (stage_geology, "geology"),
    (stage_soil, "soil"),
    (stage_era5, "era5"),
    (stage_profile, "profile"),
    (stage_synoptic, "synoptic"),
    (stage_release, "release"),
]:
    stage_app.command(fn, name=name)


def main():
    configure_logging()  # gaia-cli format and GAIA_LOG_LEVEL
    logging.getLogger("peters_dome").setLevel(logging.getLogger("gaia_cli").level)
    app()


if __name__ == "__main__":
    main()
