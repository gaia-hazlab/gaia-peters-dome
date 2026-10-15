"""``peters-dome stage <dataset>``: same parameter conventions as ``gaia stage``
(-i AOI, -s start, -e end, -o output Zarr; s3:// outputs need AWS_PROFILE)."""

import cyclopts

from .context import stage_geology, stage_glacier, stage_soil
from .imagery import stage_dem, stage_optical, stage_sar
from .met import stage_era5, stage_profile
from .release import stage_release
from .waveforms import stage_infrasound, stage_seismic

app = cyclopts.App(help="Peters Dome (Denali, 2023-09-14) landslide data container.")
stage_app = cyclopts.App(name="stage", help="Stage one dataset clipped to an AOI and time range.")
app.command(stage_app)

for fn, name in [
    (stage_seismic, "seismic"),
    (stage_infrasound, "infrasound"),
    (stage_sar, "sar"),
    (stage_optical, "optical"),
    (stage_dem, "dem"),
    (stage_glacier, "glacier"),
    (stage_geology, "geology"),
    (stage_soil, "soil"),
    (stage_era5, "era5"),
    (stage_profile, "profile"),
    (stage_release, "release"),
]:
    stage_app.command(fn, name=name)

if __name__ == "__main__":
    app()
