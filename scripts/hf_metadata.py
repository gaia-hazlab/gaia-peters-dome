"""Build the Hugging Face release metadata for data/peters_dome.zarr.

Writes into release/ (not committed):
- README.md       dataset card: event, groups with CRS / extent / time / source / licence,
                  attributions, known gaps, readiness profile, provenance, citation
- croissant.json  Croissant 1.1 + GeoCroissant 1.0: one FileSet per group of the Zarr store
- manifest.json   sha256 and size of every file in the store, plus the build provenance

Group facts (dims, sizes, CRS, bounding box, time range) are read from the store; sources and
licences come from config/release.yml (a group without a matching entry is an error).

    pixi run python scripts/hf_metadata.py [--store data/peters_dome.zarr] [--out release]
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from gaia_cli import io as gio

from peters_dome import common

CROISSANT_1_1 = "http://mlcommons.org/croissant/1.1"
GEOCROISSANT_1_0 = "http://mlcommons.org/croissant/geo/1.0"
ZARR_MIME = "application/x-zarr"
CONTEXT = {
    "@language": "en", "@vocab": "https://schema.org/", "sc": "https://schema.org/",
    "cr": "http://mlcommons.org/croissant/", "rai": "http://mlcommons.org/croissant/RAI/",
    "geocr": "http://mlcommons.org/croissant/geo/", "dct": "http://purl.org/dc/terms/",
    "citeAs": "cr:citeAs", "conformsTo": "dct:conformsTo", "containedIn": "cr:containedIn",
    "data": {"@id": "cr:data", "@type": "@json"}, "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
    "examples": {"@id": "cr:examples", "@type": "@json"}, "extract": "cr:extract", "field": "cr:field",
    "fileObject": "cr:fileObject", "fileProperty": "cr:fileProperty", "fileSet": "cr:fileSet",
    "format": "cr:format", "includes": "cr:includes", "isLiveDataset": "cr:isLiveDataset",
    "jsonPath": "cr:jsonPath", "key": "cr:key", "md5": "cr:md5", "parentField": "cr:parentField",
    "path": "cr:path", "recordSet": "cr:recordSet", "references": "cr:references", "regex": "cr:regex",
    "repeated": "cr:repeated", "replace": "cr:replace", "samplingRate": "cr:samplingRate",
    "separator": "cr:separator", "source": "cr:source", "subField": "cr:subField", "transform": "cr:transform",
    "arrayShape": "cr:arrayShape", "isArray": "cr:isArray", "column": "cr:column",
    "equivalentProperty": "cr:equivalentProperty",
}


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], check=False, capture_output=True, text=True, cwd=common.ROOT).stdout.strip()


def source_for(group: str, sources: dict) -> dict:
    keys = [k for k in sources if group == k or group.startswith(k)]
    if not keys:
        raise KeyError(f"no entry in config/release.yml covers group {group!r}")
    return sources[max(keys, key=len)]


def _bbox_lonlat(ds) -> list[float] | None:
    """[west, south, east, north] in EPSG:4326, from a raster grid or from CF geometries."""
    try:
        if {"x", "y"} <= set(ds.dims) and ("spatial_ref" in ds.coords or "spatial_ref" in ds):
            b = common.with_crs(ds).rio.transform_bounds("EPSG:4326")
            return [round(float(v), 5) for v in b]
        dec = ds.xvec.decode_cf()
        geoms = list(dec.xvec.geom_coords)
        if geoms:
            import geopandas as gpd
            crs = getattr(dec.xindexes.get(geoms[0]), "crs", None) or "EPSG:4326"
            g = gpd.GeoSeries(np.asarray(dec[geoms[0]].values).ravel(), crs=crs)
            b = g.to_crs("EPSG:4326").total_bounds
            return [round(float(v), 5) for v in b]
    except Exception:  # noqa: BLE001  extent is descriptive; never block the card on it
        return None
    return None


def _crs(ds) -> str | None:
    """EPSG of a raster grid (x/y dims + spatial_ref) or of decoded xvec geometries."""
    crs = None
    try:
        if {"x", "y"} <= set(ds.dims) and ("spatial_ref" in ds.coords or "spatial_ref" in ds):
            crs = common.with_crs(ds).rio.crs
        else:
            dec = ds.xvec.decode_cf()
            geoms = list(dec.xvec.geom_coords)
            crs = getattr(dec.xindexes.get(geoms[0]), "crs", None) if geoms else None
    except Exception:  # noqa: BLE001  descriptive only
        return None
    if crs is None:
        return None
    epsg = crs.to_epsg()
    return f"EPSG:{epsg}" if epsg else "custom (WKT in spatial_ref)"


def _time_range(ds) -> tuple[str, str] | None:
    for name in ("time", "epoch"):
        if name in ds.coords and np.issubdtype(ds[name].dtype, np.datetime64) and ds[name].size:
            t = pd.to_datetime(ds[name].values)
            return t.min().isoformat(), t.max().isoformat()
    return None


def describe(store: str, sources: dict) -> list[dict]:
    tree = gio.open_datatree(store)
    out = []
    for path, node in tree.subtree_with_keys:
        ds = node.to_dataset()
        if not ds.data_vars:
            continue
        src = source_for(path, sources)
        size = sum(p.stat().st_size for p in (Path(store) / path).rglob("*") if p.is_file())
        out.append({
            "group": path,
            "variables": sorted(map(str, ds.data_vars)),
            "dims": {str(k): int(v) for k, v in ds.sizes.items()},
            "crs": _crs(ds),
            "bbox": _bbox_lonlat(ds),
            "time": _time_range(ds),
            "bytes": size,
            **src,
        })
    return out


def manifest(store: str) -> list[dict]:
    rows = []
    for p in sorted(Path(store).rglob("*")):
        if p.is_file():
            h = hashlib.sha256()
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            rows.append({"path": str(p.relative_to(Path(store).parent)), "bytes": p.stat().st_size, "sha256": h.hexdigest()})
    return rows


def _mb(n: int) -> str:
    return f"{n / 1e6:,.1f}"


def croissant(cfg: dict, ev: dict, groups: list[dict], url: str, prov: dict) -> dict:
    wests, souths, easts, norths = zip(*[g["bbox"] for g in groups if g["bbox"]])
    starts = [g["time"][0] for g in groups if g["time"]]
    ends = [g["time"][1] for g in groups if g["time"]]
    repo = {
        "@type": "cr:FileObject", "@id": "repo", "name": "repo",
        "description": "The Hugging Face git repository.",
        "contentUrl": url, "encodingFormat": "git+https", "sha256": "https://github.com/mlcommons/croissant/issues/80",
    }
    filesets = [{
        "@type": "cr:FileSet",
        "@id": f"zarr/{g['group']}",
        "name": g["group"],
        "description": (f"Zarr v3 group {g['group']} of peters_dome.zarr. Variables: {', '.join(g['variables'])}. "
                        f"Dims: {g['dims']}. CRS: {g['crs'] or 'n/a (station or vector geometry, lon/lat)'}. "
                        f"Source: {g['source']}. Licence: {g['license']}. Attribution: {g['attribution']}."),
        "containedIn": {"@id": "repo"},
        "encodingFormat": ZARR_MIME,
        "includes": f"peters_dome.zarr/{g['group']}/**",
    } for g in groups]
    lic = cfg["license"]
    return {
        "@context": CONTEXT,
        "@type": "sc:Dataset",
        "name": cfg["hf_repo"].split("/")[-1],
        "alternateName": cfg["pretty_name"],
        "description": (f"Multimodal data container for the {ev['name']} ({ev['region']}, "
                        f"{ev['origin_time_utc']}): seismic, infrasound, SAR, optical, DEM, glacier, geology, soil and "
                        "meteorology on their native grids, plus the published force-time function and SHALTOP run, "
                        "in one Zarr v3 store with one group per dataset."),
        "conformsTo": [CROISSANT_1_1, GEOCROISSANT_1_0],
        "citeAs": prov["cite_as"],
        "license": f"https://spdx.org/licenses/{lic.upper()}.html" if lic.lower().startswith("cc") else lic,
        "usageInfo": "Each group keeps its source licence; see the per-group licence and attribution in the FileSet descriptions and the dataset card.",
        "url": url,
        "version": cfg["version"].removeprefix("v"),
        "datePublished": prov["date"],
        "creator": [{"@type": "Person", "name": c["name"], "affiliation": c.get("affiliation")} for c in cfg["creators"]],
        "publisher": {"@type": "Organization", "name": "gaia-hazlab", "url": "https://huggingface.co/gaia-hazlab"},
        "keywords": cfg["keywords"],
        "isAccessibleForFree": True,
        "isBasedOn": [
            {"@type": "ScholarlyArticle", "name": "Toney et al. (2026) JGR Earth Surface 131(3), e2025JF008577",
             "identifier": "https://doi.org/10.1029/2025JF008577"},
            {"@type": "Dataset", "name": "USGS data release (Toney et al.)", "identifier": "https://doi.org/10.5066/P13WIS6X"},
            {"@type": "SoftwareSourceCode", "name": "gaia-peters-dome", "codeRepository": "https://github.com/gaia-hazlab/gaia-peters-dome",
             "version": prov["commit"]},
            {"@type": "SoftwareSourceCode", "name": "gaia-cli", "codeRepository": "https://github.com/gaia-hazlab/gaia-cli",
             "version": prov["gaia_cli"]},
        ],
        "spatialCoverage": {"@type": "Place", "name": ev["region"],
                            "geo": {"@type": "GeoShape", "box": f"{min(souths)} {min(wests)} {max(norths)} {max(easts)}"}},
        "temporalCoverage": f"{min(starts)[:10]}/{max(ends)[:10]}",
        "geocr:coordinateReferenceSystem": sorted({g["crs"] for g in groups if g["crs"]} | {"EPSG:4326"}),
        "rai:dataCollection": "Staged from public archives with the gaia-cli load/stage contract (scripts/stage_all.sh); "
                              "rasters on their native grids, only the AOI reprojected; waveforms response-removed.",
        "rai:dataLimitations": "Not co-registered DEM differences; SHALTOP time zero assumed; soil grids mostly NaN over ice; "
                               "Synoptic station data not included. See Known gaps in the card.",
        "rai:dataUseCases": "Event reconstruction, multimodal landslide detection and characterization, teaching. "
                            "Not an operational hazard product.",
        "distribution": [repo, *filesets],
    }


def card(cfg: dict, ev: dict, groups: list[dict], prov: dict, total_bytes: int, nfiles: int) -> str:
    def fmt_bbox(b):
        return "" if not b else f"{b[0]:.2f}, {b[1]:.2f}, {b[2]:.2f}, {b[3]:.2f}"

    def fmt_time(t):
        return "static" if not t else (t[0][:16] if t[0] == t[1] else f"{t[0][:16]} to {t[1][:16]}")

    rows = "\n".join(
        f"| `{g['group']}` | {', '.join(g['variables'][:4])}{' …' if len(g['variables']) > 4 else ''} | "
        f"{' × '.join(f'{k} {v}' for k, v in g['dims'].items())} | {g['crs'] or 'lon/lat geometry'} | "
        f"{fmt_bbox(g['bbox'])} | {fmt_time(g['time'])} | {_mb(g['bytes'])} |"
        for g in groups)
    lic_rows = "\n".join(f"| `{g['group']}` | {g['source']} | {g['license']} | {g['attribution']} |" for g in groups)
    yaml_header = {
        "license": cfg["license"],
        "pretty_name": cfg["pretty_name"],
        "tags": cfg["keywords"],
        "size_categories": ["1K<n<10K"],
        "viewer": False,
    }
    return f"""---
{yaml.safe_dump(yaml_header, sort_keys=False).strip()}
---

# {cfg['pretty_name']}

One Zarr v3 store, `peters_dome.zarr` ({_mb(total_bytes)} MB, {nfiles:,} files), holding every dataset
staged for the **{ev['name']}** ({ev['region']}), studied in Toney et al. (2026), JGR Earth Surface
131(3), e2025JF008577, [doi:10.1029/2025JF008577](https://doi.org/10.1029/2025JF008577). One group per
dataset, on its native grid, following the [gaia-cli](https://github.com/gaia-hazlab/gaia-cli) staging
contract. Built by [gaia-peters-dome](https://github.com/gaia-hazlab/gaia-peters-dome) at commit
`{prov['commit']}`; report with figures at <https://gaia-hazlab.github.io/gaia-peters-dome/>.

## Event

| quantity | value | source |
|---|---|---|
| origin time | {ev['origin_time_utc']} | USGS ComCat, type "landslide", M2.6 |
| precursor | {ev['precursor_time_utc']} | ComCat "other event", M1.4 |
| source area | {ev['source_area_lat']} N, {abs(ev['source_area_lon'])} W | USGS data release |
| volume | {float(ev['volume_m3']) / 1e6:.1f} × 10⁶ m³ | data release / paper |
| peak speed | > {ev['peak_speed_m_s']} m/s | paper abstract |

## Open it

```python
import xarray as xr
from huggingface_hub import snapshot_download

path = snapshot_download("{cfg['hf_repo']}", repo_type="dataset", revision="{cfg['version']}")
tree = xr.open_datatree(f"{{path}}/peters_dome.zarr", engine="zarr")
seis = tree["stations/seismic"].to_dataset().xvec.decode_cf()   # station and vector groups are xvec CF-encoded
```

Download one group only with `allow_patterns=["peters_dome.zarr/zarr.json", "peters_dome.zarr/rasters/zarr.json",
"peters_dome.zarr/rasters/sar/**"]`: the group plus the `zarr.json` of each parent group.

## Groups

Extent is west, south, east, north in degrees. Sizes are on-disk MB.

| group | variables | dims | CRS | extent | time | MB |
|---|---|---|---|---|---|---|
{rows}

## Sources, licences and attribution

The dataset as a whole is released under {cfg['license'].upper()}; each group keeps the terms of its
source, listed here, and anyone redistributing a group must carry its attribution. The Copernicus DEM
group is distributed under its own licence ([PDF](https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/DEM/resources/license/License-COPDEM-30.pdf)),
which must be passed on with it.

| group | source | licence | attribution |
|---|---|---|---|
{lic_rows}

Licences checked on 2026-10-30; the check behind each one is recorded in `config/release.yml` of the
source repository.

## Known gaps

- `dh_post_minus_mosaic` in `datacube` is not co-registered.
- No post-event ArcticDEM strip over the AOI (latest 2020-11-03); the post-event surface is the release SETSM DEM.
- `rasters/topography/arcticdem_strips` are raw strips, not filtered or co-registered: 1.5 % of valid cells
  differ from the ArcticDEM mosaic by more than 100 m and 0.7 % by more than 500 m (maximum 6,560 m against
  a mosaic maximum of 4,600 m). Mask blunders before differencing; median valid coverage per strip is 6 %.
- `model/force_published` and `model/shaltop` are the authors' published results, staged unmodified; no
  inversion or flow simulation is rerun. SHALTOP time zero is assumed equal to the force-inversion zero time.
- Atmospheric profiles stop at 30 hPa (~24 km).
- Not included: Synoptic station precipitation (token-gated; terms not reviewed for redistribution), ice
  thickness grids, permafrost, Daymet, HRRR-Alaska, NPS oblique photos, Sentinel-1 SLC.
- Soil grids are mostly NaN over ice and bare rock, which covers the source area.

## Readiness profile (v0, self-assessed, not reviewed)

Criteria from the GAIA AI-readiness instrument. Evidence level 0 missing, 1 present but incomplete,
2 verifiable by another team.

| criterion | level | evidence |
|---|---|---|
| A1 provenance and rights | 2 | per-group source, licence and attribution above; sha256 manifest; build commit |
| A2 physical comparability | 1 | native grids and CRS kept; waveforms response-removed with stated units; DEM vertical datums differ (EGM2008 vs ellipsoid) |
| A3 quality and gaps | 1 | known gaps listed; no per-sample quality flags beyond source masks (S2 SCL, Landsat QA) |
| A4 structure and metadata | 2 | Zarr v3, CF time bounds, xvec geometries; Croissant 1.1 + GeoCroissant record |
| B1 diversity | 0 | one event; not a training corpus on its own |
| B7 independent partitioning | n/a | no splits; a single-event container |

## Provenance

| item | value |
|---|---|
| builder | gaia-peters-dome `{prov['commit']}` |
| gaia-cli | `{prov['gaia_cli']}` |
| metadata built | {prov['date']} |
| manifest | `manifest.json`: sha256 of every file in the store |

## Citation

{prov['cite_as']}

Please also cite Toney et al. (2026), doi:10.1029/2025JF008577, the USGS data release
doi:10.5066/P13WIS6X, and the attributions of the groups you use.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=common.STORE)
    ap.add_argument("--out", default="release")
    a = ap.parse_args()

    cfg = yaml.safe_load((common.ROOT / "config/release.yml").read_text())
    ev = common.event()["event"]
    groups = describe(a.store, cfg["sources"])
    files = manifest(a.store)
    total = sum(f["bytes"] for f in files)
    url = f"https://huggingface.co/datasets/{cfg['hf_repo']}"
    names = ", ".join(c["name"] for c in cfg["creators"])
    prov = {
        "commit": _git("rev-parse", "--short", "HEAD") + ("-dirty" if _git("status", "--porcelain", "--", "src", "scripts") else ""),
        "gaia_cli": "896b628",
        "date": dt.date.today().isoformat(),
        "cite_as": f"{names} ({dt.date.today().year}). {cfg['pretty_name']} ({cfg['version']}). "
                   f"GAIA HazLab, Hugging Face. {url}",
    }
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    (out / "README.md").write_text(card(cfg, ev, groups, prov, total, len(files)))
    (out / "croissant.json").write_text(json.dumps(croissant(cfg, ev, groups, url, prov), indent=1, ensure_ascii=False) + "\n")
    (out / "manifest.json").write_text(json.dumps({"provenance": prov, "groups": groups, "files": files}, indent=1, default=str) + "\n")
    print(f"{len(groups)} groups, {len(files)} files, {_mb(total)} MB -> {out}/")
    for g in groups:
        print(f"  {g['group']:38s} {g['crs'] or '-':12s} {_mb(g['bytes']):>9s} MB  {g['license'][:40]}")


if __name__ == "__main__":
    sys.exit(main())
