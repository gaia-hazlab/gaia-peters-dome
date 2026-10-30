# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface_hub>=0.34", "pyyaml"]
# ///
"""Publish data/peters_dome.zarr and release/ to the Hugging Face repo in config/release.yml.

Run scripts/hf_metadata.py first. Steps, each idempotent:
1. create the dataset repo (public) if missing;
2. upload data/peters_dome.zarr as peters_dome.zarr/ with the resumable large-folder uploader;
3. upload release/{README.md, croissant.json, manifest.json};
4. tag the resulting commit with the version (refuses to move an existing tag);
5. add the dataset to the gaia-hazlab collection named in the config (created if missing).

Before step 2 the manifest is checked against the store on disk, so the card always
describes the bytes that are uploaded. Needs an HF token with write access to gaia-hazlab.

    uv run scripts/hf_publish.py            # dry run: checks and prints the plan
    uv run scripts/hf_publish.py --yes      # publish
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml
from huggingface_hub import (
    HfApi,
    add_collection_item,
    create_collection,
    list_collections,
)

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "data" / "peters_dome.zarr"
RELEASE = ROOT / "release"


def check_manifest() -> dict:
    m = json.loads((RELEASE / "manifest.json").read_text())
    listed = {f["path"]: f for f in m["files"]}
    on_disk = {str(p.relative_to(STORE.parent)): p for p in STORE.rglob("*") if p.is_file()}
    if set(listed) != set(on_disk):
        extra, missing = set(on_disk) - set(listed), set(listed) - set(on_disk)
        sys.exit(f"manifest out of date: {len(extra)} new and {len(missing)} missing files; rerun hf_metadata.py")
    for rel, f in listed.items():
        if on_disk[rel].stat().st_size != f["bytes"]:
            sys.exit(f"manifest out of date: size of {rel} changed; rerun hf_metadata.py")
    # full hash of a sample: every 50th file plus all root metadata
    for rel in sorted(listed)[::50] + [r for r in listed if r.endswith("zarr.json")][:20]:
        h = hashlib.sha256(on_disk[rel].read_bytes()).hexdigest()
        if h != listed[rel]["sha256"]:
            sys.exit(f"manifest out of date: sha256 of {rel} changed; rerun hf_metadata.py")
    if m["provenance"]["commit"].endswith("-dirty"):
        sys.exit("metadata was built from uncommitted code; commit, then rerun hf_metadata.py")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true", help="publish (default: dry run)")
    a = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "config/release.yml").read_text())
    repo, version = cfg["hf_repo"], cfg["version"]
    m = check_manifest()
    total = sum(f["bytes"] for f in m["files"])
    print(f"{repo} {version}: {len(m['files'])} files, {total / 1e6:,.1f} MB, built from {m['provenance']['commit']}")
    if not a.yes:
        print("dry run; pass --yes to publish")
        return

    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=False, exist_ok=True)
    refs = api.list_repo_refs(repo, repo_type="dataset")
    if any(t.name == version for t in refs.tags):
        sys.exit(f"tag {version} already exists on {repo}; bump version in config/release.yml")

    api.upload_large_folder(repo_id=repo, repo_type="dataset", folder_path=str(STORE.parent),
                            allow_patterns=["peters_dome.zarr/**"])
    info = api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=str(RELEASE),
                             allow_patterns=["README.md", "croissant.json", "manifest.json"],
                             commit_message=f"{version}: card, Croissant/GeoCroissant, manifest ({m['provenance']['commit']})")
    api.create_tag(repo, repo_type="dataset", tag=version, revision=info.oid,
                   tag_message=f"gaia-peters-dome {m['provenance']['commit']}")
    print(f"tagged {version} at {info.oid}")

    owner = repo.split("/")[0]
    coll = next((c for c in list_collections(owner=owner, limit=100) if c.title == cfg["collection_title"]), None)
    if coll is None:
        coll = create_collection(title=cfg["collection_title"], namespace=owner, private=False,
                                 description="Datasets published by the GAIA hazards lab (University of Washington).")
    add_collection_item(coll.slug, item_id=repo, item_type="dataset", exists_ok=True)
    print(f"collection: https://huggingface.co/collections/{coll.slug}")


if __name__ == "__main__":
    main()
