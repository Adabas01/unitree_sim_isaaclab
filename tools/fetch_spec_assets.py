"""Fetch the object assets of G1-D task specs from ponimatkin's chiral (formerly isaaclab_biman_sim) asset set.

Specs name BEHAVIOR-1K objects (`asset: {source: behavior1k, category, model}`). chiral maps each to a SimReady /
Palatial / ForgeCAD stand-in: `assets/objects/registry.yaml` gives a single-rigid-body wrapper USD kept in its git
(`assets/objects/<category>__<model>/model.usda`, scaled per axis to the spec's box at spawn) that references the
original asset in a public Hugging Face dataset folder (registry `source`). This tool downloads those folders to
`assets/third_party/hf/<owner>__<dataset>/<folder>/` of the asset root, the layout the wrappers reference (as
chiral's tools/fetch_assets.py), and checks that every wrapper's references resolve.

The asset root lives outside git ($G1D_CHIRAL_ASSETS, default ~/adam_unitree/sim_assets/chiral_554fb5e) and is
made once from a chiral clone at a fixed commit:

    git -C ~/adam_unitree/isaaclab_biman_sim archive 554fb5e assets/objects assets/MANIFEST.csv biman/tasks/specs \
        | tar -x -C ~/adam_unitree/sim_assets/chiral_554fb5e

Usage (sim environment; specs under tasks/g1d_tasks/spec_tasks/specs/):

    python tools/fetch_spec_assets.py put_fruit_on_plate collect_eggs
    python tools/fetch_spec_assets.py --all
"""

import argparse
import os
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

import yaml

SPEC_DIR = Path(__file__).resolve().parents[1] / "tasks" / "g1d_tasks" / "spec_tasks" / "specs"
DEFAULT_ROOT = "~/adam_unitree/sim_assets/chiral_554fb5e"
IGNORE_PATTERNS = ["*.psd", "*.exr", "*8k*", "*.zip"]  # as chiral tools/fetch_assets.py


def asset_root() -> Path:
    return Path(os.environ.get("G1D_CHIRAL_ASSETS", DEFAULT_ROOT)).expanduser()


def hf_folder(source: str):
    """(repo id, folder) of a https://huggingface.co/datasets/<owner>/<name>/tree/main/<folder> URL, else None."""
    url = urlparse(source)
    parts = url.path.strip("/").split("/")
    if url.netloc != "huggingface.co" or parts[:1] != ["datasets"] or parts[3:5] != ["tree", "main"]:
        return None
    return f"{parts[1]}/{parts[2]}", unquote("/".join(parts[5:]))


def download_folder(repo: str, folder: str, local: Path) -> None:
    """Every file of one folder of a public dataset repo, skipping IGNORE_PATTERNS and files already there.
    (Not snapshot_download: it lists the whole repo, and with this env's tqdm its thread_map fails.)"""
    from concurrent.futures import ThreadPoolExecutor
    from fnmatch import fnmatch

    from huggingface_hub import HfApi, hf_hub_download

    files = [f.path for f in HfApi().list_repo_tree(repo, path_in_repo=folder, repo_type="dataset", recursive=True)
             if hasattr(f, "size") and not any(fnmatch(f.path.rsplit("/", 1)[-1].lower(), p) for p in IGNORE_PATTERNS)
             and not ((local / f.path).is_file() and (local / f.path).stat().st_size == f.size)]
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda path: hf_hub_download(repo, path, repo_type="dataset", local_dir=local), files))


def references(usd: Path) -> list[Path]:
    """Files an ASCII USD layer references (@...@), resolved against its folder."""
    return [(usd.parent / ref).resolve() for ref in re.findall(r"@([^@]+)@", usd.read_text(errors="replace"))
            if not ref.startswith(("http", "omniverse:"))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("specs", nargs="*", help="task names (file stems under spec_tasks/specs/)")
    parser.add_argument("--all", action="store_true", help="every spec under spec_tasks/specs/")
    parser.add_argument("--dry-run", action="store_true", help="only list what would be downloaded")
    args = parser.parse_args()
    root = asset_root()
    registry_file = root / "assets" / "objects" / "registry.yaml"
    if not registry_file.is_file():
        sys.exit(f"No chiral asset root at {root} (see this file's docstring to create it).")
    registry = yaml.safe_load(registry_file.read_text())
    files = sorted(SPEC_DIR.glob("*/*.yaml")) if args.all else []
    for name in args.specs:
        match = sorted(SPEC_DIR.glob(f"*/{name}.yaml"))
        if len(match) != 1:
            sys.exit(f"No unique spec {name!r} under {SPEC_DIR}")
        files += match
    if not files:
        parser.error("name specs or use --all")

    wanted, unavailable = {}, {}
    for path in files:
        for o in yaml.safe_load(path.read_text())["objects"]:
            a = o["asset"]
            if a["source"] != "behavior1k":
                continue
            key = f"{a['category']}/{a['model']}"
            entry = registry.get(key)
            folder = hf_folder(entry["source"]) if entry and entry.get("source") else None
            if entry and entry.get("articulated"):
                unavailable.setdefault(key, set()).add(f"{path.stem} (articulated, not supported here)")
            elif folder is None:
                why = "not in the registry" if not entry else f"source not on Hugging Face: {entry.get('source')}"
                unavailable.setdefault(key, set()).add(f"{path.stem} ({why})")
            else:
                wanted.setdefault(key, (entry, folder, set()))[2].add(path.stem)

    failed = []
    for key, (entry, (repo, folder), specs) in sorted(wanted.items()):
        local = root / "assets" / "third_party" / "hf" / repo.replace("/", "__")
        print(f"{key:28s} {repo}:{folder}  ({', '.join(sorted(specs))})", flush=True)
        if not args.dry_run:
            download_folder(repo, folder, local)
        wrapper = root / entry["usd"]
        missing = [p for p in references(wrapper) if not p.exists()] if wrapper.is_file() else [wrapper]
        if missing and not args.dry_run:
            failed.append(key)
            print(f"  MISSING after download: {[str(p) for p in missing]}")
    for key, specs in sorted(unavailable.items()):
        print(f"{key:28s} not fetched, the spec tasks use a stand-in of its box: {', '.join(sorted(specs))}")
    if failed:
        sys.exit(f"References still missing for {failed}")


if __name__ == "__main__":
    main()
