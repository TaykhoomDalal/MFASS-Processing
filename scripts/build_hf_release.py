#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELEASE_FILES = {"README.md", "mfass.parquet"}
COMMIT_TOKEN = "{{PROCESSING_COMMIT}}"
MANIFEST_HASH_TOKEN = "{{PROCESSING_MANIFEST_SHA256}}"


def copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(target)


def write_text(text: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(target)


def remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def require_release_allowlist(directory: Path) -> None:
    entries = {
        path.name
        for path in directory.iterdir()
        if path.name != ".git"
    }
    if entries != RELEASE_FILES:
        raise RuntimeError(
            f"{directory}: expected only {sorted(RELEASE_FILES)}, "
            f"found {sorted(entries)}"
        )
    for name in RELEASE_FILES:
        path = directory / name
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"release artifact must be a file: {path}")


def processing_provenance() -> tuple[str, str]:
    status = subprocess.check_output(
        [
            "git",
            "-C",
            str(ROOT),
            "status",
            "--porcelain",
            "--untracked-files=normal",
        ],
        text=True,
    )
    if status:
        raise RuntimeError(
            "commit all processing changes before building the HF release"
        )
    revision = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    committed_manifest = subprocess.check_output(
        ["git", "-C", str(ROOT), "show", f"{revision}:manifest.json"]
    )
    working_manifest = (ROOT / "manifest.json").read_bytes()
    if working_manifest != committed_manifest:
        raise RuntimeError("working manifest differs from the processing commit")
    digest = hashlib.sha256(committed_manifest).hexdigest()
    return revision, digest


def render_card(revision: str, manifest_hash: str) -> str:
    template = (ROOT / "MFASS/HF_DATASET_CARD.md").read_text(
        encoding="utf-8"
    )
    rendered = template.replace(COMMIT_TOKEN, revision).replace(
        MANIFEST_HASH_TOKEN, manifest_hash
    )
    if COMMIT_TOKEN in rendered or MANIFEST_HASH_TOKEN in rendered:
        raise RuntimeError("dataset card provenance placeholders remain")
    return rendered


def build_release(
    output: Path, revision: str, manifest_hash: str
) -> None:
    output = output.expanduser().absolute()
    if output in {ROOT, ROOT / "MFASS"}:
        raise RuntimeError("refusing to replace a processing directory")
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        raise RuntimeError(f"release output must be a real directory: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.mfass-staging"
    remove(staging)
    staging.mkdir()
    try:
        copy(ROOT / "MFASS/mfass.parquet", staging / "mfass.parquet")
        write_text(
            render_card(revision, manifest_hash),
            staging / "README.md",
        )
        require_release_allowlist(staging)

        output.mkdir(parents=True, exist_ok=True)
        for path in output.iterdir():
            if path.name != ".git":
                remove(path)
        for name in sorted(RELEASE_FILES):
            os.replace(staging / name, output / name)
        require_release_allowlist(output)
    finally:
        remove(staging)
    print(f"built exact compact Hugging Face release under {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    revision, manifest_hash = processing_provenance()
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/verify_outputs.py")],
        check=True,
    )
    build_release(args.output, revision, manifest_hash)


if __name__ == "__main__":
    main()
