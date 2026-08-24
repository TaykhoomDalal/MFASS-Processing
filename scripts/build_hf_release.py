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
GIT_METADATA = {".git", ".gitattributes"}
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


def require_release_allowlist(
    directory: Path, metadata: set[str]
) -> None:
    entries = {path.name for path in directory.iterdir()}
    expected = RELEASE_FILES | metadata
    if entries != expected:
        raise RuntimeError(
            f"{directory}: expected only {sorted(expected)}, "
            f"found {sorted(entries)}"
        )
    for name in RELEASE_FILES:
        path = directory / name
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"release artifact must be a file: {path}")
    require_git_metadata(directory, metadata)


def require_git_metadata(directory: Path, metadata: set[str]) -> None:
    attributes = directory / ".gitattributes"
    if ".gitattributes" in metadata and (
        not attributes.is_file() or attributes.is_symlink()
    ):
        raise RuntimeError(
            f"Git attributes must be a regular file: {attributes}"
        )
    git_path = directory / ".git"
    if ".git" in metadata and (
        git_path.is_symlink()
        or not (git_path.is_file() or git_path.is_dir())
    ):
        raise RuntimeError(f"invalid Git metadata: {git_path}")


def reject_processing_overlap(output: Path) -> None:
    root = ROOT.resolve()
    resolved = output.resolve()
    if (
        resolved == root
        or root in resolved.parents
        or resolved in root.parents
    ):
        raise RuntimeError(
            "release output must not be the processing repository, "
            "an ancestor, or a descendant"
        )


def require_clean_git_root(output: Path) -> None:
    try:
        inside = subprocess.check_output(
            [
                "git",
                "-C",
                str(output),
                "rev-parse",
                "--is-inside-work-tree",
            ],
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
        top_level = Path(
            subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(output),
                    "rev-parse",
                    "--show-toplevel",
                ],
                text=True,
                stderr=subprocess.PIPE,
            ).strip()
        ).resolve()
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"existing release target is not a Git worktree: {output}"
        ) from error
    if inside != "true" or top_level != output:
        raise RuntimeError(
            f"release target must be the Git worktree root: {output}"
        )
    status = subprocess.check_output(
        [
            "git",
            "-C",
            str(output),
            "status",
            "--porcelain",
            "--untracked-files=all",
        ],
        text=True,
    )
    if status:
        raise RuntimeError(
            f"existing Hugging Face Git repository is dirty: {output}"
        )


def inspect_destination(output: Path) -> tuple[set[str], bytes | None]:
    if not output.exists():
        return set(), None
    if not output.is_dir():
        raise RuntimeError(
            f"release output must be a real directory: {output}"
        )

    entries = {path.name for path in output.iterdir()}
    if ".git" not in entries:
        if entries:
            raise RuntimeError(
                "existing non-Git release output must be empty and "
                "disposable"
            )
        return set(), None

    unexpected = entries - RELEASE_FILES - GIT_METADATA
    if unexpected:
        raise RuntimeError(
            f"{output}: unsupported existing entries: {sorted(unexpected)}"
        )
    require_clean_git_root(output)
    metadata = entries & GIT_METADATA
    require_git_metadata(output, metadata)
    for name in RELEASE_FILES & entries:
        path = output / name
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(
                f"existing release artifact must be a file: {path}"
            )
    attributes = (
        (output / ".gitattributes").read_bytes()
        if ".gitattributes" in metadata
        else None
    )
    return metadata, attributes


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
    requested_output = output.expanduser().absolute()
    if requested_output.is_symlink():
        raise RuntimeError(f"release output must be a real directory: {output}")
    output = requested_output.resolve()
    reject_processing_overlap(output)
    metadata, attributes = inspect_destination(output)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / (
        f".{output.name}.mfass-staging-{os.getpid()}"
    )
    if staging.exists() or staging.is_symlink():
        raise RuntimeError(f"staging path already exists: {staging}")
    staging.mkdir()
    try:
        copy(ROOT / "MFASS/mfass.parquet", staging / "mfass.parquet")
        write_text(
            render_card(revision, manifest_hash),
            staging / "README.md",
        )
        require_release_allowlist(staging, set())

        if ".git" in metadata:
            for name in sorted(RELEASE_FILES):
                os.replace(staging / name, output / name)
            staging.rmdir()
        else:
            if output.exists():
                output.rmdir()
            os.replace(staging, output)
        require_release_allowlist(output, metadata)
        if attributes is not None and (
            output / ".gitattributes"
        ).read_bytes() != attributes:
            raise RuntimeError(".gitattributes changed during release build")
    finally:
        if staging.is_symlink():
            staging.unlink()
        elif staging.exists():
            shutil.rmtree(staging)
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
