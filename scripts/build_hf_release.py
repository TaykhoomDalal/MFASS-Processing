#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.common import replace_files_transactionally

RELEASE_FILES = {"README.md", "mfass.parquet"}
GIT_METADATA = {".git", ".gitattributes"}
COMPACT_OUTPUT_PATH = "MFASS/mfass.parquet"
LEGACY_PROVENANCE_TOKENS = {
    "{{PROCESSING_COMMIT}}",
    "{{PROCESSING_MANIFEST_SHA256}}",
}
STABLE_CARD_LINKS = {
    "https://github.com/TaykhoomDalal/MFASS-Processing/tree/main",
    "https://github.com/TaykhoomDalal/MFASS-Processing/blob/main/README.md",
    "https://github.com/TaykhoomDalal/MFASS-Processing/blob/main/manifest.json",
    "https://github.com/TaykhoomDalal/MFASS-Processing/blob/main/NOTICE.md",
}
COMMIT_SPECIFIC_PROCESSING_URL = re.compile(
    r"https://(?:"
    r"github\.com/TaykhoomDalal/MFASS-Processing/"
    r"(?:commit|blob|tree)/"
    r"|raw\.githubusercontent\.com/TaykhoomDalal/"
    r"MFASS-Processing/"
    r")[0-9a-f]{40}(?:/|\b)"
)


def verify_compact_artifact(
    path: Path, expected: dict, context: str
) -> None:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"{context} must be a regular file: {path}")
    if path.stat().st_size != expected["bytes"]:
        raise RuntimeError(
            f"{context} byte size changed: expected {expected['bytes']}, "
            f"got {path.stat().st_size}"
        )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    observed = digest.hexdigest()
    if observed != expected["sha256"]:
        raise RuntimeError(
            f"{context} sha256 changed: expected {expected['sha256']}, "
            f"got {observed}"
        )


def copy_verified_compact(
    source: Path, target: Path, expected: dict
) -> None:
    verify_compact_artifact(
        source, expected, "processing compact output"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    digest = hashlib.sha256()
    copied = 0
    try:
        with source.open("rb") as source_handle:
            with temporary.open("wb") as target_handle:
                for block in iter(
                    lambda: source_handle.read(1 << 20), b""
                ):
                    target_handle.write(block)
                    digest.update(block)
                    copied += len(block)
        if (
            copied != expected["bytes"]
            or digest.hexdigest() != expected["sha256"]
        ):
            raise RuntimeError(
                "processing compact output changed while being copied"
            )
        temporary.replace(target)
        verify_compact_artifact(
            target, expected, "staged compact output"
        )
    finally:
        temporary.unlink(missing_ok=True)


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


def require_clean_processing_tree() -> None:
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


def load_compact_output_authority() -> dict:
    revision = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    committed_manifest = subprocess.check_output(
        ["git", "-C", str(ROOT), "show", f"{revision}:manifest.json"]
    )
    working_manifest = (ROOT / "manifest.json").read_bytes()
    if working_manifest != committed_manifest:
        raise RuntimeError(
            "working manifest differs from the processing commit"
        )
    manifest = json.loads(committed_manifest)
    try:
        record = manifest["outputs"][COMPACT_OUTPUT_PATH]
    except (KeyError, TypeError) as error:
        raise RuntimeError(
            f"committed manifest has no {COMPACT_OUTPUT_PATH} authority"
        ) from error
    expected = {
        "bytes": record.get("bytes"),
        "sha256": record.get("sha256"),
    }
    if (
        not isinstance(expected["bytes"], int)
        or expected["bytes"] < 0
        or not isinstance(expected["sha256"], str)
        or len(expected["sha256"]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in expected["sha256"]
        )
    ):
        raise RuntimeError(
            "committed manifest has invalid compact output authority"
        )
    return expected


def render_card() -> str:
    rendered = (ROOT / "MFASS/HF_DATASET_CARD.md").read_text(
        encoding="utf-8"
    )
    remaining = {
        token for token in LEGACY_PROVENANCE_TOKENS
        if token in rendered
    }
    if remaining:
        raise RuntimeError(
            "dataset card contains legacy commit-specific placeholders: "
            f"{sorted(remaining)}"
        )
    commit_url = COMMIT_SPECIFIC_PROCESSING_URL.search(rendered)
    if commit_url:
        raise RuntimeError(
            "dataset card contains a commit-specific processing URL: "
            f"{commit_url.group(0)}"
        )
    if "Processing commit:" in rendered or "Root manifest SHA-256:" in rendered:
        raise RuntimeError(
            "dataset card contains commit-specific processing provenance"
        )
    missing_links = {
        link for link in STABLE_CARD_LINKS
        if link not in rendered
    }
    if missing_links:
        raise RuntimeError(
            "dataset card is missing stable processing links: "
            f"{sorted(missing_links)}"
        )
    return rendered


def build_release(output: Path, expected_compact: dict) -> None:
    requested_output = output.expanduser().absolute()
    if requested_output.is_symlink():
        raise RuntimeError(f"release output must be a real directory: {output}")
    output = requested_output.resolve()
    reject_processing_overlap(output)
    output_existed = output.exists()
    metadata, attributes = inspect_destination(output)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / (
        f".{output.name}.mfass-staging-{os.getpid()}"
    )
    if staging.exists() or staging.is_symlink():
        raise RuntimeError(f"staging path already exists: {staging}")
    staging.mkdir()
    try:
        copy_verified_compact(
            ROOT / COMPACT_OUTPUT_PATH,
            staging / "mfass.parquet",
            expected_compact,
        )
        write_text(
            render_card(),
            staging / "README.md",
        )
        require_release_allowlist(staging, set())
        verify_compact_artifact(
            staging / "mfass.parquet",
            expected_compact,
            "staged compact output",
        )

        def verify_update() -> None:
            require_release_allowlist(output, metadata)
            verify_compact_artifact(
                output / "mfass.parquet",
                expected_compact,
                "installed compact output",
            )
            if attributes is not None and (
                output / ".gitattributes"
            ).read_bytes() != attributes:
                raise RuntimeError(
                    ".gitattributes changed during release build"
                )

        if ".git" in metadata:
            replace_files_transactionally(
                [
                    (staging / name, output / name)
                    for name in sorted(RELEASE_FILES)
                ],
                staging / ".rollback",
                verify_update,
            )
        else:
            if not output_existed:
                output.mkdir()
            try:
                replace_files_transactionally(
                    [
                        (staging / name, output / name)
                        for name in sorted(RELEASE_FILES)
                    ],
                    staging / ".rollback",
                    verify_update,
                )
            except BaseException:
                if (
                    not output_existed
                    and output.is_dir()
                    and not output.is_symlink()
                    and not any(output.iterdir())
                ):
                    output.rmdir()
                raise
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
    require_clean_processing_tree()
    expected_compact = load_compact_output_authority()
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/verify_outputs.py")],
        check=True,
    )
    build_release(args.output, expected_compact)


if __name__ == "__main__":
    main()
