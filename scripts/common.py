"""Shared processing helpers."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Iterable


COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def reverse_complement(sequence: str) -> str:
    return sequence.translate(COMPLEMENT)[::-1]


def align_sequences(source: str, target: str) -> dict[str, Any]:
    mapping: list[int | None] = [None] * len(source)
    substitutions = insertions = deletions = 0
    for operation, a0, a1, b0, b1 in SequenceMatcher(
        None, source, target, autojunk=False
    ).get_opcodes():
        source_length, target_length = a1 - a0, b1 - b0
        if operation == "equal":
            for offset in range(source_length):
                mapping[a0 + offset] = b0 + offset
        elif operation == "replace":
            if source_length != target_length:
                raise RuntimeError(
                    "unequal replacement requires an explicit alignment rule"
                )
            substitutions += source_length
            for offset in range(source_length):
                mapping[a0 + offset] = b0 + offset
        elif operation == "insert":
            insertions += target_length
        elif operation == "delete":
            deletions += source_length
        else:
            raise RuntimeError(f"unknown alignment operation: {operation}")
    if substitutions == insertions == deletions == 0:
        status = "exact"
    elif substitutions == 1 and insertions == deletions == 0:
        status = "one_substitution"
    elif insertions == 1 and substitutions == deletions == 0:
        status = "one_hg38_insertion"
    else:
        status = "nontrivial"
    return {
        "mapping": mapping,
        "status": status,
        "substitutions": substitutions,
        "insertions": insertions,
        "deletions": deletions,
        "edit_distance": substitutions + insertions + deletions,
    }


def sha256(path: Path) -> str:
    return file_digest(path, "sha256")


def file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_parquet(frame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def replace_files_transactionally(
    replacements: Iterable[tuple[Path, Path]],
    backup_directory: Path,
    verify: Callable[[], None] | None = None,
) -> None:
    """Replace a set of files and restore every destination on failure."""
    items = list(replacements)
    if backup_directory.exists() or backup_directory.is_symlink():
        raise RuntimeError(
            f"backup path already exists: {backup_directory}"
        )
    backup_directory.mkdir()
    previous = []
    for index, (source, destination) in enumerate(items):
        if not source.is_file() or source.is_symlink():
            raise RuntimeError(f"replacement must be a regular file: {source}")
        existed = destination.exists() or destination.is_symlink()
        if existed and (
            not destination.is_file() or destination.is_symlink()
        ):
            raise RuntimeError(
                f"replacement destination must be a regular file: "
                f"{destination}"
            )
        backup = backup_directory / f"{index}-{destination.name}"
        if existed:
            shutil.copy2(destination, backup)
        previous.append((destination, backup, existed))

    installed = 0
    try:
        for source, destination in items:
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, destination)
            installed += 1
        if verify is not None:
            verify()
    except BaseException as error:
        rollback_errors = []
        for destination, backup, existed in reversed(previous[:installed]):
            try:
                if existed:
                    os.replace(backup, destination)
                else:
                    destination.unlink(missing_ok=True)
            except OSError as rollback_error:
                rollback_errors.append(
                    f"{destination}: {rollback_error}"
                )
        if rollback_errors:
            raise RuntimeError(
                "replacement failed and rollback was incomplete: "
                + "; ".join(rollback_errors)
            ) from error
        raise
