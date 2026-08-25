"""Shared processing helpers."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Iterable


COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


@contextmanager
def repository_lock(root: Path, exclusive: bool):
    directory = root / ".locks"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "processing.lock").open("a+b") as handle:
        operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        fcntl.flock(handle, operation)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


@contextmanager
def destination_lock(destination: Path):
    destination = destination.expanduser().absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    path = destination.parent / f".{destination.name}.mfass.lock"
    with path.open("a+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def unique_directory(parent: Path, prefix: str) -> Path:
    return Path(tempfile.mkdtemp(dir=parent, prefix=prefix))


@contextmanager
def isolated_fasta(path: Path, **kwargs):
    from pyfaidx import Fasta

    directory = unique_directory(path.parent, f".{path.name}.index-")
    genome = None
    try:
        genome = Fasta(
            path,
            indexname=str(directory / "index.fai"),
            rebuild=True,
            **kwargs,
        )
        yield genome
    finally:
        if genome is not None:
            genome.close()
        shutil.rmtree(directory, ignore_errors=True)


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
    removals: Iterable[Path] = (),
) -> None:
    """Replace/remove files and restore every destination on failure."""
    items = list(replacements)
    removal_items = list(removals)
    destinations = [
        destination for _source, destination in items
    ] + removal_items
    if len(destinations) != len(set(destinations)):
        raise RuntimeError("transaction destinations must be unique")
    if backup_directory.exists() or backup_directory.is_symlink():
        raise RuntimeError(
            f"backup path already exists: {backup_directory}"
        )
    backup_directory.mkdir()
    for source, _destination in items:
        if not source.is_file() or source.is_symlink():
            raise RuntimeError(f"replacement must be a regular file: {source}")
    previous = []
    for index, destination in enumerate(destinations):
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

    applied = []
    try:
        for index, (source, destination) in enumerate(items):
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, destination)
            applied.append(previous[index])
        for offset, destination in enumerate(
            removal_items, start=len(items)
        ):
            destination.unlink(missing_ok=True)
            applied.append(previous[offset])
        if verify is not None:
            verify()
    except BaseException as error:
        rollback_errors = []
        for destination, backup, existed in reversed(applied):
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
