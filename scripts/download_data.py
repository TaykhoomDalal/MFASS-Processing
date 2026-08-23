#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.common import file_digest


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifest.json"
DOWNLOAD_ATTEMPTS = 3
DOWNLOAD_TIMEOUT_SECONDS = 120
USER_AGENT = "MFASS-Processing/1.0"


def load_manifest(path: Path | None = None) -> dict[str, Any]:
    manifest_path = MANIFEST if path is None else path
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("manifest_version") != 1:
        raise RuntimeError("unsupported manifest version")
    if not isinstance(payload.get("sources"), dict):
        raise RuntimeError("manifest has no source records")
    return payload


def source_records() -> dict[str, dict[str, Any]]:
    return load_manifest()["sources"]


def materialized_sources() -> dict[str, dict[str, Any]]:
    return {
        record["local_path"]: record["materialized"]
        for record in source_records().values()
    }


def verify(path: Path, expected: dict[str, Any]) -> None:
    expected_bytes = expected.get("bytes")
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        raise RuntimeError(
            f"{path}: expected {expected_bytes} bytes, "
            f"got {path.stat().st_size}"
        )
    for algorithm in ("sha256", "md5"):
        wanted = expected.get(algorithm)
        if wanted is None:
            continue
        observed = file_digest(path, algorithm)
        if observed != wanted:
            raise RuntimeError(
                f"{path}: expected {algorithm} {wanted}, got {observed}"
            )


def fetch(url: str, temporary: Path, expected: dict[str, Any]) -> None:
    errors = []
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        temporary.unlink(missing_ok=True)
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": USER_AGENT}
            )
            with urllib.request.urlopen(
                request, timeout=DOWNLOAD_TIMEOUT_SECONDS
            ) as source:
                content_length = source.headers.get("Content-Length")
                if (
                    content_length is not None
                    and expected.get("bytes") is not None
                    and int(content_length) != expected["bytes"]
                ):
                    raise RuntimeError(
                        f"{url}: expected Content-Length "
                        f"{expected['bytes']}, got {content_length}"
                    )
                with temporary.open("wb") as target:
                    shutil.copyfileobj(source, target)
            verify(temporary, expected)
            return
        except (
            OSError,
            RuntimeError,
            TimeoutError,
            urllib.error.URLError,
        ) as error:
            temporary.unlink(missing_ok=True)
            errors.append(f"attempt {attempt}: {error}")
            if attempt < DOWNLOAD_ATTEMPTS:
                delay = 2 ** (attempt - 1)
                print(
                    f"download failed; retrying in {delay}s: {url}",
                    file=sys.stderr,
                )
                time.sleep(delay)
    raise RuntimeError(
        f"failed to acquire {url} after {DOWNLOAD_ATTEMPTS} attempts: "
        + "; ".join(errors)
    )


def download(record: dict[str, Any], target: Path) -> None:
    if target.is_file():
        verify(target, record["materialized"])
        return
    if target.exists() or target.is_symlink():
        raise RuntimeError(f"refusing to overwrite unexpected path: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.download")
    fetch(record["url"], temporary, record["download"])
    verify(temporary, record["materialized"])
    os.replace(temporary, target)


def link(
    source: Path, target: Path, expected: dict[str, Any]
) -> None:
    source = source.resolve(strict=True)
    verify(source, expected)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() and target.resolve() == source:
        verify(target, expected)
        return
    if target.is_file():
        verify(target, expected)
        return
    if target.exists() or target.is_symlink():
        raise RuntimeError(f"refusing to overwrite unexpected path: {target}")
    temporary = target.with_name(f".{target.name}.link")
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(source)
    os.replace(temporary, target)


def prepare_compressed(
    record: dict[str, Any], target: Path, reuse: Path | None
) -> None:
    if reuse is not None:
        link(reuse, target, record["materialized"])
        return
    if target.is_file():
        verify(target, record["materialized"])
        return
    if target.exists() or target.is_symlink():
        raise RuntimeError(f"refusing to overwrite unexpected path: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    compressed = target.with_name(f".{target.name}.download.gz")
    materialized = target.with_name(f".{target.name}.materializing")
    try:
        fetch(record["url"], compressed, record["download"])
        with gzip.open(compressed, "rb") as source:
            with materialized.open("wb") as destination:
                shutil.copyfileobj(source, destination)
        verify(materialized, record["materialized"])
        os.replace(materialized, target)
    finally:
        compressed.unlink(missing_ok=True)
        materialized.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--reuse-source", type=Path)
    parser.add_argument("--reuse-benchmark-root", type=Path)
    parser.add_argument("--reuse-fasta", type=Path)
    args = parser.parse_args()

    records = source_records()
    reuse_paths = {
        "mfass_measurements": args.reuse_source,
        "mfass_labels": (
            args.reuse_benchmark_root / "data/mfass_labels.csv"
            if args.reuse_benchmark_root
            else None
        ),
        "pangolin_scores": (
            args.reuse_benchmark_root / "results/scores_pangolin.csv"
            if args.reuse_benchmark_root
            else None
        ),
        "spliceai_scores": (
            args.reuse_benchmark_root / "results/scores_spliceai.csv"
            if args.reuse_benchmark_root
            else None
        ),
        "splicetransformer_scores": (
            args.reuse_benchmark_root / "results/scores_splicetx.csv"
            if args.reuse_benchmark_root
            else None
        ),
        "mmsplice_scores": (
            args.reuse_benchmark_root / "results/scores_mmsplice.csv"
            if args.reuse_benchmark_root
            else None
        ),
    }
    for name, record in records.items():
        target = args.data_root / record["local_path"]
        if record.get("compression"):
            prepare_compressed(record, target, args.reuse_fasta)
        elif reuse_paths.get(name) is not None:
            link(reuse_paths[name], target, record["materialized"])
        else:
            download(record, target)

    print(
        f"prepared sources under {args.data_root} from "
        f"{MANIFEST.name}"
    )


if __name__ == "__main__":
    main()
