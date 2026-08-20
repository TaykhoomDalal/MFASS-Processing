#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import shutil
import urllib.request
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import sha256, write_json


MFASS_COMMIT = "3b1b2bdaea828283508ba22cdd8d0c431ea70dea"
SPLICECONSENSUS_COMMIT = "cdb411fd5fc30dd811284f7c59ce1b5acf4e9c82"
MFASS_RAW = (
    "https://raw.githubusercontent.com/KosuriLab/MFASS/"
    f"{MFASS_COMMIT}/processed_data/snv/snv_data_clean.txt"
)
SPLICE_RAW = (
    "https://raw.githubusercontent.com/brhanufen/spliceconsensus/"
    f"{SPLICECONSENSUS_COMMIT}"
)
FASTA_URL = (
    "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/001/405/"
    "GCA_000001405.15_GRCh38/seqs_for_alignment_pipelines.ucsc_ids/"
    "GCA_000001405.15_GRCh38_no_alt_analysis_set.fna.gz"
)
FASTA_SHA256 = (
    "9cce8b926416dd96b152deea85188495b75f7ac8d634cc723a017067be8702b7"
)

ASSETS = {
    "source/snv_data_clean.txt": {
        "url": MFASS_RAW,
        "sha256": (
            "a637ca0e307e66ff48811ec7efa22b9ce453bc7883b04f0cacb867f7283132d8"
        ),
    },
    "published/mfass_labels.csv": {
        "url": f"{SPLICE_RAW}/data/mfass_labels.csv",
        "sha256": (
            "ce6e3b584ae8fe07d52f1f2611689c6d68ed531c4d2b5b073af94851c146cd7c"
        ),
    },
    "published/scores_pangolin.csv": {
        "url": f"{SPLICE_RAW}/results/scores_pangolin.csv",
        "sha256": (
            "c76ce4fbe4c5c979c4e1534068e72354223f0d5c62d3f7e91e1f4fccd16341fa"
        ),
    },
    "published/scores_spliceai.csv": {
        "url": f"{SPLICE_RAW}/results/scores_spliceai.csv",
        "sha256": (
            "2e923e60c7611486ebcd95797c3e436807535d9463be4685dcc8e44b13239776"
        ),
    },
    "published/scores_splicetx.csv": {
        "url": f"{SPLICE_RAW}/results/scores_splicetx.csv",
        "sha256": (
            "d482551808825d2de59952d006a8524f97efec451db01201544ef3a08e6d3a8f"
        ),
    },
    "published/scores_mmsplice.csv": {
        "url": f"{SPLICE_RAW}/results/scores_mmsplice.csv",
        "sha256": (
            "b05b07be7db19a39a040963885f5053ad164b851cebad377113747f0a227dec0"
        ),
    },
}


def verify(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"{path}: expected sha256 {expected}, got {actual}")


def download(url: str, path: Path, expected: str) -> None:
    if path.is_file():
        verify(path, expected)
        return
    if path.exists():
        raise RuntimeError(f"refusing to overwrite unexpected path: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    request = urllib.request.Request(
        url, headers={"User-Agent": "MFASS-Processing/1.0"}
    )
    with urllib.request.urlopen(request, timeout=180) as source:
        with temporary.open("wb") as target:
            shutil.copyfileobj(source, target)
    verify(temporary, expected)
    temporary.replace(path)


def link(source: Path, target: Path, expected: str) -> None:
    source = source.resolve()
    verify(source, expected)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() and target.resolve() == source:
        return
    if target.exists() or target.is_symlink():
        raise RuntimeError(f"refusing to overwrite unexpected path: {target}")
    target.symlink_to(source)


def prepare_fasta(data_root: Path, reuse: Path | None) -> Path:
    fasta = (
        data_root / "reference"
        / "GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta"
    )
    if reuse is not None:
        link(reuse, fasta, FASTA_SHA256)
        return fasta
    if fasta.is_file():
        verify(fasta, FASTA_SHA256)
        return fasta
    fasta.parent.mkdir(parents=True, exist_ok=True)
    compressed = fasta.with_suffix(".fa.gz")
    request = urllib.request.Request(
        FASTA_URL, headers={"User-Agent": "MFASS-Processing/1.0"}
    )
    with urllib.request.urlopen(request, timeout=180) as source:
        with compressed.open("wb") as target:
            shutil.copyfileobj(source, target)
    temporary = fasta.with_suffix(".fa.tmp")
    with gzip.open(compressed, "rb") as source, temporary.open("wb") as target:
        shutil.copyfileobj(source, target)
    verify(temporary, FASTA_SHA256)
    temporary.replace(fasta)
    compressed.unlink()
    return fasta


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=root / "data")
    parser.add_argument("--reuse-source", type=Path)
    parser.add_argument("--reuse-benchmark-root", type=Path)
    parser.add_argument("--reuse-fasta", type=Path)
    args = parser.parse_args()

    for relative, record in ASSETS.items():
        target = args.data_root / relative
        if relative == "source/snv_data_clean.txt" and args.reuse_source:
            link(args.reuse_source, target, record["sha256"])
        elif relative.startswith("published/") and args.reuse_benchmark_root:
            name = Path(relative).name
            source = (
                args.reuse_benchmark_root / "data/mfass_labels.csv"
                if name == "mfass_labels.csv"
                else args.reuse_benchmark_root / "results" / name
            )
            link(source, target, record["sha256"])
        else:
            download(record["url"], target, record["sha256"])

    fasta = prepare_fasta(args.data_root, args.reuse_fasta)
    manifest = {
        "mfass_commit": MFASS_COMMIT,
        "spliceconsensus_commit": SPLICECONSENSUS_COMMIT,
        "assets": {
            relative: {
                **record,
                "local_path": relative,
            }
            for relative, record in ASSETS.items()
        },
        "reference": {
            "url": FASTA_URL,
            "sha256": FASTA_SHA256,
            "local_path": str(fasta.relative_to(args.data_root)),
        },
    }
    write_json(args.data_root / "input_manifest.json", manifest)
    print(f"prepared pinned inputs under {args.data_root}")


if __name__ == "__main__":
    main()
