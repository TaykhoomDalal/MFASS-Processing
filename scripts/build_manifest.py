#!/usr/bin/env python3
"""Refresh generated output evidence in the combined root manifest."""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from MFASS.process import (
    COMPACT_COLUMNS,
    EXPECTED_EVALUABLE,
    EXPECTED_POSITIVES,
    EXPECTED_ROWS,
    validate_inputs,
)
from scripts.common import (
    replace_files_transactionally,
    sha256,
    write_json,
)
from scripts.download_data import MANIFEST, load_manifest


OUTPUT_NAMES = (
    "mfass.parquet",
    "mfass-full.parquet",
    "published_metrics.parquet",
)
SCORE_FILES = {
    "pangolin": "scores_pangolin.csv",
    "spliceai": "scores_spliceai.csv",
    "splicetransformer": "scores_splicetx.csv",
    "mmsplice": "scores_mmsplice.csv",
}


def processing_contract() -> dict:
    return {
        "task": "mfass",
        "rows": EXPECTED_ROWS,
        "split": "test",
        "identifiers": {
            "pair": {
                "column": "pair_id",
                "unique": True,
            },
            "component": {
                "column": "component_id",
                "expected_distinct": 2_198,
                "meaning": "Ensembl exon",
            },
        },
        "input_pair": {
            "reference_column": "sequence",
            "alternate_column": "alt_sequence",
            "length": 170,
            "orientation": "transcript",
            "allele_order": "GRCh38 reference then alternate",
            "deduplicate": False,
            "expected_distinct_reference_sequences": 2_199,
        },
        "targets": {
            "quantitative": {
                "primary_column": "delta_psi",
                "definition": "alternate minus reference exon inclusion",
                "replicate_columns": [
                    "delta_psi_rep1",
                    "delta_psi_rep2",
                ],
                "reference_inclusion_column": "ref_inclusion",
                "alternate_inclusion_column": "alt_inclusion",
                "nullable": True,
            },
            "classification": {
                "column": "label",
                "definition": "delta_psi <= -0.50",
                "evaluable_rows": EXPECTED_EVALUABLE,
                "positive_rows": EXPECTED_POSITIVES,
                "nullable": True,
            },
        },
        "evaluation": {
            "label_column": "source_label",
            "loss_column": "v2_dpsi",
            "loss_transform": "negative",
            "mask": "source_label.notna() & score.notna()",
            "metrics": [
                "auroc",
                "average_precision",
                "spearman_vs_loss",
            ],
            "methods": [
                {
                    "name": "pangolin",
                    "score_column": "pangolin_score",
                },
                {
                    "name": "spliceai",
                    "score_column": "spliceai_score",
                },
                {
                    "name": "splicetransformer",
                    "score_column": "splicetransformer_score",
                },
                {
                    "name": "mmsplice",
                    "score_column": "mmsplice_score",
                },
                {
                    "name": "spanr",
                    "score_column": "spanr_score",
                },
            ],
        },
        "coordinates": {
            "assembly": "GRCh38 no-alt GCA_000001405.15",
            "chromosome_names": "UCSC chr-prefixed",
            "position": "1-based",
            "alleles": "forward-genome",
            "strand_column": "strand",
            "negative_strand_assay_bases": "complement of genome alleles",
            "variant_offset": "0-based within assay sequence",
            "exon_start": "0-based within assay sequence",
            "exon_end": "exclusive within assay sequence",
        },
        "outputs": {
            "compact": {
                "path": "MFASS/mfass.parquet",
                "rows": EXPECTED_ROWS,
                "columns": len(COMPACT_COLUMNS),
            },
            "full": {
                "path": "MFASS/mfass-full.parquet",
                "rows": EXPECTED_ROWS,
                "columns": 92,
                "compact_projection": "ordered first 23 columns",
            },
            "metrics": {
                "path": "MFASS/published_metrics.parquet",
                "rows": 5,
                "columns": 8,
            },
        },
    }


def arrow_schema(path: Path) -> list[dict]:
    return [
        {
            "name": field.name,
            "type": str(field.type),
            "nullable": field.nullable,
        }
        for field in pq.read_schema(path)
    ]


def describe(path: Path) -> dict:
    parquet = pq.ParquetFile(path)
    return {
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "rows": parquet.metadata.num_rows,
        "arrow_schema": arrow_schema(path),
    }


def ordered_id_hash(values: pd.Series) -> str:
    digest = hashlib.sha256()
    for value in values.astype(str):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def metric_membership(data_root: Path) -> dict:
    source = pd.read_csv(
        data_root / "source/snv_data_clean.txt",
        sep="\t",
        usecols=["id", "category"],
    )
    pair_ids = source.loc[source.category.eq("mutant"), "id"].astype(str)
    labels = pd.read_csv(
        data_root / "published/mfass_labels.csv"
    ).set_index("id")
    if labels.index.duplicated().any() or set(labels.index) != set(pair_ids):
        raise RuntimeError("MFASS label membership does not match source")
    labels = labels.reindex(pair_ids)
    membership = {}
    for method in (
        "pangolin",
        "spliceai",
        "splicetransformer",
        "mmsplice",
        "spanr",
    ):
        if method == "spanr":
            scores = labels["spanr"]
        else:
            score_table = pd.read_csv(
                data_root / "published" / SCORE_FILES[method]
            ).set_index("id")
            if (
                score_table.index.duplicated().any()
                or set(score_table.index) != set(pair_ids)
            ):
                raise RuntimeError(
                    f"{method} score membership does not match source"
                )
            scores = score_table.reindex(pair_ids)["score"]
        mask = labels["sdv"].notna() & scores.notna()
        members = pair_ids.loc[mask.to_numpy()]
        membership[method] = {
            "rows": int(mask.sum()),
            "positives": int(labels.loc[mask, "sdv"].sum()),
            "ordered_pair_ids_sha256": ordered_id_hash(members),
        }
    return membership


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-root", type=Path, default=ROOT / "MFASS")
    parser.add_argument(
        "--full",
        action="store_true",
        help="required: verify and register compact, full, and metrics outputs",
    )
    parser.add_argument(
        "--manifest", type=Path, default=MANIFEST
    )
    args = parser.parse_args()
    if not args.full:
        parser.error(
            "--full is required; compact builds must not register an "
            "unverified mfass-full.parquet"
        )

    current = load_manifest(args.manifest)
    validate_inputs(args.data_root)
    output_root = args.output_root.expanduser().absolute()
    if output_root.is_symlink() or (
        output_root.exists() and not output_root.is_dir()
    ):
        raise RuntimeError(
            f"output root must be a real directory: {output_root}"
        )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = output_root.parent / (
        f".{output_root.name}.manifest-staging-{os.getpid()}"
    )
    if staging.exists() or staging.is_symlink():
        raise RuntimeError(f"staging path already exists: {staging}")
    staging.mkdir()
    try:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "MFASS/process.py"),
                "--full",
                "--data-root",
                str(args.data_root),
                "--output",
                str(staging),
            ],
            check=True,
        )
        entries = {path.name for path in staging.iterdir()}
        if entries != set(OUTPUT_NAMES):
            raise RuntimeError(
                f"fresh build produced unexpected artifacts: "
                f"{sorted(entries)}"
            )
        outputs = {
            f"MFASS/{name}": describe(staging / name)
            for name in OUTPUT_NAMES
        }
        contract = processing_contract()
        contract["evaluation"]["membership"] = metric_membership(
            args.data_root
        )
        payload = {
            "manifest_version": 1,
            "sources": current["sources"],
            "outputs": outputs,
            "contracts": {
                "mfass": contract,
            },
        }
        candidate = staging / "manifest.json.candidate"
        write_json(candidate, payload)
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/verify_outputs.py"),
                "--full",
                "--data-root",
                str(args.data_root),
                "--output",
                str(staging),
                "--manifest",
                str(candidate),
            ],
            check=True,
        )
        replacements = [
            (staging / name, output_root / name)
            for name in OUTPUT_NAMES
        ]
        candidate_bytes = candidate.read_bytes()
        replacements.append((candidate, args.manifest))

        def verify_install() -> None:
            for name in OUTPUT_NAMES:
                path = output_root / name
                expected = outputs[f"MFASS/{name}"]
                if (
                    not path.is_file()
                    or path.is_symlink()
                    or describe(path) != expected
                ):
                    raise RuntimeError(
                        f"installed output differs from candidate: {path}"
                    )
            if (
                not args.manifest.is_file()
                or args.manifest.is_symlink()
                or args.manifest.read_bytes() != candidate_bytes
            ):
                raise RuntimeError(
                    "installed manifest differs from verified candidate"
                )

        replace_files_transactionally(
            replacements,
            staging / ".rollback",
            verify_install,
        )
    finally:
        if staging.is_symlink():
            staging.unlink()
        elif staging.exists():
            shutil.rmtree(staging)
    print(
        f"wrote fully verified combined manifest to {args.manifest}"
    )


if __name__ == "__main__":
    main()
