#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.common import isolated_fasta, repository_lock, sha256


EXPECTED_ROWS = 28_972
EXPECTED_EVALUABLE = 27_733
EXPECTED_POSITIVES = 1_050
EXPECTED_METHODS = (
    "pangolin",
    "spliceai",
    "splicetransformer",
    "mmsplice",
    "spanr",
)
SCORE_FILES = {
    "pangolin": ("scores_pangolin.csv", "pangolin_score"),
    "spliceai": ("scores_spliceai.csv", "spliceai_score"),
    "splicetransformer": (
        "scores_splicetx.csv",
        "splicetransformer_score",
    ),
    "mmsplice": ("scores_mmsplice.csv", "mmsplice_score"),
}
SCORE_SOURCE_NAMES = {
    "pangolin": "pangolin_scores",
    "spliceai": "spliceai_scores",
    "splicetransformer": "splicetransformer_scores",
    "mmsplice": "mmsplice_scores",
}
PUBLISHED_SOURCE_NAMES = (
    "mfass_labels",
    "pangolin_scores",
    "spliceai_scores",
    "splicetransformer_scores",
    "mmsplice_scores",
)
EXPECTED_COMPACT_COLUMNS = [
    "split",
    "source_index",
    "pair_id",
    "component_id",
    "sequence",
    "alt_sequence",
    "label",
    "delta_psi",
    "delta_psi_rep1",
    "delta_psi_rep2",
    "ref_inclusion",
    "alt_inclusion",
    "chrom",
    "position",
    "ref",
    "alt",
    "strand",
    "variant_offset",
    "exon_start",
    "exon_end",
    "region",
    "splice_site_offset",
    "assay_hg38_alignment",
]
EXPECTED_OUTPUT_PATHS = {
    "MFASS/mfass.parquet",
    "MFASS/mfass-full.parquet",
    "MFASS/published_metrics.parquet",
}
REGION_NAMES = {
    "upstr_intron": "upstream_intron",
    "exon": "exon",
    "downstr_intron": "downstream_intron",
}
ALIGNMENT_NAMES = {
    "exact": "exact",
    "one_substitution": "substitution",
    "one_hg38_insertion": "insertion",
}
COMPLEMENT = str.maketrans("ACGT", "TGCA")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def compare_frames(
    name: str, actual: pd.DataFrame, expected: pd.DataFrame
) -> None:
    require(
        actual.columns.tolist() == expected.columns.tolist(),
        f"{name}: column mismatch",
    )
    actual_schema = pa.Table.from_pandas(
        actual, preserve_index=False
    ).schema.remove_metadata()
    expected_schema = pa.Table.from_pandas(
        expected, preserve_index=False
    ).schema.remove_metadata()
    require(actual_schema == expected_schema, f"{name}: schema mismatch")
    try:
        pd.testing.assert_frame_equal(
            actual,
            expected,
            check_dtype=True,
            check_exact=True,
            check_like=False,
        )
    except AssertionError as error:
        raise RuntimeError(f"{name}: data mismatch: {error}") from error


def arrow_schema(path: Path) -> list[dict]:
    return [
        {
            "name": field.name,
            "type": str(field.type),
            "nullable": field.nullable,
        }
        for field in pq.read_schema(path)
    ]


def verify_output_record(
    path: Path, expected: dict, expected_rows: int
) -> None:
    require(path.is_file(), f"missing output: {path}")
    require(
        path.stat().st_size == expected["bytes"],
        f"{path.name}: byte size changed",
    )
    observed = sha256(path)
    require(
        observed == expected["sha256"],
        f"{path.name}: expected sha256 {expected['sha256']}, got {observed}",
    )
    parquet = pq.ParquetFile(path)
    require(
        parquet.metadata.num_rows == expected_rows == expected["rows"],
        f"{path.name}: row count changed",
    )
    require(
        arrow_schema(path) == expected["arrow_schema"],
        f"{path.name}: Arrow schema changed",
    )


def load_golden_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    require(
        set(manifest) == {
            "manifest_version",
            "sources",
            "outputs",
            "contracts",
        },
        "manifest must contain only sources, outputs, and contracts metadata",
    )
    require(
        manifest["manifest_version"] == 1,
        "unsupported manifest version",
    )
    require(
        isinstance(manifest["outputs"], dict)
        and set(manifest["outputs"]) == EXPECTED_OUTPUT_PATHS,
        "manifest must contain exactly the three supported outputs",
    )
    require(
        set(manifest["sources"])
        == {
            "grch38_no_alt_reference",
            "mfass_labels",
            "mfass_measurements",
            "mmsplice_scores",
            "pangolin_scores",
            "spliceai_scores",
            "splicetransformer_scores",
        },
        "unexpected source manifest membership",
    )
    local_paths = []
    for name, record in manifest["sources"].items():
        local_paths.append(record["local_path"])
        require(
            isinstance(record.get("identity"), dict)
            and isinstance(record.get("download"), dict)
            and isinstance(record.get("materialized"), dict)
            and isinstance(record.get("rights"), dict)
            and record["rights"].get("artifact_license") == "NOASSERTION"
            and str(record.get("url", "")).startswith("https://"),
            f"invalid source record: {name}",
        )
        require(
            isinstance(record["materialized"].get("bytes"), int)
            and len(record["materialized"].get("sha256", "")) == 64,
            f"invalid materialized source identity: {name}",
        )
    require(
        len(local_paths) == len(set(local_paths)),
        "source local paths must be unique",
    )
    return manifest


def validate_inputs(golden: dict, data_root: Path) -> dict[str, str]:
    observed = {}
    for name, record in golden["sources"].items():
        relative = Path(record["local_path"])
        require(
            not relative.is_absolute() and ".." not in relative.parts,
            f"invalid source path: {relative}",
        )
        path = data_root / relative
        expected = record["materialized"]
        require(path.is_file(), f"missing pinned input: {path}")
        require(
            path.stat().st_size == expected["bytes"],
            f"{name}: source byte size changed",
        )
        digest = sha256(path)
        require(
            digest == expected["sha256"],
            f"{name}: expected sha256 {expected['sha256']}, got {digest}",
        )
        observed[str(relative)] = digest
    return observed


def load_processing_contract(golden: dict) -> dict:
    require(
        set(golden["contracts"]) == {"mfass"},
        "manifest must contain only the mfass contract",
    )
    contract = golden["contracts"].get("mfass")
    require(isinstance(contract, dict), "missing processing contract")
    require(
        contract["rows"] == EXPECTED_ROWS
        and contract["task"] == "mfass"
        and contract["split"] == "test",
        "unexpected MFASS row contract",
    )
    require(
        contract["identifiers"]
        == {
            "component": {
                "column": "component_id",
                "expected_distinct": 2_198,
                "meaning": "Ensembl exon",
            },
            "pair": {
                "column": "pair_id",
                "unique": True,
            },
        },
        "unexpected MFASS identifier contract",
    )
    require(
        contract["input_pair"]
        == {
            "alternate_column": "alt_sequence",
            "allele_order": "GRCh38 reference then alternate",
            "deduplicate": False,
            "expected_distinct_reference_sequences": 2_199,
            "length": 170,
            "orientation": "transcript",
            "reference_column": "sequence",
        },
        "unexpected MFASS input-pair contract",
    )
    require(
        contract["coordinates"]
        == {
            "alleles": "forward-genome",
            "assembly": "GRCh38 no-alt GCA_000001405.15",
            "chromosome_names": "UCSC chr-prefixed",
            "exon_end": "exclusive within assay sequence",
            "exon_start": "0-based within assay sequence",
            "negative_strand_assay_bases": "complement of genome alleles",
            "position": "1-based",
            "strand_column": "strand",
            "variant_offset": "0-based within assay sequence",
        },
        "unexpected MFASS coordinate contract",
    )
    require(
        contract["targets"]
        == {
            "classification": {
                "column": "label",
                "definition": "delta_psi <= -0.50",
                "evaluable_rows": EXPECTED_EVALUABLE,
                "nullable": True,
                "positive_rows": EXPECTED_POSITIVES,
            },
            "quantitative": {
                "alternate_inclusion_column": "alt_inclusion",
                "definition": "alternate minus reference exon inclusion",
                "nullable": True,
                "primary_column": "delta_psi",
                "reference_inclusion_column": "ref_inclusion",
                "replicate_columns": [
                    "delta_psi_rep1",
                    "delta_psi_rep2",
                ],
            },
        },
        "unexpected MFASS target contract",
    )
    expected_methods = [
        {
            "name": method,
            "score_column": (
                "spanr_score"
                if method == "spanr"
                else SCORE_FILES[method][1]
            ),
        }
        for method in EXPECTED_METHODS
    ]
    evaluation = contract["evaluation"]
    membership = evaluation.get("membership")
    require(
        {
            key: value
            for key, value in evaluation.items()
            if key != "membership"
        }
        == {
            "label_column": "source_label",
            "loss_column": "v2_dpsi",
            "loss_transform": "negative",
            "mask": "source_label.notna() & score.notna()",
            "methods": expected_methods,
            "metrics": [
                "auroc",
                "average_precision",
                "spearman_vs_loss",
            ],
        },
        "unexpected MFASS evaluation contract",
    )
    require(
        isinstance(membership, dict)
        and set(membership) == set(EXPECTED_METHODS),
        "unexpected metric-membership contract",
    )
    for method, record in membership.items():
        require(
            isinstance(record.get("rows"), int)
            and isinstance(record.get("positives"), int)
            and len(record.get("ordered_pair_ids_sha256", "")) == 64,
            f"invalid metric-membership record: {method}",
        )

    outputs = contract["outputs"]
    require(
        set(outputs) == {"compact", "full", "metrics"},
        "processing contract must contain exactly three outputs",
    )
    require(
        outputs["compact"]["path"] == "MFASS/mfass.parquet"
        and outputs["compact"]["columns"] == len(EXPECTED_COMPACT_COLUMNS),
        "unexpected compact output contract",
    )
    require(
        outputs["full"]["path"] == "MFASS/mfass-full.parquet"
        and outputs["full"]["compact_projection"]
        == "ordered first 23 columns",
        "unexpected full output contract",
    )
    require(
        outputs["metrics"]["path"] == "MFASS/published_metrics.parquet",
        "unexpected metrics output contract",
    )
    for output in outputs.values():
        record = golden["outputs"].get(output["path"])
        require(record is not None, f"missing golden record: {output['path']}")
        require(
            output["rows"] == record["rows"]
            and output["columns"] == len(record["arrow_schema"]),
            f"processing contract differs from golden output: {output['path']}",
        )

    compact_schema = golden["outputs"][
        outputs["compact"]["path"]
    ]["arrow_schema"]
    require(
        [field["name"] for field in compact_schema]
        == EXPECTED_COMPACT_COLUMNS,
        "unexpected ordered compact column contract",
    )
    compact_columns = {field["name"] for field in compact_schema}
    full_columns = {
        field["name"]
        for field in golden["outputs"][
            outputs["full"]["path"]
        ]["arrow_schema"]
    }
    metric_columns = {
        field["name"]
        for field in golden["outputs"][
            outputs["metrics"]["path"]
        ]["arrow_schema"]
    }
    require(
        {
            "pair_id",
            "component_id",
            "sequence",
            "alt_sequence",
            "label",
            "delta_psi",
            "delta_psi_rep1",
            "delta_psi_rep2",
            "ref_inclusion",
            "alt_inclusion",
            "chrom",
            "position",
            "ref",
            "alt",
            "strand",
            "variant_offset",
            "exon_start",
            "exon_end",
        }
        <= compact_columns,
        "processing contract names missing compact columns",
    )
    require(
        {
            contract["evaluation"]["label_column"],
            contract["evaluation"]["loss_column"],
            *(
                method["score_column"]
                for method in contract["evaluation"]["methods"]
            ),
        }
        <= full_columns,
        "processing contract names missing full evaluation columns",
    )
    require(
        {"method", "n", "positives", "score_column"}
        | set(contract["evaluation"]["metrics"])
        <= metric_columns,
        "processing contract names missing metric columns",
    )
    return contract


def contract_methods(contract: dict) -> list[str]:
    return [
        method["name"]
        for method in contract["evaluation"]["methods"]
    ]


def ordered_id_hash(values: pd.Series) -> str:
    digest = hashlib.sha256()
    for value in values.astype(str):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def optional_float(value: object) -> float | None:
    return None if pd.isna(value) else float(value)


def load_oracle_source(data_root: Path, golden: dict) -> pd.DataFrame:
    source_path = (
        data_root
        / golden["sources"]["mfass_measurements"]["local_path"]
    )
    source = pd.read_csv(
        source_path,
        sep="\t",
        low_memory=False,
    )
    source["source_index"] = source.index.astype("int64")
    source = source.loc[source.category.eq("mutant")].copy()
    source.reset_index(drop=True, inplace=True)
    require(
        len(source) == EXPECTED_ROWS and not source.id.duplicated().any(),
        "unexpected canonical mutant row set",
    )
    return source


def classify_region_alignment(
    assay_sequence: str, hg38_sequence: str
) -> dict:
    if assay_sequence == hg38_sequence:
        return {
            "status": "exact",
            "substitutions": 0,
            "insertions": 0,
            "deletions": 0,
            "edit_distance": 0,
            "insertion_offsets": (),
        }
    if len(assay_sequence) == len(hg38_sequence):
        mismatches = sum(
            left != right
            for left, right in zip(assay_sequence, hg38_sequence)
        )
        if mismatches == 1:
            return {
                "status": "one_substitution",
                "substitutions": 1,
                "insertions": 0,
                "deletions": 0,
                "edit_distance": 1,
                "insertion_offsets": (),
            }
    if len(hg38_sequence) == len(assay_sequence) + 1:
        insertion_offsets = tuple(
            offset
            for offset in range(len(hg38_sequence))
            if (
                hg38_sequence[:offset]
                + hg38_sequence[offset + 1:]
                == assay_sequence
            )
        )
        if insertion_offsets:
            return {
                "status": "one_hg38_insertion",
                "substitutions": 0,
                "insertions": 1,
                "deletions": 0,
                "edit_distance": 1,
                "insertion_offsets": insertion_offsets,
            }
    raise RuntimeError("unexpected independent lifted-region alignment")


def build_region_oracle(
    source: pd.DataFrame,
    genome,
    identity_orientation: dict[str, bool],
) -> dict[str, dict]:
    validations = {}
    for exon, rows in source.groupby("ensembl_id", sort=False):
        require(
            rows[
                [
                    "start_hg38_0based",
                    "end_hg38_0based",
                    "strand",
                    "natural_seq",
                ]
            ].nunique(dropna=False).eq(1).all(),
            f"{exon}: inconsistent region source fields",
        )
        row = rows.iloc[0]
        region_start = int(row.start_hg38_0based)
        region_end = int(row.end_hg38_0based)
        hg38_strand = str(row.strand)
        if not identity_orientation[exon]:
            hg38_strand = "-" if hg38_strand == "+" else "+"
        forward_sequence = str(
            genome[row.chr][region_start:region_end]
        ).upper()
        hg38_sequence = (
            forward_sequence
            if hg38_strand == "+"
            else forward_sequence.translate(COMPLEMENT)[::-1]
        )
        validation = classify_region_alignment(
            str(row.natural_seq).upper(), hg38_sequence
        )
        for variant in rows.itertuples(index=False):
            assay_offset = int(variant.rel_position) - 1
            hg38_offset = (
                int(variant.snp_position_hg38_1based) - 1 - region_start
                if hg38_strand == "+"
                else region_end - int(variant.snp_position_hg38_1based)
            )
            if validation["status"] == "one_hg38_insertion":
                mapped_offsets = {
                    assay_offset + (assay_offset >= insertion_offset)
                    for insertion_offset in validation["insertion_offsets"]
                }
            else:
                mapped_offsets = {assay_offset}
            require(
                hg38_offset in mapped_offsets,
                f"{variant.id}: independent assay/hg38 offsets disagree",
            )
        validations[exon] = {
            **validation,
            "hg38_strand": hg38_strand,
        }
    return validations


def nullable_source_label(series: pd.Series) -> pd.Series:
    return series.map({
        True: True,
        False: False,
        "TRUE": True,
        "FALSE": False,
        "True": True,
        "False": False,
        1: True,
        0: False,
    }).astype("boolean")


def load_published_oracle(
    data_root: Path,
    golden: dict,
    source: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.Series]]:
    pair_ids = source["id"].astype(str)
    label_path = (
        data_root / golden["sources"]["mfass_labels"]["local_path"]
    )
    labels = pd.read_csv(label_path)
    require(
        labels.columns.tolist() == ["id", "sdv", "dpsi", "spanr"],
        "unexpected published-label columns",
    )
    labels["id"] = labels["id"].astype(str)
    labels = labels.set_index("id")
    require(
        not labels.index.duplicated().any()
        and set(labels.index) == set(pair_ids),
        "published labels do not contain canonical pair membership",
    )
    labels = labels.reindex(pair_ids).reset_index(drop=True)

    source_labels = nullable_source_label(source["strong_lof"])
    published_labels = labels["sdv"].map({
        0.0: False,
        1.0: True,
    }).astype("boolean")
    try:
        pd.testing.assert_series_equal(
            source_labels.reset_index(drop=True),
            published_labels.reset_index(drop=True),
            check_names=False,
            check_dtype=True,
            check_exact=True,
        )
    except AssertionError as error:
        raise RuntimeError(
            f"published labels do not reproduce source strong_lof: {error}"
        ) from error
    require(
        np.array_equal(
            source["v2_dpsi"].to_numpy(float),
            labels["dpsi"].to_numpy(float),
            equal_nan=True,
        ),
        "published dPSI does not reproduce the source measurements",
    )

    source_oracle = source.drop(columns=["source_index"]).rename(columns={
        "sequence": "source_sequence",
        "label": "source_region",
        "strand": "source_strand",
    })
    source_oracle["published_label"] = labels["sdv"].to_numpy()
    source_oracle["published_dpsi"] = labels["dpsi"].to_numpy()
    source_oracle["spanr_score"] = labels["spanr"].to_numpy()
    scores = {"spanr": labels["spanr"]}
    for method, (_filename, score_column) in SCORE_FILES.items():
        source_name = SCORE_SOURCE_NAMES[method]
        score_path = (
            data_root / golden["sources"][source_name]["local_path"]
        )
        score_table = pd.read_csv(score_path)
        require(
            score_table.columns.tolist() == ["id", "score"],
            f"unexpected {method} score columns",
        )
        score_table["id"] = score_table["id"].astype(str)
        score_table = score_table.set_index("id")
        require(
            not score_table.index.duplicated().any()
            and set(score_table.index) == set(pair_ids),
            f"{method} scores do not contain canonical pair membership",
        )
        score_series = score_table.reindex(pair_ids)["score"].reset_index(
            drop=True
        )
        source_oracle[score_column] = score_series.to_numpy()
        scores[method] = score_series
    source_oracle["source_label"] = source_labels.reset_index(drop=True)
    return source_oracle, labels, scores


def build_output_oracles(
    data_root: Path, golden: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source = load_oracle_source(data_root, golden)
    fasta_path = (
        data_root
        / golden["sources"]["grch38_no_alt_reference"]["local_path"]
    )
    with isolated_fasta(
        fasta_path,
        one_based_attributes=False,
    ) as genome:
        reference_bases = pd.Series(
            [
                str(
                    genome[row.chr][
                        int(row.snp_position_hg38_1based) - 1
                    ]
                ).upper()
                for row in source.itertuples(index=False)
            ],
            index=source.index,
        )
        source_refs = source.ref_allele.str.upper()
        orientation_votes = pd.DataFrame({
            "component_id": source.ensembl_id,
            "direct": reference_bases.eq(source_refs).astype(int),
            "reverse": reference_bases.eq(
                source_refs.str.translate(COMPLEMENT)
            ).astype(int),
        }).groupby("component_id", sort=False)[["direct", "reverse"]].sum()
        require(
            not orientation_votes.direct.eq(
                orientation_votes.reverse
            ).any(),
            "ambiguous independent exon-orientation vote",
        )
        identity_orientation = orientation_votes.direct.gt(
            orientation_votes.reverse
        ).to_dict()
        region_validations = build_region_oracle(
            source, genome, identity_orientation
        )

    source_oracle, _labels, _scores = load_published_oracle(
        data_root, golden, source
    )
    compact_rows = []
    audit_rows = []
    for row, reference_base in zip(
        source.itertuples(index=False), reference_bases
    ):
        source_ref = str(row.ref_allele).upper()
        source_alt = str(row.alt_allele).upper()
        identity = identity_orientation[row.ensembl_id]
        region_validation = region_validations[row.ensembl_id]
        genomic_ref = (
            source_ref
            if identity
            else source_ref.translate(COMPLEMENT)
        )
        genomic_alt = (
            source_alt
            if identity
            else source_alt.translate(COMPLEMENT)
        )
        effect_sign = 1.0
        action = "identity" if identity else "reverse_complement"
        if reference_base == genomic_alt:
            genomic_ref, genomic_alt = genomic_alt, genomic_ref
            effect_sign = -1.0
            action += "_reference_swap"
        else:
            require(
                reference_base == genomic_ref,
                f"{row.id}: neither source allele matches GRCh38",
            )

        strand = str(row.strand)
        if not identity:
            strand = "-" if strand == "+" else "+"
        sequence = str(row.natural_seq).upper()
        alt_sequence = str(row.original_seq).upper()
        if effect_sign < 0:
            sequence, alt_sequence = alt_sequence, sequence
        variant_offset = int(row.rel_position) - 1
        differences = [
            index
            for index, (reference, alternate) in enumerate(
                zip(sequence, alt_sequence)
            )
            if reference != alternate
        ]
        expected_ref = (
            genomic_ref
            if strand == "+"
            else genomic_ref.translate(COMPLEMENT)
        )
        expected_alt = (
            genomic_alt
            if strand == "+"
            else genomic_alt.translate(COMPLEMENT)
        )
        require(
            len(sequence) == len(alt_sequence) == 170
            and differences == [variant_offset]
            and sequence[variant_offset] == expected_ref
            and alt_sequence[variant_offset] == expected_alt,
            f"{row.id}: source-derived assay geometry is invalid",
        )

        delta_psi = optional_float(row.v2_dpsi)
        replicate_1 = optional_float(row.v2_dpsi_R1)
        replicate_2 = optional_float(row.v2_dpsi_R2)
        ref_inclusion, alt_inclusion = (
            (row.nat_v2_index, row.v2_index)
            if effect_sign > 0
            else (row.v2_index, row.nat_v2_index)
        )
        ref_inclusion = optional_float(ref_inclusion)
        alt_inclusion = optional_float(alt_inclusion)
        normalized_delta = (
            None
            if delta_psi is None
            else effect_sign * delta_psi
        )
        require(
            (normalized_delta is None)
            == (ref_inclusion is None or alt_inclusion is None),
            f"{row.id}: target and inclusion null masks disagree",
        )
        if normalized_delta is not None:
            require(
                np.isclose(
                    normalized_delta,
                    alt_inclusion - ref_inclusion,
                    rtol=0,
                    atol=1e-6,
                ),
                f"{row.id}: delta_psi is not alternate minus reference",
            )
        exon_start = int(
            row.intron1_len
            if row.strand == "+"
            else row.intron2_len
        )
        compact_rows.append({
            "split": "test",
            "source_index": int(row.source_index),
            "pair_id": str(row.id),
            "component_id": str(row.ensembl_id),
            "sequence": sequence,
            "alt_sequence": alt_sequence,
            "label": (
                None
                if normalized_delta is None
                else normalized_delta <= -0.5
            ),
            "delta_psi": normalized_delta,
            "delta_psi_rep1": (
                None
                if replicate_1 is None
                else effect_sign * replicate_1
            ),
            "delta_psi_rep2": (
                None
                if replicate_2 is None
                else effect_sign * replicate_2
            ),
            "ref_inclusion": ref_inclusion,
            "alt_inclusion": alt_inclusion,
            "chrom": str(row.chr),
            "position": int(row.snp_position_hg38_1based),
            "ref": genomic_ref,
            "alt": genomic_alt,
            "strand": strand,
            "variant_offset": variant_offset,
            "exon_start": exon_start,
            "exon_end": exon_start + int(row.exon_len),
            "region": REGION_NAMES[str(row.label)],
            "splice_site_offset": int(row.rel_position_feature),
            "assay_hg38_alignment": ALIGNMENT_NAMES[
                region_validation["status"]
            ],
        })
        audit_rows.append({
            "orientation_action": action,
            "region_edit_distance": int(
                region_validation["edit_distance"]
            ),
            "region_substitutions": int(
                region_validation["substitutions"]
            ),
            "region_insertions": int(region_validation["insertions"]),
            "region_deletions": int(region_validation["deletions"]),
            "distance_to_splice_site": abs(
                int(row.rel_position_feature)
            ),
            "is_evaluable": delta_psi is not None,
        })

    compact_oracle = pd.DataFrame(compact_rows)
    compact_oracle["label"] = compact_oracle["label"].astype("boolean")
    for column in (
        "delta_psi",
        "delta_psi_rep1",
        "delta_psi_rep2",
        "ref_inclusion",
        "alt_inclusion",
    ):
        compact_oracle[column] = compact_oracle[column].astype("float32")
    audit_oracle = pd.DataFrame(audit_rows)
    audit_oracle["is_evaluable"] = audit_oracle[
        "is_evaluable"
    ].astype("boolean")
    full_oracle = pd.concat(
        [compact_oracle, source_oracle, audit_oracle],
        axis=1,
    )
    require(
        not full_oracle.columns.duplicated().any(),
        "independent full oracle has duplicate columns",
    )
    return compact_oracle, full_oracle, source


def verify_metrics_provenance(
    metrics: pd.DataFrame, golden: dict
) -> None:
    identities = [
        golden["sources"][name]["identity"]
        for name in PUBLISHED_SOURCE_NAMES
    ]
    repositories = {identity.get("repository") for identity in identities}
    revisions = {identity.get("revision") for identity in identities}
    require(
        len(repositories) == 1
        and None not in repositories
        and len(revisions) == 1
        and None not in revisions,
        "published metric inputs do not share one pinned source identity",
    )
    repository = str(next(iter(repositories))).rstrip("/")
    revision = str(next(iter(revisions)))
    require(
        len(revision) == 40
        and all(character in "0123456789abcdef" for character in revision),
        "published metric source revision is not a pinned Git commit",
    )
    values = metrics["source"].drop_duplicates().tolist()
    require(
        values == [f"SpliceConsensus {revision}"],
        "published metric source text or revision changed",
    )
    require(
        "SpliceConsensus".casefold()
        == repository.rsplit("/", 1)[-1].casefold(),
        "published metric source text differs from its pinned repository",
    )


def independent_membership_checks(
    data_root: Path,
    compact: pd.DataFrame,
    metrics: pd.DataFrame,
    full: pd.DataFrame | None,
    contract: dict,
    golden: dict,
) -> None:
    source_path = (
        data_root
        / golden["sources"]["mfass_measurements"]["local_path"]
    )
    source = pd.read_csv(
        source_path,
        sep="\t",
        usecols=["id", "category"],
    )
    pair_ids = source.loc[source.category.eq("mutant"), "id"].astype(str)
    identifiers = contract["identifiers"]
    row_key = identifiers["pair"]["column"]
    component_key = identifiers["component"]["column"]
    require(
        compact[row_key].tolist() == pair_ids.tolist(),
        f"compact {row_key} order differs from the canonical mutant rows",
    )
    require(
        compact[row_key].is_unique,
        f"{row_key} must be the unique row key",
    )
    require(
        compact[component_key].notna().all()
        and compact[component_key].nunique()
        == identifiers["component"]["expected_distinct"],
        "component identifier membership changed",
    )
    reference_column = contract["input_pair"]["reference_column"]
    require(
        compact[reference_column].nunique()
        == contract["input_pair"][
            "expected_distinct_reference_sequences"
        ],
        "shared reference sequences were unexpectedly deduplicated or changed",
    )
    require(
        compact[reference_column].duplicated().any(),
        "MFASS reference sequences are expected to repeat across variant pairs",
    )

    label_path = (
        data_root / golden["sources"]["mfass_labels"]["local_path"]
    )
    labels = pd.read_csv(label_path).set_index("id")
    require(
        not labels.index.duplicated().any()
        and set(labels.index) == set(pair_ids),
        "published labels do not contain the canonical pair_id membership",
    )
    labels = labels.reindex(pair_ids)
    require(
        labels["sdv"].dropna().isin([0, 1]).all(),
        "published labels contain values other than 0, 1, or null",
    )
    source_labels = labels["sdv"].map({0.0: False, 1.0: True}).astype(
        "boolean"
    )
    metrics_by_method = metrics.set_index("method")
    require(
        metrics_by_method.index.is_unique,
        "published metrics methods must be unique",
    )
    require(
        metrics["method"].tolist()
        == contract_methods(contract),
        "published metrics method order changed",
    )

    if full is not None:
        require(
            full["id"].astype(str).tolist() == pair_ids.tolist(),
            "full source rows do not preserve canonical pair_id order",
        )
        try:
            pd.testing.assert_series_equal(
                full["source_label"].reset_index(drop=True),
                source_labels.reset_index(drop=True),
                check_names=False,
                check_dtype=True,
                check_exact=True,
            )
        except AssertionError as error:
            raise RuntimeError(
                f"full source-label membership changed: {error}"
            ) from error

    method_contracts = {
        item["name"]: item
        for item in contract["evaluation"]["methods"]
    }
    for method in contract_methods(contract):
        if method == "spanr":
            scores = labels["spanr"]
            score_column = method_contracts[method]["score_column"]
        else:
            _filename, _score_column = SCORE_FILES[method]
            score_column = method_contracts[method]["score_column"]
            source_name = SCORE_SOURCE_NAMES[method]
            score_path = (
                data_root / golden["sources"][source_name]["local_path"]
            )
            score_table = pd.read_csv(score_path).set_index("id")
            require(
                not score_table.index.duplicated().any()
                and set(score_table.index) == set(pair_ids),
                f"{method} scores do not contain canonical pair membership",
            )
            scores = score_table.reindex(pair_ids)["score"]

        if full is not None:
            observed_scores = full[score_column].to_numpy(float)
            require(
                np.array_equal(
                    observed_scores, scores.to_numpy(float), equal_nan=True
                ),
                f"full {method} score membership or order changed",
            )

        mask = source_labels.notna() & scores.notna()
        members = pair_ids.loc[mask.to_numpy()]
        expected_membership = contract["evaluation"]["membership"][method]
        require(
            int(mask.sum()) == expected_membership["rows"],
            f"{method}: evaluable mask row count changed",
        )
        require(
            int(source_labels.loc[mask].sum())
            == expected_membership["positives"],
            f"{method}: evaluable mask positive count changed",
        )
        require(
            ordered_id_hash(members)
            == expected_membership["ordered_pair_ids_sha256"],
            f"{method}: ordered metric row membership changed",
        )

        row = metrics_by_method.loc[method]
        require(
            int(row["n"]) == int(mask.sum())
            and int(row["positives"]) == int(source_labels.loc[mask].sum()),
            f"{method}: published metric mask counts changed",
        )
        require(
            row["score_column"] == score_column,
            f"{method}: score-column identity changed",
        )
        binary_labels = source_labels.loc[mask].astype(bool).to_numpy()
        selected_scores = scores.loc[mask].to_numpy(float)
        loss = -labels.loc[mask, "dpsi"].to_numpy(float)
        reproduced = {
            "auroc": roc_auc_score(binary_labels, selected_scores),
            "average_precision": average_precision_score(
                binary_labels, selected_scores
            ),
            "spearman_vs_loss": pd.Series(selected_scores).rank().corr(
                pd.Series(loss).rank()
            ),
        }
        for column, value in reproduced.items():
            require(
                np.isclose(
                    float(row[column]), value, rtol=0, atol=1e-12
                ),
                f"{method}: independently reproduced {column} changed",
            )


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full",
        action="store_true",
        help="require and verify mfass-full.parquet",
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "MFASS")
    parser.add_argument("--manifest", type=Path, default=ROOT / "manifest.json")
    args = parser.parse_args()

    golden = load_golden_manifest(args.manifest)
    contract = load_processing_contract(golden)
    outputs = contract["outputs"]
    before = validate_inputs(golden, args.data_root)

    compact_path = args.output / Path(outputs["compact"]["path"]).name
    full_path = args.output / Path(outputs["full"]["path"]).name
    metrics_path = args.output / Path(outputs["metrics"]["path"]).name
    verify_output_record(
        compact_path,
        golden["outputs"][outputs["compact"]["path"]],
        outputs["compact"]["rows"],
    )
    verify_output_record(
        metrics_path,
        golden["outputs"][outputs["metrics"]["path"]],
        outputs["metrics"]["rows"],
    )
    if args.full:
        verify_output_record(
            full_path,
            golden["outputs"][outputs["full"]["path"]],
            outputs["full"]["rows"],
        )

    compact = pd.read_parquet(compact_path)
    metrics = pd.read_parquet(metrics_path)
    full = pd.read_parquet(full_path) if args.full else None
    if full is not None:
        compare_frames(
            "compact ordered projection of full",
            compact,
            full.loc[:, EXPECTED_COMPACT_COLUMNS],
        )

    require(
        compact.columns.tolist() == EXPECTED_COMPACT_COLUMNS,
        "unexpected compact column contract",
    )
    require(
        len(compact) == contract["rows"],
        "unexpected compact row count",
    )
    require(
        int(compact.delta_psi.notna().sum())
        == contract["targets"]["classification"]["evaluable_rows"],
        "unexpected evaluable row count",
    )
    require(
        int(compact.label.fillna(False).sum())
        == contract["targets"]["classification"]["positive_rows"],
        "unexpected positive-label count",
    )
    require(
        compact.split.eq(contract["split"]).all(),
        "unexpected split value",
    )
    require(
        metrics.shape
        == (
            outputs["metrics"]["rows"],
            outputs["metrics"]["columns"],
        ),
        "published metrics must have five rows and eight columns",
    )
    if full is not None:
        require(
            full.shape
            == (outputs["full"]["rows"], outputs["full"]["columns"]),
            "full output must have 28,972 rows and 92 columns",
        )

    compact_oracle, full_oracle, _source = build_output_oracles(
        args.data_root, golden
    )
    compare_frames(
        "independent source-derived compact oracle",
        compact,
        compact_oracle,
    )
    if full is not None:
        compare_frames(
            "independent source/score/alignment/audit full oracle",
            full,
            full_oracle,
        )
    verify_metrics_provenance(metrics, golden)
    independent_membership_checks(
        args.data_root, compact, metrics, full, contract, golden
    )
    after = validate_inputs(golden, args.data_root)
    require(before == after, "pinned input changed during verification")
    detail = (
        ", 92-column full output and ordered compact projection"
        if args.full
        else ""
    )
    print(
        "verified golden hashes/schemas, the 23-column compact contract"
        f"{detail}, independent alignment/source/score/audit values, "
        "28,972 pair_id rows, repeated assay references, and five "
        "independently masked published baselines with pinned provenance"
    )


def main() -> None:
    if os.environ.get("MFASS_PROCESSING_LOCK_HELD") == "1":
        _main()
        return
    with repository_lock(ROOT, exclusive=False):
        _main()


if __name__ == "__main__":
    main()
