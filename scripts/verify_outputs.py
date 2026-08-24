#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pyfaidx import Fasta
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.common import sha256


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
ORACLE_SOURCE_COLUMNS = [
    "id",
    "ensembl_id",
    "chr",
    "strand",
    "intron1_len",
    "exon_len",
    "intron2_len",
    "ref_allele",
    "alt_allele",
    "rel_position",
    "snp_position_hg38_1based",
    "label",
    "rel_position_feature",
    "original_seq",
    "natural_seq",
    "category",
    "v2_dpsi_R1",
    "v2_dpsi_R2",
    "v2_index",
    "nat_v2_index",
    "v2_dpsi",
]
REGION_NAMES = {
    "upstr_intron": "upstream_intron",
    "exon": "exon",
    "downstr_intron": "downstream_intron",
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


def build_compact_oracle(
    data_root: Path, golden: dict
) -> pd.DataFrame:
    source_path = (
        data_root
        / golden["sources"]["mfass_measurements"]["local_path"]
    )
    source = pd.read_csv(
        source_path,
        sep="\t",
        usecols=ORACLE_SOURCE_COLUMNS,
        low_memory=False,
    )
    source["source_index"] = source.index.astype("int64")
    source = source.loc[source.category.eq("mutant")].reset_index(drop=True)
    require(
        len(source) == EXPECTED_ROWS and not source.id.duplicated().any(),
        "unexpected canonical mutant row set",
    )

    fasta_path = (
        data_root
        / golden["sources"]["grch38_no_alt_reference"]["local_path"]
    )
    genome = Fasta(
        fasta_path,
        one_based_attributes=False,
        rebuild=False,
    )
    try:
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
    finally:
        genome.close()

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

    rows = []
    for row, reference_base in zip(
        source.itertuples(index=False), reference_bases
    ):
        source_ref = str(row.ref_allele).upper()
        source_alt = str(row.alt_allele).upper()
        identity = identity_orientation[row.ensembl_id]
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
        if reference_base == genomic_alt:
            genomic_ref, genomic_alt = genomic_alt, genomic_ref
            effect_sign = -1.0
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
        rows.append({
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
        })

    oracle = pd.DataFrame(rows)
    oracle["label"] = oracle["label"].astype("boolean")
    for column in (
        "delta_psi",
        "delta_psi_rep1",
        "delta_psi_rep2",
        "ref_inclusion",
        "alt_inclusion",
    ):
        oracle[column] = oracle[column].astype("float32")
    return oracle


def independent_membership_checks(
    data_root: Path,
    compact: pd.DataFrame,
    metrics: pd.DataFrame,
    full: pd.DataFrame | None,
    contract: dict,
) -> None:
    source = pd.read_csv(
        data_root / "source/snv_data_clean.txt",
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

    labels = pd.read_csv(
        data_root / "published/mfass_labels.csv"
    ).set_index("id")
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
            filename, _score_column = SCORE_FILES[method]
            score_column = method_contracts[method]["score_column"]
            score_table = pd.read_csv(
                data_root / "published" / filename
            ).set_index("id")
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


def main() -> None:
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

    oracle = build_compact_oracle(args.data_root, golden)
    compare_frames(
        "independent source-derived compact oracle",
        compact.loc[:, oracle.columns],
        oracle,
    )
    independent_membership_checks(
        args.data_root, compact, metrics, full, contract
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
        f"{detail}, 28,972 pair_id rows, repeated assay references, "
        "and five independently masked published baselines"
    )


if __name__ == "__main__":
    main()
