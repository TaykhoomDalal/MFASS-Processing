#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from MFASS.process import (
    COMPACT_COLUMNS,
    EXPECTED_EVALUABLE,
    EXPECTED_POSITIVES,
    EXPECTED_ROWS,
    build_frames,
    load_inputs,
    published_metrics,
    validate_inputs,
)
from scripts.common import sha256
from scripts.download_data import load_manifest


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
    manifest = load_manifest(path)
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
        outputs["compact"]["path"] == "MFASS/mfass.parquet"
        and outputs["compact"]["columns"] == len(COMPACT_COLUMNS),
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

    compact_columns = {
        field["name"]
        for field in golden["outputs"][
            outputs["compact"]["path"]
        ]["arrow_schema"]
    }
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


def independent_compact_checks(
    compact: pd.DataFrame, contract: dict
) -> None:
    quantitative = contract["targets"]["quantitative"]
    classification = contract["targets"]["classification"]
    primary = quantitative["primary_column"]
    expected_quantitative = (
        compact[quantitative["alternate_inclusion_column"]]
        - compact[quantitative["reference_inclusion_column"]]
    )
    require(
        compact[primary].notna().equals(expected_quantitative.notna()),
        "quantitative target null mask differs from inclusion measurements",
    )
    require(
        np.allclose(
            compact.loc[compact[primary].notna(), primary],
            expected_quantitative.loc[compact[primary].notna()],
            rtol=0,
            atol=1e-6,
        ),
        "quantitative target is not alternate minus reference inclusion",
    )

    expected_labels = compact[primary].le(-0.5).where(
        compact[primary].notna()
    ).astype("boolean")
    try:
        pd.testing.assert_series_equal(
            compact[classification["column"]],
            expected_labels,
            check_names=False,
            check_dtype=True,
            check_exact=True,
        )
    except AssertionError as error:
        raise RuntimeError(
            f"compact label is not delta_psi <= -0.50: {error}"
        ) from error

    input_pair = contract["input_pair"]
    sequence_length = input_pair["length"]
    reference_column = input_pair["reference_column"]
    alternate_column = input_pair["alternate_column"]
    pair_column = contract["identifiers"]["pair"]["column"]
    for row in compact.itertuples(index=False):
        reference_sequence = getattr(row, reference_column)
        alternate_sequence = getattr(row, alternate_column)
        pair_id = getattr(row, pair_column)
        differences = [
            index
            for index, (reference, alternate) in enumerate(
                zip(reference_sequence, alternate_sequence)
            )
            if reference != alternate
        ]
        require(
            len(reference_sequence)
            == len(alternate_sequence)
            == sequence_length
            and differences == [row.variant_offset],
            f"{pair_id}: invalid {sequence_length} bp assay pair",
        )
        expected_ref = (
            row.ref
            if row.strand == "+"
            else row.ref.translate(COMPLEMENT)
        )
        expected_alt = (
            row.alt
            if row.strand == "+"
            else row.alt.translate(COMPLEMENT)
        )
        require(
            reference_sequence[row.variant_offset] == expected_ref
            and alternate_sequence[row.variant_offset] == expected_alt,
            f"{pair_id}: assay/genome allele orientation mismatch",
        )


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
    before = validate_inputs(args.data_root)
    source = load_inputs(args.data_root)
    fasta = (
        args.data_root / "reference"
        / "GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta"
    )
    expected_compact, expected_full, _orientation = build_frames(
        source, fasta
    )
    expected_metrics = published_metrics(source)
    after = validate_inputs(args.data_root)
    require(before == after, "pinned input changed during verification")

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
    compare_frames("compact parquet", compact, expected_compact)
    compare_frames("published metrics", metrics, expected_metrics)
    if full is not None:
        compare_frames("full parquet", full, expected_full)
        compare_frames(
            "compact ordered projection of full",
            compact,
            full.loc[:, COMPACT_COLUMNS],
        )

    require(
        compact.columns.tolist() == COMPACT_COLUMNS,
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

    independent_compact_checks(compact, contract)
    independent_membership_checks(
        args.data_root, compact, metrics, full, contract
    )
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
