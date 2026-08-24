#!/usr/bin/env python3
"""Build compact and exhaustive MFASS benchmark parquets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyfaidx import Fasta
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.common import (
    COMPLEMENT,
    align_sequences,
    reverse_complement,
    sha256,
    write_parquet,
)
from scripts.download_data import materialized_sources


EXPECTED_ROWS = 28_972
EXPECTED_EVALUABLE = 27_733
EXPECTED_POSITIVES = 1_050
COMPACT_COLUMNS = [
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
SCORE_FILES = {
    "pangolin_score": "scores_pangolin.csv",
    "spliceai_score": "scores_spliceai.csv",
    "splicetransformer_score": "scores_splicetx.csv",
    "mmsplice_score": "scores_mmsplice.csv",
}
EXPECTED_METRICS = {
    "pangolin": (27_733, 0.8882, 0.4208, 0.1060),
    "spliceai": (27_733, 0.8193, 0.3208, 0.0946),
    "splicetransformer": (27_733, 0.7857, 0.3174, 0.0839),
    "mmsplice": (27_733, 0.7582, 0.2558, 0.0897),
    "spanr": (27_663, 0.7479, 0.2279, 0.0681),
}


def validate_inputs(data_root: Path) -> dict[str, str]:
    expected = materialized_sources()
    for relative, record in expected.items():
        path = data_root / relative
        if not path.is_file():
            raise RuntimeError(f"missing pinned input: {path}")
        observed_bytes = path.stat().st_size
        if observed_bytes != record["bytes"]:
            raise RuntimeError(
                f"{path}: expected {record['bytes']} bytes, "
                f"got {observed_bytes}"
            )
        observed = sha256(path)
        if observed != record["sha256"]:
            raise RuntimeError(
                f"{path}: expected sha256 {record['sha256']}, got {observed}"
            )
    return {
        relative: record["sha256"]
        for relative, record in expected.items()
    }


def nullable_bool(series: pd.Series) -> pd.Series:
    mapping = {
        True: True,
        False: False,
        "TRUE": True,
        "FALSE": False,
        "True": True,
        "False": False,
        1: True,
        0: False,
    }
    return series.map(mapping).astype("boolean")


def complement(base: str) -> str:
    return base.translate(COMPLEMENT)


def infer_exon_orientation(
    frame: pd.DataFrame, reference_bases: list[str]
) -> dict[str, str]:
    probes = frame[["ensembl_id", "ref_allele"]].copy()
    probes["reference_base"] = reference_bases
    probes["direct"] = (
        probes.reference_base == probes.ref_allele.str.upper()
    ).astype(int)
    probes["reverse_complement"] = (
        probes.reference_base
        == probes.ref_allele.str.upper().map(complement)
    ).astype(int)
    counts = probes.groupby("ensembl_id", sort=False)[
        ["direct", "reverse_complement"]
    ].sum()
    if (counts.direct.eq(counts.reverse_complement)).any():
        tied = counts[counts.direct.eq(counts.reverse_complement)].index.tolist()
        raise RuntimeError(f"ambiguous liftover orientation: {tied[:10]}")
    return {
        exon: (
            "identity"
            if row.direct > row.reverse_complement
            else "reverse_complement"
        )
        for exon, row in counts.iterrows()
    }


def validate_regions(
    frame: pd.DataFrame,
    genome: Fasta,
    orientation: dict[str, str],
) -> dict[str, dict]:
    validations = {}
    for exon, rows in frame.groupby("ensembl_id", sort=False):
        row = rows.iloc[0]
        region_start = int(row.start_hg38_0based)
        region_end = int(row.end_hg38_0based)
        forward_sequence = str(
            genome[row.chr][region_start:region_end]
        ).upper()
        hg38_strand = str(row.strand)
        if orientation[exon] == "reverse_complement":
            hg38_strand = "-" if hg38_strand == "+" else "+"
        oriented_sequence = (
            forward_sequence
            if hg38_strand == "+"
            else reverse_complement(forward_sequence)
        )
        alignment = align_sequences(
            str(row.natural_seq).upper(), oriented_sequence
        )
        if alignment["status"] == "nontrivial":
            raise RuntimeError(f"{exon}: unexpected lifted-region alignment")
        for variant in rows.itertuples(index=False):
            assay_offset = int(variant.rel_position) - 1
            hg38_offset = (
                int(variant.snp_position_hg38_1based) - 1 - region_start
                if hg38_strand == "+"
                else region_end - int(variant.snp_position_hg38_1based)
            )
            if alignment["mapping"][assay_offset] != hg38_offset:
                raise RuntimeError(
                    f"{variant.id}: assay and hg38 variant offsets disagree"
                )
        validations[exon] = {
            **alignment,
            "hg38_strand": hg38_strand,
        }
    return validations


def load_inputs(data_root: Path) -> pd.DataFrame:
    source_path = data_root / "source/snv_data_clean.txt"
    source = pd.read_csv(source_path, sep="\t", low_memory=False)
    source["source_index"] = np.arange(len(source), dtype=np.int64)
    source = source[source.category.eq("mutant")].copy()
    if len(source) != EXPECTED_ROWS or source.id.duplicated().any():
        raise RuntimeError("unexpected MFASS mutant row set")

    labels = pd.read_csv(data_root / "published/mfass_labels.csv").rename(
        columns={
            "sdv": "published_label",
            "dpsi": "published_dpsi",
            "spanr": "spanr_score",
        }
    )
    if labels.id.duplicated().any() or set(labels.id) != set(source.id):
        raise RuntimeError("published label IDs do not match MFASS")
    frame = source.merge(labels, on="id", validate="one_to_one")
    for column, filename in SCORE_FILES.items():
        scores = pd.read_csv(data_root / "published" / filename).rename(
            columns={"score": column}
        )
        if scores.id.duplicated().any() or scores[column].isna().any():
            raise RuntimeError(f"invalid score file: {filename}")
        frame = frame.merge(scores, on="id", validate="one_to_one")

    source_label = nullable_bool(frame.strong_lof)
    published_label = nullable_bool(frame.published_label)
    if not source_label.equals(published_label):
        raise RuntimeError("published labels do not reproduce MFASS strong_lof")
    if not np.allclose(
        frame.v2_dpsi,
        frame.published_dpsi,
        equal_nan=True,
        atol=1e-12,
        rtol=0,
    ):
        raise RuntimeError("published dPSI does not reproduce MFASS")
    frame["source_label"] = source_label
    return frame


def build_frames(
    frame: pd.DataFrame, fasta: Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    genome = Fasta(fasta, one_based_attributes=False, rebuild=True)
    reference_bases = [
        str(genome[row.chr][int(row.snp_position_hg38_1based) - 1]).upper()
        for row in frame.itertuples(index=False)
    ]
    orientation = infer_exon_orientation(frame, reference_bases)
    region_validations = validate_regions(frame, genome, orientation)

    compact_rows = []
    audit_rows = []
    actions: dict[str, int] = {}
    reverse_exons = {
        exon for exon, value in orientation.items()
        if value == "reverse_complement"
    }

    for row, reference_base in zip(
        frame.itertuples(index=False), reference_bases
    ):
        source_ref = str(row.ref_allele).upper()
        source_alt = str(row.alt_allele).upper()
        exon_orientation = orientation[row.ensembl_id]
        region_validation = region_validations[row.ensembl_id]
        hg38_ref = (
            source_ref
            if exon_orientation == "identity"
            else complement(source_ref)
        )
        hg38_alt = (
            source_alt
            if exon_orientation == "identity"
            else complement(source_alt)
        )
        effect_sign = 1.0
        action = exon_orientation
        if reference_base == hg38_alt:
            hg38_ref, hg38_alt = hg38_alt, hg38_ref
            effect_sign = -1.0
            action += "_reference_swap"
        elif reference_base != hg38_ref:
            raise RuntimeError(
                f"{row.id}: hg38 base {reference_base} matches neither allele"
            )
        actions[action] = actions.get(action, 0) + 1

        assay_ref = str(row.natural_seq).upper()
        assay_alt = str(row.original_seq).upper()
        variant_offset = int(row.rel_position) - 1
        differences = [
            index
            for index, (left, right) in enumerate(zip(assay_ref, assay_alt))
            if left != right
        ]
        transcript_ref = (
            source_ref if row.strand == "+" else complement(source_ref)
        )
        transcript_alt = (
            source_alt if row.strand == "+" else complement(source_alt)
        )
        if (
            len(assay_ref) != 170
            or len(assay_alt) != 170
            or differences != [variant_offset]
            or assay_ref[variant_offset] != transcript_ref
            or assay_alt[variant_offset] != transcript_alt
        ):
            raise RuntimeError(f"{row.id}: invalid assay sequence pair")

        hg38_strand = region_validation["hg38_strand"]
        assay_exon_start = int(
            row.intron1_len if row.strand == "+" else row.intron2_len
        )
        assay_exon_end = assay_exon_start + int(row.exon_len)
        source_effect = (
            None if pd.isna(row.v2_dpsi) else float(row.v2_dpsi)
        )
        normalized_effect = (
            None if source_effect is None else effect_sign * source_effect
        )
        normalized_label = (
            None if normalized_effect is None else normalized_effect <= -0.5
        )
        source_rep1 = (
            None if pd.isna(row.v2_dpsi_R1) else float(row.v2_dpsi_R1)
        )
        source_rep2 = (
            None if pd.isna(row.v2_dpsi_R2) else float(row.v2_dpsi_R2)
        )
        normalized_sequence, normalized_alt_sequence = (
            (assay_ref, assay_alt)
            if effect_sign == 1
            else (assay_alt, assay_ref)
        )
        ref_inclusion, alt_inclusion = (
            (row.nat_v2_index, row.v2_index)
            if effect_sign == 1
            else (row.v2_index, row.nat_v2_index)
        )
        expected_ref = (
            hg38_ref if hg38_strand == "+" else complement(hg38_ref)
        )
        expected_alt = (
            hg38_alt if hg38_strand == "+" else complement(hg38_alt)
        )
        if (
            normalized_sequence[variant_offset] != expected_ref
            or normalized_alt_sequence[variant_offset] != expected_alt
        ):
            raise RuntimeError(f"{row.id}: normalized assay allele mismatch")

        compact_rows.append({
            "split": "test",
            "source_index": int(row.source_index),
            "pair_id": str(row.id),
            "component_id": str(row.ensembl_id),
            "sequence": normalized_sequence,
            "alt_sequence": normalized_alt_sequence,
            "label": normalized_label,
            "delta_psi": normalized_effect,
            "delta_psi_rep1": (
                None if source_rep1 is None else effect_sign * source_rep1
            ),
            "delta_psi_rep2": (
                None if source_rep2 is None else effect_sign * source_rep2
            ),
            "ref_inclusion": (
                None if pd.isna(ref_inclusion) else float(ref_inclusion)
            ),
            "alt_inclusion": (
                None if pd.isna(alt_inclusion) else float(alt_inclusion)
            ),
            "chrom": str(row.chr),
            "position": int(row.snp_position_hg38_1based),
            "ref": hg38_ref,
            "alt": hg38_alt,
            "strand": hg38_strand,
            "variant_offset": variant_offset,
            "exon_start": assay_exon_start,
            "exon_end": assay_exon_end,
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
            "distance_to_splice_site": abs(int(row.rel_position_feature)),
            "is_evaluable": source_effect is not None,
        })

    validation_counts: dict[str, int] = {}
    validation_exons: dict[str, list[str]] = {}
    for exon, validation in region_validations.items():
        status = validation["status"]
        validation_counts[status] = validation_counts.get(status, 0) + 1
        if status != "exact":
            validation_exons.setdefault(status, []).append(exon)
    compact = typed_compact(pd.DataFrame(compact_rows))
    source = frame.reset_index(drop=True).rename(columns={
        "sequence": "source_sequence",
        "label": "source_region",
        "strand": "source_strand",
    })
    source = source.drop(columns=["source_index"])
    audit = pd.DataFrame(audit_rows)
    audit["is_evaluable"] = audit["is_evaluable"].astype("boolean")
    full = pd.concat([compact, source, audit], axis=1)
    if full.columns.duplicated().any():
        duplicates = full.columns[full.columns.duplicated()].tolist()
        raise RuntimeError(f"duplicate full-output columns: {duplicates}")
    return compact, full, {
        "actions": actions,
        "reverse_complement_exons": sorted(reverse_exons),
        "region_validation_counts": validation_counts,
        "nonexact_region_exons": validation_exons,
    }


def typed_compact(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame[COMPACT_COLUMNS]
    frame["label"] = frame["label"].astype("boolean")
    for column in (
        "delta_psi",
        "delta_psi_rep1",
        "delta_psi_rep2",
        "ref_inclusion",
        "alt_inclusion",
    ):
        frame[column] = frame[column].astype("float32")
    return frame


def published_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    score_columns = {
        "pangolin": "pangolin_score",
        "spliceai": "spliceai_score",
        "splicetransformer": "splicetransformer_score",
        "mmsplice": "mmsplice_score",
        "spanr": "spanr_score",
    }
    rows = []
    for method, score_column in score_columns.items():
        subset = frame[
            frame.source_label.notna() & frame[score_column].notna()
        ]
        labels = subset.source_label.astype(bool).to_numpy()
        scores = subset[score_column].to_numpy(float)
        loss = -subset.v2_dpsi.to_numpy(float)
        spearman = pd.Series(scores).rank().corr(pd.Series(loss).rank())
        rows.append({
            "method": method,
            "n": len(subset),
            "positives": int(labels.sum()),
            "auroc": roc_auc_score(labels, scores),
            "average_precision": average_precision_score(labels, scores),
            "spearman_vs_loss": spearman,
            "score_column": score_column,
            "source": (
                "SpliceConsensus "
                "cdb411fd5fc30dd811284f7c59ce1b5acf4e9c82"
            ),
        })
    metrics = pd.DataFrame(rows)
    for row in metrics.itertuples(index=False):
        expected = EXPECTED_METRICS[row.method]
        actual = (
            row.n,
            round(row.auroc, 4),
            round(row.average_precision, 4),
            round(row.spearman_vs_loss, 4),
        )
        if actual != expected:
            raise RuntimeError(
                f"{row.method}: expected {expected}, reproduced {actual}"
            )
    return metrics


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=root / "data")
    parser.add_argument("--output", type=Path, default=root / "MFASS")
    parser.add_argument(
        "--full",
        action="store_true",
        help="also write all source, score, and audit columns",
    )
    args = parser.parse_args()

    input_hashes = validate_inputs(args.data_root)
    frame = load_inputs(args.data_root)
    fasta = (
        args.data_root / "reference"
        / "GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta"
    )
    compact, full, _orientation = build_frames(frame, fasta)
    metrics = published_metrics(frame)
    if validate_inputs(args.data_root) != input_hashes:
        raise RuntimeError("pinned input changed during processing")
    if (
        len(compact) != EXPECTED_ROWS
        or int(compact.delta_psi.notna().sum()) != EXPECTED_EVALUABLE
        or int(compact.label.fillna(False).sum()) != EXPECTED_POSITIVES
        or not compact.split.eq("test").all()
    ):
        raise RuntimeError("unexpected final MFASS counts")

    compact_path = args.output / "mfass.parquet"
    full_path = args.output / "mfass-full.parquet"
    metrics_path = args.output / "published_metrics.parquet"
    if not args.full and (full_path.exists() or full_path.is_symlink()):
        if full_path.is_dir() and not full_path.is_symlink():
            raise RuntimeError(
                f"refusing compact output with directory at {full_path}"
            )
        full_path.unlink()
    write_parquet(compact, compact_path)
    outputs = [compact_path]
    if args.full:
        write_parquet(full, full_path)
        outputs.append(full_path)
    write_parquet(metrics, metrics_path)
    outputs.append(metrics_path)
    names = ", ".join(path.name for path in outputs)
    print(f"wrote {len(compact):,} rows to {names}")


if __name__ == "__main__":
    main()
