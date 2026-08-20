#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import sha256
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


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full",
        action="store_true",
        help="require and verify mfass-full.parquet",
    )
    args = parser.parse_args()
    data_root = root / "data"
    before = validate_inputs(data_root)
    source = load_inputs(data_root)
    fasta = (
        data_root / "reference"
        / "GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta"
    )
    expected_compact, expected_full, orientation = build_frames(source, fasta)
    expected_metrics = published_metrics(source)
    after = validate_inputs(data_root)
    require(before == after, "pinned input changed during verification")

    compact_path = root / "MFASS/mfass.parquet"
    full_path = root / "MFASS/mfass-full.parquet"
    metrics_path = root / "MFASS/published_metrics.parquet"
    manifest_path = root / "MFASS/manifest.json"
    for path in (compact_path, metrics_path, manifest_path):
        require(path.is_file(), f"missing output: {path}")

    compact = pd.read_parquet(compact_path)
    metrics = pd.read_parquet(metrics_path)
    compare_frames("compact parquet", compact, expected_compact)
    compare_frames("published metrics", metrics, expected_metrics)

    require(
        compact.columns.tolist() == COMPACT_COLUMNS,
        "unexpected compact column contract",
    )
    require(len(compact) == EXPECTED_ROWS, "unexpected compact row count")
    require(
        int(compact.delta_psi.notna().sum()) == EXPECTED_EVALUABLE,
        "unexpected evaluable row count",
    )
    require(
        int(compact.label.fillna(False).sum()) == EXPECTED_POSITIVES,
        "unexpected positive-label count",
    )
    require(compact.split.eq("test").all(), "unexpected split value")
    require(
        set(metrics.method) == {
            "pangolin",
            "spliceai",
            "splicetransformer",
            "mmsplice",
            "spanr",
        }
        and len(metrics) == 5
        and metrics.method.is_unique,
        "published metrics must contain exactly five unique methods",
    )

    manifest = json.loads(manifest_path.read_text())
    verify_full = args.full or "mfass-full.parquet" in manifest["outputs"]
    if verify_full:
        require(full_path.is_file(), f"missing output: {full_path}")
        full = pd.read_parquet(full_path)
        compare_frames("full parquet", full, expected_full)
    if args.full:
        require(
            "mfass-full.parquet" in manifest["outputs"],
            "manifest does not include full output",
        )
    expected_inputs = {
        **{
            f"data/{relative}": digest
            for relative, digest in before.items()
        },
        "data/input_manifest.json": sha256(data_root / "input_manifest.json"),
    }
    require(manifest["inputs"] == expected_inputs, "input manifest mismatch")
    require(
        manifest["orientation"] == orientation,
        "orientation manifest mismatch",
    )
    require(manifest["rows"] == EXPECTED_ROWS, "manifest row mismatch")
    require(
        manifest["evaluable_rows"] == EXPECTED_EVALUABLE,
        "manifest evaluable-row mismatch",
    )
    require(
        manifest["source_positive_labels"] == EXPECTED_POSITIVES,
        "manifest positive-label mismatch",
    )
    require(
        manifest["normalized_positive_labels"] == EXPECTED_POSITIVES,
        "manifest normalized-label mismatch",
    )
    require(
        manifest["compact_columns"] == COMPACT_COLUMNS,
        "manifest compact-column mismatch",
    )
    for name, expected in manifest["outputs"].items():
        observed = sha256(root / "MFASS" / name)
        require(
            observed == expected,
            f"{name}: expected sha256 {expected}, got {observed}",
        )
    print(
        "reconstructed and verified the 23-column MFASS contract"
        f"{', full source/audit output' if verify_full else ''}, "
        "28,972 variants, and five published baselines"
    )


if __name__ == "__main__":
    main()
