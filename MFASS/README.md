# MFASS processing outputs

`process.py` writes one compact dataset and, with `--full`, one exhaustive
provenance dataset.

## `mfass.parquet`

The public file contains 28,972 rows and 23 columns:

- `split` is the first column and is always `test`;
- `sequence` and `alt_sequence` are the exact 170 bp transcript-oriented
  assay constructs in canonical hg38 reference-to-alternate order;
- `delta_psi` is alternate minus reference exon inclusion;
- `label` is nullable and equals `delta_psi <= -0.50`; and
- `chrom`, `position`, `ref`, and `alt` are sufficient to extract any desired
  genomic context from the pinned GRCh38 reference.

`pair_id` is the unique row key. Reference `sequence` values are intentionally
shared by different variants from the same exon and must not be deduplicated.
The assay strings are transcript-oriented, while `ref` and `alt` are
forward-genome GRCh38 bases.

## `mfass-full.parquet`

Generated with:

```bash
bash scripts/run_all.sh --full
```

The full file has 28,972 rows and 92 columns. Its first 23 columns are the
ordered compact contract; it then adds every source column, released row-level
baseline score, and liftover/alignment audit field. It does not store
generated 2,114 bp windows.

## `published_metrics.parquet`

This file has five rows and eight columns. It reproduces the released
Pangolin, SpliceAI, SpliceTransformer, MMSplice, and SPANR benchmark metrics
against the original MFASS label.

## Verify

```bash
bash scripts/run_all.sh --full
```

The verifier checks the tracked top-level `manifest.json`, reconstructs
the outputs, checks that compact is the ordered projection of full, and
independently reconstructs each metric mask and ordered `pair_id` membership.

See `../README.md` and the verifier-consumed contract embedded in
`../manifest.json` for the coordinate, generated-data, environment,
exception, and limitation contracts.
