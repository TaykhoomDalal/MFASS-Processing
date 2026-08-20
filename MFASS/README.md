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

## `mfass-full.parquet`

Generated with:

```bash
python MFASS/process.py --full
```

The full file starts with the compact contract and adds every source column,
released row-level baseline score, and liftover/alignment audit field. It does
not store generated 2,114 bp windows.

## `published_metrics.parquet`

This file reproduces the released Pangolin, SpliceAI, SpliceTransformer,
MMSplice, and SPANR benchmark metrics against the original MFASS label.

## Verify

```bash
python scripts/verify_outputs.py --full
```
