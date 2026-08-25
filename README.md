# MFASS parquet processing

This repository converts the published Multiplexed Functional Assay of
Splicing (MFASS) SNV table into one model-ready parquet.

It:

1. downloads and hashes the canonical measurements, baseline scores, and
   GRCh38 reference;
2. preserves the exact 170 bp assay constructs while normalizing their order
   to the current hg38 reference and alternate alleles;
3. validates every sequence pair, lifted coordinate, strand correction, and
   complete assay-region alignment; and
4. reproduces the published Pangolin, SpliceAI, SpliceTransformer, MMSplice,
   and SPANR metrics.

No source datasets or model weights are committed.

## Outputs

```text
MFASS/
|-- mfass.parquet
|-- mfass-full.parquet       # with --full
`-- published_metrics.parquet
```

`mfass.parquet` contains 28,972 variants and the 23-column public contract.
`split` is the first column and is always `test`, matching the DART-Eval
schema. The paired sequences are the exact 170 bp transcript-oriented assay
constructs, ordered so `sequence` carries the canonical hg38 reference allele
and `alt_sequence` carries the alternate allele.

`mfass-full.parquet` starts with the same public contract, then preserves all
54 source columns, released baseline scores, and orientation/alignment audit
fields, for 92 columns total. The three colliding source names are exposed as
`source_sequence`, `source_region`, and `source_strand`.

`published_metrics.parquet` has five rows and eight columns. It records the
reproduced Pangolin, SpliceAI, SpliceTransformer, MMSplice, and SPANR metrics
and their evaluation counts.

**Row identity:** `pair_id` is the unique row key. `sequence` is not:
variants from the same exon intentionally reuse the same reference assay
construct. There are 2,199 distinct `sequence` values among 28,972 rows.
Never deduplicate these rows by sequence.

Neither output stores an arbitrary fixed genomic window. Any desired context
can be extracted from `chrom`, 1-based `position`, `ref`, and `alt` using the
pinned GRCh38 reference.

The 170 bp assay strings are transcript-oriented. In contrast, `chrom`,
`position`, `ref`, and `alt` describe forward-genome GRCh38. Thus, for a
minus-strand row, the assay base is the complement of the corresponding
forward-genome allele.

Source row order is preserved, and the compact table must be the ordered first
23-column projection of the full table. Within each 170 bp assay string,
`variant_offset` and `exon_start` are zero-based and `exon_end` is exclusive.

## Hugging Face

The compact output is published as
[`Taykhoom/mfass`](https://huggingface.co/datasets/Taykhoom/mfass):

```python
from datasets import load_dataset

dataset = load_dataset("Taykhoom/mfass", split="test")
```

## Source findings

The canonical table contains 28,972 mutant rows from 2,198 exons. Of these,
27,733 have the two replicate measurements used by the published benchmark,
and 1,050 are labeled splice-disrupting variants.

Every sequence pair is exactly 170 bp, differs at one nucleotide, and places
that nucleotide at `variant_offset`. Six exons containing 97 variants changed
orientation during liftover. One additional variant has a genuine
hg19-to-hg38 reference swap; its sequence order, inclusion values, and signed
effects are reversed so the compact contract remains uniformly hg38
reference-to-alternate.

The complete lifted assay interval is also aligned to the measured construct.
There are 2,195 exact exon matches, two one-base assembly substitutions, and
one one-base hg38 insertion. No row is dropped.

## Limitations and explicit exceptions

- The pipeline starts from the authors' processed table and released scores;
  it does not reproduce raw-read processing or rerun the five models.
- The source has 1,239 rows without the two-replicate benchmark label/dPSI.
  Four metrics therefore use 27,733 rows; SPANR has 70 additional missing
  evaluable scores and uses 27,663.
- `split` is always `test`; MFASS does not publish train/validation splits.
- The full-output names `source_sequence`, `source_region`, and
  `source_strand` intentionally resolve collisions with compact columns.
- The six reverse-complemented exons, single reference-swap variant, and three
  non-exact whole-assay alignments described above are reported exceptions,
  not silently corrected or dropped.

## Reproducibility

The measurement table is pinned to KosuriLab/MFASS commit
`3b1b2bdaea828283508ba22cdd8d0c431ea70dea`.

Baseline scores are pinned to `brhanufen/spliceconsensus` commit
`cdb411fd5fc30dd811284f7c59ce1b5acf4e9c82`.

The reference is NCBI's complete GRCh38 no-alt analysis set
`GCA_000001405.15`, using UCSC chromosome names. Every download is checked by
content hash and byte size; the compressed NCBI FASTA is also checked against
NCBI's published MD5.

[`manifest.json`](manifest.json) pins the source files and expected generated
outputs used by the downloader and verifier. Generated files under `data/`
and `MFASS/` remain ignored. After an intentional reviewed output change,
refresh the full candidate build and recorded hashes with:

```bash
python scripts/build_manifest.py --full
```

The manifest builder creates all three outputs in fresh staging, records only
those candidate artifacts, verifies the complete candidate, and then replaces
the outputs and manifest as one rollback-safe transaction. Existing Parquets
are never used as manifest evidence.

Network acquisition uses three bounded attempts, a 120-second socket timeout,
and bounded backoff. Downloads are checked by expected bytes and digest;
reused local files receive the same materialized checks. The compressed NCBI
FASTA is checked by its published MD5 before decompression and by SHA-256
afterward. Downloads, symlinks, Parquets, JSON, and the exact two-file Hugging
Face release are staged before replacement so incomplete files do not become
published artifacts. Multi-file manifest and existing-Git release updates
restore every prior destination if any replacement or final check fails.

## Build

```bash
bash scripts/create_environment.sh --locked
bash scripts/run_all.sh
```

The compact output is the default. Build and verify the exhaustive output
with:

```bash
bash scripts/run_all.sh --full
```

Compact processing removes any stale `mfass-full.parquet`; only `--full`
produces all three supported processing outputs. Manifest refresh remains
full-only.

`run_all.sh` always invokes the named `mfass-processing` Mamba environment.
`environment.yml` is the maintained solve;
`environment-lock-linux-64.txt` plus `requirements-lock.txt` are the audited
Linux lock (the separate requirements lock exists because pyfaidx is installed
with pip).
To select one input root and reuse existing sources consistently:

```bash
bash scripts/run_all.sh --full \
  --data-root /path/to/mfass-data \
  --reuse-source /path/to/snv_data_clean.txt \
  --reuse-benchmark-root /path/to/spliceconsensus \
  --reuse-fasta /path/to/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta
```

Build the compact Hugging Face release:

```bash
python scripts/build_hf_release.py --output /path/to/mfass
```

The release builder loads the compact byte size and SHA-256 from the committed
root manifest, verifies the processing artifact while copying, rechecks fresh
staging, and checks the installed Parquet inside the rollback-protected
transaction. It leaves only `README.md` and `mfass.parquet` (plus Git metadata)
in the Hugging Face repository. It rejects the processing repository, all of
its ancestors and descendants, and nonempty non-Git destinations. An existing
Hugging Face Git target must be its clean worktree root and may contain only
the two release files, `.git`, and optional `.gitattributes`; the attributes
file is preserved byte-for-byte, Git metadata is left in place, and README
plus Parquet updates roll back together on failure. The processing repository
must have a clean committed HEAD so the release is built from a reviewed,
verified tree. The card uses stable `main` repository and documentation links
rather than commit hashes or commit-specific URLs. No manifest file is added
to the Hugging Face repository.

Run the release race and rollback regressions with:

```bash
python -m unittest scripts.test_build_hf_release
```

## Published baseline reproduction

| Method | Rows | AUROC | Average precision |
|---|---:|---:|---:|
| Pangolin | 27,733 | 0.8882 | 0.4208 |
| SpliceAI | 27,733 | 0.8193 | 0.3208 |
| SpliceTransformer | 27,733 | 0.7857 | 0.3174 |
| MMSplice | 27,733 | 0.7582 | 0.2558 |
| SPANR | 27,663 | 0.7479 | 0.2279 |

## Citation

Cheung, R. et al. *A multiplexed assay for exon recognition reveals that an
unappreciated fraction of rare genetic variants cause large-effect splicing
disruptions*. Molecular Cell 73, 183-194.e8 (2019).
https://doi.org/10.1016/j.molcel.2018.10.037

## License

The processing code is MIT licensed. Artifact-level status is `NOASSERTION`;
the Hugging Face `license: other` metadata does not grant a new license. See
[NOTICE.md](NOTICE.md) for authoritative upstream links.

Permission for third-party redistribution of the processed MFASS measurements
and derivatives in this repackaged form remains unresolved; obtain
clarification or permission from the MFASS rights holders before
redistribution.
