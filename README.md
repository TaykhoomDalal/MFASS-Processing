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

`pair_id` is the unique row key. `sequence` is not: variants from the same
exon intentionally reuse the same reference assay construct. There are 2,199
distinct `sequence` values among 28,972 rows, and these rows must not be
deduplicated.

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
The machine-readable version of this row, sequence, coordinate, output, and
evaluation contract is embedded in
[`manifest.json`](manifest.json).

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

[`manifest.json`](manifest.json) is the single tracked provenance authority:

- `sources` owns source identities, URLs, hashes, and byte sizes and is
  consumed by acquisition;
- `outputs` records golden hashes, bytes, rows, and Arrow schemas; and
- `contracts` records the verifier-consumed input-pair, target, evaluation,
  identifier, coordinate, and sequence semantics, including ordered metric
  membership.

Generated files under `data/` and `MFASS/` are ignored and are not used as
mutable provenance evidence. There are no separate source/input/output
manifest variants. Normal builds do not rewrite the tracked manifest. After
an intentional reviewed output change, first run a full build, then refresh
its generated evidence with:

```bash
python scripts/build_manifest.py --full
```

The manifest builder refuses compact-only registration. It stages a candidate
manifest, runs full verification against compact, full, and metrics outputs,
and installs it only after verification succeeds. Thus a compact build can
never hash or register a leftover `mfass-full.parquet`.

Network acquisition uses three bounded attempts, a 120-second socket timeout,
and bounded backoff. Downloads are checked by expected bytes and digest;
reused local files receive the same materialized checks. The compressed NCBI
FASTA is checked by its published MD5 before decompression and by SHA-256
afterward. Downloads, symlinks, Parquets, JSON, and the exact two-file Hugging
Face release are staged before replacement so incomplete files do not become
published artifacts.

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

The release builder verifies the compact output, stages a fresh exact
allowlist, and leaves only `README.md` and `mfass.parquet` (plus Git metadata)
in the Hugging Face repository. It refuses a dirty processing repository and
must be run after the parent processing commit. The rendered card embeds that
exact commit, its raw root-manifest URL, and the committed manifest SHA-256;
no HF-side manifest file is created.

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

The processing code is MIT licensed. No license is asserted for the combined
data artifacts (`NOASSERTION`); see [NOTICE.md](NOTICE.md) for authoritative
upstream links and the unresolved MFASS redistribution question.
