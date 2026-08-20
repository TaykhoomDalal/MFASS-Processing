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
|-- published_metrics.parquet
`-- manifest.json
```

`mfass.parquet` contains 28,972 variants and the 23-column public contract.
`split` is the first column and is always `test`, matching the DART-Eval
schema. The paired sequences are the exact 170 bp transcript-oriented assay
constructs, ordered so `sequence` carries the canonical hg38 reference allele
and `alt_sequence` carries the alternate allele.

`mfass-full.parquet` starts with the same public contract, then preserves all
54 source columns, released baseline scores, and orientation/alignment audit
fields. The three colliding source names are exposed as `source_sequence`,
`source_region`, and `source_strand`.

Neither output stores an arbitrary fixed genomic window. Any desired context
can be extracted from `chrom`, 1-based `position`, `ref`, and `alt` using the
pinned GRCh38 reference.

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

## Reproducibility

The measurement table is pinned to KosuriLab/MFASS commit
`3b1b2bdaea828283508ba22cdd8d0c431ea70dea`.

Baseline scores are pinned to `brhanufen/spliceconsensus` commit
`cdb411fd5fc30dd811284f7c59ce1b5acf4e9c82`.

The reference is NCBI's complete GRCh38 no-alt analysis set
`GCA_000001405.15`, using UCSC chromosome names. Every download is checked by
SHA-256.

## Build

```bash
bash scripts/create_environment.sh
mamba activate mfass-processing
bash scripts/run_all.sh
```

To reuse existing inputs:

```bash
python scripts/download_data.py \
  --reuse-source /path/to/snv_data_clean.txt \
  --reuse-benchmark-root /path/to/spliceconsensus \
  --reuse-fasta /path/to/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta
python MFASS/process.py --full
python scripts/verify_outputs.py --full
```

Build the compact Hugging Face release:

```bash
python scripts/build_hf_release.py --output /path/to/mfass
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

Chong, R. et al. *A multiplexed assay for exon recognition reveals that an
unappreciated fraction of rare genetic variants cause large-effect splicing
disruptions*. Molecular Cell 73, 183-194.e8 (2019).
https://doi.org/10.1016/j.molcel.2018.10.037

## License

The processing code is MIT licensed. The MFASS measurements and external
scores retain their original terms; see [NOTICE.md](NOTICE.md).
