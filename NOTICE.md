# Data and rights notice

The processing code in this repository is MIT licensed under
[`LICENSE`](LICENSE). That license does not assign rights to downloaded source
data or to the generated data artifacts.

## Artifact-level status

The following data artifacts are marked **`NOASSERTION`**:

- the canonical MFASS measurement input;
- the released MFASS labels and baseline-score inputs;
- `mfass.parquet`, `mfass-full.parquet`, and
  `published_metrics.parquet`; and
- the compact Hugging Face release.

`NOASSERTION` is a rights-status marker, not a license and not a restriction
invented by this repository.

## Authoritative upstream links

### MFASS measurements and labels

The pinned KosuriLab/MFASS tree contains no license file or repository license
declaration:

- <https://github.com/KosuriLab/MFASS/tree/3b1b2bdaea828283508ba22cdd8d0c431ea70dea>
- GEO GSE120695:
  <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE120695>
- Cheung et al. (2019):
  <https://doi.org/10.1016/j.molcel.2018.10.037>

Public availability and required citation do not by themselves establish a
right to redistribute the processed measurements or derivatives. The
unresolved rights question is whether the MFASS rights holders authorize
third-party redistribution in this repackaged form; obtain clarification or
permission when required.

### Released baseline scores

The pinned `brhanufen/spliceconsensus` repository declares MIT:

- repository:
  <https://github.com/brhanufen/spliceconsensus/tree/cdb411fd5fc30dd811284f7c59ce1b5acf4e9c82>
- license:
  <https://github.com/brhanufen/spliceconsensus/blob/cdb411fd5fc30dd811284f7c59ce1b5acf4e9c82/LICENSE>

The score files also reflect MFASS labels, model outputs, and cited upstream
methods. This repository does not infer a broader license for those underlying
materials from the SpliceConsensus repository license.

### GRCh38 reference

The reference is NCBI assembly `GCA_000001405.15`:

- assembly:
  <https://www.ncbi.nlm.nih.gov/datasets/genome/GCA_000001405.15/>
- NCBI policies:
  <https://www.ncbi.nlm.nih.gov/home/about/policies/>

NCBI states that it places no restrictions on molecular-data use or
distribution, while also stating that it cannot transfer rights that may be
claimed by submitters. This repository therefore makes no additional rights
assertion.
