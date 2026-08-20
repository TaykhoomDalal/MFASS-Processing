#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(target)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    subprocess.run(
        [sys.executable, str(root / "scripts/verify_outputs.py")],
        check=True,
    )
    copy(root / "MFASS/mfass.parquet", args.output / "mfass.parquet")
    copy(root / "HF_DATASET_CARD.md", args.output / "README.md")
    for omitted in (
        "mfass-assay.parquet",
        "mfass-hg38.parquet",
        "mfass-full.parquet",
        "published_metrics.parquet",
    ):
        path = args.output / omitted
        if path.exists():
            path.unlink()
    print(f"built Hugging Face release under {args.output}")


if __name__ == "__main__":
    main()
