#!/usr/bin/env python3
import os
import shutil
import unittest
from pathlib import Path

from scripts.common import isolated_fasta


ROOT = Path(__file__).resolve().parents[1]


class FastaIndexTest(unittest.TestCase):
    def setUp(self) -> None:
        self.scratch = (
            ROOT.parent
            / f".mfass-fasta-index-{os.getpid()}-{self._testMethodName}"
        )
        self.scratch.mkdir()
        self.addCleanup(shutil.rmtree, self.scratch, True)

    def test_newer_bogus_adjacent_index_is_never_trusted(self) -> None:
        fasta = self.scratch / "reference.fasta"
        fasta.write_text(">chr1\nACGTACGT\n>chr2\nTTGCAACC\n")
        bogus = fasta.with_suffix(fasta.suffix + ".fai")
        bogus.write_text("chr1\t999\t0\t999\t1000\n")
        future = fasta.stat().st_mtime + 3600
        os.utime(bogus, (future, future))
        before = bogus.read_bytes()

        with isolated_fasta(
            fasta, one_based_attributes=False
        ) as genome:
            self.assertEqual(str(genome["chr1"][:]), "ACGTACGT")
            self.assertEqual(str(genome["chr2"][2:6]), "GCAA")

        self.assertEqual(bogus.read_bytes(), before)
        self.assertFalse(list(self.scratch.glob(".reference.fasta.index-*")))


if __name__ == "__main__":
    unittest.main()
