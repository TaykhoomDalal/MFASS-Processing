#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from scripts import build_hf_release as hf


REPOSITORY = Path(__file__).resolve().parents[1]


def tree_digest(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(path.rglob("*")):
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        if item.is_symlink():
            digest.update(os.readlink(item).encode())
        elif item.is_file():
            digest.update(item.read_bytes())
    return digest.hexdigest()


class CompactAuthorityTest(unittest.TestCase):
    def test_committed_manifest_authorizes_current_compact(self) -> None:
        expected = hf.load_compact_output_authority()
        hf.verify_compact_artifact(
            REPOSITORY / hf.COMPACT_OUTPUT_PATH,
            expected,
            "processing compact output",
        )


class CardReferenceTest(unittest.TestCase):
    def test_every_processing_reference_must_be_main(self) -> None:
        card = (
            REPOSITORY / "MFASS/HF_DATASET_CARD.md"
        ).read_text(encoding="utf-8")
        invalid_refs = [
            "deadbee",
            "ABCDEF0123456789ABCDEF0123456789ABCDEF01",
            "v1.0.0",
            "release",
            "{{PROCESSING_COMMIT}}",
        ]
        for reference in invalid_refs:
            with self.subTest(reference=reference):
                url = (
                    "https://github.com/TaykhoomDalal/"
                    f"MFASS-Processing/blob/{reference}/README.md"
                )
                with self.assertRaisesRegex(
                    RuntimeError, "non-main processing references"
                ):
                    hf.validate_processing_urls(card + "\n" + url)

    def test_abbreviated_commit_url_is_rejected(self) -> None:
        card = (
            REPOSITORY / "MFASS/HF_DATASET_CARD.md"
        ).read_text(encoding="utf-8")
        with self.assertRaisesRegex(
            RuntimeError, "commit-specific processing URL"
        ):
            hf.validate_processing_urls(
                card
                + "\nhttps://github.com/TaykhoomDalal/"
                "MFASS-Processing/commit/deadbee"
            )

    def test_current_card_uses_only_main(self) -> None:
        card = hf.render_card()
        self.assertIn("Rockie Chong et al.", card)


class HuggingFaceReleaseRaceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.scratch = (
            REPOSITORY.parent
            / (
                f".mfass-hf-race-test-{os.getpid()}-"
                f"{self._testMethodName}"
            )
        )
        if self.scratch.exists() or self.scratch.is_symlink():
            raise RuntimeError(f"scratch path exists: {self.scratch}")
        self.addCleanup(self.cleanup_scratch)
        self.processing = self.scratch / "processing"
        source_directory = self.processing / "MFASS"
        source_directory.mkdir(parents=True)
        self.source = source_directory / "mfass.parquet"
        self.source.write_bytes(b"verified compact parquet\n")
        shutil.copyfile(
            REPOSITORY / "MFASS/HF_DATASET_CARD.md",
            source_directory / "HF_DATASET_CARD.md",
        )
        self.expected = {
            "bytes": self.source.stat().st_size,
            "sha256": hashlib.sha256(self.source.read_bytes()).hexdigest(),
        }
        self.target = self.scratch / "target"
        self.target.mkdir()
        subprocess.run(
            ["git", "-C", str(self.target), "init", "--quiet"],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(self.target),
                "config",
                "user.name",
                "MFASS test",
            ],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(self.target),
                "config",
                "user.email",
                "mfass-test@example.invalid",
            ],
            check=True,
        )
        (self.target / "README.md").write_bytes(b"baseline readme\n")
        (self.target / "mfass.parquet").write_bytes(
            b"baseline parquet\n"
        )
        (self.target / ".gitattributes").write_bytes(
            b"*.parquet -text\n"
        )
        subprocess.run(
            ["git", "-C", str(self.target), "add", "."],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(self.target),
                "commit",
                "--quiet",
                "-m",
                "baseline",
            ],
            check=True,
        )
        self.before = {
            name: (self.target / name).read_bytes()
            for name in ("README.md", "mfass.parquet", ".gitattributes")
        }
        self.git_before = tree_digest(self.target / ".git")

    def cleanup_scratch(self) -> None:
        if self.scratch.exists():
            shutil.rmtree(self.scratch)

    def assert_target_unchanged(self) -> None:
        for name, content in self.before.items():
            self.assertEqual(
                (self.target / name).read_bytes(),
                content,
            )
        self.assertEqual(
            tree_digest(self.target / ".git"),
            self.git_before,
        )
        self.assertFalse(
            list(
                self.target.parent.glob(
                    f".{self.target.name}.mfass-staging-*"
                )
            )
        )

    def test_new_release_installs_verified_bytes(self) -> None:
        output = self.scratch / "new-target"
        with mock.patch.object(hf, "ROOT", self.processing):
            hf.build_release(output, self.expected)
        self.assertEqual(
            (output / "mfass.parquet").read_bytes(),
            self.source.read_bytes(),
        )
        card = (output / "README.md").read_text(encoding="utf-8")
        self.assertFalse(hf.COMMIT_SPECIFIC_PROCESSING_URL.search(card))
        self.assertTrue(
            all(link in card for link in hf.STABLE_CARD_LINKS)
        )

    def test_existing_git_target_updates_without_metadata_changes(
        self,
    ) -> None:
        attributes = (self.target / ".gitattributes").read_bytes()
        with mock.patch.object(hf, "ROOT", self.processing):
            hf.build_release(self.target, self.expected)
        self.assertEqual(
            (self.target / "mfass.parquet").read_bytes(),
            self.source.read_bytes(),
        )
        self.assertEqual(
            (self.target / "README.md").read_bytes(),
            (self.processing / "MFASS/HF_DATASET_CARD.md").read_bytes(),
        )
        self.assertEqual(
            (self.target / ".gitattributes").read_bytes(),
            attributes,
        )
        self.assertEqual(
            tree_digest(self.target / ".git"),
            self.git_before,
        )

    def test_source_replacement_during_copy_is_rejected(self) -> None:
        original_verify = hf.verify_compact_artifact
        raced = False

        def verify_then_replace(path, expected, context):
            nonlocal raced
            original_verify(path, expected, context)
            if (
                not raced
                and Path(path) == self.source
                and context == "processing compact output"
            ):
                raced = True
                replacement = self.source.with_suffix(".race")
                replacement.write_bytes(
                    b"concurrent ignored replacement\n"
                )
                replacement.replace(self.source)

        with (
            mock.patch.object(hf, "ROOT", self.processing),
            mock.patch.object(
                hf,
                "verify_compact_artifact",
                side_effect=verify_then_replace,
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "changed while being copied",
            ):
                hf.build_release(self.target, self.expected)
        self.assertTrue(raced)
        self.assert_target_unchanged()

    def test_installed_corruption_rolls_back_git_target(self) -> None:
        original_verify = hf.verify_compact_artifact
        corrupted = False

        def corrupt_installed(path, expected, context):
            nonlocal corrupted
            path = Path(path)
            if (
                not corrupted
                and path == self.target / "mfass.parquet"
                and context == "installed compact output"
            ):
                corrupted = True
                path.write_bytes(b"concurrent installed replacement\n")
            original_verify(path, expected, context)

        with (
            mock.patch.object(hf, "ROOT", self.processing),
            mock.patch.object(
                hf,
                "verify_compact_artifact",
                side_effect=corrupt_installed,
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "installed compact output",
            ):
                hf.build_release(self.target, self.expected)
        self.assertTrue(corrupted)
        self.assert_target_unchanged()

    def test_concurrent_writers_do_not_mix_or_clobber_release(self) -> None:
        output = self.scratch / "concurrent-target"
        original_copy = hf.copy_verified_compact
        active = 0
        maximum_active = 0
        active_lock = threading.Lock()
        errors = []

        def slow_copy(*args, **kwargs):
            nonlocal active, maximum_active
            with active_lock:
                active += 1
                maximum_active = max(maximum_active, active)
            try:
                time.sleep(0.1)
                return original_copy(*args, **kwargs)
            finally:
                with active_lock:
                    active -= 1

        def build() -> None:
            try:
                hf.build_release(output, self.expected)
            except RuntimeError as error:
                errors.append(error)

        with (
            mock.patch.object(hf, "ROOT", self.processing),
            mock.patch.object(
                hf, "copy_verified_compact", side_effect=slow_copy
            ),
        ):
            threads = [threading.Thread(target=build) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(10)
                self.assertFalse(thread.is_alive(), "release writer hung")

        self.assertEqual(maximum_active, 1)
        self.assertEqual(len(errors), 1)
        self.assertEqual(
            (output / "mfass.parquet").read_bytes(),
            self.source.read_bytes(),
        )
        self.assertEqual(
            (output / "README.md").read_bytes(),
            (self.processing / "MFASS/HF_DATASET_CARD.md").read_bytes(),
        )


if __name__ == "__main__":
    unittest.main()
