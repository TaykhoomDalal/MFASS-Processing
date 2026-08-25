#!/usr/bin/env python3
import os
import shutil
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from scripts import build_manifest, common
from MFASS import process
from scripts.common import repository_lock


ROOT = Path(__file__).resolve().parents[1]


class GenerationLockTest(unittest.TestCase):
    def setUp(self) -> None:
        self.scratch = (
            ROOT.parent
            / f".mfass-lock-test-{os.getpid()}-{self._testMethodName}"
        )
        self.scratch.mkdir()
        self.addCleanup(shutil.rmtree, self.scratch, True)

    def test_locked_reader_cannot_observe_mixed_generation(self) -> None:
        first = self.scratch / "first"
        second = self.scratch / "second"
        first.write_text("old")
        second.write_text("old")
        first_replaced = threading.Event()
        release_writer = threading.Event()
        observed = []

        def writer() -> None:
            with repository_lock(self.scratch, exclusive=True):
                first.write_text("new")
                first_replaced.set()
                release_writer.wait(10)
                second.write_text("new")

        def reader() -> None:
            with repository_lock(self.scratch, exclusive=False):
                observed.append((first.read_text(), second.read_text()))

        writer_thread = threading.Thread(target=writer)
        writer_thread.start()
        self.assertTrue(first_replaced.wait(5))
        reader_thread = threading.Thread(target=reader)
        reader_thread.start()
        time.sleep(0.1)
        self.assertTrue(reader_thread.is_alive())
        release_writer.set()
        writer_thread.join(5)
        reader_thread.join(5)
        self.assertFalse(writer_thread.is_alive())
        self.assertFalse(reader_thread.is_alive())
        self.assertEqual(observed, [("new", "new")])

    def test_manifest_entrypoint_serializes_writers(self) -> None:
        active = 0
        maximum_active = 0
        active_lock = threading.Lock()

        def slow_refresh() -> None:
            nonlocal active, maximum_active
            with active_lock:
                active += 1
                maximum_active = max(maximum_active, active)
            try:
                time.sleep(0.1)
            finally:
                with active_lock:
                    active -= 1

        with (
            mock.patch.object(build_manifest, "ROOT", self.scratch),
            mock.patch.object(
                build_manifest, "_main", side_effect=slow_refresh
            ),
        ):
            threads = [
                threading.Thread(target=build_manifest.main)
                for _ in range(2)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(5)
                self.assertFalse(thread.is_alive(), "manifest writer hung")
        self.assertEqual(maximum_active, 1)

    def test_concurrent_run_and_process_serialize_without_deadlock(
        self,
    ) -> None:
        active = 0
        maximum_active = 0
        active_lock = threading.Lock()
        official_started = threading.Event()

        def slow_process() -> None:
            nonlocal active, maximum_active
            with active_lock:
                active += 1
                maximum_active = max(maximum_active, active)
            try:
                time.sleep(0.15)
            finally:
                with active_lock:
                    active -= 1

        def official_run() -> None:
            with repository_lock(self.scratch, exclusive=True):
                official_started.set()
                process.run(lock_held=True)

        def direct_process() -> None:
            official_started.wait(5)
            process.run(lock_held=False)

        with (
            mock.patch.object(process, "ROOT", self.scratch),
            mock.patch.object(
                process, "_main", side_effect=slow_process
            ),
        ):
            official = threading.Thread(target=official_run)
            direct = threading.Thread(target=direct_process)
            official.start()
            direct.start()
            official.join(5)
            direct.join(5)
        self.assertFalse(official.is_alive(), "official run deadlocked")
        self.assertFalse(direct.is_alive(), "direct process deadlocked")
        self.assertEqual(maximum_active, 1)

    def prepare_compact_transaction(self):
        output = self.scratch / "output"
        staging = self.scratch / "staging"
        output.mkdir()
        staging.mkdir()
        old = {
            "mfass.parquet": b"old compact\n",
            "published_metrics.parquet": b"old metrics\n",
            "mfass-full.parquet": b"old full\n",
        }
        for name, content in old.items():
            (output / name).write_bytes(content)
        (staging / "mfass.parquet").write_bytes(b"new compact\n")
        (staging / "published_metrics.parquet").write_bytes(
            b"new metrics\n"
        )
        return output, staging, old

    def test_compact_deletion_failure_rolls_back_every_output(
        self,
    ) -> None:
        output, staging, old = self.prepare_compact_transaction()
        full_path = output / "mfass-full.parquet"
        original_unlink = Path.unlink

        def fail_full_unlink(path, *args, **kwargs):
            if path == full_path:
                raise OSError("injected full deletion failure")
            return original_unlink(path, *args, **kwargs)

        with mock.patch.object(
            Path, "unlink", autospec=True, side_effect=fail_full_unlink
        ):
            with self.assertRaisesRegex(
                OSError, "injected full deletion failure"
            ):
                process.publish_outputs(
                    staging,
                    output,
                    ["mfass.parquet", "published_metrics.parquet"],
                    remove_full=True,
                )
        for name, content in old.items():
            self.assertEqual((output / name).read_bytes(), content)

    def test_compact_reader_waits_for_atomic_generation(self) -> None:
        output, staging, _old = self.prepare_compact_transaction()
        first_replaced = threading.Event()
        release_writer = threading.Event()
        observed = []
        original_replace = common.os.replace

        def pause_after_first(source, destination):
            result = original_replace(source, destination)
            if Path(destination) == output / "mfass.parquet":
                first_replaced.set()
                release_writer.wait(5)
            return result

        def writer() -> None:
            with repository_lock(self.scratch, exclusive=True):
                process.publish_outputs(
                    staging,
                    output,
                    ["mfass.parquet", "published_metrics.parquet"],
                    remove_full=True,
                )

        def reader() -> None:
            first_replaced.wait(5)
            with repository_lock(self.scratch, exclusive=False):
                observed.append((
                    (output / "mfass.parquet").read_bytes(),
                    (output / "published_metrics.parquet").read_bytes(),
                    (output / "mfass-full.parquet").exists(),
                ))

        with mock.patch.object(
            common.os, "replace", side_effect=pause_after_first
        ):
            writer_thread = threading.Thread(target=writer)
            reader_thread = threading.Thread(target=reader)
            writer_thread.start()
            reader_thread.start()
            self.assertTrue(first_replaced.wait(5))
            time.sleep(0.1)
            self.assertTrue(reader_thread.is_alive())
            release_writer.set()
            writer_thread.join(5)
            reader_thread.join(5)
        self.assertFalse(writer_thread.is_alive())
        self.assertFalse(reader_thread.is_alive())
        self.assertEqual(
            observed,
            [(b"new compact\n", b"new metrics\n", False)],
        )


if __name__ == "__main__":
    unittest.main()
