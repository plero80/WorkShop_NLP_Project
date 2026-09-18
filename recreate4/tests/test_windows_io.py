"""Deterministic retry tests and native Windows atomic replacement checks."""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python_helper"))
import windows_io


def windows_error(code):
    error = OSError(f"synthetic Windows error {code}")
    error.winerror = code
    return error


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.delays = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.delays.append(seconds)
        self.now += seconds


class ReplacementRetryTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.report = mock.Mock()

    def wrap(self, replace, **options):
        return windows_io.retrying_replace(replace, platform_name="nt",
            monotonic=self.clock.monotonic, sleep=self.clock.sleep,
            report=self.report, **options)

    def test_success_preserves_arguments_keyword_arguments_and_result(self):
        result = object()
        replace = mock.Mock(return_value=result)
        source, destination = object(), object()
        self.assertIs(self.wrap(replace)(source, destination, src_dir_fd=3, dst_dir_fd=4), result)
        replace.assert_called_once_with(source, destination, src_dir_fd=3, dst_dir_fd=4)
        self.assertEqual(self.clock.delays, [])
        self.report.assert_not_called()

    def test_each_supported_error_retries_and_succeeds_without_argument_changes(self):
        for code in (5, 32, 33):
            with self.subTest(winerror=code):
                replace = mock.Mock(side_effect=[windows_error(code), None])
                result = self.wrap(replace)(src="status.tmp", dst="status.json")
                self.assertIsNone(result)
                self.assertEqual(replace.call_args_list,
                    [mock.call(src="status.tmp", dst="status.json")] * 2)
        self.assertEqual(self.clock.delays, [.05] * 3)
        self.assertEqual(self.report.call_args_list, [mock.call("status.json", 8.0)] * 3)

    def test_exponential_backoff_is_capped_and_reports_once(self):
        replace = mock.Mock(side_effect=[windows_error(5)] * 6 + [None])
        self.wrap(replace)("source", "target")
        self.assertEqual(self.clock.delays, [.05, .1, .2, .4, .5, .5])
        self.report.assert_called_once_with("target", 8.0)

    def test_persistent_failure_is_bounded_and_raises_original_error(self):
        first, later = windows_error(32), windows_error(5)
        calls = []
        def replace(*args, **kwargs):
            calls.append((args, kwargs))
            raise first if len(calls) == 1 else later
        with self.assertRaises(OSError) as caught:
            self.wrap(replace, max_wait=.23)("source", "target")
        self.assertIs(caught.exception, first)
        self.assertAlmostEqual(sum(self.clock.delays), .23)
        self.assertLessEqual(max(self.clock.delays), .1)
        self.assertEqual(len(calls), 4)
        self.report.assert_called_once()

    def test_nonretryable_windows_errors_do_not_sleep(self):
        for code in (2, 3, 17, 80, 112):
            with self.subTest(winerror=code):
                error = windows_error(code)
                replace = mock.Mock(side_effect=error)
                with self.assertRaises(OSError) as caught:
                    self.wrap(replace)("source", "target")
                self.assertIs(caught.exception, error)
                replace.assert_called_once()
        self.assertEqual(self.clock.delays, [])
        self.report.assert_not_called()

    def test_permission_error_without_windows_code_is_not_retried(self):
        error = PermissionError("ordinary POSIX permission failure")
        with self.assertRaises(PermissionError) as caught:
            self.wrap(mock.Mock(side_effect=error))("source", "target")
        self.assertIs(caught.exception, error)
        self.assertEqual(self.clock.delays, [])

    def test_nonretryable_error_after_initial_lock_propagates_immediately(self):
        error = windows_error(112)
        replace = mock.Mock(side_effect=[windows_error(5), error])
        with self.assertRaises(OSError) as caught:
            self.wrap(replace)("source", "target")
        self.assertIs(caught.exception, error)
        self.assertEqual(self.clock.delays, [.05])

    def test_non_os_exception_is_unchanged(self):
        error = TypeError("wrong arguments")
        with self.assertRaises(TypeError) as caught:
            self.wrap(mock.Mock(side_effect=error))("source", "target")
        self.assertIs(caught.exception, error)
        self.assertEqual(self.clock.delays, [])

    def test_non_windows_function_is_returned_unchanged(self):
        original = mock.Mock()
        self.assertIs(windows_io.retrying_replace(original, platform_name="posix"), original)

    def test_already_wrapped_function_is_not_wrapped_twice(self):
        wrapped = self.wrap(mock.Mock())
        self.assertIs(self.wrap(wrapped), wrapped)

    def test_invalid_retry_limits_are_rejected(self):
        for options in ({"max_wait": 0}, {"max_wait": 10.01}, {"max_wait": float("inf")},
                        {"initial_delay": -1}, {"max_delay": float("nan")}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.wrap(mock.Mock(), **options)

    def test_context_restores_original_after_failure_and_nested_use(self):
        original = os.replace
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            with windows_io.replacement_retries():
                installed = os.replace
                if os.name == "nt":
                    self.assertIsNot(installed, original)
                else:
                    self.assertIs(installed, original)
                with windows_io.replacement_retries():
                    self.assertIs(os.replace, installed)
                self.assertIs(os.replace, installed)
                raise RuntimeError("synthetic")
        self.assertIs(os.replace, original)


@unittest.skipUnless(os.name == "nt", "Native Windows file sharing integration")
class WindowsReplacementIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="recreate3 atomic IO ")
        self.addCleanup(self.temporary.cleanup)
        self.source = Path(self.temporary.name) / "status.json.tmp"
        self.destination = Path(self.temporary.name) / "status.json"
        self.source.write_bytes(b'{"update":218}')
        self.destination.write_bytes(b'{"update":217}')
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD,
            wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        self.kernel.CreateFileW.restype = wintypes.HANDLE
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel.CloseHandle.restype = wintypes.BOOL

    def hold_destination_without_delete_sharing(self):
        # Ordinary readers can continue opening the file; deletion/replacement
        # is blocked until this handle closes, matching the observed WinError 5.
        handle = self.kernel.CreateFileW(str(self.destination), 0x80000000,
                                        0x1 | 0x2, None, 3, 0x80, None)
        if handle == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        return handle

    def test_real_locked_destination_retries_then_atomically_replaces_after_reader_closes(self):
        handle = self.hold_destination_without_delete_sharing()
        retry_seen = threading.Event()
        closed = threading.Event()
        def reader():
            try:
                retry_seen.wait(3)
                time.sleep(.08)
            finally:
                self.kernel.CloseHandle(handle)
                closed.set()
        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        reports = []
        def report(destination, budget):
            self.assertEqual(self.destination.read_bytes(), b'{"update":217}')
            self.assertEqual(self.source.read_bytes(), b'{"update":218}')
            reports.append((destination, budget))
            retry_seen.set()
        try:
            replacement = windows_io.retrying_replace(os.replace, max_wait=2, report=report)
            replacement(self.source, self.destination)
        finally:
            retry_seen.set()
            thread.join(timeout=5)
        self.assertTrue(closed.is_set())
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(reports), 1)
        self.assertFalse(self.source.exists())
        self.assertEqual(self.destination.read_bytes(), b'{"update":218}')

    def test_real_persistent_lock_keeps_both_files_and_raises_windows_error(self):
        handle = self.hold_destination_without_delete_sharing()
        try:
            replace = windows_io.retrying_replace(os.replace, max_wait=.12,
                initial_delay=.02, max_delay=.04, report=lambda *args: None)
            with self.assertRaises(OSError) as caught:
                replace(self.source, self.destination)
            self.assertIn(caught.exception.winerror, (5, 32, 33))
            self.assertEqual(self.destination.read_bytes(), b'{"update":217}')
            self.assertEqual(self.source.read_bytes(), b'{"update":218}')
        finally:
            self.kernel.CloseHandle(handle)


if __name__ == "__main__":
    unittest.main()
