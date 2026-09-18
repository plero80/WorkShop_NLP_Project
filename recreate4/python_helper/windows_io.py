"""Bounded retries for Windows readers temporarily preventing atomic replacement.

Keep the original write-then-replace protocol: never remove the destination or
pretend a failed replacement succeeded. No training or checkpoint format changes
are needed, and non-Windows replacement behavior remains untouched.
"""
from __future__ import annotations

from contextlib import contextmanager
import functools
import math
import os
import sys
import time


RETRYABLE_WINERRORS = frozenset((5, 32, 33))


def _report_retry(destination, max_wait):
    name = os.path.basename(os.fsdecode(destination)) if destination is not None else "file"
    print(f"[Windows I/O] Temporary access conflict replacing {name}; "
          f"retrying for up to {max_wait:g} seconds.", file=sys.stderr, flush=True)


def retrying_replace(replace, *, platform_name=None, monotonic=time.monotonic,
                     sleep=time.sleep, max_wait=8.0, initial_delay=0.05,
                     max_delay=0.5, report=_report_retry):
    """Wrap one replacement function, preserving its arguments and return value.

    Only Windows access/sharing/lock errors are retried. A permanently locked
    file still fails with the original exception after at most eight seconds of
    waiting by default. Injected clock and sleep functions support CPU-only tests.
    """
    for name, value in (("max_wait", max_wait), ("initial_delay", initial_delay),
                        ("max_delay", max_delay)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    if max_wait > 10:
        raise ValueError("Windows replacement retries must not exceed ten seconds.")
    if (os.name if platform_name is None else platform_name) != "nt":
        return replace
    if getattr(replace, "_recreate3_windows_replace_retry", False) is True:
        return replace

    @functools.wraps(replace)
    def wrapped(*args, **kwargs):
        deadline = monotonic() + max_wait
        delay = min(initial_delay, max_delay)
        original_error = None
        original_traceback = None
        while True:
            try:
                return replace(*args, **kwargs)
            except OSError as error:
                if getattr(error, "winerror", None) not in RETRYABLE_WINERRORS:
                    raise
                if original_error is None:
                    original_error = error
                    original_traceback = error.__traceback__
                    destination = args[1] if len(args) > 1 else kwargs.get("dst")
                    report(destination, max_wait)
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise original_error.with_traceback(original_traceback)
                sleep(min(delay, remaining))
                delay = min(delay * 2, max_delay)

    wrapped._recreate3_windows_replace_retry = True
    return wrapped


@contextmanager
def replacement_retries():
    """Temporarily protect process-wide os.replace on Windows, once per process.

    Install before entering the original CLI so Path.replace and runtime atomic
    saves use the same bounded policy. Nested callers reuse the installed wrapper.
    """
    previous = os.replace
    replacement = retrying_replace(previous)
    if replacement is previous:
        yield
        return
    os.replace = replacement
    try:
        yield
    finally:
        os.replace = previous
