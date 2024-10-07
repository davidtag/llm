"""Utilities for profiling code."""

import time


class Profile:
    """Helper context manager for profiling code."""

    def __init__(self) -> None:
        self.start = 0
        self.end = 0
        self.duration = 0

    def __enter__(self):
        """Enter the context."""
        self.start = time.monotonic()
        return self

    def __exit__(self, type, value, traceback):
        """Exit the context."""
        self.end = time.monotonic()
        self.duration = self.end - self.start

    def scale_by(self, num_runs: int) -> None:
        """Scale the profiled duration by a number of runs."""
        self.duration /= num_runs

    @property
    def seconds(self) -> float:
        """The profiled duration in seconds."""
        return self.duration

    @property
    def milliseconds(self) -> float:
        """The profiled duration in milliseconds."""
        return self.seconds * 1000.0

    @property
    def milliseconds_formatted(self) -> str:
        """The profiled duration in milliseconds, formatted as a string."""
        return f"{self.milliseconds:.1f}ms"
