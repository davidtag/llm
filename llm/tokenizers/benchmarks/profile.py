"""Utilities for profiling code."""

from __future__ import annotations

import time
from typing import Optional, Type
from types import TracebackType


class Profile:
    """Helper context manager for profiling code."""

    def __init__(self) -> None:
        """Initialize the profile."""
        self.start: float = 0
        self.end: float = 0
        self.duration: float = 0

    def __enter__(self) -> Profile:
        """Enter the context."""
        self.start = time.monotonic()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
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
