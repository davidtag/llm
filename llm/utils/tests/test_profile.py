"""Unit tests for profile.py."""

import time
import unittest

from llm.utils.profile import Profile


class TestProfile(unittest.TestCase):
    """Unit tests for Profile."""

    def test_basic(self) -> None:
        """Test the ability to use Profile as a context manager."""
        with Profile() as prof:
            time.sleep(0.1)

        self.assertGreater(prof.seconds, 0.1)
        self.assertGreater(prof.milliseconds, 100)

        prof.scale_by(2)

        self.assertGreater(prof.seconds, 0.05)
        self.assertGreater(prof.milliseconds, 50)

        self.assertTrue(prof.milliseconds_formatted.endswith("ms"))

    def test_exception(self) -> None:
        """Test that exceptions flow through the exception manager and profiling is still recorded."""
        with self.assertRaises(ValueError) as cm:
            with Profile() as prof:
                time.sleep(0.1)
                raise ValueError("some error")

        self.assertEqual(str(cm.exception), "some error")
        self.assertGreater(prof.milliseconds, 100)
