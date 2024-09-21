"""Unit tests for math.py."""

import unittest

import numpy as np

from llm.utils.math import (
    relu,
    log_sum_exp,
    log_softmax,
    softmax,
)


class TestReLU(unittest.TestCase):
    """Unit tests for relu()."""

    def test_positive_values_are_identity(self) -> None:
        """Test that an all-positive input is mapped as an indentity method."""
        x = np.random.random(size=(2, 3, 5))
        a = relu(x)
        np.testing.assert_equal(a, x)

    def test_negative_values_are_zeroed(self) -> None:
        """Test that negative values get zeroed out."""
        x = np.ones(shape=(2, 3, 5)) * -1
        a = relu(x)
        expected = np.zeros_like(x)
        np.testing.assert_equal(a, expected)

    def test_random(self) -> None:
        """Test behavior on a random input."""
        x = np.random.standard_normal(size=(2, 3, 5))
        a = relu(x)
        expected = np.copy(x)
        expected[expected < 0] = 0
        np.testing.assert_equal(a, expected)


class TestLogSumExp(unittest.TestCase):
    """Unit tests for log_sum_exp()."""

    def test_one_dimension(self) -> None:
        """Test a one-dimensional input."""
        x = np.array([0, 0])
        y = log_sum_exp(x)
        self.assertEqual(x.shape, (2,))
        self.assertEqual(y.shape, (1,))
        self.assertAlmostEqual(y[0], np.log(2))  # log(e^0 + e^0)

    def test_two_dimensions(self) -> None:
        """Test a two-dimensional input."""
        x = np.array(
            [
                [0, 0],
                [1, 1],
            ]
        )
        y = log_sum_exp(x)
        self.assertEqual(x.shape, (2, 2))
        self.assertEqual(y.shape, (2, 1))
        self.assertAlmostEqual(y[0][0], np.log(2))  # log(e^0 + e^0)
        self.assertAlmostEqual(y[1][0], np.log(2) + 1)  # log(e^1 + e^1) = log(2*e) = log(2) + 1

    def test_three_dimensions(self) -> None:
        """Test a three-dimensional input."""
        x = np.array(
            [
                [
                    [0, 0],
                    [1, 1],
                ],
                [
                    [2, 2],
                    [-1, -1],
                ],
            ]
        )
        y = log_sum_exp(x)
        self.assertEqual(x.shape, (2, 2, 2))
        self.assertEqual(y.shape, (2, 2, 1))
        np.testing.assert_almost_equal(
            y.flatten(),
            [
                np.log(2),
                np.log(2) + 1,
                np.log(2) + 2,
                np.log(2) - 1,
            ],
        )

    def test_large_values(self) -> None:
        """Test numerical stability on large input values."""
        x = np.array(
            [
                [-0.5, 100.0],
                [-1.3, 900.0],
            ]
        )
        y = log_sum_exp(x)
        self.assertEqual(y[0][0], 100)
        self.assertEqual(y[1][0], 900)


class TestLogSoftmax(unittest.TestCase):
    """Unit tests for log_softmax()."""

    def test_one_dimension(self) -> None:
        """Test a one-dimensional input."""
        x = np.array([0, 0])
        y = log_softmax(x)
        self.assertEqual(x.shape, (2,))
        self.assertEqual(y.shape, (2,))
        # softmax probabilities are 1/2 each
        np.testing.assert_almost_equal(y, [-np.log(2), -np.log(2)])

    def test_two_dimensions(self) -> None:
        """Test a two-dimensional input."""
        x = np.array(
            [
                [0, 0],
                [1, 1],
            ]
        )
        y = log_softmax(x)
        self.assertEqual(x.shape, (2, 2))
        self.assertEqual(y.shape, (2, 2))
        # softmax probabilities are 1/2 each
        expected = np.array([-np.log(2)] * 4).reshape(2, 2)
        np.testing.assert_almost_equal(y, expected)

    def test_three_dimensions(self) -> None:
        """Test a three-dimensional input."""
        x = np.array(
            [
                [
                    [0, 0, 0],
                    [1, 1, 1],
                ],
                [
                    [2, 2, 2],
                    [-1, -1, -1],
                ],
            ]
        )
        y = log_softmax(x)
        self.assertEqual(x.shape, (2, 2, 3))
        self.assertEqual(y.shape, (2, 2, 3))
        # softmax probabilities are 1/3 each
        expected = np.array([-np.log(3)] * 12).reshape(2, 2, 3)
        np.testing.assert_almost_equal(y, expected)

    def test_large_values(self) -> None:
        """Test numerical stability on large input values."""
        x = np.array(
            [
                [-0.5, 100.0],
                [900.0, -1.3],
            ]
        )
        y = log_softmax(x)
        # softmax puts prob ~1 on large values, log of which is 0
        self.assertAlmostEqual(y[0][1], 0)
        self.assertAlmostEqual(y[1][0], 0)
        self.assertLess(y[0][0], -100)
        self.assertLess(y[1][1], -100)


class TestSoftmax(unittest.TestCase):
    """Unit tests for softmax()."""

    def test_one_dimension(self) -> None:
        """Test a one-dimensional input."""
        x = np.array([0, 0])
        y = softmax(x)
        self.assertEqual(x.shape, (2,))
        self.assertEqual(y.shape, (2,))
        np.testing.assert_almost_equal(y, [0.5, 0.5])

    def test_two_dimensions(self) -> None:
        """Test a two-dimensional input."""
        x = np.array(
            [
                [0, 0],
                [1, 1],
            ]
        )
        y = softmax(x)
        self.assertEqual(x.shape, (2, 2))
        self.assertEqual(y.shape, (2, 2))
        expected = np.array([0.5] * 4).reshape(2, 2)
        np.testing.assert_almost_equal(y, expected)

    def test_three_dimensions(self) -> None:
        """Test a three-dimensional input."""
        x = np.array(
            [
                [
                    [0, 0, 0],
                    [1, 1, 1],
                ],
                [
                    [2, 2, 2],
                    [-1, -1, -1],
                ],
            ]
        )
        y = softmax(x)
        self.assertEqual(x.shape, (2, 2, 3))
        self.assertEqual(y.shape, (2, 2, 3))
        expected = np.array([1 / 3] * 12).reshape(2, 2, 3)
        np.testing.assert_almost_equal(y, expected)

    def test_large_values(self) -> None:
        """Test numerical stability on large input values."""
        x = np.array(
            [
                [-0.5, 100.0],
                [900.0, -1.3],
            ]
        )
        y = softmax(x)
        expected = np.array(
            [
                [0, 1],
                [1, 0],
            ]
        )
        np.testing.assert_almost_equal(y, expected)

    def test_random_values(self) -> None:
        """Test behavior on a random input."""
        x = np.random.standard_normal(size=(5, 9))
        y = softmax(x)
        # all values of y should be positive and sum to 1
        self.assertTrue(np.all(y > 0))
        y_sum = np.sum(y, axis=-1)
        expected_y_sum = np.ones(shape=(5,))
        np.testing.assert_almost_equal(y_sum, expected_y_sum)
