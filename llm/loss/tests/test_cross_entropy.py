"""Unit tests for cross_entropy.py."""

import unittest

import numpy as np

from llm.loss.cross_entropy import CrossEntropyLoss


class TestCrossEntropyLoss(unittest.TestCase):
    """Unit tests for CrossEntropyLoss."""

    def test_forward_equal_probabilties(self) -> None:
        """Test the forward pass where output probabilities are uniform."""
        loss_fn = CrossEntropyLoss()

        logits = np.array([[0, 0]])  # all probabilties become 1/2
        targets = np.array([1])
        loss = loss_fn.forward(logits, targets)
        self.assertAlmostEqual(loss, np.log(2))  # -log(1/2)

        logits = np.array([[0, 0, 0]])  # all probabilties become 1/3
        targets = np.array([2])
        loss = loss_fn.forward(logits, targets)
        self.assertAlmostEqual(loss, np.log(3))  # -log(1/3)

    def test_forward_equal_probabilties_3d(self) -> None:
        """Test the forward pass where output probabilities are uniform (3d input)."""
        loss_fn = CrossEntropyLoss()

        logits = np.zeros((16, 32, 2))
        targets = np.ones((16, 32)).astype(np.int32)

        loss = loss_fn.forward(logits, targets)
        self.assertAlmostEqual(loss, np.log(2))  # -log(1/2)

        logits = np.zeros((16, 32, 3))
        targets = np.ones((16, 32)).astype(np.int32)

        loss = loss_fn.forward(logits, targets)
        self.assertAlmostEqual(loss, np.log(3))  # -log(1/3)

    def test_zero_loss(self) -> None:
        """Test the forward & backward pass when the loss is 0."""
        loss_fn = CrossEntropyLoss()

        logits = np.array(
            [
                [0.3, 100.0],  # almost all prob on 1
                [900.0, -0.5],  # almost all prob on 0
            ]
        )
        targets = np.array([1, 0])
        loss = loss_fn.forward(logits, targets)
        self.assertAlmostEqual(float(loss), 0)

        dlogits = loss_fn.backward()
        self.assertEqual(dlogits.shape, logits.shape)
        expected = np.zeros_like(logits)
        np.testing.assert_almost_equal(dlogits, expected)

    def test_zero_loss_3d(self) -> None:
        """Test the forward & backward pass when the loss is 0 (3d input)."""
        loss_fn = CrossEntropyLoss()

        logits = np.array(
            [
                [
                    [0.3, 100.0],  # almost all prob on 1
                    [900.0, -0.5],  # almost all prob on 0
                    [800.0, -2.5],  # almost all prob on 0
                ],
                [
                    [456.0, 0.2],  # almost all prob on 0
                    [-0.3, 99.9],  # almost all prob on 1
                    [-1.8, 107.0],  # almost all prob on 1
                ],
            ]
        )
        targets = np.array(
            [
                [
                    1,
                    0,
                    0,
                ],
                [
                    0,
                    1,
                    1,
                ],
            ]
        )
        loss = loss_fn.forward(logits, targets)
        self.assertAlmostEqual(float(loss), 0)

        dlogits = loss_fn.backward()
        self.assertEqual(dlogits.shape, logits.shape)
        expected = np.zeros_like(logits)
        np.testing.assert_almost_equal(dlogits, expected)

    def test_large_loss(self) -> None:
        """Test numerical stability when the loss function is large."""
        loss_fn = CrossEntropyLoss()

        logits = np.array(
            [
                [0.3, 100.0],  # almost all prob on 1
                [900.0, -0.5],  # almost all prob on 0
                [45.0, -0.5],  # almost all prob on 0
            ]
        )
        targets = np.array([0, 1, 1])
        loss = loss_fn.forward(logits, targets)
        self.assertGreater(float(loss), 100)

        dlogits = loss_fn.backward()
        self.assertEqual(dlogits.shape, logits.shape)
        expected = (1 / 3) * np.array(
            [
                [-1, 1],
                [1, -1],
                [1, -1],
            ]
        )
        np.testing.assert_almost_equal(dlogits, expected)

    def test_backward_random_step(self) -> None:
        """Test correctness of the backward pass using a random step in the input."""
        loss_fn = CrossEntropyLoss()

        logits = np.random.standard_normal(size=(5, 3))
        targets = np.array([0, 1, 1, 0, 2])
        loss = loss_fn.forward(logits, targets)

        dlogits = loss_fn.backward()
        step = 0.01 * np.random.standard_normal(size=logits.shape)
        expected_change = np.sum(step * dlogits)
        logits2 = logits + step
        loss2 = loss_fn.forward(logits2, targets)
        actual_change = loss2 - loss
        self.assertAlmostEqual(actual_change, expected_change, places=3)

    def test_backward_random_step_3d(self) -> None:
        """Test correctness of the backward pass using a random step in the input (3d input)."""
        loss_fn = CrossEntropyLoss()

        logits = np.random.standard_normal(size=(16, 5, 3))
        targets = np.random.randint(low=0, high=2, size=(16, 5))
        loss = loss_fn.forward(logits, targets)

        dlogits = loss_fn.backward()
        step = 0.01 * np.random.standard_normal(size=logits.shape)
        expected_change = np.sum(step * dlogits)
        logits2 = logits + step
        loss2 = loss_fn.forward(logits2, targets)
        actual_change = loss2 - loss
        self.assertAlmostEqual(actual_change, expected_change, places=3)
