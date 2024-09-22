"""Unit tests for block.py."""

import unittest

import numpy as np

from llm.layers.block import Block


# TODO(dtag): Test masked_attention=True
class TestBlock(unittest.TestCase):
    """Unit tests for Block."""

    def setUp(self) -> None:
        np.random.seed(31415926)
        self.data = np.array(
            [
                [-0.80672381, -0.08818247, 0.002],
                [0.63413982, 1.32233656, 0.332],
                [0.1814214, -0.50674539, -0.0223],
                [1.16085551, -0.15033837, -0.332],
            ]
        )

    def test_n_params(self) -> None:
        """Test the layer reports the correct number of parameters."""
        model = Block(d_model=13, d_k=17, d_v=37, h=7, d_ff=256)
        self.assertEqual(
            model.n_params,
            (
                # MultiHeadAttention
                7 * 13 * (17 + 17 + 37 + 37)
                # LayerNorm
                + 13 * 2
                # FeedForward
                + 13 * 256
                + 256
                + 256 * 13
                + 13
                # LayerNorm
                + 13 * 2
            ),
        )

    def test_forward(self) -> None:
        """Test the forward pass."""
        model = Block(d_model=3)

        out = model.forward(self.data)
        self.assertEqual(out.shape, (4, 3))

    def test_backward_at_zero(self) -> None:
        """Test the backward pass with upstream gradient being 0."""
        model = Block(d_model=3)

        out = model.forward(self.data)

        dout = np.zeros_like(out)
        model.backward(dout)
        dx = model.cache["dx"]

        self.assertEqual(dx.shape, (4, 3))

        self.assertTrue(np.all(dx == 0))

    def test_backward_at_one_dx(self) -> None:
        """Test the backward pass for dx with upstream gradient being 1."""
        model = Block(d_model=3)

        out = model.forward(self.data)
        loss = out.sum()

        dout = np.ones_like(out)  # derivative of loss is 1 for each element
        model.backward(dout)
        dx = model.cache["dx"]

        step = 0.01
        expected_change = step * dx.sum()
        x = self.data + step
        out2 = model.forward(x)
        loss2 = out2.sum()
        actual_change = loss2 - loss
        self.assertAlmostEqual(actual_change, expected_change, places=2)

    def test_backward_random_dx(self) -> None:
        """Test the backward pass for dx gradients with random step."""
        model = Block(d_model=3)

        out = model.forward(self.data)
        loss = out.sum()

        dout = np.ones_like(out)  # derivative of loss is 1 for each element
        model.backward(dout)
        dx = model.cache["dx"]

        step = 0.01 * np.random.random(size=self.data.shape)
        expected_change = np.sum(step * dx)
        x = self.data + step
        out2 = model.forward(x)
        loss2 = out2.sum()
        actual_change = loss2 - loss
        self.assertAlmostEqual(actual_change, expected_change, places=2)
