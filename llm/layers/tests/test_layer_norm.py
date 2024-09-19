"""Unit tests for layer_norm.py."""

import unittest

import numpy as np

from llm.layers.layer_norm import LayerNorm


class TestLayerNorm(unittest.TestCase):
    """Unit tests for LayerNorm."""

    def setUp(self) -> None:
        self.data = np.array(
            [
                [-0.80672381, -0.08818247, 0.002],
                [0.63413982, 1.32233656, 0.332],
                [0.1814214, -0.50674539, -0.0223],
                [1.16085551, -0.15033837, -0.332],
            ]
        )

    def test_n_params(self) -> None:
        model = LayerNorm(n_input=3)
        self.assertEqual(model.n_params, 6)

        model = LayerNorm(n_input=32)
        self.assertEqual(model.n_params, 64)

    def test_forward(self) -> None:
        model = LayerNorm(n_input=3)

        out = model.forward(self.data)
        self.assertEqual(out.shape, (4, 3))

    def test_backward_at_zero(self) -> None:
        model = LayerNorm(n_input=3)

        out = model.forward(self.data)

        dout = np.zeros_like(out)
        model.backward(dout)
        dx = model.cache["dx"]
        dgamma = model.cache["dgamma"]
        dbeta = model.cache["dbeta"]

        self.assertEqual(dx.shape, (4, 3))
        self.assertEqual(dgamma.shape, (1, 3))
        self.assertEqual(dbeta.shape, (1, 3))

        self.assertTrue(np.all(dx == 0))
        self.assertTrue(np.all(dgamma == 0))
        self.assertTrue(np.all(dbeta == 0))

    def test_backward_at_one_dx(self) -> None:
        model = LayerNorm(n_input=3)

        out = model.forward(self.data)
        loss = out.sum()

        dout = np.ones_like(out)  # derivative of loss is 1 for each element
        model.backward(dout)
        dx = model.cache["dx"]

        step = 0.1
        expected_change = step * dx.sum()
        self.assertAlmostEqual(
            # layer norm is translation invariant to its input
            expected_change,
            0,
            places=9,
        )
        x = self.data + step
        out2 = model.forward(x)
        loss2 = out2.sum()
        actual_change = loss2 - loss
        self.assertAlmostEqual(actual_change, expected_change, places=9)

    def test_backward_at_one_dgamma(self) -> None:
        model = LayerNorm(n_input=3)

        out = model.forward(self.data)
        loss = out.sum()

        dout = np.ones_like(out)  # derivative of loss is 1 for each element
        model.backward(dout)
        dgamma = model.cache["dgamma"]

        step = 0.01
        expected_change = step * dgamma.sum()
        self.assertAlmostEqual(
            # gamma multiplies a normalized input, so it's sensitivity to input
            # parameters is also normalized.
            expected_change,
            0,
            places=9,
        )
        model.gamma += step
        out2 = model.forward(self.data)
        loss2 = out2.sum()
        actual_change = loss2 - loss
        self.assertAlmostEqual(actual_change, expected_change, places=9)

    def test_backward_at_one_dbeta(self) -> None:
        model = LayerNorm(n_input=3)

        out = model.forward(self.data)
        loss = out.sum()

        dout = np.ones_like(out)  # derivative of loss is 1 for each element
        model.backward(dout)
        dbeta = model.cache["dbeta"]

        # beta is directly added to each element of the output, so it's gradient
        # with a simple additive loss is just the number of data points in each
        # dimension.
        expected_dbeta = np.array([[4, 4, 4]])
        np.testing.assert_array_equal(dbeta, expected_dbeta)

        step = 0.01
        expected_change = step * dbeta.sum()
        model.beta += step
        out2 = model.forward(self.data)
        loss2 = out2.sum()
        actual_change = loss2 - loss
        self.assertAlmostEqual(actual_change, expected_change, places=9)

    def test_backward_random_dx(self) -> None:
        model = LayerNorm(n_input=3)

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
        self.assertAlmostEqual(actual_change, expected_change, places=3)

    def test_backward_random_dgamma(self) -> None:
        model = LayerNorm(n_input=3)

        out = model.forward(self.data)
        loss = out.sum()

        dout = np.ones_like(out)  # derivative of loss is 1 for each element
        model.backward(dout)
        dgamma = model.cache["dgamma"]

        step = 0.1 * np.random.normal(size=model.gamma.shape)
        expected_change = np.sum(step * dgamma)
        model.gamma += step
        out2 = model.forward(self.data)
        loss2 = out2.sum()
        actual_change = loss2 - loss
        self.assertAlmostEqual(actual_change, expected_change, places=9)
