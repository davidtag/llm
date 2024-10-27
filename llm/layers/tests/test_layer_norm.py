"""Unit tests for layer_norm.py."""

import operator
import unittest

import numpy as np

from llm.constants import DEFAULT_DTYPE
from llm.layers.layer_norm import LayerNorm


class TestLayerNorm(unittest.TestCase):
    """Unit tests for LayerNorm."""

    def setUp(self) -> None:
        self.data = np.array(  # shape = (4, 3)
            [
                [-0.80672381, -0.08818247, 0.002],
                [0.63413982, 1.32233656, 0.332],
                [0.1814214, -0.50674539, -0.0223],
                [1.16085551, -0.15033837, -0.332],
            ]
        )
        self.data_3d = np.array([self.data, 2 * self.data])  # shape = (2, 4, 3)

    def test_n_params(self) -> None:
        """Test the layer reports the correct number of parameters."""
        model = LayerNorm(n_input=3)
        self.assertEqual(model.n_params, 3 + 3)

        model = LayerNorm(n_input=32)
        self.assertEqual(model.n_params, 32 + 32)

    def test_get_parameters(self) -> None:
        """Test the get_parameters() method."""
        model = LayerNorm(n_input=3)
        params = model.get_parameters()
        self.assertSetEqual(set(params.keys()), {"gamma", "beta"})
        gamma = params["gamma"]
        beta = params["beta"]
        assert isinstance(gamma, np.ndarray)
        assert isinstance(beta, np.ndarray)
        self.assertEqual(gamma.shape, (1, 3))
        self.assertEqual(beta.shape, (1, 3))
        self.assertEqual(gamma.dtype, DEFAULT_DTYPE)
        self.assertEqual(beta.dtype, DEFAULT_DTYPE)

    def test_load_parameters(self) -> None:
        """Test the load_paramters() method."""
        model = LayerNorm(n_input=3)
        out1 = model.forward(self.data)

        gamma_new = np.zeros((1, 3), dtype=DEFAULT_DTYPE)
        beta_new = np.zeros((1, 3), dtype=DEFAULT_DTYPE)
        params = {
            "gamma": gamma_new,
            "beta": beta_new,
        }

        model.load_parameters(params)
        out2 = model.forward(self.data)
        np.testing.assert_array_compare(operator.__ne__, out1, out2)

    def test_get_parameters_and_load_parameters_roundtrip(self) -> None:
        """Test that the return value of get_parameters() can be loaded."""
        model = LayerNorm(n_input=3)
        out1 = model.forward(self.data)

        params = model.get_parameters()

        model.load_parameters(params)
        out2 = model.forward(self.data)
        np.testing.assert_array_equal(out1, out2)

    def test_forward(self) -> None:
        """Test the forward pass."""
        model = LayerNorm(n_input=3)

        out = model.forward(self.data)
        self.assertEqual(out.shape, (4, 3))

    def test_forward_3d(self) -> None:
        """Test the forward pass (3d input)."""
        model = LayerNorm(n_input=3)

        out = model.forward(self.data_3d)
        self.assertEqual(out.shape, (2, 4, 3))

    def test_backward_at_zero(self) -> None:
        """Test the backward pass with upstream gradient being 0."""
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

    def test_backward_at_zero_3d(self) -> None:
        """Test the backward pass with upstream gradient being 0 (3d input)."""
        model = LayerNorm(n_input=3)

        out = model.forward(self.data_3d)

        dout = np.zeros_like(out)
        model.backward(dout)
        dx = model.cache["dx"]
        dgamma = model.cache["dgamma"]
        dbeta = model.cache["dbeta"]

        self.assertEqual(dx.shape, (2, 4, 3))
        self.assertEqual(dgamma.shape, (1, 3))
        self.assertEqual(dbeta.shape, (1, 3))

        self.assertTrue(np.all(dx == 0))
        self.assertTrue(np.all(dgamma == 0))
        self.assertTrue(np.all(dbeta == 0))

    def test_backward_at_one_dx(self) -> None:
        """Test the backward pass for dx with upstream gradient being 1."""
        model = LayerNorm(n_input=3)
        model.gamma = np.random.standard_normal((1, 3))  # change gamma to make test non-trivial

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

    def test_backward_at_one_dx_3d(self) -> None:
        """Test the backward pass for dx with upstream gradient being 1 (3d input)."""
        model = LayerNorm(n_input=3)
        model.gamma = np.random.standard_normal((1, 3))  # change gamma to make test non-trivial

        out = model.forward(self.data_3d)
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
        x = self.data_3d + step
        out2 = model.forward(x)
        loss2 = out2.sum()
        actual_change = loss2 - loss
        self.assertAlmostEqual(actual_change, expected_change, places=9)

    def test_backward_at_one_dgamma(self) -> None:
        """Test the backward pass for dgamma with upstream gradient being 1."""
        model = LayerNorm(n_input=3)
        model.gamma = np.random.standard_normal((1, 3))  # change gamma to make test non-trivial

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

    def test_backward_at_one_dgamma_3d(self) -> None:
        """Test the backward pass for dgamma with upstream gradient being 1 (3d input)."""
        model = LayerNorm(n_input=3)
        model.gamma = np.random.standard_normal((1, 3))  # change gamma to make test non-trivia

        out = model.forward(self.data_3d)
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
        out2 = model.forward(self.data_3d)
        loss2 = out2.sum()
        actual_change = loss2 - loss
        self.assertAlmostEqual(actual_change, expected_change, places=9)

    def test_backward_at_one_dbeta(self) -> None:
        """Test the backward pass for dbeta with upstream gradient being 1."""
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

    def test_backward_at_one_dbeta_3d(self) -> None:
        """Test the backward pass for dbeta with upstream gradient being 1 (3d input)."""
        model = LayerNorm(n_input=3)

        out = model.forward(self.data_3d)
        loss = out.sum()

        dout = np.ones_like(out)  # derivative of loss is 1 for each element
        model.backward(dout)
        dbeta = model.cache["dbeta"]

        # beta is directly added to each element of the output, so it's gradient
        # with a simple additive loss is just the number of data points in each
        # dimension.
        expected_dbeta = np.array([[8, 8, 8]])
        np.testing.assert_array_equal(dbeta, expected_dbeta)

        step = 0.01
        expected_change = step * dbeta.sum()
        model.beta += step
        out2 = model.forward(self.data_3d)
        loss2 = out2.sum()
        actual_change = loss2 - loss
        self.assertAlmostEqual(actual_change, expected_change, places=9)

    def test_backward_random_dx(self) -> None:
        """Test the backward pass for dx with random step."""
        model = LayerNorm(n_input=3)
        model.gamma = np.random.standard_normal((1, 3))  # change gamma to make test non-trivial

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

    def test_backward_random_dx_3d(self) -> None:
        """Test the backward pass for dx with random step (3d input)."""
        model = LayerNorm(n_input=3)
        model.gamma = np.random.standard_normal((1, 3))  # change gamma to make test non-trivial

        out = model.forward(self.data_3d)
        loss = out.sum()

        dout = np.ones_like(out)  # derivative of loss is 1 for each element
        model.backward(dout)
        dx = model.cache["dx"]

        step = 0.01 * np.random.random(size=self.data_3d.shape)
        expected_change = np.sum(step * dx)
        x = self.data_3d + step
        out2 = model.forward(x)
        loss2 = out2.sum()
        actual_change = loss2 - loss
        self.assertAlmostEqual(actual_change, expected_change, places=3)

    def test_backward_random_dgamma(self) -> None:
        """Test the backward pass for dgamma with random step."""
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

    def test_backward_random_gradient_and_step_dx_3d(self) -> None:
        """Test the backward pass for dx with random gradient and step (3d input)."""
        model = LayerNorm(n_input=3)
        model.gamma = np.random.standard_normal((1, 3))
        model.beta = np.random.standard_normal((1, 3))
        weight = np.random.standard_normal((3, 1))

        out = model.forward(self.data_3d)
        loss = np.matmul(out, weight).sum()

        dout = np.ones_like(out) * weight.T
        model.backward(dout)
        dx = model.cache["dx"]

        dx_mean = dx.sum(axis=-1)  # This is always true
        np.testing.assert_almost_equal(dx_mean, np.zeros_like(dx_mean))
        dx_std = dx.std(axis=-1)
        with self.assertRaises(AssertionError):
            np.testing.assert_almost_equal(dx_std, np.zeros_like(dx_std))

        step = 0.01 * np.random.random(size=self.data_3d.shape)
        expected_change = np.sum(step * dx)
        x = self.data_3d + step
        out2 = model.forward(x)
        loss2 = np.matmul(out2, weight).sum()
        actual_change = loss2 - loss
        self.assertAlmostEqual(actual_change, expected_change, places=3)
