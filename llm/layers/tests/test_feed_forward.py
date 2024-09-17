"""Unit tests for feed_forward.py."""

import unittest

import numpy as np

from llm.layers.feed_forward import FeedForward


class TestFeedForward(unittest.TestCase):
    """Unit tests for FeedForward."""

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
        model = FeedForward(n_input=3, n_hidden=13, n_output=5)
        layer_1_params = 3 * 13 + 13
        layer_2_params = 13 * 5 + 5
        total_params = layer_1_params + layer_2_params
        self.assertEqual(model.n_params, total_params)

        model = FeedForward(n_input=50, n_hidden=130, n_output=73)
        layer_1_params = 50 * 130 + 130
        layer_2_params = 130 * 73 + 73
        total_params = layer_1_params + layer_2_params
        self.assertEqual(model.n_params, total_params)

    def test_get_parameters(self) -> None:
        """Test the get_parameters() method."""
        model = FeedForward(n_input=3, n_hidden=13, n_output=5)
        params = model.get_parameters()
        self.assertSetEqual(set(params.keys()), {"layer_1", "layer_2"})

    def test_get_parameters_and_load_parameters_roundtrip(self) -> None:
        """Test that the return value of get_parameters() can be loaded."""
        model = FeedForward(n_input=3, n_hidden=13, n_output=5)
        out1 = model.forward(self.data)

        params = model.get_parameters()

        model.load_parameters(params)
        out2 = model.forward(self.data)
        np.testing.assert_array_equal(out1, out2)

    def test_forward(self) -> None:
        """Test the forward pass."""
        model = FeedForward(n_input=3, n_hidden=13, n_output=5)

        out = model.forward(self.data)
        self.assertEqual(out.shape, (4, 5))

    def test_forward_3d(self) -> None:
        """Test the forward pass (3d input)."""
        model = FeedForward(n_input=3, n_hidden=13, n_output=5)

        out = model.forward(self.data_3d)
        self.assertEqual(out.shape, (2, 4, 5))

    def test_backward_at_zero(self) -> None:
        """Test the backward pass with upstream gradient being 0."""
        model = FeedForward(n_input=3, n_hidden=13, n_output=5)

        out = model.forward(self.data)

        dout = np.zeros_like(out)
        model.backward(dout)
        dx = model.cache["dx"]
        dw1 = model.layer_1.cache["dw"]
        db1 = model.layer_1.cache["db"]
        dw2 = model.layer_2.cache["dw"]
        db2 = model.layer_2.cache["db"]

        self.assertEqual(dx.shape, (4, 3))
        self.assertEqual(dw1.shape, (3, 13))
        self.assertEqual(db1.shape, (1, 13))
        self.assertEqual(dw2.shape, (13, 5))
        self.assertEqual(db2.shape, (1, 5))

        self.assertTrue(np.all(dx == 0))
        self.assertTrue(np.all(dw1 == 0))
        self.assertTrue(np.all(db1 == 0))
        self.assertTrue(np.all(dw2 == 0))
        self.assertTrue(np.all(db2 == 0))

    def test_backward_at_zero_3d(self) -> None:
        """Test the backward pass with upstream gradient being 0 (3d input)."""
        model = FeedForward(n_input=3, n_hidden=13, n_output=5)

        out = model.forward(self.data_3d)

        dout = np.zeros_like(out)
        model.backward(dout)
        dx = model.cache["dx"]
        dw1 = model.layer_1.cache["dw"]
        db1 = model.layer_1.cache["db"]
        dw2 = model.layer_2.cache["dw"]
        db2 = model.layer_2.cache["db"]

        self.assertEqual(dx.shape, (2, 4, 3))
        self.assertEqual(dw1.shape, (3, 13))
        self.assertEqual(db1.shape, (1, 13))
        self.assertEqual(dw2.shape, (13, 5))
        self.assertEqual(db2.shape, (1, 5))

        self.assertTrue(np.all(dx == 0))
        self.assertTrue(np.all(dw1 == 0))
        self.assertTrue(np.all(db1 == 0))
        self.assertTrue(np.all(dw2 == 0))
        self.assertTrue(np.all(db2 == 0))

    def test_backward_at_one_dx(self) -> None:
        """Test the backward pass for dx with upstream gradient being 1."""
        model = FeedForward(n_input=3, n_hidden=13, n_output=5)

        out = model.forward(self.data)
        loss = out.sum()

        dout = np.ones_like(out)  # derivative of loss is 1 for each element
        model.backward(dout)
        dx = model.cache["dx"]

        step = 0.001
        expected_change = step * dx.sum()
        x = self.data + step
        out2 = model.forward(x)
        loss2 = out2.sum()
        actual_change = loss2 - loss
        self.assertAlmostEqual(actual_change, expected_change, places=2)

    def test_backward_at_one_dx_3d(self) -> None:
        """Test the backward pass for dx with upstream gradient being 1 (3d input)."""
        model = FeedForward(n_input=3, n_hidden=13, n_output=5)

        out = model.forward(self.data_3d)
        loss = out.sum()

        dout = np.ones_like(out)  # derivative of loss is 1 for each element
        model.backward(dout)
        dx = model.cache["dx"]

        step = 0.001
        expected_change = step * dx.sum()
        x = self.data_3d + step
        out2 = model.forward(x)
        loss2 = out2.sum()
        actual_change = loss2 - loss
        self.assertAlmostEqual(actual_change, expected_change, places=2)

    def test_backward_random_dx(self) -> None:
        """Test the backward pass for dx with random step."""
        model = FeedForward(n_input=3, n_hidden=13, n_output=5)

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

    def test_backward_random_dx_3d(self) -> None:
        """Test the backward pass for dx with random step (3d input)."""
        model = FeedForward(n_input=3, n_hidden=13, n_output=5)

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
        self.assertAlmostEqual(actual_change, expected_change, places=2)
