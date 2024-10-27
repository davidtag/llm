"""Unit tests for multi_head_attention.py."""

import operator
import unittest

import numpy as np

from llm.constants import DEFAULT_DTYPE
from llm.layers.multi_head_attention import MultiHeadAttention


class TestMultiHeadAttention(unittest.TestCase):
    """Unit tests for MultiHeadAttention."""

    def setUp(self) -> None:
        np.random.seed(31415926)
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
        model = MultiHeadAttention(d_model=13, d_k=17, d_v=37, h=7)
        self.assertEqual(model.n_params, 7 * 13 * (17 + 17 + 37 + 37))

        model = MultiHeadAttention(d_model=14, d_k=64, d_v=37, h=8)
        self.assertEqual(model.n_params, 8 * 14 * (64 + 64 + 37 + 37))

    def test_get_parameters(self) -> None:
        """Test the get_parameters() method."""
        model = MultiHeadAttention(d_model=13, d_k=17, d_v=37, h=7)
        params = model.get_parameters()
        self.assertSetEqual(set(params.keys()), {"w_q", "w_k", "w_v", "w_o"})
        w_q = params["w_q"]
        w_k = params["w_k"]
        w_v = params["w_v"]
        w_o = params["w_o"]
        assert isinstance(w_q, np.ndarray)
        assert isinstance(w_k, np.ndarray)
        assert isinstance(w_v, np.ndarray)
        assert isinstance(w_o, np.ndarray)
        self.assertEqual(w_q.shape, (7, 13, 17))
        self.assertEqual(w_k.shape, (7, 13, 17))
        self.assertEqual(w_v.shape, (7, 13, 37))
        self.assertEqual(w_o.shape, (259, 13))
        self.assertEqual(w_q.dtype, DEFAULT_DTYPE)
        self.assertEqual(w_k.dtype, DEFAULT_DTYPE)
        self.assertEqual(w_v.dtype, DEFAULT_DTYPE)
        self.assertEqual(w_o.dtype, DEFAULT_DTYPE)

    def test_load_parameters(self) -> None:
        """Test the load_paramters() method."""
        model = MultiHeadAttention(d_model=3, d_k=13, d_v=17, h=16)
        out1 = model.forward(self.data)

        w_q = np.zeros((16, 3, 13), dtype=DEFAULT_DTYPE)
        w_k = np.zeros((16, 3, 13), dtype=DEFAULT_DTYPE)
        w_v = np.zeros((16, 3, 17), dtype=DEFAULT_DTYPE)
        w_o = np.zeros((272, 3), dtype=DEFAULT_DTYPE)

        params = {
            "w_q": w_q,
            "w_k": w_k,
            "w_v": w_v,
            "w_o": w_o,
        }

        model.load_parameters(params)
        out2 = model.forward(self.data)
        np.testing.assert_array_compare(operator.__ne__, out1, out2)

    def test_get_parameters_and_load_parameters_roundtrip(self) -> None:
        """Test that the return value of get_parameters() can be loaded."""
        model = MultiHeadAttention(d_model=3, d_k=13, d_v=17, h=16)
        out1 = model.forward(self.data)

        params = model.get_parameters()

        model.load_parameters(params)
        out2 = model.forward(self.data)
        np.testing.assert_array_equal(out1, out2)

    def test_forward(self) -> None:
        """Test the forward pass."""
        model = MultiHeadAttention(d_model=3, d_k=13, d_v=17, h=16)

        out = model.forward(self.data)
        self.assertEqual(out.shape, (4, 3))

    def test_forward_3d(self) -> None:
        """Test the forward pass."""
        model = MultiHeadAttention(d_model=3, d_k=13, d_v=17, h=16, masked=True)

        out = model.forward(self.data_3d)
        self.assertEqual(out.shape, (2, 4, 3))

    def test_backward_at_zero(self) -> None:
        """Test the backward pass with upstream gradient being 0."""
        model = MultiHeadAttention(d_model=3, d_k=13, d_v=17, h=16)

        out = model.forward(self.data)

        dout = np.zeros_like(out)
        model.backward(dout)
        dx = model.cache["dx"]
        dw_q = model.cache["dw_q"]
        dw_k = model.cache["dw_k"]
        dw_v = model.cache["dw_v"]
        dw_o = model.cache["dw_o"]

        self.assertEqual(dx.shape, (4, 3))
        self.assertEqual(dw_q.shape, (16, 3, 13))
        self.assertEqual(dw_k.shape, (16, 3, 13))
        self.assertEqual(dw_v.shape, (16, 3, 17))
        self.assertEqual(dw_o.shape, (16 * 17, 3))

        self.assertTrue(np.all(dx == 0))
        self.assertTrue(np.all(dw_q == 0))
        self.assertTrue(np.all(dw_k == 0))
        self.assertTrue(np.all(dw_v == 0))
        self.assertTrue(np.all(dw_o == 0))

    def test_backward_at_zero_3d(self) -> None:
        """Test the backward pass with upstream gradient being 0 (3d input)."""
        model = MultiHeadAttention(d_model=3, d_k=13, d_v=17, h=16, masked=True)

        out = model.forward(self.data_3d)

        dout = np.zeros_like(out)
        model.backward(dout)
        dx = model.cache["dx"]
        dw_q = model.cache["dw_q"]
        dw_k = model.cache["dw_k"]
        dw_v = model.cache["dw_v"]
        dw_o = model.cache["dw_o"]

        self.assertEqual(dx.shape, (2, 4, 3))
        self.assertEqual(dw_q.shape, (16, 3, 13))
        self.assertEqual(dw_k.shape, (16, 3, 13))
        self.assertEqual(dw_v.shape, (16, 3, 17))
        self.assertEqual(dw_o.shape, (16 * 17, 3))

        self.assertTrue(np.all(dx == 0))
        self.assertTrue(np.all(dw_q == 0))
        self.assertTrue(np.all(dw_k == 0))
        self.assertTrue(np.all(dw_v == 0))
        self.assertTrue(np.all(dw_o == 0))

    def test_backward_at_one_dx(self) -> None:
        """Test the backward pass for dx with upstream gradient being 1."""
        model = MultiHeadAttention(d_model=3)

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
        places = 2 if abs(expected_change) < 0.1 else 1
        self.assertAlmostEqual(actual_change, expected_change, places=places)

    def test_backward_at_one_dw_q(self) -> None:
        """Test the backward pass for dw_q with upstream gradient being 1."""
        model = MultiHeadAttention(d_model=3)

        out = model.forward(self.data)
        loss = out.sum()

        dout = np.ones_like(out)  # derivative of loss is 1 for each element
        model.backward(dout)
        dw_q = model.cache["dw_q"]

        step = 0.01
        expected_change = step * dw_q.sum()
        model.w_q += step
        out2 = model.forward(self.data)
        loss2 = out2.sum()
        actual_change = loss2 - loss
        self.assertAlmostEqual(actual_change, expected_change, places=1)

    def test_backward_at_one_dw_q_3d(self) -> None:
        """Test the backward pass for dw_q with upstream gradient being 1 (3d input)."""
        model = MultiHeadAttention(d_model=3)

        out = model.forward(self.data_3d)
        loss = out.sum()

        dout = np.ones_like(out)  # derivative of loss is 1 for each element
        model.backward(dout)
        dw_q = model.cache["dw_q"]

        step = 0.01
        expected_change = step * dw_q.sum()
        model.w_q += step
        out2 = model.forward(self.data_3d)
        loss2 = out2.sum()
        actual_change = loss2 - loss
        self.assertAlmostEqual(actual_change, expected_change, places=1)

    def test_backward_at_one_dw_k(self) -> None:
        """Test the backward pass for dw_k with upstream gradient being 1."""
        model = MultiHeadAttention(d_model=3)

        out = model.forward(self.data)
        loss = out.sum()

        dout = np.ones_like(out)  # derivative of loss is 1 for each element
        model.backward(dout)
        dw_k = model.cache["dw_k"]

        step = 0.01
        expected_change = step * dw_k.sum()
        model.w_k += step
        out2 = model.forward(self.data)
        loss2 = out2.sum()
        actual_change = loss2 - loss
        self.assertAlmostEqual(actual_change, expected_change, places=1)

    def test_backward_at_one_dw_v(self) -> None:
        """Test the backward pass for dw_v with upstream gradient being 1."""
        model = MultiHeadAttention(d_model=3)

        out = model.forward(self.data)
        loss = out.sum()

        dout = np.ones_like(out)  # derivative of loss is 1 for each element
        model.backward(dout)
        dw_v = model.cache["dw_v"]

        step = 0.01
        expected_change = step * dw_v.sum()
        model.w_v += step
        out2 = model.forward(self.data)
        loss2 = out2.sum()
        actual_change = loss2 - loss
        self.assertAlmostEqual(actual_change, expected_change, places=9)

    def test_backward_at_one_dw_o(self) -> None:
        """Test the backward pass for dw_o with upstream gradient being 1."""
        model = MultiHeadAttention(d_model=3)

        out = model.forward(self.data)
        loss = out.sum()

        dout = np.ones_like(out)  # derivative of loss is 1 for each element
        model.backward(dout)
        dw_o = model.cache["dw_o"]

        step = 0.01
        expected_change = step * dw_o.sum()
        model.w_o += step
        out2 = model.forward(self.data)
        loss2 = out2.sum()
        actual_change = loss2 - loss
        self.assertAlmostEqual(actual_change, expected_change, places=9)

    def test_backward_random_dx(self) -> None:
        """Test the backward pass for dx with random step."""
        model = MultiHeadAttention(d_model=3)

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
        places = 2 if abs(expected_change) < 0.1 else 1
        self.assertAlmostEqual(actual_change, expected_change, places=places)

    def test_backward_random_dx_3d(self) -> None:
        """Test the backward pass for dx with random step (3d input)."""
        model = MultiHeadAttention(d_model=3, masked=True)

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
        places = 2 if abs(expected_change) < 0.1 else 1
        self.assertAlmostEqual(actual_change, expected_change, places=places)
