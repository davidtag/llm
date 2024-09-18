"""Unit tests for multi_head_attention.py."""

import unittest

import numpy as np

from llm.layers.multi_head_attention import MultiHeadAttention


class TestMultiHeadAttention(unittest.TestCase):
    """Unit tests for MultiHeadAttention."""

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
        model = MultiHeadAttention(d_model=13, d_k=17, d_v=37, h=7)
        self.assertEqual(model.n_params, 7 * 13 * (17 + 17 + 37 + 37))

        model = MultiHeadAttention(d_model=14, d_k=64, d_v=37, h=8)
        self.assertEqual(model.n_params, 8 * 14 * (64 + 64 + 37 + 37))

    def test_forward(self) -> None:
        model = MultiHeadAttention(d_model=3, d_k=13, d_v=17, h=16)

        out = model.forward(self.data)
        self.assertEqual(out.shape, (4, 3))

    def test_backward_at_zero(self) -> None:
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

    def test_backward_at_one_dx(self) -> None:
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
        self.assertAlmostEqual(actual_change, expected_change, places=2)  # TODO(dtag): flaky

    def test_backward_at_one_dw_q(self) -> None:
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

    def test_backward_at_one_dw_k(self) -> None:
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
        self.assertAlmostEqual(actual_change, expected_change, places=2)

    def test_backward_at_one_dw_v(self) -> None:
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
        self.assertAlmostEqual(actual_change, expected_change, places=2)
