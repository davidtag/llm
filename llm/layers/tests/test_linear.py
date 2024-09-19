"""Unit tests for linear.py."""

import unittest

import numpy as np

from llm.layers.linear import Linear


class TestLinear(unittest.TestCase):
    """Unit tests for Linear."""

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
        model = Linear(n_input=3, n_output=5)
        self.assertEqual(model.n_params, 3 * 5 + 5)

        model = Linear(n_input=50, n_output=73)
        self.assertEqual(model.n_params, 50 * 73 + 73)

    def test_forward(self) -> None:
        model = Linear(n_input=3, n_output=2)

        out = model.forward(self.data)
        self.assertEqual(out.shape, (4, 2))

    def test_backward_at_zero(self) -> None:
        model = Linear(n_input=3, n_output=2)

        out = model.forward(self.data)

        dout = np.zeros_like(out)
        model.backward(dout)
        dx = model.cache["dx"]
        dw = model.cache["dw"]
        db = model.cache["db"]

        self.assertEqual(dx.shape, (4, 3))
        self.assertEqual(dw.shape, (3, 2))
        self.assertEqual(db.shape, (1, 2))

        self.assertTrue(np.all(dx == 0))
        self.assertTrue(np.all(dw == 0))
        self.assertTrue(np.all(db == 0))

    def test_backward_at_one_dx(self) -> None:
        model = Linear(n_input=3, n_output=2)

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
        self.assertAlmostEqual(actual_change, expected_change, places=9)

    def test_backward_at_one_dw(self) -> None:
        model = Linear(n_input=3, n_output=2)

        out = model.forward(self.data)
        loss = out.sum()

        dout = np.ones_like(out)  # derivative of loss is 1 for each element
        model.backward(dout)
        dw = model.cache["dw"]

        step = 0.01
        expected_change = step * dw.sum()
        model.w += step
        out2 = model.forward(self.data)
        loss2 = out2.sum()
        actual_change = loss2 - loss
        self.assertAlmostEqual(actual_change, expected_change, places=9)

    def test_backward_at_one_db(self) -> None:
        model = Linear(n_input=3, n_output=2)

        out = model.forward(self.data)
        loss = out.sum()

        dout = np.ones_like(out)  # derivative of loss is 1 for each element
        model.backward(dout)
        db = model.cache["db"]

        step = 0.01
        expected_change = step * db.sum()
        model.b += step
        out2 = model.forward(self.data)
        loss2 = out2.sum()
        actual_change = loss2 - loss
        self.assertAlmostEqual(actual_change, expected_change, places=9)

    def test_backward_random_dx(self) -> None:
        model = Linear(n_input=3, n_output=2)

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
        self.assertAlmostEqual(actual_change, expected_change, places=9)
