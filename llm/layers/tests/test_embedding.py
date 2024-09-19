"""Unit tests for embedding.py."""

import unittest

import numpy as np

from llm.layers.embedding import Embedding


class TestEmbedding(unittest.TestCase):
    """Unit tests for Embedding."""

    def setUp(self) -> None:
        self.data = np.array([0, 13, 19, 17, 19, 255])

    def test_n_params(self) -> None:
        model = Embedding(vocab_size=1000, d_model=512)
        self.assertEqual(model.n_params, 512_000)

    def test_forward(self) -> None:
        model = Embedding(vocab_size=256, d_model=512)

        out = model.forward(self.data)
        self.assertEqual(out.shape, (6, 512))

        np.testing.assert_array_equal(out[2], out[4])
        self.assertFalse(np.all(out[2] == out[3]))

    def test_backward_at_zero(self) -> None:
        model = Embedding(vocab_size=256, d_model=512)

        out = model.forward(self.data)

        dout = np.zeros_like(out)
        model.backward(dout)
        dembedding_mat = model.cache["dembedding_mat"]

        self.assertEqual(dembedding_mat.shape, (256, 512))
        self.assertTrue(np.all(dembedding_mat == 0))

    def test_backward_at_one(self) -> None:
        model = Embedding(vocab_size=256, d_model=512)

        out = model.forward(self.data)

        dout = np.ones_like(out)  # derivative of loss is 1 for each element
        model.backward(dout)
        dembedding_mat = model.cache["dembedding_mat"]

        # Input sequence elements have gradients equal to # of times vocab items appears in input
        self.assertTrue(np.all(dembedding_mat[0] == 1))
        self.assertTrue(np.all(dembedding_mat[13] == 1))
        self.assertTrue(np.all(dembedding_mat[17] == 1))
        self.assertTrue(np.all(dembedding_mat[19] == 2))
        self.assertTrue(np.all(dembedding_mat[255] == 1))

        # Other entries have no gradients
        self.assertTrue(np.all(dembedding_mat[2] == 0))
        self.assertTrue(np.all(dembedding_mat[16] == 0))
        self.assertTrue(np.all(dembedding_mat[137] == 0))

    def test_backward_random(self) -> None:
        model = Embedding(vocab_size=256, d_model=512)

        out = model.forward(self.data)
        loss = out.sum()

        dout = np.ones_like(out)  # derivative of loss is 1 for each element
        model.backward(dout)
        dembedding_mat = model.cache["dembedding_mat"]

        step = np.random.normal(loc=0, scale=0.01, size=model.embedding_mat.shape)
        expected_change = np.sum(step * dembedding_mat)
        model.embedding_mat += step
        out2 = model.forward(self.data)
        loss2 = out2.sum()
        actual_change = loss2 - loss
        self.assertAlmostEqual(actual_change, expected_change, places=9)
