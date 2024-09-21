"""Unit tests for embedding.py."""

import unittest

import numpy as np

from llm.layers.embedding import Embedding


class TestEmbedding(unittest.TestCase):
    """Unit tests for Embedding."""

    def setUp(self) -> None:
        self.data = np.array([0, 13, 19, 17, 19, 255])

    def test_n_params(self) -> None:
        """Test the layer reports the correct number of parameters."""
        model = Embedding(vocab_size=1000, context_window=256, d_model=512)
        self.assertEqual(model.n_params, 1000 * 512 + 256 * 512)

    def test_forward(self) -> None:
        """Test the forward pass."""
        model = Embedding(vocab_size=256, context_window=128, d_model=512)

        out = model.forward(self.data)
        self.assertEqual(out.shape, (6, 512))

        np.testing.assert_array_equal(out[2], out[4])
        self.assertFalse(np.all(out[2] == out[3]))

    def test_backward_at_zero(self) -> None:
        """Test the backward pass with upstream gradient being 0."""
        model = Embedding(vocab_size=256, context_window=128, d_model=512)

        out = model.forward(self.data)

        dout = np.zeros_like(out)
        model.backward(dout)
        dtoken_embedding_matrix = model.cache["dtoken_embedding_matrix"]

        self.assertEqual(dtoken_embedding_matrix.shape, (256, 512))
        self.assertTrue(np.all(dtoken_embedding_matrix == 0))

    def test_backward_at_one(self) -> None:
        """Test the backward pass with upstream gradient being 1."""
        model = Embedding(vocab_size=256, context_window=128, d_model=512)

        out = model.forward(self.data)

        dout = np.ones_like(out)  # derivative of loss is 1 for each element
        model.backward(dout)
        dtoken_embedding_matrix = model.cache["dtoken_embedding_matrix"]

        # Input sequence elements have gradients equal to # of times vocab items appears in input
        self.assertTrue(np.all(dtoken_embedding_matrix[0] == 1))
        self.assertTrue(np.all(dtoken_embedding_matrix[13] == 1))
        self.assertTrue(np.all(dtoken_embedding_matrix[17] == 1))
        self.assertTrue(np.all(dtoken_embedding_matrix[19] == 2))
        self.assertTrue(np.all(dtoken_embedding_matrix[255] == 1))

        # Other entries have no gradients
        self.assertTrue(np.all(dtoken_embedding_matrix[2] == 0))
        self.assertTrue(np.all(dtoken_embedding_matrix[16] == 0))
        self.assertTrue(np.all(dtoken_embedding_matrix[137] == 0))

    def test_backward_random(self) -> None:
        """Test the backward pass with upstream gradient being random."""
        model = Embedding(vocab_size=256, context_window=128, d_model=512)

        out = model.forward(self.data)
        loss = out.sum()

        dout = np.ones_like(out)  # derivative of loss is 1 for each element
        model.backward(dout)
        dtoken_embedding_matrix = model.cache["dtoken_embedding_matrix"]

        step = np.random.normal(loc=0, scale=0.01, size=model.token_embedding_matrix.shape)
        expected_change = np.sum(step * dtoken_embedding_matrix)
        model.token_embedding_matrix += step
        out2 = model.forward(self.data)
        loss2 = out2.sum()
        actual_change = loss2 - loss
        self.assertAlmostEqual(actual_change, expected_change, places=9)
