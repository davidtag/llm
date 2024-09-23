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
        model = Embedding(vocab_size=1000, context_size=256, d_model=512)
        self.assertEqual(model.n_params, 1000 * 512 + 256 * 512)

    def test_forward(self) -> None:
        """Test the forward pass."""
        model = Embedding(vocab_size=256, context_size=128, d_model=512)

        out = model.forward(self.data)
        self.assertEqual(out.shape, (6, 512))

    def test_backward_at_zero(self) -> None:
        """Test the backward pass with upstream gradient being 0."""
        model = Embedding(vocab_size=256, context_size=128, d_model=512)

        out = model.forward(self.data)

        dout = np.zeros_like(out)
        model.backward(dout)

        dtoken_embedding_matrix = model.cache["dtoken_embedding_matrix"]
        dposition_embedding_matrix = model.cache["dposition_embedding_matrix"]

        self.assertEqual(dtoken_embedding_matrix.shape, (256, 512))
        self.assertEqual(dposition_embedding_matrix.shape, (128, 512))

        self.assertTrue(np.all(dtoken_embedding_matrix == 0))
        self.assertTrue(np.all(dposition_embedding_matrix == 0))

    def test_backward_at_one(self) -> None:
        """Test the backward pass with upstream gradient being 1."""
        model = Embedding(vocab_size=256, context_size=128, d_model=512)

        out = model.forward(self.data)

        dout = np.ones_like(out)  # derivative of loss is 1 for each element
        model.backward(dout)

        dtoken_embedding_matrix = model.cache["dtoken_embedding_matrix"]
        dposition_embedding_matrix = model.cache["dposition_embedding_matrix"]

        # Token Embedding Matrix
        # -> Input sequence elements have gradients equal to # of times vocab items appears in input
        self.assertTrue(np.all(dtoken_embedding_matrix[0] == 1))
        self.assertTrue(np.all(dtoken_embedding_matrix[13] == 1))
        self.assertTrue(np.all(dtoken_embedding_matrix[17] == 1))
        self.assertTrue(np.all(dtoken_embedding_matrix[19] == 2))
        self.assertTrue(np.all(dtoken_embedding_matrix[255] == 1))
        # -> Other entries have no gradients
        self.assertTrue(np.all(dtoken_embedding_matrix[2] == 0))
        self.assertTrue(np.all(dtoken_embedding_matrix[16] == 0))
        self.assertTrue(np.all(dtoken_embedding_matrix[137] == 0))

        # Position Embedding Matrix
        # -> All positions up to the size of the input have gradient of 1
        self.assertTrue(np.all(dposition_embedding_matrix[:6] == 1))
        # -> All positions after it have gradeint 0 because they don't contribute to the loss
        self.assertTrue(np.all(dposition_embedding_matrix[6:] == 0))

    def test_backward_dtoken_random(self) -> None:
        """Test the backward pass for token gradients with random step."""
        model = Embedding(vocab_size=256, context_size=128, d_model=512)

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

    def test_backward_dposition_random(self) -> None:
        """Test the backward pass for position gradients with random step."""
        model = Embedding(vocab_size=256, context_size=128, d_model=512)

        out = model.forward(self.data)
        loss = out.sum()

        dout = np.ones_like(out)  # derivative of loss is 1 for each element
        model.backward(dout)
        dposition_embedding_matrix = model.cache["dposition_embedding_matrix"]

        step = np.random.normal(loc=0, scale=0.01, size=model.position_embedding_matrix.shape)
        expected_change = np.sum(step * dposition_embedding_matrix)
        model.position_embedding_matrix += step
        out2 = model.forward(self.data)
        loss2 = out2.sum()
        actual_change = loss2 - loss
        self.assertAlmostEqual(actual_change, expected_change, places=9)
