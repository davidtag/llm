"""Unit tests for transformer.py."""

import unittest

import numpy as np

from llm.models.transformer import Transformer


class TestTransformer(unittest.TestCase):
    """Unit tests for Transformer."""

    def setUp(self) -> None:
        self.data = np.array([0, 2, 7, 2, 9, 4, 11])
        self.model = Transformer(vocab_size=13, n_blocks=2, d_model=10, d_k=17, d_v=37, h=7, d_ff=256)

    def test_n_params(self) -> None:
        embedding_params = 13 * 10
        block_stack_params = 2 * (
            # MultiHeadAttention
            7 * 10 * (17 + 17 + 37 + 37)
            # LayerNorm
            + 10 * 2
            # FeedForward
            + 10 * 256
            + 256
            + 256 * 10
            + 10
            # LayerNorm
            + 10 * 2
        )
        unembedding_params = 10 * 13 + 13
        total_params = embedding_params + block_stack_params + unembedding_params

        self.assertEqual(self.model.n_params, total_params)

    def test_forward(self) -> None:
        out = self.model.forward(self.data)
        self.assertEqual(out.shape, (7, 13))

    def test_backward_at_zero(self) -> None:
        out = self.model.forward(self.data)

        dout = np.zeros_like(out)
        self.model.backward(dout)

        self.assertTrue(np.all(self.model.unembedding_layer.cache["dx"] == 0))
        self.assertTrue(np.all(self.model.encoder.cache["dx"] == 0))
        self.assertTrue(np.all(self.model.embedding_layer.cache["dembedding_mat"] == 0))

    def test_predict(self) -> None:
        probs = self.model.predict(self.data)
        self.assertEqual(probs.shape, (13,))
        self.assertTrue(np.all(probs > 0))
        self.assertAlmostEqual(probs.sum(), 1)

    def test_generate(self) -> None:
        output = self.model.generate(self.data, max_tokens=5)
        self.assertTrue(output.shape, (5,))
        self.assertGreaterEqual(min(output), 0)
        self.assertLess(max(output), 13)
