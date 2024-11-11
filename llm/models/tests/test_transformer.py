"""Unit tests for transformer.py."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock

import numpy as np

from llm.optimizers import Optimizer
from llm.models.transformer import Transformer


class TestTransformer(unittest.TestCase):
    """Unit tests for Transformer."""

    def setUp(self) -> None:
        self.data = np.array(  # shape = (2, 7)
            [
                [3, 2, 9, 1, 8, 4, 1],
                [0, 2, 7, 2, 9, 4, 9],
            ]
        )
        self.model = Transformer(
            vocab_size=13, context_size=128, n_blocks=2, d_model=10, d_k=17, d_v=37, h=7, d_ff=256
        )

    def test_n_params(self) -> None:
        """Test the layer reports the correct number of parameters."""
        embedding_params = 13 * 10 + 128 * 10
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
        final_norm_params = 10 + 10
        unembedding_params = 10 * 13 + 13
        total_params = embedding_params + block_stack_params + final_norm_params + unembedding_params

        self.assertEqual(self.model.n_params, total_params)

    def test_get_parameters(self) -> None:
        """Test the get_parameters() method."""
        params = self.model.get_parameters()
        self.assertSetEqual(
            set(params.keys()), {"embedding_layer", "decoder", "final_norm", "unembedding_layer"}
        )

    def test_get_parameters_and_load_parameters_roundtrip(self) -> None:
        """Test that the return value of get_parameters() can be loaded."""
        out1 = self.model.forward(self.data)

        params = self.model.get_parameters()

        self.model.load_parameters(params)
        out2 = self.model.forward(self.data)
        np.testing.assert_array_equal(out1, out2)

    def test_forward(self) -> None:
        """Test the forward pass."""
        out = self.model.forward(self.data)
        self.assertEqual(out.shape, (2, 7, 13))

    def test_backward_at_zero(self) -> None:
        """Test the backward pass with upstream gradient being 0."""
        out = self.model.forward(self.data)

        dout = np.zeros_like(out)
        self.model.backward(dout)

        self.assertTrue(np.all(self.model.unembedding_layer.cache["dx"] == 0))
        # TODO(dtag): add test for final norm
        self.assertTrue(np.all(self.model.decoder.cache["dx"] == 0))
        self.assertTrue(np.all(self.model.embedding_layer.cache["dtoken_embedding_matrix"] == 0))
        self.assertTrue(np.all(self.model.embedding_layer.cache["dposition_embedding_matrix"] == 0))

    def test_predict(self) -> None:
        """Test the predict method."""
        probs = self.model.predict(self.data[0])
        self.assertEqual(probs.shape, (13,))  # probability distribution over the vocab
        self.assertTrue(np.all(probs > 0))
        self.assertAlmostEqual(probs.sum(), 1)

    def test_generate(self) -> None:
        """Test the generate method."""
        output = self.model.generate(self.data[0], max_tokens=5)
        self.assertTrue(output.shape, (5,))
        self.assertGreaterEqual(np.min(output), 0)
        self.assertLess(np.max(output), 13)

    def test_save_and_load(self) -> None:
        """Test the ability to save and load checkpoint files."""
        with tempfile.NamedTemporaryFile() as f:
            path = Path(f.name)
            self.model.save(path)

            self.model.load(path)

            optimizer = MagicMock(spec=Optimizer)
            model2 = Transformer.load_for_training(path, optimizer)
            self.assertTrue(model2.enable_grad)
            self.assertIs(model2.optimizer, optimizer)

            model3 = Transformer.load_for_eval(path)
            self.assertFalse(model3.enable_grad)
            self.assertIsNone(model3.optimizer)

            with self.assertRaises(ValueError):  # load fails because there's a size mismatch
                model4 = Transformer(vocab_size=1, context_size=1, n_blocks=1)
                model4.load(path)
