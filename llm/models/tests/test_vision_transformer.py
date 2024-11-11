"""Unit tests for vision_transformer.py."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock

import numpy as np

from llm.optimizers.base import Optimizer
from llm.models.vision_transformer import VisionTransformer


class TestVisionTransformer(unittest.TestCase):
    """Unit tests for VisionTransformer."""

    def setUp(self) -> None:
        self.data = np.random.standard_normal(size=(3, 16, 16, 3))  # batch of 3 normalized images
        self.C = 13
        self.model = VisionTransformer(
            n_classes=self.C,
            patch_size=8,
            canonical_width=16,
            canonical_height=16,
            n_channel=3,
            n_blocks=2,
            d_model=10,
            d_k=17,
            d_v=37,
            h=7,
            d_ff=256,
        )

    def test_n_params(self) -> None:
        """Test the layer reports the correct number of parameters."""
        embedding_params = 10 + 192 * 10 + (4 + 1) * 10
        initial_norm_params = 10 + 10
        encoder_params = 2 * (
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
        prediction_params = 10 * 13 + 13
        total_params = (
            embedding_params + initial_norm_params + encoder_params + final_norm_params + prediction_params
        )

        self.assertEqual(self.model.n_params, total_params)

    def test_get_parameters(self) -> None:
        """Test the get_parameters() method."""
        params = self.model.get_parameters()
        self.assertSetEqual(
            set(params.keys()),
            {"embedding_layer", "initial_norm", "encoder", "final_norm", "prediction_head"},
        )

    def test_get_parameters_and_load_parameters_roundtrip(self) -> None:
        """Test that the return value of get_parameters() can be loaded."""
        out1 = self.model.forward(self.data)

        params = self.model.get_parameters()

        self.model.load_parameters(params)
        out2 = self.model.forward(self.data)
        np.testing.assert_array_equal(out1, out2)

    def test_represent(self) -> None:
        """Test the forward pass up to the pre-final layer."""
        out = self.model.represent(self.data)
        self.assertEqual(out.shape, (3, 10))

    def test_forward(self) -> None:
        """Test the forward pass."""
        out = self.model.forward(self.data)
        self.assertEqual(out.shape, (3, self.C))

    def test_backward_at_zero(self) -> None:
        """Test the backward pass with upstream gradient being 0."""
        out = self.model.forward(self.data)

        dout = np.zeros_like(out)
        self.model.backward(dout)

        self.assertTrue(np.all(self.model.prediction_head.cache["dx"] == 0))
        self.assertTrue(np.all(self.model.final_norm.cache["dx"] == 0))
        self.assertTrue(np.all(self.model.encoder.cache["dx"] == 0))
        self.assertTrue(np.all(self.model.initial_norm.cache["dx"] == 0))
        self.assertTrue(np.all(self.model.embedding_layer.cache["dposition_embedding_matrix"] == 0))
        self.assertTrue(np.all(self.model.embedding_layer.cache["dpatch_proj"] == 0))
        self.assertTrue(np.all(self.model.embedding_layer.cache["dclass_token"] == 0))

    def test_save_and_load(self) -> None:
        """Test the ability to save and load checkpoint files."""
        with tempfile.NamedTemporaryFile() as f:
            path = Path(f.name)
            self.model.save(path)

            self.model.load(path)

            optimizer = MagicMock(spec=Optimizer)
            model2 = VisionTransformer.load_for_training(path, optimizer)
            self.assertTrue(model2.enable_grad)
            self.assertIs(model2.optimizer, optimizer)

            model3 = VisionTransformer.load_for_eval(path)
            self.assertFalse(model3.enable_grad)
            self.assertIsNone(model3.optimizer)

            with self.assertRaises(ValueError):  # load fails because there's a size mismatch
                model4 = VisionTransformer(n_classes=7, n_blocks=1)
                model4.load(path)
