"""Unit tests for image_embedding.py."""

import operator
import unittest

import numpy as np

from llm.constants import DEFAULT_DTYPE
from llm.layers.image_embedding import ImageEmbedding


class TestImageEmbedding(unittest.TestCase):
    """Unit tests for ImageEmbedding."""

    def setUp(self) -> None:
        self.data = np.random.standard_normal(size=(3, 16, 16, 3))  # batch of 3 normalized images

    def test_failed_init(self) -> None:
        """Test paramter validation failures in the constructore."""
        with self.assertRaises(ValueError) as cm:
            ImageEmbedding(patch_size=17, canonical_height=137, canonical_width=137)
        self.assertEqual(str(cm.exception), "Width must be a multiple of patch size")

        with self.assertRaises(ValueError) as cm:
            ImageEmbedding(patch_size=16, canonical_height=137, canonical_width=224)
        self.assertEqual(str(cm.exception), "Height must be a multiple of patch size")

    def test_n_params(self) -> None:
        """Test the layer reports the correct number of parameters."""
        model = ImageEmbedding(patch_size=8, canonical_width=16, canonical_height=16, n_channel=3, d_model=64)
        self.assertEqual(
            model.n_params,
            (
                1 * 64  # class_token
                + 192 * 64  # patch_proj: 192 = 8 * 8 * 3
                + (4 + 1) * 64  # position_embedding_matrix
            ),
        )

    def test_get_parameters(self) -> None:
        """Test the get_parameters() method."""
        model = ImageEmbedding(patch_size=8, canonical_width=16, canonical_height=16, n_channel=3, d_model=64)
        params = model.get_parameters()
        self.assertSetEqual(set(params.keys()), {"class_token", "patch_proj", "position_embedding_matrix"})
        token = params["class_token"]
        proj = params["patch_proj"]
        pos = params["position_embedding_matrix"]
        assert isinstance(token, np.ndarray)
        assert isinstance(proj, np.ndarray)
        assert isinstance(pos, np.ndarray)
        self.assertEqual(token.shape, (1, 64))
        self.assertEqual(proj.shape, (192, 64))
        self.assertEqual(pos.shape, (5, 64))
        self.assertEqual(token.dtype, DEFAULT_DTYPE)
        self.assertEqual(proj.dtype, DEFAULT_DTYPE)
        self.assertEqual(pos.dtype, DEFAULT_DTYPE)

    def test_load_parameters(self) -> None:
        """Test the load_paramters() method."""
        model = ImageEmbedding(patch_size=8, canonical_width=16, canonical_height=16, n_channel=3, d_model=64)
        out1 = model.forward(self.data)

        token_new = np.zeros((1, 64), dtype=DEFAULT_DTYPE)
        proj_new = np.zeros((192, 64), dtype=DEFAULT_DTYPE)
        pos_new = np.zeros((5, 64), dtype=DEFAULT_DTYPE)
        params = {
            "class_token": token_new,
            "patch_proj": proj_new,
            "position_embedding_matrix": pos_new,
        }

        model.load_parameters(params)
        out2 = model.forward(self.data)
        np.testing.assert_array_compare(operator.__ne__, out1, out2)

    def test_load_parameters_failure(self) -> None:
        """Test validation in load_parameters()."""
        model = ImageEmbedding(patch_size=16)

        with self.assertRaises(ValueError) as cm:
            model.load_parameters({})
        self.assertEqual(str(cm.exception), "Missing parameters")

        with self.assertRaises(ValueError) as cm:
            model.load_parameters(
                {
                    "class_token": 0,  # type: ignore
                    "patch_proj": 0,  # type: ignore
                    "position_embedding_matrix": 0,  # type: ignore
                }
            )
        self.assertEqual(str(cm.exception), "Invalid shape for parameters map")

    def test_get_parameters_and_load_parameters_roundtrip(self) -> None:
        """Test that the return value of get_parameters() can be loaded."""
        model = ImageEmbedding(patch_size=8, canonical_width=16, canonical_height=16, n_channel=3, d_model=64)
        out1 = model.forward(self.data)

        params = model.get_parameters()

        model.load_parameters(params)
        out2 = model.forward(self.data)
        np.testing.assert_array_equal(out1, out2)

    def test_forward(self) -> None:
        """Test the forward pass."""
        model = ImageEmbedding(patch_size=8, canonical_width=16, canonical_height=16, n_channel=3, d_model=64)

        out = model.forward(self.data)
        self.assertEqual(out.shape, (3, 4 + 1, 64))  # W/H of 16 + patch_size=8 -> 4 patches; +1 class token

    def test_backward_at_zero(self) -> None:
        """Test the backward pass with upstream gradient being 0."""
        model = ImageEmbedding(patch_size=8, canonical_width=16, canonical_height=16, n_channel=3, d_model=64)

        out = model.forward(self.data)

        dout = np.zeros_like(out)
        model.backward(dout)

        dclass_token = model.cache["dclass_token"]
        dpatch_proj = model.cache["dpatch_proj"]
        dposition_embedding_matrix = model.cache["dposition_embedding_matrix"]

        self.assertEqual(dclass_token.shape, (1, 64))
        self.assertEqual(dpatch_proj.shape, (192, 64))
        self.assertEqual(dposition_embedding_matrix.shape, (5, 64))

        self.assertTrue(np.all(dclass_token == 0))
        self.assertTrue(np.all(dpatch_proj == 0))
        self.assertTrue(np.all(dposition_embedding_matrix == 0))

    def test_backward_at_one(self) -> None:
        """Test the backward pass with upstream gradient being 1."""
        model = ImageEmbedding(patch_size=8, canonical_width=16, canonical_height=16, n_channel=3, d_model=64)

        out = model.forward(self.data)
        loss = out.sum()

        dout = np.ones_like(out)  # derivative of loss is 1 for each element
        model.backward(dout)

        dclass_token = model.cache["dclass_token"]
        dpatch_proj = model.cache["dpatch_proj"]
        dposition_embedding_matrix = model.cache["dposition_embedding_matrix"]

        # Class Token
        # -> All positions have gradient of 3 (batch_size) because they're directly added to the output
        self.assertTrue(np.all(dclass_token == 3))

        # Position Embedding Matrix
        # -> All positions have gradient of 3 (batch_size) because each position is represented in the input
        self.assertTrue(np.all(dposition_embedding_matrix == 3))

        # Patch Projection Matrix
        step = 0.01
        expected_change = step * dpatch_proj.sum()
        model.patch_proj += step
        out2 = model.forward(self.data)
        loss2 = out2.sum()
        actual_change = loss2 - loss
        self.assertAlmostEqual(actual_change, expected_change, places=9)

    def test_backward_dclass_token_random(self) -> None:
        """Test the backward pass for the class_token gradient with random step."""
        model = ImageEmbedding(patch_size=8, canonical_width=16, canonical_height=16, n_channel=3, d_model=64)

        out = model.forward(self.data)
        loss = out.sum()

        dout = np.ones_like(out)
        model.backward(dout)
        dclass_token = model.cache["dclass_token"]

        step = np.random.normal(loc=0, scale=0.01, size=model.class_token.shape)
        expected_change = np.sum(step * dclass_token)
        model.class_token += step
        out2 = model.forward(self.data)
        loss2 = out2.sum()
        actual_change = loss2 - loss
        self.assertAlmostEqual(actual_change, expected_change, places=9)

    def test_backward_dpatch_proj_random(self) -> None:
        """Test the backward pass for the patch_proj gradient with random step."""
        model = ImageEmbedding(patch_size=8, canonical_width=16, canonical_height=16, n_channel=3, d_model=64)

        out = model.forward(self.data)
        loss = out.sum()

        dout = np.ones_like(out)
        model.backward(dout)
        dpatch_proj = model.cache["dpatch_proj"]

        step = np.random.normal(loc=0, scale=0.01, size=model.patch_proj.shape)
        expected_change = np.sum(step * dpatch_proj)
        model.patch_proj += step
        out2 = model.forward(self.data)
        loss2 = out2.sum()
        actual_change = loss2 - loss
        self.assertAlmostEqual(actual_change, expected_change, places=9)

    def test_backward_dposition_random(self) -> None:
        """Test the backward pass for position gradients with random step."""
        model = ImageEmbedding(patch_size=8, canonical_width=16, canonical_height=16, n_channel=3, d_model=64)

        out = model.forward(self.data)
        loss = out.sum()

        dout = np.ones_like(out)
        model.backward(dout)
        dposition_embedding_matrix = model.cache["dposition_embedding_matrix"]

        step = np.random.normal(loc=0, scale=0.01, size=model.position_embedding_matrix.shape)
        expected_change = np.sum(step * dposition_embedding_matrix)
        model.position_embedding_matrix += step
        out2 = model.forward(self.data)
        loss2 = out2.sum()
        actual_change = loss2 - loss
        self.assertAlmostEqual(actual_change, expected_change, places=9)
