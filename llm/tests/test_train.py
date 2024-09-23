# pylint: disable=invalid-name
"""A set of integration tests of various layers/losses/model/optimizers, testing training end-to-end."""

import unittest

import numpy as np

from llm.optimizers import Adam, StochasticGradientDescent
from llm.layers import Block, BlockStack, Embedding, FeedForward, LayerNorm, Linear, MultiHeadAttention
from llm.loss import CrossEntropyLoss
from llm.models import Transformer
from llm.utils.math import softmax


class TestTrainingEndToEnd(unittest.TestCase):
    """Tests a full training pipeline for various layers/losses/models/optmizers."""

    def setUp(self) -> None:
        self.data = np.array(
            [
                [-0.80672381, -0.08818247, 0.002, 0.005, -0.3],
                [0.63413982, 1.32233656, 0.332, -0.332, 1.223],
                [0.1814214, -0.50674539, -0.0223, 0.33546, 0.223],
                [1.16085551, -0.15033837, -0.332, -1.334, -0.09],
            ]
        )
        self.N = self.data.shape[0]
        self.D = self.data.shape[1]
        self.C = 3
        self.targets = np.array([0, 1, 1, 2])

    def assert_probabilites_match_targets(self, probabilities: np.ndarray, decimal: int = 2) -> None:
        """Assert that probabilities for the true labels are 1 within a decimal error."""
        assert probabilities.shape == (self.N, self.C)
        probabilities_for_true_labels = probabilities[np.arange(self.N), self.targets]
        expected = np.ones(shape=(self.N,))
        np.testing.assert_almost_equal(probabilities_for_true_labels, expected, decimal=decimal)

    def test_train_with_sgd(self, num_iters: int = 200) -> None:
        """Test that we can overfit a small training dataset using the SGD optimizer."""
        optimizer = StochasticGradientDescent(lr=1)
        model = FeedForward(n_input=self.D, n_hidden=32, n_output=self.C, optimizer=optimizer)
        loss_fn = CrossEntropyLoss()

        initial_loss = float("inf")
        last_loss = float("inf")

        for i in range(num_iters):
            # Forward Pass
            logits = model.forward(self.data)
            loss = loss_fn.forward(logits, self.targets)

            if i == 0:
                initial_loss = loss
            if i == num_iters - 1:
                last_loss = loss
                break

            # Backward Pass
            dlogits = loss_fn.backward()
            model.backward(dlogits)
            model.step()

        # Loss Improvement
        self.assertLess(last_loss, initial_loss)
        self.assertLess(last_loss, 0.01)

        # Predictions are Correct
        logits = model.forward(self.data)
        probabilities = softmax(logits)
        self.assert_probabilites_match_targets(probabilities, decimal=2)  # probs ~0.99

    def test_train_with_adam(self, num_iters: int = 75) -> None:
        """Test that we can overfit a small training dataset using the Adam optimizer."""
        optimizer = Adam(lr=0.5)
        model = FeedForward(n_input=self.D, n_hidden=32, n_output=self.C, optimizer=optimizer)
        loss_fn = CrossEntropyLoss()

        initial_loss = float("inf")
        last_loss = float("inf")

        for i in range(num_iters):
            # Forward Pass
            logits = model.forward(self.data)
            loss = loss_fn.forward(logits, self.targets)

            if i == 0:
                initial_loss = loss
            if i == num_iters - 1:
                last_loss = loss
                break

            # Backward Pass
            dlogits = loss_fn.backward()
            model.backward(dlogits)
            model.step()

        # Loss Improvement
        self.assertLess(last_loss, initial_loss)
        self.assertLess(last_loss, 1e-4)  # Adam much better than SGB at getting to minima

        # Predictions are Correct
        logits = model.forward(self.data)
        probabilities = softmax(logits)
        self.assert_probabilites_match_targets(probabilities, decimal=4)  # probs ~0.9999

    def test_train_with_layer_norm_and_adam(self, num_iters: int = 75) -> None:
        """Test that we can overfit a small training dataset using a normalization layer."""
        optimizer = Adam(lr=0.01)
        norm = LayerNorm(n_input=self.D, optimizer=optimizer)
        model = FeedForward(n_input=self.D, n_hidden=32, n_output=self.C, optimizer=optimizer)
        loss_fn = CrossEntropyLoss()

        initial_loss = float("inf")
        last_loss = float("inf")

        for i in range(num_iters):
            # Forward Pass
            data_norm = norm.forward(self.data)
            logits = model.forward(data_norm)
            loss = loss_fn.forward(logits, self.targets)

            if i == 0:
                initial_loss = loss
            if i == num_iters - 1:
                last_loss = loss
                break

            # Backward Pass
            dlogits = loss_fn.backward()
            model.backward(dlogits)
            norm.backward(model.cache["dx"])
            model.step()
            norm.step()

        # Loss Improvement
        self.assertLess(last_loss, initial_loss)
        self.assertLess(last_loss, 0.01)

        # Predictions are Correct
        data_norm = norm.forward(self.data)
        logits = model.forward(data_norm)
        probabilities = softmax(logits)
        self.assert_probabilites_match_targets(probabilities, decimal=2)

    def test_train_multihead_attention(self, num_iters: int = 50) -> None:
        """Test that we can overfit a small training dataset using a multi-head attention layer."""
        optimizer = Adam(lr=0.05)
        layer_1 = MultiHeadAttention(d_model=self.D, optimizer=optimizer)
        layer_2 = Linear(n_input=self.D, n_output=self.C, optimizer=optimizer)  # needed to project to output
        loss_fn = CrossEntropyLoss()

        initial_loss = float("inf")
        last_loss = float("inf")

        for i in range(num_iters):
            # Forward Pass
            hidden = layer_1.forward(self.data)
            logits = layer_2.forward(hidden)
            loss = loss_fn.forward(logits, self.targets)

            if i == 0:
                initial_loss = loss
            if i == num_iters - 1:
                last_loss = loss
                break

            # Backward Pass
            dlogits = loss_fn.backward()
            layer_2.backward(dlogits)
            layer_1.backward(layer_2.cache["dx"])
            layer_2.step()
            layer_1.step()

        # Loss Improvement
        self.assertLess(last_loss, initial_loss)
        self.assertLess(last_loss, 1e-4)

        # Predictions are Correct
        hidden = layer_1.forward(self.data)
        logits = layer_2.forward(hidden)
        probabilities = softmax(logits)
        self.assert_probabilites_match_targets(probabilities, decimal=2)

    def test_train_block(self, num_iters: int = 200) -> None:
        """Test that we can overfit a small training dataset using a Block layer."""
        optimizer = Adam(lr=0.05)
        layer_1 = Block(d_model=self.D, optimizer=optimizer)
        layer_2 = Linear(n_input=self.D, n_output=self.C, optimizer=optimizer)  # needed to project to output
        loss_fn = CrossEntropyLoss()

        initial_loss = float("inf")
        last_loss = float("inf")

        for i in range(num_iters):
            # Forward Pass
            hidden = layer_1.forward(self.data)
            logits = layer_2.forward(hidden)
            loss = loss_fn.forward(logits, self.targets)

            if i == 0:
                initial_loss = loss
            if i == num_iters - 1:
                last_loss = loss
                break

            # Backward Pass
            dlogits = loss_fn.backward()
            layer_2.backward(dlogits)
            layer_1.backward(layer_2.cache["dx"])
            layer_2.step()
            layer_1.step()

        # Loss Improvement
        self.assertLess(last_loss, initial_loss)
        self.assertLess(last_loss, 1e-3)

        # Predictions are Correct
        hidden = layer_1.forward(self.data)
        logits = layer_2.forward(hidden)
        probabilities = softmax(logits)
        self.assert_probabilites_match_targets(probabilities, decimal=2)

    def test_train_block_stack(self, num_iters: int = 1_000) -> None:
        """Test that we can overfit a small training dataset using a BlockStack layer."""

        optimizer = Adam(lr=0.01, beta_1=0.9, beta_2=0.98, epsilon=1e-9)  # 3,4 blocks=> lr=0.000175,2000 iter
        layer_1 = BlockStack(n_blocks=2, d_model=self.D, optimizer=optimizer)
        layer_2 = Linear(n_input=self.D, n_output=self.C, optimizer=optimizer)  # needed to project to output
        loss_fn = CrossEntropyLoss()

        initial_loss = float("inf")
        last_loss = float("inf")

        for i in range(num_iters):
            # Forward Pass
            hidden = layer_1.forward(self.data)
            logits = layer_2.forward(hidden)
            loss = loss_fn.forward(logits, self.targets)

            if i == 0:
                initial_loss = loss
            if i == num_iters - 1:
                last_loss = loss
                break

            # Backward Pass
            dlogits = loss_fn.backward()
            layer_2.backward(dlogits)
            layer_1.backward(layer_2.cache["dx"])
            layer_2.step()
            layer_1.step()

        # Loss Improvement
        self.assertLess(last_loss, initial_loss)
        self.assertLess(last_loss, 0.01)

        # Predictions are Correct
        hidden = layer_1.forward(self.data)
        logits = layer_2.forward(hidden)
        probabilities = softmax(logits)
        self.assert_probabilites_match_targets(probabilities, decimal=2)

    def test_train_embedding(self, num_iters: int = 50) -> None:
        """Test that we can train an embedding layer."""
        optimizer = Adam(lr=0.5)
        model = Embedding(vocab_size=self.C, context_window=self.N, d_model=self.C, optimizer=optimizer)
        loss_fn = CrossEntropyLoss()

        initial_loss = float("inf")
        last_loss = float("inf")

        for i in range(num_iters):
            # Forward Pass
            logits = model.forward(self.targets)
            loss = loss_fn.forward(logits, self.targets)

            if i == 0:
                initial_loss = loss
            if i == num_iters - 1:
                last_loss = loss
                break

            # Backward Pass
            dlogits = loss_fn.backward()
            model.backward(dlogits)
            model.step()

        # Loss Improvement
        self.assertLess(last_loss, initial_loss)
        self.assertLess(last_loss, 1e-4)

        # Predictions are Correct
        logits = model.forward(self.targets)
        probabilities = softmax(logits)
        self.assert_probabilites_match_targets(probabilities, decimal=4)

    def test_train_embedding_with_linear(self, num_iters: int = 50) -> None:
        """Test that we can train an embedding layer when stacked with a linear layer."""
        optimizer = Adam(lr=0.1)
        layer_1 = Embedding(vocab_size=500, context_window=128, d_model=512, optimizer=optimizer)
        layer_2 = Linear(n_input=512, n_output=self.C, optimizer=optimizer)
        loss_fn = CrossEntropyLoss()

        initial_loss = float("inf")
        last_loss = float("inf")

        data = np.array([234, 496, 210, 8])

        for i in range(num_iters):
            # Forward Pass
            hidden = layer_1.forward(data)
            logits = layer_2.forward(hidden)
            loss = loss_fn.forward(logits, self.targets)

            if i == 0:
                initial_loss = loss
            if i == num_iters - 1:
                last_loss = loss
                break

            # Backward Pass
            dlogits = loss_fn.backward()
            layer_2.backward(dlogits)
            layer_1.backward(layer_2.cache["dx"])
            # layer_2.step()  # intentionaly don't train layer_2, just the embedding
            layer_1.step()

        # Loss Improvement
        self.assertLess(last_loss, initial_loss)
        self.assertLess(last_loss, 1e-6)

        # Predictions are Correct
        hidden = layer_1.forward(data)
        logits = layer_2.forward(hidden)
        probabilities = softmax(logits)
        self.assert_probabilites_match_targets(probabilities, decimal=4)

    def test_train_transformer_depth_1(self, num_iters: int = 50) -> None:
        """Test that we can train a Transformer model with depth 1."""
        optimizer = Adam(lr=0.001)
        model = Transformer(vocab_size=self.C, context_window=self.N, n_blocks=1, optimizer=optimizer)
        loss_fn = CrossEntropyLoss()

        initial_loss = float("inf")
        last_loss = float("inf")

        for i in range(num_iters):
            # Forward Pass
            logits = model.forward(self.targets)
            loss = loss_fn.forward(logits, self.targets)

            if i == 0:
                initial_loss = loss
            if i == num_iters - 1:
                last_loss = loss
                break

            # Backward Pass
            dlogits = loss_fn.backward()
            model.backward(dlogits)
            model.step()

        # Loss Improvement
        self.assertLess(last_loss, initial_loss)
        self.assertLess(last_loss, 1e-3)

        # Predictions are Correct
        logits = model.forward(self.targets)
        probabilities = softmax(logits)
        self.assert_probabilites_match_targets(probabilities, decimal=4)

    def test_train_transformer_vocab_size_2(self, num_epochs: int = 10, context_window: int = 16) -> None:
        """Test that we can train a Transformer model on a small vocab size with predictable pattern."""
        optimizer = Adam(lr=0.03)
        model = Transformer(
            vocab_size=2,
            context_window=context_window,
            n_blocks=1,
            d_model=64,
            d_k=8,
            d_v=8,
            h=8,
            d_ff=256,
            optimizer=optimizer,
        )
        loss_fn = CrossEntropyLoss()

        training_sequence = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        self.assertEqual(len(training_sequence), context_window + 1)

        initial_loss = float("inf")
        last_loss = float("inf")

        for i in range(num_epochs):
            for j in range(context_window):
                data = training_sequence[j:-1]
                targets = training_sequence[(j + 1) :]

                # Forward Pass
                logits = model.forward(data)
                loss = loss_fn.forward(logits, targets)

                if i == 0 and j == 0:
                    initial_loss = loss
                if i == num_epochs - 1 and j == context_window - 1:
                    last_loss = loss
                    break

                # Backward Pass
                dlogits = loss_fn.backward()
                model.backward(dlogits)
                model.step()

        # Loss Improvement
        self.assertLess(last_loss, initial_loss)
        self.assertLess(last_loss, 1e-3)

        # Predictions are Correct
        self.assertAlmostEqual(model.predict(np.array([0]))[1], 1.0, places=4)  # (0,) -> 1
        self.assertAlmostEqual(model.predict(np.array([1]))[0], 1.0, places=4)  # (1,) -> 0
        self.assertAlmostEqual(model.predict(np.array([0, 1]))[0], 1.0, places=4)  # (0, 1,) -> 0
        self.assertAlmostEqual(model.predict(np.array([1, 0]))[1], 1.0, places=4)  # (1, 0,) -> 1
        self.assertAlmostEqual(model.predict(np.array([0, 1, 0]))[1], 1.0, places=4)
        self.assertAlmostEqual(model.predict(np.array([1, 0, 1]))[0], 1.0, places=4)
        self.assertAlmostEqual(model.predict(np.array([0, 1, 0, 1]))[0], 1.0, places=4)
        self.assertAlmostEqual(model.predict(np.array([0, 1, 0, 1, 0]))[1], 1.0, places=4)

        # Generation is correct
        np.testing.assert_array_equal(
            model.generate(np.array([0, 1, 0, 1, 0, 1, 0, 1]), max_tokens=10, is_random=False),
            np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
        )
        np.testing.assert_array_equal(
            model.generate(np.array([0, 1, 0, 1, 0, 1, 0, 1, 0]), max_tokens=10, is_random=False),
            np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0]),
        )
