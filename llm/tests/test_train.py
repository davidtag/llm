"""An integration test of FeedForward and CrossEntropyLoss, testing training end-to-end."""

import unittest

import numpy as np

from llm.optimizers.adam import Adam
from llm.optimizers.sgd import StochasticGradientDescent
from llm.layers.block import Block
from llm.layers.block_stack import BlockStack
from llm.layers.feed_forward import FeedForward
from llm.layers.layer_norm import LayerNorm
from llm.layers.linear import Linear
from llm.layers.multi_head_attention import MultiHeadAttention
from llm.loss.cross_entropy import CrossEntropyLoss
from llm.utils.math import softmax


class TestTrainFeedForwardWithCrossEntropyLoss(unittest.TestCase):
    """Tests a full training pipeline of FeedForward and CrossEntropyLoss."""

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

        initial_loss = None
        last_loss = None

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

    def test_train_with_adam(self, num_iters: int = 50) -> None:
        """Test that we can overfit a small training dataset using the Adam optimizer."""
        optimizer = Adam(lr=0.5)
        model = FeedForward(n_input=self.D, n_hidden=32, n_output=self.C, optimizer=optimizer)
        loss_fn = CrossEntropyLoss()

        initial_loss = None
        last_loss = None

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

    def test_train_with_layer_norm_and_adam(self, num_iters: int = 50) -> None:
        """Test that we can overfit a small training dataset using a normalization layer."""
        optimizer = Adam(lr=0.5)
        norm = LayerNorm(n_input=self.D, optimizer=optimizer)
        model = FeedForward(n_input=self.D, n_hidden=32, n_output=self.C, optimizer=optimizer)
        loss_fn = CrossEntropyLoss()

        initial_loss = None
        last_loss = None

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
        self.assert_probabilites_match_targets(probabilities, decimal=2)  # probs ~0.99

    def test_train_multihead_attention(self, num_iters: int = 50) -> None:
        """Test that we can overfit a small training dataset using a multi-head attention layer."""
        optimizer = Adam(lr=0.05)
        layer_1 = MultiHeadAttention(d_model=self.D, optimizer=optimizer)
        layer_2 = Linear(n_input=self.D, n_output=self.C, optimizer=optimizer)  # needed to project to output
        loss_fn = CrossEntropyLoss()

        initial_loss = None
        last_loss = None

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
        self.assert_probabilites_match_targets(probabilities, decimal=2)  # probs ~0.99

    def test_train_block(self, num_iters: int = 200) -> None:
        """Test that we can overfit a small training dataset using a Block layer."""
        optimizer = Adam(lr=0.05)
        layer_1 = Block(d_model=self.D, optimizer=optimizer)
        layer_2 = Linear(n_input=self.D, n_output=self.C, optimizer=optimizer)  # needed to project to output
        loss_fn = CrossEntropyLoss()

        initial_loss = None
        last_loss = None

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
        self.assert_probabilites_match_targets(probabilities, decimal=2)  # probs ~0.99

    def test_train_block_stack(self, num_iters: int = 1_000) -> None:
        """Test that we can overfit a small training dataset using a BlockStack layer."""

        optimizer = Adam(lr=0.01, beta_1=0.9, beta_2=0.98, epsilon=1e-9)  # 3,4 blocks=> lr=0.000175,2000 iter
        layer_1 = BlockStack(n_blocks=2, d_model=self.D, optimizer=optimizer)
        layer_2 = Linear(n_input=self.D, n_output=self.C, optimizer=optimizer)  # needed to project to output
        loss_fn = CrossEntropyLoss()

        initial_loss = None
        last_loss = None

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
        self.assert_probabilites_match_targets(probabilities, decimal=2)  # probs ~0.99


if __name__ == "__main__":
    unittest.main()
