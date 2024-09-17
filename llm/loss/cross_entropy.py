"""Implementation of a cross-entropy loss function."""

import numpy as np

from llm.utils.math import log_softmax, softmax


class CrossEntropyLoss:
    """Implements the cross-entropy, or negative loglikelihood, loss function."""

    def __init__(
        self,
        enable_grad: bool = True,
    ) -> None:
        """Initialize the loss head."""
        self.enable_grad = enable_grad
        self.cache: dict[str, np.ndarray] = {}

    def forward(self, logits: np.ndarray, targets: np.ndarray) -> np.float64:
        """Compute the loss for a set of logits and targets."""
        assert logits.ndim >= 2 and targets.ndim == logits.ndim - 1
        assert logits.shape[:-1] == targets.shape

        n_classes = logits.shape[-1]
        n_predictions = targets.size

        logits_stacked = logits.reshape(n_predictions, n_classes)
        targets_stacked = targets.reshape(n_predictions)

        log_probabilities = log_softmax(logits_stacked)
        loss_elems = log_probabilities[np.arange(n_predictions), targets_stacked]
        loss = -np.mean(loss_elems)

        if self.enable_grad:
            self.cache["logits"] = logits
            self.cache["targets"] = targets

        return loss

    def backward(self) -> np.ndarray:
        """Compute the gradient of the loss with respect to the logits."""
        assert self.enable_grad, "Cannot compute the backward pass with enable_grad=False"
        logits = self.cache["logits"]
        targets = self.cache["targets"]

        n_classes = logits.shape[-1]
        n_predictions = targets.size

        logits_stacked = logits.reshape(n_predictions, n_classes)
        targets_stacked = targets.reshape(n_predictions)

        I_target = np.zeros_like(logits_stacked)
        I_target[np.arange(n_predictions), targets_stacked] = 1
        probabilities = softmax(logits_stacked)
        dlogits = (1 / n_predictions) * (probabilities - I_target)

        dlogits = dlogits.reshape(logits.shape)

        return dlogits
