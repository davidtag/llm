"""Implementation of a cross-entropy loss function."""

import numpy as np

from llm.utils.math import log_softmax, softmax


class CrossEntropyLoss(object):
    """Implements the cross-entropy, or negative loglikelihood, loss function."""

    def __init__(
        self,
        enable_grad: bool = True,
    ) -> None:
        """Initialize the loss head."""
        self.enable_grad = enable_grad
        self.cache = {}

    def forward(self, logits: np.ndarray, targets: np.ndarray) -> float:
        """Compute the loss for a set of logits and targets."""
        assert logits.ndim == 2 and targets.ndim == 1
        assert logits.shape[0] == targets.shape[0]

        n = logits.shape[0]
        log_probabilities = log_softmax(logits)
        loss_elems = log_probabilities[np.arange(n), targets]
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

        n = logits.shape[0]
        I_target = np.zeros_like(logits)
        I_target[np.arange(n), targets] = 1
        probabilities = softmax(logits)
        dlogits = (1 / n) * (probabilities - I_target)

        return dlogits
