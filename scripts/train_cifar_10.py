"""Train a VisionTransformer (ViT) on the CIFAR-10 image classification task."""

import numpy as np

from llm.data.loaders import load_cifar_10_split
from llm.data.registry import ImageDataRegistry
from llm.loss.cross_entropy import CrossEntropyLoss
from llm.models import VisionTransformer
from llm.optimizers.adam import Adam


def _get_data_statistics(x: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    x_mean = np.mean(x, axis=0, keepdims=True)
    x_var = np.mean(np.square(x - x_mean), axis=0, keepdims=True)
    x_std = np.sqrt(x_var + eps)
    return x_mean, x_std


def _load_data() -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    registry = ImageDataRegistry()

    Xs, Ys = [], []
    for file in registry.train_files:
        X, Y = load_cifar_10_split(file)
        Xs.append(X)
        Ys.append(Y)
    X_train = np.vstack(Xs)
    Y_train = np.concatenate(Ys)
    assert X_train.shape == (50_000, 32, 32, 3)
    assert Y_train.shape == (50_000,)

    X_test, Y_test = load_cifar_10_split(registry.test_file)
    assert X_test.shape == (10_000, 32, 32, 3)
    assert Y_test.shape == (10_000,)

    m, s = _get_data_statistics(X_train)

    X_train = (X_train - m) / s
    X_test = (X_test - m) / s

    print(f"Loaded {X_train.shape[0]:,} train examples")
    print(f"Loaded {X_test.shape[0]:,} test examples")
    return ((X_train, Y_train), (X_test, Y_test))


def _get_accuracy(logits: np.ndarray, targets: np.ndarray) -> float:
    assert logits.shape[0] == targets.shape[0]
    predictions = np.argmax(logits, axis=1)
    accuracy = 100 * np.mean(predictions == targets)
    return accuracy


def _initialize_model_for_training(
    lr: float,
) -> VisionTransformer:
    optimizer = Adam(lr=lr)
    model = VisionTransformer(
        n_classes=10,
        patch_size=8,
        canonical_width=32,
        canonical_height=32,
        n_channel=3,
        n_blocks=2,
        d_model=64,
        d_k=8,
        d_v=8,
        h=8,
        d_ff=128,
        optimizer=optimizer,
    )
    print(f"Initialized model with n_params={model.n_params:,}")
    return model


def _train_model(
    model: VisionTransformer,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 32,
    num_batches: int = 20,
    test_evaluation_freq: int = 30,
    test_evaluation_batch_size: int = 512,
) -> VisionTransformer:
    assert x_train.shape[0] == y_train.shape[0]
    assert x_test.shape[0] == y_test.shape[0]

    loss_fn = CrossEntropyLoss()

    def get_train_batch() -> tuple[np.ndarray, np.ndarray]:
        idxs = np.random.randint(low=0, high=x_train.shape[0], size=batch_size)
        return x_train[idxs], y_train[idxs]

    def get_test_batch() -> tuple[np.ndarray, np.ndarray]:
        idxs = np.random.randint(low=0, high=x_test.shape[0], size=test_evaluation_batch_size)
        return x_test[idxs], y_test[idxs]

    def get_test_loss_and_accuracy() -> tuple[float, float]:
        data, targets = get_test_batch()
        logits = model.forward(data)
        loss = loss_fn.forward(logits, targets)
        accuracy = _get_accuracy(logits, targets)
        return float(loss), accuracy

    print("-- Training --------------------------------------------------------------")
    for i in range(num_batches):
        data_i, targets_i = get_train_batch()

        # Forward Pass
        logits = model.forward(data_i)
        loss = loss_fn.forward(logits, targets_i)

        # Backward Pass
        dlogits = loss_fn.backward()
        model.backward(dlogits)
        model.step()

        # Report batch loss and accuracy
        accuracy = _get_accuracy(logits, targets_i)
        if i % test_evaluation_freq == 0:
            test_loss, test_accuracy = get_test_loss_and_accuracy()
            test_str = f"| {test_loss=:6.3f}  {test_accuracy=:5.1f}%"
        else:
            test_str = ""
        print(f"  {i + 1:6}/{num_batches}: {loss=:6.3f}  {accuracy=:5,.1f}% {test_str}")

    return model


def main() -> None:
    """Entrypoint."""
    (X_train, Y_train), (X_test, Y_test) = _load_data()
    model = _initialize_model_for_training(lr=0.001)

    _train_model(
        model=model,
        x_train=X_train,
        y_train=Y_train,
        x_test=X_test,
        y_test=Y_test,
        batch_size=128,
        num_batches=50_000,
    )


if __name__ == "__main__":
    main()
