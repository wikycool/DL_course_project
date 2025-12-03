"""Dataset loading and batching utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Generator, Literal, Tuple

import numpy as np


try:  # Optional dependency
    from keras.datasets import fashion_mnist, mnist, cifar10  # type: ignore
except Exception:  # pragma: no cover - keras is optional for CPU-only setups
    fashion_mnist = mnist = cifar10 = None


ArrayLike = np.ndarray


def _load_from_keras(dataset: Literal["fashion_mnist", "mnist", "cifar10"], flatten: bool) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    if fashion_mnist is None:
        raise ImportError("Keras is not available. Install 'tensorflow' or switch to source='local'.")

    if dataset == "mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    elif dataset == "fashion_mnist":
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    elif dataset == "cifar10":
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_train = y_train.flatten()
        y_test = y_test.flatten()
    else:
        raise ValueError(f"Unsupported dataset '{dataset}'.")

    if flatten:
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)

    return x_train.astype(np.float32), y_train.astype(np.int64), x_test.astype(np.float32), y_test.astype(np.int64)


def _download_fashion_mnist_local(data_dir: Path) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    import gzip
    import urllib.request

    base_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }

    data_dir.mkdir(parents=True, exist_ok=True)

    def load_images(filename: str) -> ArrayLike:
        path = data_dir / filename
        if not path.exists():
            urllib.request.urlretrieve(base_url + filename, path)
        with gzip.open(path, "rb") as fh:
            fh.read(16)
            images = np.frombuffer(fh.read(), dtype=np.uint8).reshape(-1, 784)
        return images.astype(np.float32)

    def load_labels(filename: str) -> ArrayLike:
        path = data_dir / filename
        if not path.exists():
            urllib.request.urlretrieve(base_url + filename, path)
        with gzip.open(path, "rb") as fh:
            fh.read(8)
            labels = np.frombuffer(fh.read(), dtype=np.uint8)
        return labels.astype(np.int64)

    x_train = load_images(files["train_images"])
    y_train = load_labels(files["train_labels"])
    x_test = load_images(files["test_images"])
    y_test = load_labels(files["test_labels"])

    return x_train, y_train, x_test, y_test


def _download_cifar10_local(data_dir: Path) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    import pickle
    import tarfile
    import urllib.request

    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    tar_path = data_dir / "cifar-10-python.tar.gz"
    extract_path = data_dir / "cifar-10-batches-py"

    data_dir.mkdir(parents=True, exist_ok=True)

    if not extract_path.exists():
        if not tar_path.exists():
            urllib.request.urlretrieve(url, tar_path)
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(data_dir)

    def load_batch(filepath: Path) -> Tuple[ArrayLike, ArrayLike]:
        with filepath.open("rb") as fh:
            batch = pickle.load(fh, encoding="bytes")
        images = batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        labels = np.array(batch[b"labels"], dtype=np.int64)
        return images.astype(np.float32), labels

    x_train_list = []
    y_train_list = []
    for i in range(1, 6):
        batch_path = extract_path / f"data_batch_{i}"
        x_batch, y_batch = load_batch(batch_path)
        x_train_list.append(x_batch)
        y_train_list.append(y_batch)

    x_train = np.concatenate(x_train_list, axis=0).reshape(-1, 32 * 32 * 3)
    y_train = np.concatenate(y_train_list, axis=0)

    x_test, y_test = load_batch(extract_path / "test_batch")
    x_test = x_test.reshape(x_test.shape[0], -1)

    return x_train, y_train, x_test, y_test


def load_dataset(
    dataset: Literal["fashion_mnist", "mnist", "cifar10"] = "fashion_mnist",
    source: Literal["keras", "local"] = "keras",
    data_dir: str | Path = "data",
    normalize: bool = True,
    flatten: bool = True,
) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    """Load a dataset either via Keras or local download."""

    data_dir = Path(data_dir)

    if source == "keras":
        x_train, y_train, x_test, y_test = _load_from_keras(dataset, flatten)
    elif source == "local":
        if dataset in {"fashion_mnist", "mnist"}:
            x_train, y_train, x_test, y_test = _download_fashion_mnist_local(data_dir / dataset)
        elif dataset == "cifar10":
            x_train, y_train, x_test, y_test = _download_cifar10_local(data_dir / dataset)
        else:
            raise ValueError(f"Unsupported dataset '{dataset}'.")
    else:
        raise ValueError(f"Unsupported source '{source}'.")

    if normalize:
        x_train = x_train / 255.0
        x_test = x_test / 255.0

    return x_train, y_train, x_test, y_test


def train_val_split(
    x: ArrayLike,
    y: ArrayLike,
    validation_split: float = 0.1,
    seed: int | None = 42,
) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    """Split training data into train/validation subsets."""

    if not 0 < validation_split < 1:
        raise ValueError("validation_split must be in (0, 1).")

    rng = np.random.default_rng(seed)
    indices = np.arange(x.shape[0])
    rng.shuffle(indices)

    split = int(x.shape[0] * (1 - validation_split))
    train_idx, val_idx = indices[:split], indices[split:]

    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def one_hot_encode(labels: ArrayLike, num_classes: int) -> ArrayLike:
    """Convert label indices to one-hot encoding."""
    one_hot = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    one_hot[np.arange(labels.shape[0]), labels] = 1.0
    return one_hot


def create_batches(
    x: ArrayLike,
    y: ArrayLike,
    batch_size: int,
    shuffle: bool = True,
    seed: int | None = None,
) -> Generator[Tuple[ArrayLike, ArrayLike], None, None]:
    """Generate mini-batches from the dataset."""

    n_samples = x.shape[0]
    indices = np.arange(n_samples)

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_idx = indices[start:end]
        yield x[batch_idx], y[batch_idx]
