"""High-dimensional vectors labelled by orientation in a fixed 3D subspace."""

from dataclasses import dataclass

import numpy as np


@dataclass
class PairedAxes:
    # c=0: left-handed, c=1: right-handed. Same j identifies a pair.
    coordinates: np.ndarray  # (2, number_of_pairs, 3, dim): u, v, w
    projection: np.ndarray  # (dim, 3); shared task basis, not a model input

    def flat(self):
        """Flatten in chirality-major order: i = c * number_of_pairs + j."""
        count = self.coordinates.shape[1]
        return (
            self.coordinates.reshape(2 * count, -1).copy(),
            np.repeat(np.arange(2, dtype=np.int64), count),
        )


def make_projection(dim, seed=17):
    if not isinstance(dim, (int, np.integer)) or dim < 3:
        raise ValueError("dim must be an integer >= 3")
    if dim == 3:
        return np.eye(3)  # Recover the original physical 3D orientation labels.
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(dim, 3)))
    return q * np.where(np.diag(r) >= 0, 1., -1.)


def generate_pairs(sample_count, rng, dim=16, projection_seed=17, noise_std=1.0):
    if sample_count <= 0 or sample_count % 2:
        raise ValueError("sample_count must be positive and even (two samples per pair)")
    projection = make_projection(dim, projection_seed)
    if not np.isfinite(noise_std) or noise_std < 0:
        raise ValueError("noise_std must be finite and nonnegative")
    count = sample_count // 2
    coordinates = np.empty((2, count, 3, dim), dtype=np.float32)
    for j in range(count):
        # Draw a Haar-distributed right-handed 3D orthonormal frame.
        while True:
            a = rng.normal(size=3)
            if np.linalg.norm(a) < 1e-12:
                continue
            a /= np.linalg.norm(a)
            b = rng.normal(size=3)
            b -= np.dot(a, b) * a
            if np.linalg.norm(b) < 1e-12:
                continue
            b /= np.linalg.norm(b)
            frame = np.stack((a, b, np.cross(a, b)))
            break
        signal = frame @ projection.T
        noise = np.zeros((3, dim))
        if dim > 3 and noise_std > 0:
            gaussian = rng.normal(scale=noise_std, size=(3, dim))
            noise = gaussian - (gaussian @ projection) @ projection.T
        # No normalization: projected axes must retain unit length and |det|=1.
        coordinates[1, j] = signal + noise
        signal[2] *= -1
        coordinates[0, j] = signal + noise
    return PairedAxes(coordinates, projection)


@dataclass
class AxesSplit:
    coordinates: np.ndarray  # (sample_count, 3, dim), chirality-major order
    labels: np.ndarray
    pair_ids: np.ndarray  # Global j; exactly one occurrence per split.
    projection: np.ndarray

    def flat(self):
        return self.coordinates.reshape(len(self.labels), -1).copy(), self.labels.copy()


def generate_datasets(train_size=100, test_size=100, seed=42, dim=16, projection_seed=17,
                      noise_std=1.0):
    """Put exactly one member of every pair in each split, balanced by label."""
    if train_size != test_size or train_size <= 0 or train_size % 2:
        raise ValueError("Opposite-pair splitting requires equal, positive, even split sizes")
    rng = np.random.default_rng(seed)
    pairs = generate_pairs(2 * train_size, rng, dim, projection_seed, noise_std)
    # Randomly choose which chirality enters training, with exact class balance.
    train_labels = rng.permutation(np.repeat(np.arange(2, dtype=np.int64), train_size // 2))
    pair_ids = np.arange(train_size)
    splits = []
    for labels in (train_labels, 1 - train_labels):
        order = np.argsort(labels, kind="stable")
        splits.append(AxesSplit(
            pairs.coordinates[labels[order], pair_ids[order]].copy(),
            labels[order].copy(), pair_ids[order].copy(), pairs.projection.copy(),
        ))
    return tuple(splits)
