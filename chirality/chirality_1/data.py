"""Paired 3D orthonormal frames with a Gaussian origin and chirality labels."""

from dataclasses import dataclass

import numpy as np


@dataclass
class PairedAxes:
    # c=0: left-handed, c=1: right-handed. Same j identifies a pair.
    coordinates: np.ndarray  # (2, number_of_pairs, 4, 3): origin, u, v, w

    def flat(self):
        """Flatten in chirality-major order: i = c * number_of_pairs + j."""
        count = self.coordinates.shape[1]
        return (
            self.coordinates.reshape(2 * count, -1).copy(),
            np.repeat(np.arange(2, dtype=np.int64), count),
        )


def generate_pairs(sample_count, rng, origin_mean=0.0, origin_std=1.0):
    if sample_count <= 0 or sample_count % 2:
        raise ValueError("sample_count must be positive and even (two samples per pair)")
    count = sample_count // 2
    coordinates = np.empty((2, count, 4, 3), dtype=np.float32)
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
        origin = rng.normal(loc=origin_mean, scale=origin_std, size=3)
        coordinates[:, j, 0] = origin
        coordinates[1, j, 1:] = frame
        frame[2] *= -1
        coordinates[0, j, 1:] = frame
    return PairedAxes(coordinates)


@dataclass
class AxesSplit:
    coordinates: np.ndarray  # (sample_count, 4, 3), chirality-major order
    labels: np.ndarray
    pair_ids: np.ndarray  # Global j; exactly one occurrence per split.

    def flat(self):
        return self.coordinates.reshape(len(self.labels), -1).copy(), self.labels.copy()


def generate_datasets(train_size=100, test_size=100, seed=42, origin_mean=0.0, origin_std=1.0):
    """Put exactly one member of every pair in each split, balanced by label."""
    if train_size != test_size or train_size <= 0 or train_size % 2:
        raise ValueError("Opposite-pair splitting requires equal, positive, even split sizes")
    rng = np.random.default_rng(seed)
    pairs = generate_pairs(2 * train_size, rng, origin_mean, origin_std)
    # Randomly choose which chirality enters training, with exact class balance.
    train_labels = rng.permutation(np.repeat(np.arange(2, dtype=np.int64), train_size // 2))
    pair_ids = np.arange(train_size)
    splits = []
    for labels in (train_labels, 1 - train_labels):
        order = np.argsort(labels, kind="stable")
        splits.append(AxesSplit(
            pairs.coordinates[labels[order], pair_ids[order]].copy(),
            labels[order].copy(), pair_ids[order].copy(),
        ))
    return tuple(splits)
