"""Display both chiralities of a pair, as coordinates and binary voxels."""

import matplotlib.pyplot as plt
import numpy as np

from data import generate_datasets


# Use the experiment settings directly; importing train does not start training.
from train import TRAIN_SIZE, TEST_SIZE, DATA_SEED, ORIGIN_MEAN, ORIGIN_STD
PAIR_INDEX = 0  # Zero-based j, not the flattened sample index.
GRID_SIZE = 10
CELL_SIZE = 0.25


def voxelize(vectors, grid_size=GRID_SIZE, cell_size=CELL_SIZE):
    """Binary union of axis segments; translate origin to the grid center.

    Visit the cells between every segment/grid-plane intersection. Translation
    is for display only: the model receives the original Gaussian origin.
    """
    if grid_size <= 0 or cell_size <= 0 or grid_size * cell_size <= 2:
        raise ValueError("Display grid must have positive sizes and span > 2 units")
    center = np.full(3, grid_size * cell_size / 2)
    occupied = np.zeros((grid_size,) * 3, dtype=bool)
    boundaries = np.arange(grid_size + 1) * cell_size
    for vector in vectors:
        times = [0.0, 1.0]
        for k in range(3):
            if abs(vector[k]) > 1e-12:
                crossings = (boundaries - center[k]) / vector[k]
                times.extend(crossings[(crossings > 0) & (crossings < 1)])
        times = np.unique(times)
        samples = np.r_[0., (times[:-1] + times[1:]) / 2, 1.]
        points = center + samples[:, None] * vector
        indices = np.floor(points / cell_size).astype(int)
        if np.any(indices < 0) or np.any(indices >= grid_size):
            raise ValueError("Axis lies outside display grid")
        occupied[tuple(indices.T)] = True
    return occupied


def plot_pair(dataset, pair_index=0, grid_size=GRID_SIZE, cell_size=CELL_SIZE, other=None):
    """Return a figure for pair j. Caller may show it or save it."""
    if other is None:
        if not 0 <= pair_index < dataset.coordinates.shape[1] or dataset.coordinates.ndim != 4:
            raise ValueError("For split datasets, pass other=test to display the counterpart")
        coordinates = dataset.coordinates[:, pair_index]
        split_names = ["", ""]
    else:
        coordinates = np.empty((2, 4, 3))
        split_names = ["", ""]
        labels = []
        for split, name in ((dataset, "train"), (other, "test")):
            matches = np.flatnonzero(split.pair_ids == pair_index)
            if len(matches) != 1:
                raise IndexError("pair_index must occur exactly once in each split")
            i = matches[0]
            c = int(split.labels[i])
            labels.append(c)
            coordinates[c] = split.coordinates[i]
            split_names[c] = f" [{name}]"
        if labels[0] == labels[1]:
            raise ValueError("Pair members must have opposite labels")
    fig = plt.figure(figsize=(11, 9))
    for c, title in enumerate(("Left-handed (c=0)", "Right-handed (c=1)")):
        origin = coordinates[c, 0]
        vectors = coordinates[c, 1:]
        ax = fig.add_subplot(2, 2, c + 1, projection="3d")
        for vector, color, name in zip(vectors, ("red", "green", "blue"), ("u", "v", "w")):
            ax.quiver(*origin, *vector, color=color, label=name, arrow_length_ratio=0.15)
        ax.scatter(*origin, color="black", s=15)
        for setter, coordinate in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), origin):
            setter(coordinate - 1.2, coordinate + 1.2)
        ax.set(xlabel="x", ylabel="y", zlabel="z", title=title + split_names[c])
        ax.set_box_aspect((1, 1, 1))
        ax.legend()
        voxels = voxelize(vectors, grid_size, cell_size)
        ax = fig.add_subplot(2, 2, c + 3, projection="3d")
        ax.voxels(voxels, facecolors="steelblue", edgecolor="gray", alpha=0.7)
        ax.set(xlabel="voxel x", ylabel="voxel y", zlabel="voxel z",
               title=f"Binary union: {voxels.sum()} occupied cells")
        ax.set_box_aspect((1, 1, 1))
    fig.suptitle(f"Pair j={pair_index}; 3D frame with Gaussian origin")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    train, test = generate_datasets(TRAIN_SIZE, TEST_SIZE, DATA_SEED,
                                    origin_mean=ORIGIN_MEAN, origin_std=ORIGIN_STD)
    plot_pair(train, PAIR_INDEX, other=test)
    plt.show()
