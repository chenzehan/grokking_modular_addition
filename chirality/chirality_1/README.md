# 3D chirality with a Gaussian origin

Run `train.py` in the IDE. Parameters are at the top of the file.
Dependencies: NumPy, PyTorch, Matplotlib.

Each input is `[origin, u, v, w]`, flattened to 12 values. The origin is
sampled componentwise from a normal distribution controlled by `ORIGIN_MEAN`
and `ORIGIN_STD` at the top of `train.py` (defaults 0 and 1). Both accept a
scalar or a three-component tuple; setting the standard deviation to zero
fixes the origin at the mean. Visualization uses the same settings.
The vectors u and v are
random orthogonal unit vectors; w is their cross product for the right-handed
sample and its negative for the left-handed sample. Both members of a pair
share the origin, u, and v. There is no high-dimensional projection or nuisance noise.

`generate_pairs(sample_count, rng)` returns coordinates of shape `(2, pairs, 4, 3)`.
Indexing is chirality-major: `coordinates[c, j]`, with c=0 for left and c=1
for right. Flattened index i = c * number_of_pairs + j.

`generate_datasets(train_size=100, test_size=100, seed=42)` retains the paired
split: one member of each pair enters training and its opposite enters testing.
Split sizes must be equal, positive, and even. Both splits have exact class
balance and are sorted by chirality. `pair_ids` identifies counterparts across
splits. Each split has coordinates `(samples, 4, 3)` and flattened inputs
`(samples, 12)`. Data is generated in memory once per run and stays fixed
throughout training; datasets are not saved. Testing measures generalization
to the opposite member of each pair, rather than independent new frames.

Defaults: 100 training samples, 100 test samples, two hidden ReLU layers of
width 1024, SGD, and evaluation every 5 epochs. Live plots show train/test loss
and accuracy, including initialization at epoch -1. Seeds control reproducibility.

Run `visualize.py` to inspect `PAIR_INDEX`. `plot_pair(train, j, other=test)`
shows both chiralities at their actual shared origin and their binary voxel
representations. `plot_pair(pairs, j)` also accepts a complete paired dataset.
Voxelization translates the origin to the grid center for display only; the
model receives the original Gaussian origin. The voxel grid is 10 x 10 x 10
with cell size 0.25.

Each run creates a timestamped `runs/frame_3d_origin_sgd_...` directory with
`config.json`, incrementally flushed `results.csv`, and, on completion,
`model_final.pt` and `curves.png` when plotting is enabled. Existing runs are
left intact. `task.md` describes this task; `task_d.md` records the previous
high-dimensional experiment, whose implementation is backed up in `chirality_2/`.
