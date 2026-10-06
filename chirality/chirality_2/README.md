# 3D Projection Orientation Experiment with High-Dimensional Inputs

Run `train.py` directly in an IDE. All experimental parameters are specified at the top of the file, with no command-line overrides.

Dependencies: NumPy, PyTorch, and Matplotlib. `DIM` controls the dimension of each vector, so the total input dimension is `3*DIM`.

By default, the network has one hidden layer `(1024,)` and uses SGD (`MOMENTUM=0.0`). Modify `HIDDEN_SIZES` to change the number and width of hidden layers.

Training/test accuracy and cross-entropy loss are displayed every `INTERVAL`. `epoch=-1` denotes the initialization metrics before any parameter update. These metrics are immediately written to the CSV file and displayed on the live curves; the final plot also includes this point.

All subsequent records, starting from epoch 0, correspond to metrics measured after the update for that epoch.

`data.py` generates the training and test data in memory. A new dataset is generated once for each experiment and remains fixed throughout training.

The training and test sets must have equal sizes, and each size must be a positive even number. The two members of every chirality pair are always split between the sets: one goes into the training set and the other into the test set.

The chirality assigned to the training member of each pair is chosen randomly, while each set is kept exactly balanced between the two classes.

The underlying complete paired data `coordinates[c,j]` have shape `(3,DIM)` and contain, in order, `u`, `v`, and `w`. No origin vector is included in the input.

A random right-handed orthonormal frame `(a,b,c)` is first generated in 3D, with `c=a×b`.

We then construct `u=Aa+ξu`, `v=Ab+ξv`, and `w±=±Ac+ξw`. The two members of each pair share the same `ξ`, and `A.T ξ=0`.

The 3D projections always form orthonormal frames, with determinant exactly ±1 up to floating-point error.

The projection basis `A` has shape `(DIM,3)` and satisfies `A.T @ A = I`. It is controlled independently by `PROJECTION_SEED` and remains fixed across all samples and across the training/test split. For `DIM=3`, we explicitly set `A=I` and the noise to zero, recovering the original 3D orthonormal-frame task without an origin input.

The label is determined by the sign of `det([A.T u, A.T v, A.T w])`: negative corresponds to class 0 and positive to class 1.

The rule remains cubic; it is not a `DIM`-dimensional determinant. `A` is not provided to the model, so the model must learn the relevant subspace.

The noise is generated as `ξ=(I-AA.T)g`, where `g~N(0,NOISE_STD² I)`. `NOISE_STD` can be adjusted at the top of `train.py`.

The final high-dimensional vectors are not normalized, so the signal projection retains unit length. The full high-dimensional vectors are therefore generally neither orthogonal nor unit length.

The expected squared norm of each vector is `1+(DIM-3)*NOISE_STD²`. Increasing `DIM` therefore also increases the total noise energy. When comparing experiments across dimensions, `NOISE_STD` can be used to control this effect. Setting it to 0 disables the noise entirely, leaving only an embedding of the 3D frame.

The Euclidean distance between the two inputs in each pair is always 2. Thus, oppositely labeled paired samples cannot become nearly identical due to an accidentally near-degenerate random projection.

After splitting, `AxesSplit.coordinates` has shape `(number of samples,3,DIM)` and is ordered by chirality. `labels` stores the chirality labels, while `pair_ids` stores the global pair index `j`. Do not use row indices after splitting to match paired samples.

After flattening, the input has shape `(number of samples,3*DIM)`.

The training and test sets share the paired structure. The test set therefore measures generalization across paired reflections, rather than generalization to entirely new and independently generated triples.

Changing `DATA_SEED` generates a new dataset while keeping `A` fixed. `NN_SEED` controls network initialization and minibatch ordering.

Run `visualize.py` to inspect the global pair specified by `PAIR_INDEX` (zero-based).

The function `plot_pair(train,j,other=test)` returns a four-panel Figure showing the 3D projections in `A` and the corresponding binary voxel plots for both members of the pair. The titles indicate whether each member belongs to the training or test set. The visualization directly imports the data configuration from `train.py` to ensure consistency, but does not start training.

Call `figure.savefig(...)` to save the figure.

The voxel visualization uses a 10×10×10 grid with a cell size of 0.25 and places the origin at the center of the grid. It displays the unit orthonormal projected axes and does not show the noise in the orthogonal complement. The model itself receives the full high-dimensional coordinates.

Each training run creates a separate subdirectory under `runs/`, named using `DIM` and a timestamp. The actual parameters and projection basis `A` are saved to `config.json`.

At every evaluation, the results are immediately written to and flushed from `results.csv`, including loss, accuracy, and the total parameter norm.

After normal completion, the final model is saved as `model_final.pt`. If plotting is enabled, `curves.png` is also saved.

If training is interrupted, all CSV records written up to that point are retained. The final model is generated only after normal completion. Dataset files are not saved.

The original data can be reconstructed using the same data seed and parameters. Whether grokking occurs must be determined experimentally; the default parameters do not guarantee it.