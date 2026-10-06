# Supplementary Code

This archive contains research code for the modular-arithmetic and frame-chirality experiments in the accompanying paper: synthetic-data generation, training, feature-kernel calculations, online measurements, and plotting routines.

Parameters are normally edited near the top of each script or passed to its training function; there is no shared command-line interface. The selected `__main__` experiment is not necessarily the experiment needed for a particular paper figure.

## Contents and reproduction scope

```text
package/
  README.md
  add_mod/                  Modular arithmetic: training, kernels, and analysis
  chirality/
    chirality_1/            3D frames with a shared Gaussian origin
    chirality_2/            Frames embedded in a noisy higher-dimensional space
```

All datasets are generated synthetically; no dataset download is required. Saved runs, checkpoints, and metric files are **not included**, for either modular arithmetic or chirality. Training and feature-kernel calculations can run from this archive; figures based on training trajectories require regenerated experiment outputs. Chirality trainers create their own `runs/` directories when executed.

This is a source-code supplement, not an automated reproduction of every paper figure. See the coverage table and experiment instructions below before launching an experiment.

## Environment

Lightweight checks were run with Python 3.10.6, NumPy 2.2.6, PyTorch 2.9.0+cu130, and Matplotlib 3.10.9. Training was checked on CPU; this does not establish CUDA reproducibility. NumPy 2.x is needed by the current `np.concat` calls.

Create an environment with `python -m venv .venv`. Activate it using `.venv\Scripts\activate` on Windows or `source .venv/bin/activate` on Linux/macOS, then install the dependencies:

```text
python -m pip install "numpy==2.2.6" "matplotlib==3.10.9" "torch==2.9.0"
```

Use a PyTorch build appropriate for the intended CPU/CUDA environment. The modular trainer with `_gpu` in its name supports CPU via `device="cpu"`; its default selects CUDA when available. For noninteractive runs, pass `show_plot=None`. The chirality trainers instead use `SHOW_PLOT=False` near the top of `train.py`.

Run modular scripts from `add_mod/` and each chirality script from its own directory. The scripts use local imports and relative experiment paths. Some modular training modules also change the working directory and create a default output directory when imported.

## Quick checks without existing checkpoints

### Small feature-kernel calculation

From `add_mod/`:

```text
python -c "from num_verify_kernels import kernel_values; print(kernel_values(7, 64, type='fourier')); print(kernel_values(7, 64, type='sqrwave'))"
```

The five returned averages are diagonal, feature, input collision, modular collision, and background, in that order. This small example checks execution, not the large-modulus asymptotic coefficients.

`num_verify_kernels.py` currently selects square-wave features in its main block; use `type="fourier"` for Fourier features. Its full settings include `(p, N) = (59, 2048), (113, 4096), (151, 8192)`. It creates dense matrices of shape `p**2` by `p**2`, so memory grows as `p**4`. At `p=151`, one float64 matrix alone takes about 4.2 GB, and several are retained. Start with the small example above.

### Short modular-addition training run

Save this as `smoke_train.py` inside `add_mod/` and run `python smoke_train.py`:

```python
from pathlib import Path
import numpy as np
from add_mod_2_sgd_thm3_gpu import train_with_balanced_data

out = Path("checkpoints_sgd_thm3/smoke_test")
out.mkdir(parents=True, exist_ok=True)
history = train_with_balanced_data(
    p=7, hidden_size=16, train_select=3,
    lr=0.1, weight_decay=2e-4,
    epochs=3, checkpoint_interval=1, batch_number=4,
    bias=False, show_plot=None,
    save_model=False, save_final=True,
    filename=str(out), device="cpu",
    data_seed=42, nn_seed=42,
)
np.savetxt(
    out / "loss_acc.txt", np.column_stack(history),
    header="train_loss test_loss train_accuracy test_accuracy",
)
```

This writes three metric rows and `model_epoch_2.pt`. It checks training and output generation, not grokking. The return order is `(train_losses, test_losses, train_accuracies, test_accuracies)`. Save these arrays explicitly: several original main blocks have their metric-saving line commented out.

For a paper-scale run, select the corresponding figure's settings, for example `p=59`, `hidden_size=512`, `train_select=29`, `lr=2`, `weight_decay=2e-4`, `batch_number=4`, and a substantially longer run. Choose an appropriate checkpoint interval and a new output directory. A smoke test does not establish convergence of these longer experiments.

## Modular-arithmetic code map

| File | Role |
| --- | --- |
| `mod_tensors.py` | Complete, balanced, and random datasets; two-hot inputs, centered labels, and general modular operations. |
| `model.py` | NumPy feature models, Gaussian/Fourier/square-wave initializers, and kernel utilities. |
| `model_network.py` | Load two-layer checkpoints; compute logits, residuals, energies, and effective widths. |
| `add_mod_2_sgd_thm3.py` | CPU training implementation. |
| `add_mod_2_sgd_thm3_gpu.py` | CPU/CUDA training; explicit parameter and output-path overrides. |
| `add_mod_2_sgd_init.py` | Evaluate the untrained initialization. |
| `monitor_test_acc.py` | Feature-neuron counts, kernel alignment, cosine similarities, and accuracy checkpoints. |
| `monitor_gap_seed.py` | Accuracy-threshold measurements, effective widths, and model/label-kernel/residual-kernel accuracies. |
| `monitor_early.py` | Record early-stage quantities at finer temporal resolution. |
| `monitor_init.py` | Prepend initialization measurements to an existing alignment history. |
| `op_mod_2_sgd.py` | Train on explicitly supplied modular datasets and initial feature models. |
| `op_monitor_test_acc.py` | Multiplication and square-sum experiments with online measurements. |
| `num_verify_kernels.py` | Numerical Fourier and square-wave feature-kernel coefficients. |
| `num_verify_scaling.py` | Train across parameter settings and data-split seeds; save final losses for the mean-loss scaling figure. |
| `loss_sc_gap_seed.py` | Analyze accuracy-threshold checkpoints, collision scores, and width scaling. |
| `loss_scaling_random.py` | Analyze label-group loss differences under a random split. |
| `loss_scaling_grok_group.py` | Analyze within-label loss fluctuations and effective width from checkpoints; not the mean-loss sweep driver. |
| `plot_figures.py` | Main figures and the ReLU/quadratic comparison. |
| `plot_figures_apdx.py` | Appendix figures, including the model/kernel accuracy comparison. |
| `plot_figures_kernel.py` | Appendix kernel visualizations: Gaussian/Fourier quadratic kernels, ReLU square-wave kernels, random multilayer kernels, and collision structure. |
| `model_plot_dis.py` | Interactive plotting utilities used by the trainers. |

### Parameters and conventions

- `p` is the modular-addition class count and modulus; `hidden_size` is width `N`; `train_select` is the number `q` of training samples per label. Balanced training uses `M=p*q` and fraction `q/p`. Changing `fraction_train` alone does not change the balanced split.
- For disjoint, equally sized balanced training and test sets, use `q <= floor(p/2)`. The generator only warns above this range and then produces overlapping sets; these runs must not be interpreted as held-out generalization experiments.
- Random sampling is available through `ModAddDataRandom`, but selecting it requires changing the dataset branch and using the same split in the monitors. It is not a runtime flag in the balanced trainer.
- `batch_number=4` means approximately four batches per epoch, not four samples per batch. With four updates and learning rate `eta=2`, gradient-flow time is approximately eight times the number of completed epochs.
- Training record `epoch=0` is evaluated **after** the first epoch of updates. Initialization uses `epoch=-1`; plotting commonly adds one to recorded indices. `monitor_init.py` needs an existing `align.txt` and is not a standalone first-run script.
- Use `bias=False` for the main bias-free model. The standard PyTorch trainer retains `nn.Linear` initialization in the first layer and zeros the output layer. The explicitly normalized ensembles in `model.py` are separate feature-kernel experiments; their weight scales need not match the trainer defaults.
- `data_seed` controls the split; `nn_seed` controls network initialization and training randomness. When changing `p`, `q`, or seeds, update the monitor's dataset too: several monitors read module-level settings rather than training keyword arguments.
- The multiplication and square-sum examples use nonzero residues `np.arange(1, p)` and have `p-1` classes. Use the actual class count for chance baselines and centered labels.
- The quadratic comparison requires selecting `SqrActivation` in the model definition and matching the analysis activation. The main trainer defaults to ReLU.

### Saving and plotting

Plotting functions expect experiment directories below `add_mod/checkpoints_sgd_thm3/`, including `aligns/`, `early/`, `gap_seed/`, `scaling/`, `random/`, and `tasks/`. Historical names use both `(29)` and `(0.50)`; these strings are not interchangeable, and the latter is not a reliable record of the exact fraction. Match plotting paths to the outputs you generated.

| Output | Meaning |
| --- | --- |
| `loss_acc.txt` | Usually space-separated train loss, test loss, train accuracy, test accuracy. |
| `results.txt` | Often the same quantities with comma separation; check the writer and reader. |
| `model_epoch_*.pt` | Model parameters at evaluation epochs. |
| `model_accit_*.pt` | Model parameters at monitored accuracy thresholds. |
| `align.txt`, `align_init.txt` | Alignments, squared weight norm, loss fluctuations, and effective widths. |
| `cos_sim.txt` | Residual and output-locking cosine similarities. |
| `kernel_gaps.txt` | Accuracy-crossing epochs, alignments, and effective widths. |
| `kernel_accs_S.txt` | Epoch, label-kernel train/test accuracies, residual-kernel train/test accuracies. |
| `seeded_final_loss.txt` | Final losses across data-split seeds for a parameter setting. |

Use `save_model=True` for figures needing intermediate checkpoints or neuron trajectories; `save_final=True` alone is insufficient. Saving occurs inside the evaluation block. Choose an epoch count whose last epoch is evaluated, e.g. `epochs=30001` with `checkpoint_interval=50`.

Figure destinations are sibling directories of `add_mod/`. Create them before saving, from `add_mod/`:

```text
python -c "from pathlib import Path; [Path('../', name).mkdir(exist_ok=True) for name in ('figures', 'figures_apdx', 'raw_figures')]"
```

Select the desired plotting function rather than assuming its main block produces all figures. After generating the required metrics, for example:

```text
python -c "from plot_figures_apdx import kernel_acc_compare_plot; kernel_acc_compare_plot()"
```

### Mean-loss scaling sweep

`num_verify_scaling.py` generates `seeded_final_loss.txt` under `checkpoints_sgd_thm3/scaling/`, with one final training loss per data-split seed (`0`--`9`). Its active blocks sweep the training fraction at `p=113` for widths `512` and `2048`; the weight-decay and problem-size sweeps are provided as commented blocks. Enable the required blocks before running `python num_verify_scaling.py`, then use `plot_figures.loss_scaling_plot()` after generating all settings it reads. The width-512 fraction sweep skips `q=56`, so that setting must be generated separately if its output is not already available.

The active fraction sweep includes values above `0.5`, where the equally sized balanced splits overlap as described above. These runs measure training-loss scaling and should not be used to assess held-out test accuracy. This script launches full training runs across multiple settings and seeds; it is not a quick check.

### Appendix kernel figures

After creating the figure output directories, run the desired functions from `add_mod/`:

```text
python -c "from plot_figures_kernel import kernel_gauss_fourier_sqr, kernel_sqrwave; kernel_gauss_fourier_sqr(); kernel_sqrwave()"
```

These generate `kernel_gauss_fourier_sqr.pdf` and `kernel_sqrwave.pdf`, together with SVG versions, in `package/figures_apdx/`. The same script provides `kernel_multilayer_34()` and `kernel_illus()` for random multilayer kernels and kernel collision structure. These routines construct features directly and do not require trained checkpoints; the multilayer kernel plots do not reproduce the multilayer training trajectories.

## Coverage of paper results

| Result | Included entry points | Additional requirements |
| --- | --- | --- |
| Grokking, alignment, and neuron trajectories | Trainers, monitors, `plot_figures.py` | Matching histories and intermediate checkpoints. |
| Gaussian/Fourier kernel illustration | `model.py`, `plot_figures.kernel_gauss_fourier` | No trained checkpoint; some save calls are commented out. |
| Appendix kernel visualizations | `plot_figures_kernel.py` | Create output directories and select the desired function; no trained checkpoint needed. |
| Fourier/square-wave coefficient tables | `num_verify_kernels.py` | Select ensemble and size; large calculations need substantial RAM. |
| Equilibrium mean-loss scaling | `num_verify_scaling.py` and `plot_figures.loss_scaling_plot` | Select the required sweep blocks and generate all `seeded_final_loss.txt` inputs. |
| Finite-width transition | `monitor_gap_seed.py`, `loss_sc_gap_seed.py`, plotting functions | Enable the width/seed sweep; the current main block selects a different experiment. |
| Model/kernel accuracy comparison | `MonitorKernelAcc` and `kernel_acc_compare_plot` | Matching `loss_acc.txt` and `kernel_accs_S.txt`. |
| Balanced/random and ReLU/quadratic comparisons | Dataset classes, trainers, analysis and plotting functions | Select data/activation branches and save both runs. |
| Other modular tasks and Fourier initialization | `op_mod_2_sgd.py`, `op_monitor_test_acc.py` | Select the task/initialization and matching output paths. |
| Multilayer experiments | `plot_figures_kernel.kernel_multilayer_34` and appendix training-curve plotting function | Random-kernel plots run directly; a dedicated multilayer training configuration is not included. |
| Chirality | Both chirality trainers | Generate runs first, then update the combined appendix plotting paths. |

## Chirality experiments

Run `python train.py` from either `chirality/chirality_1/` or `chirality/chirality_2/`. Both generate data in memory, use SGD without momentum, and create a new timestamped run directory.

| Setting | `chirality_1` | `chirality_2` |
| --- | --- | --- |
| Input | Origin plus three 3D axes (12 values) | Three axes embedded in dimension 7 (21 values) |
| Training/test samples | 450 / 450 | 1600 / 1600 |
| Hidden layers | One ReLU layer, width 1024 | One ReLU layer, width 1024 |
| Learning rate | 0.02 | 0.05 |
| Weight decay | `1e-4` | `1e-4` |
| Batch size | 100 | 100 |
| Epochs / evaluation interval | 20001 / 5 | 20001 / 5 |
| Data / network seeds | 42 / 42 | 42 / 42 |
| Additional settings | Origin standard deviation 2 | Noise standard deviation 1; projection seed 17 |

These are the current `train.py` settings. Some defaults described in local task notes and READMEs belong to earlier experiments; use the actual script and the `config.json` generated by each new run to identify its settings.

Training and test sets contain opposite members of each paired frame, with balanced classes. Testing measures generalization across this paired split rather than independent new frames. In setting 2, the projection is fixed, noise lies in its orthogonal complement, and the projection is not supplied to the network.

Each run saves `config.json` and incrementally written `results.csv`, then `model_final.pt` on normal completion and `curves.png` when plotting is enabled. `SHOW_PLOT=False` disables both the live figure and its final image. Initialization is recorded at epoch `-1`.
