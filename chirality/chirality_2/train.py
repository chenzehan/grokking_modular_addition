"""MLP for high-dimensional projected orientation classification."""

import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn

from data import generate_datasets


# ========================= IDE experiment settings =========================
TRAIN_SIZE = 1600  # Must equal TEST_SIZE and be even; half of each chirality.
TEST_SIZE = 1600
DIM = 7  # Dimension of each vector; model input dimension = 3 * DIM.
PROJECTION_SEED = 17  # Fixed hidden 3D subspace, independent of DATA_SEED.
NOISE_STD = 1.0  # Gaussian std in directions orthogonal to the task subspace.
DATA_SEED = 42
NN_SEED = 42
HIDDEN_SIZES = (1024,)  # One hidden layer; use (1024, 1024) for two.
EPOCHS = 20_001
LEARNING_RATE = 5e-2
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.0  # Plain SGD by default.
BATCH_SIZE = 100
INTERVAL = 5
DEVICE = "auto"
SHOW_PLOT = True
OUTPUT_DIR = Path(__file__).resolve().parent / "runs"
# ==========================================================================


def make_model(hidden_sizes=None, dim=None):
    hidden_sizes = HIDDEN_SIZES if hidden_sizes is None else hidden_sizes
    dim = DIM if dim is None else dim
    if not hidden_sizes or any(size <= 0 for size in hidden_sizes):
        raise ValueError("Positive hidden widths are required")
    layers = []
    width = 3 * dim
    for hidden in hidden_sizes:
        layers.extend((nn.Linear(width, hidden), nn.ReLU()))
        width = hidden
    layers.append(nn.Linear(width, 2))
    return nn.Sequential(*layers)


@torch.no_grad()
def evaluate(model, x, y):
    model.eval()
    logits = model(x)
    return nn.functional.cross_entropy(logits, y).item(), (logits.argmax(1) == y).float().mean().item()


def main():
    if EPOCHS <= 0 or INTERVAL <= 0 or BATCH_SIZE <= 0:
        raise ValueError("EPOCHS, INTERVAL and BATCH_SIZE must be positive")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if DEVICE == "auto" else DEVICE)
    train, test = generate_datasets(TRAIN_SIZE, TEST_SIZE, DATA_SEED,
                                    dim=DIM, projection_seed=PROJECTION_SEED, noise_std=NOISE_STD)
    x_train, y_train = (torch.as_tensor(a, device=device) for a in train.flat())
    x_test, y_test = (torch.as_tensor(a, device=device) for a in test.flat())
    torch.manual_seed(NN_SEED)
    model = make_model().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE,
                               weight_decay=WEIGHT_DECAY, momentum=MOMENTUM)
    config = {name: value for name, value in globals().items() if name.isupper()}
    config.update(optimizer="SGD", device=str(device), input_order="u,v,w",
                  input_size=3 * DIM, projection=train.projection.tolist(),
                  task="orthonormal_3d_frame_with_shared_orthogonal_noise",
                  split_strategy="opposite_pair_members_across_train_test",
                  labels={"0": "left", "1": "right"})
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = Path(OUTPUT_DIR) / f"frame_dim{DIM}_noise{NOISE_STD:g}_sgd_lr{LEARNING_RATE:g}_wd{WEIGHT_DECAY:g}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    print(f"Optimizer=SGD, device={device}, epochs={EPOCHS}, lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY}, momentum={MOMENTUM}")
    print(f"DIM={DIM}, input={3 * DIM}, train={TRAIN_SIZE}, test={TEST_SIZE}, hidden={HIDDEN_SIZES}\nResults: {run_dir}")

    figure = None
    if SHOW_PLOT:
        from matplotlib import pyplot as plt
        plt.ion()
        figure, axes = plt.subplots(1, 2, figsize=(12, 4))
        lines = []
        for ax, title in zip(axes, ("Cross-entropy loss", "Accuracy")):
            lines.extend(ax.plot([], [], label=label)[0]
                         for label in ("train", "test"))
            ax.set(title=title, xlabel="Epoch (-1 = initialization)")
            ax.legend()
            ax.grid(alpha=0.3)
        axes[1].set_ylim(0, 1.02)
        figure.tight_layout()
        plt.show(block=False)

    history = []
    fields = ["epoch", "train_loss", "test_loss", "train_accuracy", "test_accuracy", "weight_norm"]
    # Flush after each evaluation, so completed observations survive interruption.
    with (run_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        handle.flush()

        def record_metrics(epoch):
            tl, ta = evaluate(model, x_train, y_train)
            vl, va = evaluate(model, x_test, y_test)
            norm = sum(p.detach().square().sum().item() for p in model.parameters()) ** 0.5
            row = [epoch, tl, vl, ta, va, norm]
            history.append(row)
            writer.writerow(row)
            handle.flush()
            print(f"Epoch {epoch:6d}: loss={tl:.5f}/{vl:.5f}, acc={ta:.3f}/{va:.3f}, norm={norm:.3f}")
            if figure is not None and plt.fignum_exists(figure.number):
                records = np.asarray(history)
                for line, column in zip(lines, (1, 2, 3, 4)):
                    line.set_data(records[:, 0], records[:, column])
                for ax in axes:
                    ax.relim()
                    ax.autoscale_view()
                figure.canvas.draw_idle()
                plt.pause(0.01)

        # Evaluate and display the untouched initialization before any updates.
        record_metrics(-1)
        for epoch in range(EPOCHS):
            model.train()
            order = torch.randperm(TRAIN_SIZE, device=device)
            for batch in order.split(BATCH_SIZE):
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(model(x_train[batch]), y_train[batch])
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite loss at epoch {epoch}; check hyperparameters")
                loss.backward()
                optimizer.step()
            if epoch % INTERVAL == 0 or epoch == EPOCHS - 1:
                record_metrics(epoch)
    torch.save({"model_state_dict": model.state_dict(), "config": config,
                "epoch": EPOCHS - 1}, run_dir / "model_final.pt")
    if figure is not None:
        figure.savefig(run_dir / "curves.png", dpi=160)
        plt.ioff()
        plt.show()
    return run_dir


if __name__ == "__main__":
    main()
