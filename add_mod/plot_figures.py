import numpy as np
from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerTuple

import model
import mod_tensors
import model_network

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
ENV_DIR = Path(__file__).resolve().parent
FIGURE_SAVE_DIR = Path(__file__).resolve().parent.parent.joinpath("figures")
RAW_FIGURE_SAVE_DIR = Path(__file__).resolve().parent.parent.joinpath("raw_figures")
FIGURE_APDX_SAVE_DIR = Path(__file__).resolve().parent.parent.joinpath("figures_apdx")

def energy_single_neuron(filename = "checkpoints_sgd_thm3/59(0.50)-512-2", epochs=10000, ref_epoch=1000, interval=50, cmap="viridis", rand=1, 
                         ax=None, hide_cbar=False, hide_y=False,
                         ):
    show = ax is None
    if show:
        _, ax = plt.subplots()
    N = 512
    p = 59
    Epochs = np.arange(0, epochs + 1, interval)
    E = np.zeros((len(Epochs), N))
    # cn = p**2 / 2 * (1/6 - 1/8)
    data = mod_tensors.ModAddDataTrainValid(p=p, select=29)
    for i, ep in enumerate(Epochs):
        m = model_network.ModelNetworkCE(p=p, N=N, sigma=mod_tensors.ReLU(), from_file=filename + f"/model_epoch_{ep}.pt")
        F = m.activate(data.X)
        E[i, :] = 0.5 * np.linalg.norm(data.Y @ F.T, axis=0)**2 / np.linalg.norm(m.W, axis=-1)**2
    cn = np.max(E)
    E /= cn # normalize to 1

    if rand != 1:
        rng = np.random.default_rng(42)
        random_selc = rng.permutation(N)[0: int(rand * N)]
        E = E[:, random_selc]
        print(E.shape)
    selected = np.flatnonzero(E[-1, :] >= 0.5)
    r_selected = np.flatnonzero(E[-1, :] < 0.5)
    ref_index = ref_epoch // interval
    ref_values = E[ref_index, :]

    value_map = lambda x: x

    if len(selected) > 0:
        ref_min = np.min(ref_values)
        ref_max = np.max(ref_values)
        if ref_max > ref_min:
            color_values = (ref_values - ref_min) / (ref_max - ref_min)
        else:
            color_values = np.full_like(ref_values, 0.5)

        color_values = np.clip(value_map(color_values), 0.0, 1.0)
        colormap = plt.get_cmap(cmap)
        for j, color_value in zip(selected, color_values[selected]):
            ax.plot(Epochs, E[:, j], color=colormap(color_value))
        for j, color_value in zip(r_selected, color_values[r_selected]):
            ax.plot(Epochs, E[:, j], color=colormap(color_value), alpha=0.1)

        if not hide_cbar:
            colorbar = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=0.0, vmax=1.0), cmap=colormap)
            colorbar.set_array([])
            ax.figure.colorbar(colorbar, ax=ax, label=f"Normalized energy at epoch {Epochs[ref_index]}")
    if hide_y:
        ax.tick_params(axis="y", labelleft=False)
    if show:
        plt.show()

def early_late_comparison_between_relu_and_sqr():
    cmap = "viridis"
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12, 3), gridspec_kw={"wspace": 0.05})
    for ax in (ax3, ax4):
        pos = ax.get_position()
        ax.set_position([pos.x0 + 0.05, pos.y0, pos.width, pos.height])
    energy_single_neuron(ax=ax1, hide_cbar=True, cmap=cmap)
    energy_single_neuron(ax=ax2, hide_cbar=True, hide_y=True, cmap=cmap, filename = "checkpoints_sgd_thm3/sqrs/ref59(0.50)-512-2")
    ax1.set_xlabel("epochs", fontsize=12)
    ax2.set_xlabel("epochs", fontsize=12)
    ax1.set_ylabel(r"normalized energy", fontsize=12)
    ax1.set_title("(a) ReLU neuron energies", fontsize=12)
    ax2.set_title(r"(b) $x^2$ neuron energies", fontsize=12)
    # colorbar = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=0.0, vmax=1.0), cmap=cmap)
    # colorbar.set_array([])
    # fig.colorbar(colorbar, ax=[ax1, ax2], pad=0.02, label="Normalized energy at reference epoch")
    # plt.savefig(RAW_FIGURE_SAVE_DIR.joinpath("early_late_comparison_between_relu_and_sqr.svg"), bbox_inches="tight")
    acc_plot_comparison_between_relu_and_sqr(ax3, ax4, title_labels=["(c) ", "(d) "])
    # plt.savefig(FIGURE_SAVE_DIR.joinpath("early_late_comparison_between_relu_and_sqr.svg"), bbox_inches="tight")
    # plt.savefig(FIGURE_SAVE_DIR.joinpath("early_late_comparison_between_relu_and_sqr.pdf"), bbox_inches="tight")
    plt.show()

def early_late_comparison_between_relu_and_sqr_apdx():
    cmap = "viridis"
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10), gridspec_kw={"wspace": 0.05})
    # for ax in (ax3, ax4):
    #     pos = ax.get_position()
    #     ax.set_position([pos.x0 + 0.05, pos.y0, pos.width, pos.height])
    energy_single_neuron(ax=ax1, hide_cbar=True, cmap=cmap)
    energy_single_neuron(ax=ax2, hide_cbar=True, hide_y=True, cmap=cmap, filename = "checkpoints_sgd_thm3/sqrs/ref59(0.50)-512-2")
    ax1.set_xlabel("epochs", fontsize=12)
    ax2.set_xlabel("epochs", fontsize=12)
    ax1.set_ylabel(r"normalized energy", fontsize=12)
    ax1.set_title("(a) ReLU neuron energies", fontsize=12)
    ax2.set_title("(b) Quadratic neuron energies", fontsize=12)
    # colorbar = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=0.0, vmax=1.0), cmap=cmap)
    # colorbar.set_array([])
    # fig.colorbar(colorbar, ax=[ax1, ax2], pad=0.02, label="Normalized energy at reference epoch")
    # plt.savefig(RAW_FIGURE_SAVE_DIR.joinpath("early_late_comparison_between_relu_and_sqr.svg"), bbox_inches="tight")
    acc_plot_comparison_between_relu_and_sqr(ax3, ax4, title_labels=["(c) ", "(d) "])
    plt.savefig(FIGURE_APDX_SAVE_DIR.joinpath("early_late_comparison_between_relu_and_sqr.svg"), bbox_inches="tight")
    plt.savefig(FIGURE_APDX_SAVE_DIR.joinpath("early_late_comparison_between_relu_and_sqr.pdf"), bbox_inches="tight")
    plt.show()

def acc_plot_comparison_between_relu_and_sqr(ax1=None, ax2=None, title_labels=["", ""]):
    self_plot = (ax1 is None)
    acc_relu = np.loadtxt(ENV_DIR.joinpath("checkpoints_sgd_thm3/59(0.50)-512-2/loss_acc.txt"), delimiter=' ')
    acc_sqr = np.loadtxt(ENV_DIR.joinpath("checkpoints_sgd_thm3/sqrs/ref59(0.50)-512-2/loss_acc.txt"), delimiter=' ')
    Ref_epochs = 1 + np.arange(0, 10001, 50)
    Epochs = np.insert(Ref_epochs, 0, 0)
    if self_plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"wspace": 0.05})
    ax1.plot(Epochs, acc_relu[:, 2], color="#201E8F", label="train")
    ax1.plot(Epochs, acc_relu[:, 3], color="#DB2E28", label="test")
    ax2.plot(Epochs, acc_sqr[:, 2], color="#201E8F", label="train")
    ax2.plot(Epochs, acc_sqr[:, 3], color="#DB2E28", label="test")
    ax1.legend(loc=4, frameon=False, fontsize=12)
    ax2.legend(loc=4, frameon=False, fontsize=12)
    ax2.set_yticklabels([])
    ax1.set_xlabel("epochs", fontsize=12)
    ax2.set_xlabel("epochs", fontsize=12)
    ax1.set_ylabel("accuracy", fontsize=12)
    ax1.set_title(title_labels[0] + "ReLU accuracy", fontsize=12)
    ax2.set_title(title_labels[1] + "Quadratic accuracy", fontsize=12)
    
    if self_plot:
        plt.show()

def kernel_plot_tt_tv(axes=None, title_labels=[""]*3, p=17, init="gauss"):
    self_plot = (axes is None)
    q = p // 2
    N = 512
    data = mod_tensors.ModAddDataTrainValid(p=p, select=q)
    if init == "gauss":
        supscr = 'g'
        w_init = model.rand_init(p, std=1/np.sqrt(2))
    elif init == "fourier":
        supscr = 'f'
        w_init = model.fourier_init_rand_phase(p)
    elif init == "sqrwave":
        supscr = 's'
        w_init = model.sqrwave_init_rand_phase(p)
    else:
        supscr = '-'
        w_init = init
    m = model.ModelCE(p, N=N, w_init=w_init, sigma=model.ReLU(), from_file=None)
    F = m.activate(data.X)
    F_t = m.activate(data.X_train)
    F_v = m.activate(data.X_valid)
    K = F.T @ F
    K_tt = F_t.T @ F_t / N
    K_tv = F_t.T @ F_v / N
    H = np.kron(np.eye(p), np.ones((q, q)))
    Kb_tv = (1 / q**2) * H.T @ K_tv @ H
    if self_plot:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4), gridspec_kw={"wspace": 0.05})
    else:
        ax1, ax2, ax3 = axes
    ax1.imshow(K_tt)
    ax2.imshow(K_tv)
    ax3.imshow(Kb_tv)

    Ticks = np.arange(0, p * q, 4 * q)
    ax1.set_xticks(Ticks)
    ax2.set_xticks(Ticks)
    ax3.set_xticks(Ticks)
    ax1.set_yticks(Ticks)
    ax2.set_yticks([])
    ax3.set_yticks([])
    GridTicks = np.arange(q, p * q, q) - 0.5
    for ax in (ax1, ax2, ax3):
        ax.set_xticks(GridTicks, minor=True)
        ax.set_yticks(GridTicks, minor=True)
        ax.grid(which="minor", color="lightgray", linewidth=0.25, alpha=0.4)
        ax.tick_params(which="minor", bottom=False, left=False)
    ax1.set_title(title_labels[0] + rf"$\frac{{1}}{{N}}K_{{tt}}^{supscr}$", fontsize=12)
    ax2.set_title(title_labels[1] + rf"$\frac{{1}}{{N}}K_{{tv}}^{supscr}$", fontsize=12)
    ax3.set_title(title_labels[2] + rf"$\frac{{1}}{{N}}\overline{{K}}_{{tv}}^{supscr}$", fontsize=12)
    # plt.savefig(RAW_FIGURE_SAVE_DIR.joinpath(f"kernel_tt_tv_{init}.svg"), bbox_inches="tight")
    if self_plot:
        plt.show()

def kernel_gauss_fourier():
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), gridspec_kw={"wspace": 0.05})
    kernel_plot_tt_tv(axes=axes[0], title_labels=["(a) ", "(b) ", "(c) "])
    kernel_plot_tt_tv(axes=axes[1], title_labels=["(d) ", "(e) ", "(f) "], init="fourier")
    for group in axes.T:
        images = [ax.images[0] for ax in group.flat]
        norm = plt.Normalize(min(im.get_array().min() for im in images),
                             max(im.get_array().max() for im in images))
        for im in images:
            im.set_norm(norm)
        fig.colorbar(images[0], ax=group.ravel().tolist(), orientation="horizontal",
                     fraction=0.04, pad=0.05, shrink=0.8, aspect=20)
    # plt.savefig(RAW_FIGURE_SAVE_DIR.joinpath(f"kernel_gauss_fourier.svg"), bbox_inches="tight", dpi=800)
    # plt.savefig(FIGURE_SAVE_DIR.joinpath(f"kernel_gauss_fourier.svg"), bbox_inches="tight", dpi=800)
    # plt.savefig(FIGURE_SAVE_DIR.joinpath(f"kernel_gauss_fourier.pdf"), bbox_inches="tight", dpi=800)
    plt.show()

def acc_plot_width(ax=None, title_label=""):
    self_plot = (ax is None)
    if self_plot:
        fig, ax = plt.subplots(1, 1)
    p = 59
    Ns = np.array([256, 512, 1024, 2048])
    Ref_epochs = 1 + np.arange(0, 10001, 50)
    Epochs = np.insert(Ref_epochs, 0, 0)

    cmap = plt.get_cmap('Reds')
    colors = cmap(np.linspace(0.3, 0.8, len(Ns), endpoint=True)[::-1])
    cmap_2 = plt.get_cmap('seismic')
    colors_2 = cmap_2(np.linspace(0., 0.3, len(Ns), endpoint=True))

    ax.plot([-600, 1900], [1/p, 1/p], linestyle='--', color='gray', alpha=0.7, linewidth=1.4) # chance level
    ax.text(2000, 0.02, r"$\frac{1}{p}$", fontsize=16, color='black', ha='left', va='center')

    for i, N in enumerate(Ns):
        filename = f"checkpoints_sgd_thm3/aligns/{p}(29)-{N}-2/loss_acc.txt"
        d = np.loadtxt(ENV_DIR.joinpath(filename))
        ax.plot(Epochs, d[:len(Epochs), 2], color=colors_2[i], zorder=5-i)
        ax.plot(Epochs, d[:len(Epochs), 3], color=colors[i], label=f"$N={N}$", zorder=10-i)
    ax.legend(loc=4, frameon=False, fontsize=12)

    ax.set_xlabel("epochs", fontsize=12)
    ax.set_ylabel("accuracy", fontsize=12)
    ax.set_xlim(-500, 10500)
    ax.set_title(title_label + "Grokking with different widths")
    if self_plot:
        plt.show()

def alignment_increase_with_time(ax=None, title_label="", Ns = [256, 512, 1024, 2048]):
    # data = mod_tensors.ModAddDataTrainValid(p=59, select=29)
    def kernel_a_tv(filename, Acc = [0, 5, 9]):
        kd = np.loadtxt(ENV_DIR.joinpath(filename + "/kernel_gaps.txt"), delimiter=' ')
        a_tv = [(kd[i, 0], kd[i, 2] / kd[i, 3]) for i in Acc]
        return a_tv
    self_plot = (ax is None)
    if self_plot:
        fig, ax = plt.subplots(1, 1)
    # colors = [f"C{i}" for i in range(6)]
    # cmap = plt.get_cmap('plasma')
    # colors = [cmap(i) for i in np.linspace(0, 1, 5, endpoint=True)]
    # cmap = plt.get_cmap('Reds')
    # colors = [cmap(i) for i in np.linspace(0.3, 0.8, len(Ns), endpoint=True)[::-1]]
    cmap = plt.get_cmap('Reds')
    cmap_2 = plt.get_cmap('plasma')
    colors = cmap(np.linspace(0.3, 0.8, len(Ns), endpoint=True)[::-1]) * 0.7 + cmap_2(np.linspace(0, 0.8, len(Ns), endpoint=True)) * 0.3
    Ns = np.array(Ns)
    Ep_select = slice(0, 161)
    line_handles = {}
    for i, N in enumerate(Ns):
        filename = f"checkpoints_sgd_thm3/aligns/59(29)-{N}-2/align_init.txt"
        # epochs alignment W_norm_sqr alignment_tt alignment_tv std effective_width effective_feature_width
        al = np.loadtxt(ENV_DIR.joinpath(filename), delimiter=' ')
        ax.plot(al[Ep_select, 0] + 1, al[Ep_select, 4] / al[Ep_select, 2], label=f"$N={N}$", color=colors[i], zorder=10-i)

        filename_gap = f"checkpoints_sgd_thm3/59(0.50)-{N}-2"
        a_tv_s = kernel_a_tv(filename_gap)
        line_handles[f"{i}d"], = ax.plot(a_tv_s[0][0] + 1, a_tv_s[0][1], 'D', color=colors[i], zorder=10-i)
        line_handles[f"{i}o"], = ax.plot(a_tv_s[1][0] + 1, a_tv_s[1][1], 'o', color=colors[i], zorder=10-i)
        line_handles[f"{i}s"], = ax.plot(a_tv_s[2][0] + 1, a_tv_s[2][1], 's', color=colors[i], zorder=10-i)
    
    # lg_h, lg_l = ax.get_legend_handles_labels()
    # ax.legend(lg_h + [(line_handles["0d"], line_handles["0o"], line_handles["0s"])], lg_l + [r"$\rho=1/p, 50\%, 90\%$"],
    #           handler_map={tuple: HandlerTuple(ndivide=None)},
    #     loc=4, frameon=False, fontsize=12)
    ax.legend([line_handles["1d"], line_handles["1o"], line_handles["1s"]], [r"$\rho=1/p$", r"$\rho=0.5$", r"$\rho=0.9$"],
                  handler_map={tuple: HandlerTuple(ndivide=None)},
            loc=4, frameon=False, fontsize=12)
    ax.set_xlabel("epochs", fontsize=12)
    ax.set_ylabel(r"$a_{tv}$", fontsize=14, labelpad=-5)
    ax.set_title(title_label + "Alignment evolution")

    # ax.legend(loc=4, bbox_to_anchor=(0.9, 0.07), frameon=False, fontsize=12)
    # 定义三个条目:marker, 颜色, 文字
    # items = [
    #     ('D', colors[0], r"$1/p$"),
    #     ('o', colors[0], "50%"),
    #     ('s', colors[0], "90%"),
    # ]

    # # 用相对坐标排版(axes 坐标系)
    # y = 0.065
    # x = 0.64
    # dx_marker_text = 0.02   # marker 到文字的间距
    # dx_item = 0.12          # 条目之间的间距
    # dy_item = -0.004

    # for marker, color, text in items:
    #     ax.plot(x, y, marker=marker, color=color, markersize=8,
    #             transform=ax.transAxes, clip_on=False)
    #     ax.text(x + dx_marker_text, y + dy_item, text,
    #             transform=ax.transAxes, ha='left', va='center', fontsize=12)
    #     x += dx_item

    # plt.savefig(RAW_FIGURE_SAVE_DIR.joinpath(f"alignment_increase_with_time.svg"), bbox_inches="tight")
    if self_plot:
        plt.show()

def loss_scaling_plot():
    # color_1 = "darkslategray"
    # color_2 = "slategray"
    color_0 = "#1134a5"
    color_1 = "#173aae"
    color_2 = "#485a96"
    color_3 = "#6895d1"

    # NOTE: ax1 and ax2 exchanged here
    fig, (ax2, ax1, ax3) = plt.subplots(1, 3, figsize=(14, 4), gridspec_kw={"wspace": 0.2})
    Loss_p = []
    lmb = 2e-4
    Ps = np.array([37, 47, 59, 67, 89, 97, 113, 131])
    for p in Ps:
        q = p // 2
        # train_loss = np.loadtxt(ENV_DIR.joinpath(f"checkpoints_sgd_thm3/scaling/{p}({q})-512-2/results.txt"), delimiter=',')[-1, 0]
        train_loss = np.loadtxt(ENV_DIR.joinpath(f"checkpoints_sgd_thm3/scaling/{p}({q})-512-2/seeded_final_loss.txt")).mean()
        Loss_p.append(train_loss)
    ax1.plot(Ps, Loss_p, 's', color=color_0)
    ax1.plot(Ps, 4.9 * lmb * (Ps - 1), color=color_1, label=r"$4.9\lambda p$")
    ax1.plot(Ps, 5.1 * lmb * (Ps - 1), linestyle='--', linewidth=1.4, color=color_2, label=r"$5.1\lambda p$")
    ax1.set_xlabel(r"$p$", fontsize=14)
    ax1.legend(loc=4, frameon=False, fontsize=12)
    ax1.set_title("(b)")

    Lmbs = np.arange(0.5e-4, 4.1e-4, 0.5e-4)
    p = 59
    q = p // 2
    Loss_l = []
    for lmb in Lmbs:
        # train_loss = np.loadtxt(ENV_DIR.joinpath(f"checkpoints_sgd_thm3/scaling/[lambda={lmb:.2e}]{p}({q})-512-2/results.txt"), delimiter=',')[-1, 0]
        train_loss = np.loadtxt(ENV_DIR.joinpath(f"checkpoints_sgd_thm3/scaling/[lambda={lmb:.2e}]{p}({q})-512-2/seeded_final_loss.txt")).mean()
        Loss_l.append(train_loss)
    ax2.plot(Lmbs * 1e4, Loss_l, 's', color=color_0)
    ax2.plot(Lmbs * 1e4, 4.9 * Lmbs * (p - 1), color=color_1)
    ax2.plot(Lmbs * 1e4, 5.1 * Lmbs * (p - 1), linestyle='--', linewidth=1.4, color=color_2)
    ax2.set_xlabel(r"$\lambda(\times10^{-4})$", fontsize=14)
    ax2.set_ylabel("loss", fontsize=12)
    ax2.set_title("(a)")

    Alphas = np.arange(0.3, 0.71, 0.1)
    p = 113
    lmb = 2e-4
    Loss_a = []
    Loss_aN = []
    for alpha in Alphas:
        q = int(p * alpha)
        # train_loss = np.loadtxt(ENV_DIR.joinpath(f"checkpoints_sgd_thm3/scaling/{p}({q})-512-2/results.txt"), delimiter=',')[-1, 0]
        train_loss = np.loadtxt(ENV_DIR.joinpath(f"checkpoints_sgd_thm3/scaling/{p}({q})-512-2/seeded_final_loss.txt")).mean()
        Loss_a.append(train_loss)
        train_loss = np.loadtxt(ENV_DIR.joinpath(f"checkpoints_sgd_thm3/scaling/{p}({q})-2048-2/seeded_final_loss.txt")).mean()
        Loss_aN.append(train_loss)
    ax3.plot(Alphas, Loss_a, 's', color=color_0, label=r"$N=512$")
    ax3.plot(Alphas, Loss_aN, 's', color=color_3, label=r"$N=2048$")
    ax3.plot(Alphas, 4.9 * lmb * (p - 1) * np.ones(Alphas.shape), color=color_1)
    ax3.plot(Alphas, 5.1 * lmb * (p - 1) * np.ones(Alphas.shape), linewidth=1.4, linestyle='--', color=color_2)
    ax3.set_ylim(0.055, 0.145)
    ax3.set_yticks(np.arange(0.06, 0.141, 0.02))
    ax3.set_xlabel(r"$\alpha$", fontsize=14)
    ax3.legend(loc=4, frameon=False, fontsize=12)
    ax3.set_title("(c)")
    # plt.savefig(RAW_FIGURE_SAVE_DIR.joinpath(f"loss_scaling_plot.svg"), bbox_inches="tight")
    plt.savefig(FIGURE_SAVE_DIR.joinpath(f"loss_scaling_plot.svg"), bbox_inches="tight")
    plt.savefig(FIGURE_SAVE_DIR.joinpath(f"loss_scaling_plot.pdf"), bbox_inches="tight")
    plt.show()

def dyn_cos_similarity_of_ansatz_with_time():
    p = 59
    q = p // 2
    N = 512
    Epochs = np.arange(0, 10001, 50)
    data = mod_tensors.ModAddDataTrainValid(p=p, select=q)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"wspace": 0.05})
    S_Y_sim = []
    V_YF_sim = []
    V_SF_sim = []
    for ep in Epochs:
        filename = f"checkpoints_sgd_thm3/{p}(0.50)-{N}-2/model_epoch_{ep}.pt"
        m = model_network.ModelNetworkCE(p=p, N=N, from_file=ENV_DIR.joinpath(filename))
        S = (data.Y_train + 1 / p) - m.forward(data.X_train)
        S_Y_sim.append(model.cos_similarity(S, data.Y_train))
        F_t = m.activate(data.X_train)
        V_YF_sim.append(model.cos_similarity(m.V, data.Y_train @ F_t.T))
        V_SF_sim.append(model.cos_similarity(m.V, S @ F_t.T))
    ax1.plot(Epochs, S_Y_sim)
    ax2.plot(Epochs, V_YF_sim)
    ax2.plot(Epochs, V_SF_sim)
    ax1.set_title(r"cos sim of $S$ and $\tilde{Y}$")
    ax2.set_title(r"cos sim of $V$ and $\tilde{Y}F^T$ and $SF^T$")
    plt.savefig(RAW_FIGURE_SAVE_DIR.joinpath(f"dyn_cos_similarity_of_ansatz_with_time.svg"), bbox_inches="tight")
    plt.show()

def cos_similarity_of_ansatz_with_time(ax1=None, ax2=None, title_labels=["", ""]):
    self_plot = (ax1 is None)
    p = 59
    q = p // 2
    N = 512
    if self_plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"wspace": 0.05})
    d = np.loadtxt(ENV_DIR.joinpath(f"checkpoints_sgd_thm3/aligns/{p}({q})-{N}-2/cos_sim.txt"))
    Epochs = d[:, 0] + 1 # offset of init
    ax1.plot(Epochs, d[:, 1], label=f"$N={N}$")
    legend_y1, = ax2.plot(Epochs, d[:, 2], label=rf"$\tilde{{Y}}F^T, N={N}$")
    legend_s1, = ax2.plot(Epochs, d[:, 3], label=f"$SF^T, N={N}$")
    if not self_plot:
        pos = ax1.get_position()
        ax1.set_position([pos.x0 + 0.01, pos.y0, pos.width, pos.height])
        pos = ax2.get_position()
        ax2.set_position([pos.x0 - 0.018, pos.y0, pos.width, pos.height])
    
    N = 2048
    d = np.loadtxt(ENV_DIR.joinpath(f"checkpoints_sgd_thm3/aligns/{p}({q})-{N}-2/cos_sim.txt"))
    Epochs = d[:, 0]
    ax1.plot(Epochs, d[:, 1], alpha=0.5, color="C0", label=f"$N={N}$")
    legend_y2, = ax2.plot(Epochs, d[:, 2], alpha=0.5, color="C0", label=rf"$\tilde{{Y}}F^T, N={N}$")
    legend_s2, = ax2.plot(Epochs, d[:, 3], alpha=0.5, color="C1", label=f"$SF^T, N={N}$")
    ax1.legend(loc=4, frameon=False, fontsize=12)
    ax2.legend([legend_y1, legend_s1, (legend_y1, legend_y2)], [r"$V,\tilde{Y}F^T$", r"$V,SF^T$", r"$N=512,2048$"],
          handler_map={tuple: HandlerTuple(ndivide=None)},
          loc=4, frameon=False, fontsize=12)
    # ax1.set_title(r"$\cos\left<S,\tilde{Y}\right>$")
    # ax2.set_title(r"$\cos\left<V,\tilde{Y}F^T\right>$ and $\cos\left<V,SF^T\right>$")
    ax1.set_title(title_labels[0] + "Residual alignment")
    ax2.set_title(title_labels[1] + "Output-layer locking")
    ax1.set_xlabel("epochs", fontsize=12)
    ax2.set_xlabel("epochs", fontsize=12)
    ax1.set_xticks([0, 10000, 20000, 30000])
    ax2.set_xticks([0, 10000, 20000, 30000])
    ax2.set_yticklabels([])
    ax1.set_ylim(0.89, 1.01)
    ax2.set_ylim(0.89, 1.01)
    ax1.set_ylabel("cosine similarity", fontsize=12)
    # plt.savefig(RAW_FIGURE_SAVE_DIR.joinpath(f"dyn_cos_similarity_of_ansatz_with_time.svg"), bbox_inches="tight")
    if self_plot:
        plt.show()

def cos_similarity_of_ansatz_with_time_combined(ax=None, title_label="", fontsize=12):
    self_plot = (ax is None)
    if self_plot:
        fig, ax = plt.subplots(1, 1)
    p = 59
    q = p // 2
    N = 512
    d = np.loadtxt(ENV_DIR.joinpath(f"checkpoints_sgd_thm3/aligns/{p}({q})-{N}-2/cos_sim.txt"))
    Epochs = d[:, 0] + 1 # offset of init
    ax.plot(Epochs, d[:, 1], label=r"$S, \tilde{Y}$", zorder=2)
    legend_y1, = ax.plot(Epochs, d[:, 2], label=r"$V, \tilde{{Y}}F^T$", zorder=1, color="C1")
    legend_s1, = ax.plot(Epochs, d[:, 3], label=r"$V, SF^T$", zorder=1, color="C2")
    
    ax.legend(loc=4, frameon=False, fontsize=fontsize)
    # ax.legend([legend_y1, legend_s1, (legend_y1, legend_y2)], [r"$V,\tilde{Y}F^T$", r"$V,SF^T$", r"$N=512,2048$"],
    #       handler_map={tuple: HandlerTuple(ndivide=None)},
    #       loc=4, frameon=False, fontsize=12)
    # ax1.set_title(r"$\cos\left<S,\tilde{Y}\right>$")
    # ax2.set_title(r"$\cos\left<V,\tilde{Y}F^T\right>$ and $\cos\left<V,SF^T\right>$")
    ax.set_title(title_label + "Alignment", fontsize=fontsize)
    ax.set_xlabel("epochs", fontsize=fontsize)
    ax.set_xticks([0, 10000, 20000, 30000])
    ax.set_ylim(0.89, 1.01)
    ax.tick_params(axis='both', labelsize=fontsize-3)
    ax.set_ylabel("cosine similarity", fontsize=fontsize)
    # plt.savefig(RAW_FIGURE_SAVE_DIR.joinpath(f"dyn_cos_similarity_of_ansatz_with_time.svg"), bbox_inches="tight")
    if self_plot:
        plt.show()

def loss_acc_std_plot(ax1=None, ax2=None, title_labels=["", ""]):
    p = 59
    frac = 0.5
    N = 512
    self_plot = (ax1 is None)
    d = np.loadtxt(ENV_DIR.joinpath("checkpoints_sgd_thm3/59(0.50)-512-2/loss_acc.txt"), delimiter=' ')
    Ref_epochs = 1 + np.arange(0, 10001, 50)
    Epochs = np.insert(Ref_epochs, 0, 0)
    if self_plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot([-600, 1900], [1/p, 1/p], linestyle='--', color='gray', alpha=0.7, linewidth=1.4) # chance level
    ax1.text(2100, 0.03, r"$\frac{1}{p}$", fontsize=16, color='black', ha='left', va='center')
    
    ax2.plot([-600, 1900], [np.log(p), np.log(p)], linestyle='--', color='gray', alpha=0.7, linewidth=1.4) # chance level
    ax2.text(2100, np.log(p) - 0.05, r"$\log p$", fontsize=12, color='black', ha='left', va='center')
    
    ax1.plot(Epochs, d[:, 2], color="#201E8F", label="train")
    ax1.plot(Epochs, d[:, 3], color="#DB2E28", label="test")
    ax2.plot(Epochs, d[:, 0], color="#201E8F", label="train")
    ax2.plot(Epochs, d[:, 1], color="#DB2E28", label="test")
    ax1.legend(loc=4, frameon=False, fontsize=12)
    ax2.legend(loc=1, frameon=False, fontsize=12)
    ax1.set_xlabel("epochs", fontsize=12)
    ax2.set_xlabel("epochs", fontsize=12)
    ax1.set_ylabel("accuracy", fontsize=12)
    ax2.set_ylabel("loss", fontsize=12)
    ax1.set_title(title_labels[0] + "Train/test accuracy")
    ax2.set_title(title_labels[1] + "Train/test loss")

    ax1.set_xlim(-500, 10500)
    ax2.set_xlim(-500, 10500)
    
    if self_plot:
        plt.show()

def acc_inset_plot(ax=None, title_label="", fontsize=12):
    p = 59
    frac = 0.5
    N = 512
    self_plot = (ax is None)
    d = np.loadtxt(ENV_DIR.joinpath("checkpoints_sgd_thm3/59(0.50)-512-2/loss_acc.txt"), delimiter=' ')
    Ref_epochs = 1 + np.arange(0, 10001, 50)
    Epochs = np.insert(Ref_epochs, 0, 0)
    if self_plot:
        fig, ax = plt.subplots(1, 1)
    color1 = "#201E8F"
    color2 = "#DB2E28"

    ax.plot([-600, 1900], [1/p, 1/p], linestyle='--', color='gray', alpha=0.7, linewidth=1.4) # chance level
    # ax.text(2100, 0.03, r"$\frac{1}{p}$", fontsize=16, color='black', ha='left', va='center')
    
    ax.plot(Epochs, d[:, 2], color=color1, label="train")
    ax.plot(Epochs, d[:, 3], color=color2, label="test")
    ax.legend(loc=4, frameon=False, fontsize=fontsize)

    axin = ax.inset_axes([0.53, 0.38, 0.45, 0.45])
    ep_lim = 550 // 50 + 2
    axin.plot([-600, 1900], [1/p, 1/p], linestyle='--', color='gray', alpha=0.7, linewidth=1.4)
    axin.plot(Epochs[:ep_lim], d[:ep_lim, 2], color=color1)
    axin.plot(Epochs[:ep_lim], d[:ep_lim, 3], color=color2)
    axin.set_ylim(-0.01, 0.06)
    axin.set_xlim(-50, 450)
    axin.set_xticks([0, 200, 400])
    axin.set_yticks([0, 1/p], ["0", r"$\frac{1}{p}$"])
    axin.tick_params(axis='both', labelsize=fontsize-3)
    axin_ylabel_p = axin.get_yticklabels()[1]
    axin_ylabel_p.set_fontsize(fontsize + 3)
    import matplotlib.transforms as mtransforms
    offset = mtransforms.ScaledTranslation(0, 0.05, axin.figure.dpi_scale_trans)
    axin_ylabel_p.set_transform(axin_ylabel_p.get_transform() + offset)

    ax.tick_params(axis='both', labelsize=fontsize-3)

    ax.set_xlabel("epochs", fontsize=fontsize)
    ax.set_ylabel("accuracy", fontsize=fontsize)
    ax.set_title(title_label + "Train/test accuracy", fontsize=fontsize)

    ax.set_xlim(-500, 10500)
    
    if self_plot:
        plt.show()

def loss_std_plot(ax=None, title_label="", fontsize=12):
    p = 59
    frac = 0.5
    N = 512
    self_plot = (ax is None)
    d = np.loadtxt(ENV_DIR.joinpath("checkpoints_sgd_thm3/59(0.50)-512-2/loss_acc.txt"), delimiter=' ')
    Ref_epochs = 1 + np.arange(0, 10001, 50)
    Epochs = np.insert(Ref_epochs, 0, 0)
    if self_plot:
        fig, ax = plt.subplots(1, )

    ax.plot([-600, 1900], [np.log(p), np.log(p)], linestyle='--', color='gray', alpha=0.7, linewidth=1.4) # chance level
    ax.text(2100, np.log(p) - 0.05, r"$\log p$", fontsize=fontsize, color='black', ha='left', va='center')

    ax.plot(Epochs, d[:, 0], color="#201E8F")
    ax.plot(Epochs, d[:, 1], color="#DB2E28")
    ax.set_xlabel("epochs", fontsize=fontsize)
    ax.set_ylabel("loss", fontsize=fontsize)
    ax.set_title(title_label + "Train/test loss", fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize-3)

    ax.set_xlim(-500, 10500)
    
    if self_plot:
        plt.show()

def loss_acc_and_cos_sim():
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(14, 3))
    for ax in (ax3, ax4):
        pos = ax.get_position()
        ax.set_position([pos.x0 + 0.015, pos.y0, pos.width, pos.height])
    loss_acc_std_plot(ax1, ax2, title_labels=["(a) ", "(b) "])
    cos_similarity_of_ansatz_with_time(ax3, ax4, title_labels=["(c) ", "(d) "])
    plt.savefig(FIGURE_SAVE_DIR.joinpath("loss_acc_and_cos_sim.svg"), bbox_inches="tight")
    plt.savefig(FIGURE_SAVE_DIR.joinpath("loss_acc_and_cos_sim.pdf"), bbox_inches="tight")
    plt.show()

def gap_width_scaling(ax=None, title_label="", p=59, q=29, Ns = [256, 512, 1024, 2048, 4096]):
    self_plot = (ax is None)
    if self_plot:
        fig, ax = plt.subplots(1, 1)
    color_0 = ["#1134a5", "#86164C"]
    color_2 = ["#485a96", "#A42D67"]

    Ns = np.array(Ns)
    gaps = []
    N_effs = []
    N_eff_fs = []
    Acc = [5, 9]
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # seeds mean
    for N in Ns:
        gap_sum = np.zeros(len(Acc))
        n_eff_sum = np.zeros(len(Acc))
        n_eff_f_sum = np.zeros(len(Acc))
        for seed in seeds:
            data = np.loadtxt(ENV_DIR / f"checkpoints_sgd_thm3/gap_seed/{p}({q})-{N}-2/{seed}/kernel_gaps.txt", delimiter=" ", skiprows=1)
            gap_sum += data[Acc, 2] / data[Acc, 3] - data[0, 2] / data[0, 3]
            # gap_sum += data[accit, 0] - data[0, 0]
            n_eff_sum += data[Acc, 4]
            n_eff_f_sum += data[Acc, 5]
        gaps.append(gap_sum / len(seeds))
        N_effs.append(n_eff_sum / len(seeds))
        N_eff_fs.append(n_eff_f_sum / len(seeds))
    gaps = np.array(gaps)
    N_effs = np.array(N_effs)
    N_eff_fs = np.array(N_eff_fs)

    # log linear fit
    slopes = []
    for i, acc in enumerate(Acc):
        slope, intercept = np.polyfit(np.log(N_eff_fs[:, i]), np.log(gaps[:, i]), 1)
        slopes.append(slope)
        print(slope)

        ax.loglog(N_eff_fs[:, i], gaps[:, i], 's', color=color_0[i])
        # rf"$\log\Delta_\rho={slope:.2f}\log N_{{\rm eff}}^F+{intercept:.2f}$"
        ax.loglog(N_eff_fs[:, i], np.exp(slope * np.log(N_eff_fs[:, i]) + intercept), '-', label=rf"$\gamma={-slope:.2f}, \rho={acc * 0.1:.1f}$", color=color_2[i])
    ax.legend(loc=1, frameon=False, fontsize=12)

    ax.set_xticks([120, 200, 300, 400, 500, 600], [120, 200, 300, 400, 500, 600])
    # ax.set_yticks(np.arange(6, 12.1), np.arange(6, 13, dtype=int))
    ax.set_yticks(np.arange(6, 20.1, 2), np.arange(6, 21, 2, dtype=int))
    ax.set_xlabel(r"$N_{\rm eff}^F$", fontsize=12)
    ax.set_ylabel(r"$\Delta_\rho$", fontsize=12)
    ax.set_title(title_label + "Alignment gap scaling")
    # ax.set_title(f'Gap Scaling vs N for p={p}, q={q}')
    if self_plot:
        plt.show()

def N_eff_with_N(ax=None, title_label="", p=59, q=29, Ns = [256, 512, 1024, 2048, 4096]):
    self_plot = (ax is None)
    if self_plot:
        fig, ax = plt.subplots(1, 1)

    cmap = plt.get_cmap('plasma')
    colors = [cmap(i) for i in np.linspace(0., 0.6, 3, endpoint=True)]
    n_effs_end = []
    n_eff_fs_end = []
    n_eff_fs_p0 = []
    n_eff_fs_p5 = []
    accit = 0
    for i, N in enumerate(Ns):
        d = np.loadtxt(ENV_DIR / f"checkpoints_sgd_thm3/aligns/{p}({q})-{N}-2/align.txt", delimiter=" ", skiprows=1)
        k = np.loadtxt(ENV_DIR / f"checkpoints_sgd_thm3/aligns/{p}({q})-{N}-2/kernel_gaps.txt", delimiter=" ", skiprows=1)
        n_eff_fs_end.append(d[-1, 7] / N)
        n_effs_end.append(d[-1, 6] / N)
        n_eff_fs_p0.append(k[0, -1] / N)
        n_eff_fs_p5.append(k[5, -1] / N)
    ax.plot(Ns, n_eff_fs_p0, 's-', color=colors[0], label=r"$\rho=1/p$", zorder=3)
    ax.plot(Ns, n_eff_fs_p5, 's-', color=colors[1], label=r"$\rho=0.5$", zorder=2)
    ax.plot(Ns, n_eff_fs_end, 's-', color=colors[2], label=r"$\rm epoch=30001$", zorder=1)
    # ax.plot(Ns, n_effs_end, 's-')
    ax.legend(loc=1, frameon=False, fontsize=12)
    ax.set_xlabel("$N$", fontsize=12)
    ax.set_ylabel(r"$N_{\rm eff}^F/N$", fontsize=12, labelpad=-2)
    ax.set_xticks(Ns, [i if i != 512 else "" for i in Ns])
    ax.set_title(title_label + "Feature effective widths")

    if self_plot:
        plt.show()

def finite_width_dynamics():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    acc_plot_width(ax1, title_label="(a) ")
    alignment_increase_with_time(ax2, title_label="(b) ")
    # N_eff_with_N(ax3, title_label="(c) ")
    gap_width_scaling(ax3, title_label="(c) ")
    plt.savefig(FIGURE_SAVE_DIR.joinpath("finite_width_dynamics.svg"), bbox_inches="tight")
    plt.savefig(FIGURE_SAVE_DIR.joinpath("finite_width_dynamics.pdf"), bbox_inches="tight")
    plt.show()

def acc_cossim_earlylate():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    pos = ax1.get_position()
    ax1.set_position([pos.x0 - 0.01, pos.y0, pos.width, pos.height])
    acc_inset_plot(ax1, "(a) ")
    cos_similarity_of_ansatz_with_time_combined(ax2, "(b) ")
    energy_single_neuron(ax=ax3, hide_cbar=True)
    ax3.set_xlabel("epochs", fontsize=12)
    ax3.set_ylabel("normalized energy", fontsize=12)
    ax3.set_title("(c) Neuron energies", fontsize=12)
    plt.savefig(FIGURE_SAVE_DIR.joinpath("acc_cossim_earlylate.svg"), bbox_inches="tight")
    plt.savefig(FIGURE_SAVE_DIR.joinpath("acc_cossim_earlylate.pdf"), bbox_inches="tight")
    plt.show()

def acc_loss_cossim_earlylate():
    fontsize=15
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(18, 4))
    pos = ax1.get_position()
    ax1.set_position([pos.x0 - 0.012, pos.y0, pos.width, pos.height])
    pos = ax2.get_position()
    ax2.set_position([pos.x0 - 0.015, pos.y0, pos.width, pos.height])
    pos = ax3.get_position()
    ax3.set_position([pos.x0 - 0.005, pos.y0, pos.width, pos.height])
    acc_inset_plot(ax1, "(a) ", fontsize=fontsize)
    loss_std_plot(ax2, "(b) ", fontsize=fontsize)
    cos_similarity_of_ansatz_with_time_combined(ax3, "(c) ", fontsize=fontsize)
    energy_single_neuron(ax=ax4, hide_cbar=True)
    ax4.set_xlabel("epochs", fontsize=fontsize)
    ax4.set_ylabel("normalized energy", fontsize=fontsize)
    ax4.set_title("(d) Neuron energies", fontsize=fontsize)
    ax4.tick_params(axis='both', labelsize=fontsize-3)
    plt.savefig(FIGURE_SAVE_DIR.joinpath("acc_loss_cossim_earlylate.svg"), bbox_inches="tight")
    plt.savefig(FIGURE_SAVE_DIR.joinpath("acc_loss_cossim_earlylate.pdf"), bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    # acc_plot_comparison_between_relu_and_sqr()
    # kernel_plot_tt_tv(init="gauss")
    # kernel_gauss_fourier()
    # alignment_increase_with_time()
    # acc_plot_width()
    # gap_width_scaling()
    # loss_acc_and_cos_sim()
    loss_scaling_plot()
    # finite_width_dynamics()
    # acc_loss_cossim_earlylate()
    # acc_inset_plot()
    # early_late_comparison_between_relu_and_sqr_apdx()
