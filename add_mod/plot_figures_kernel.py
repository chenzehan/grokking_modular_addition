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
CODE_DIR = Path(__file__).resolve().parent.parent
FIGURE_APDX_SAVE_DIR = Path(__file__).resolve().parent.parent.joinpath("figures_apdx")
RAW_FIGURE_SAVE_DIR = Path(__file__).resolve().parent.parent.joinpath("raw_figures")

def kernel_plot_tt_tv(fig=None, axes=None, title_labels=[""]*3, p=17, init="gauss", sigma=model.ReLU(), indp_cbar=True):
    self_plot = (axes is None)
    q = p // 2
    N = 512
    data = mod_tensors.ModAddDataTrainValid(p=p, select=q)
    if init == "gauss":
        supscr = 'g'
        w_init = model.rand_init(p, std=0.5)
    elif init == "fourier":
        supscr = 'f'
        w_init = model.fourier_init_rand_phase(p)
    elif init == "sqrwave":
        supscr = 'sq'
        w_init = model.sqrwave_init_rand_phase(p)
    else:
        supscr = '-'
        w_init = init
    m = model.ModelCE(p, N=N, w_init=w_init, sigma=sigma, from_file=None)
    F = m.activate(data.X)
    F_t = m.activate(data.X_train)
    F_v = m.activate(data.X_valid)
    K = F.T @ F / N
    K_tt = F_t.T @ F_t / N
    K_tv = F_t.T @ F_v / N
    H = np.kron(np.eye(p), np.ones((q, q)))
    Kb_tv = (1 / q**2) * H.T @ K_tv @ H
    if self_plot:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), gridspec_kw={"wspace": 0.05})
    ax1, ax2, ax3 = axes
    for ax, k in zip(axes, [K_tt, K_tv, Kb_tv]):
        im = ax.imshow(k)
        fig.colorbar(im, ax=ax, shrink=0.78, aspect=30)

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
    ax1.set_title(title_labels[0] + rf"$\frac{{1}}{{N}}K_{{tt}}^{{{supscr}}}$", fontsize=12)
    ax2.set_title(title_labels[1] + rf"$\frac{{1}}{{N}}K_{{tv}}^{{{supscr}}}$", fontsize=12)
    ax3.set_title(title_labels[2] + rf"$\frac{{1}}{{N}}\overline{{K}}_{{tv}}^{{{supscr}}}$", fontsize=12)
    # plt.savefig(RAW_FIGURE_SAVE_DIR.joinpath(f"kernel_tt_tv_{init}.svg"), bbox_inches="tight")
    if self_plot:
        plt.show()

def kernel_gauss_fourier_sqr():
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), gridspec_kw={"wspace": 0.05, "hspace": 0.01})
    kernel_plot_tt_tv(fig, axes=axes[0], title_labels=["(a) ", "(b) ", "(c) "], sigma=model.Sqr())
    kernel_plot_tt_tv(fig, axes=axes[1], title_labels=["(d) ", "(e) ", "(f) "], init="fourier", sigma=model.Sqr())
    plt.savefig(RAW_FIGURE_SAVE_DIR.joinpath(f"kernel_gauss_fourier_sqr.svg"), bbox_inches="tight", dpi=800)
    plt.savefig(FIGURE_APDX_SAVE_DIR.joinpath(f"kernel_gauss_fourier_sqr.svg"), bbox_inches="tight", dpi=800)
    plt.savefig(FIGURE_APDX_SAVE_DIR.joinpath(f"kernel_gauss_fourier_sqr.pdf"), bbox_inches="tight", dpi=800)
    plt.show()

def kernel_sqrwave():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), gridspec_kw={"wspace": 0.05})
    kernel_plot_tt_tv(fig, axes=axes, title_labels=["(a) ", "(b) ", "(c) "], init="sqrwave", sigma=model.ReLU())
    # plt.savefig(RAW_FIGURE_SAVE_DIR.joinpath(f"kernel_gauss_fourier_sqr.svg"), bbox_inches="tight", dpi=800)
    plt.savefig(FIGURE_APDX_SAVE_DIR.joinpath(f"kernel_sqrwave.svg"), bbox_inches="tight", dpi=800)
    plt.savefig(FIGURE_APDX_SAVE_DIR.joinpath(f"kernel_sqrwave.pdf"), bbox_inches="tight", dpi=800)
    plt.show()

def kernel_multilayer(ax=None, p=11, N=512, layer=2, clim=(0.15, 1.45), hide_y=False, title=""):
    self_plot = (ax is None)
    if self_plot:
        fig, ax = plt.subplots(1, 1)
    q = p // 2
    data = mod_tensors.ModAddDataTrainValid(p=p, select=q)
    m = model.ModelCE(p, N=N, w_init=model.rand_init(p), sigma=model.ReLU(), from_file=None)
    F = m.activate(data.X)
    rng = np.random.default_rng(seed=42)
    for l in range(2, layer):
        M = rng.normal(0, 1/np.sqrt(N), size=(N, N))
        F = M @ F
    K = F.T @ F / N
    im = ax.imshow(K)
    if self_plot:
        fig.colorbar(im, ax=ax)
        plt.show()
    else:
        im.set_clim(clim)
        if hide_y:
            ax.set_yticklabels([])
        ax.set_title(title)
    return im

def kernel_multilayer_34():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"wspace": 0.05})
    im1 = kernel_multilayer(ax1, layer=3, title="(a) 3 layers")
    im2 = kernel_multilayer(ax2, layer=4, title="(b) 4 layers", hide_y=True)
    fig.colorbar(im2, ax=[ax1, ax2])
    plt.savefig(FIGURE_APDX_SAVE_DIR.joinpath(f"kernel_multilayer_34.svg"), bbox_inches="tight", dpi=800)
    plt.savefig(FIGURE_APDX_SAVE_DIR.joinpath(f"kernel_multilayer_34.pdf"), bbox_inches="tight", dpi=800)
    plt.show()

def kernel_illus():
    fig, (axes, axes2) = plt.subplots(2, 3, figsize=(12, 9), gridspec_kw={"wspace": 0.05, "hspace": 0.05})
    p = 11
    data = mod_tensors.ModAddDataComplete(p)
    X = data.X
    Y = data.Y
    B = Y.T @ Y
    C = X.T @ X
    C_od = C.copy()
    np.fill_diagonal(C_od, np.nan)
    pairs = np.array(data.get_as_pairs())
    d = (pairs[:,0] - pairs[:,1]) % p
    D = np.eye(p)[d].T
    C_mod = D.T @ D
    C_mm = C_mod.copy()
    np.fill_diagonal(C_mm, np.nan)
    for ax, k, title in zip(axes, [B, C_od, C_mm], [r"(a) $B=Y^\top Y$", r"(b) $C=X^\top X$", r"(c) $C_{\rm mod}=D^\top D$"]):
        ax.imshow(k)
        ax.set_title(title)
    axes[1].set_yticklabels([])
    axes[2].set_yticklabels([])

    m = model.ModelCE(p, N=2048, sigma=model.Sqr())
    mf = model.ModelCE(p, N=2048, sigma=model.Sqr(), w_init=model.fourier_init_rand_phase(p))
    F = m.activate(X)
    K = F.T @ F
    K_od = K.copy()
    np.fill_diagonal(K_od, np.nan)
    Ff = mf.activate(X)
    Kf = Ff.T @ Ff
    Kf_od = Kf.copy()
    np.fill_diagonal(Kf_od, np.nan)
    Kfm = Kf.copy()
    Kfm[B > 0] = np.nan
    for ax, k, title in zip(axes2, [Kf_od, K_od, Kfm], [r"(d) $K^f$", r"(e) $K^g$", r"(f) $K^f$"]):
        ax.imshow(k)
        ax.set_title(title)
    axes2[1].set_yticklabels([])
    axes2[2].set_yticklabels([])
    plt.savefig(FIGURE_APDX_SAVE_DIR.joinpath(f"kernel_illus.svg"), bbox_inches="tight", dpi=800)
    plt.savefig(FIGURE_APDX_SAVE_DIR.joinpath(f"kernel_illus.pdf"), bbox_inches="tight", dpi=800)
    
    plt.show()

if __name__ == "__main__":
    kernel_multilayer_34()
    # kernel_multilayer(layer=3)
    # kernel_illus()
    # kernel_sqrwave()