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

def final_freq_phase_dis():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    p = 59
    ep = 30000
    Freq = [0] * (p // 2 + 2)
    Phi = []
    for seed in range(10):
        filename = f"checkpoints_sgd_thm3/final/{p}(29)-2048-2/{seed}/model_epoch_{ep}.pt"
        m = model_network.ModelNetworkCE(p=p, N=2048, from_file=filename)
        for w in m.W:
            f1 = np.fft.fft(w[:p])[:p // 2 + 1]
            f2 = np.fft.fft(w[p:])[:p // 2 + 1]
            fi1 = np.argmax(np.abs(f1))
            fi2 = np.argmax(np.abs(f2))
            if np.abs(f1[fi1]) < 0.5:
                continue
            phi1 = np.angle(f1[fi1])
            phi2 = np.angle(f2[fi2])
            Freq[fi1] += 1
            Phi.append((phi2 - phi1) % (2 * np.pi))

    print(np.sum(Freq), np.sum(Freq[1:]))
    ax1.bar(np.arange(1, p // 2 + 2), Freq[1:])
    ax1.set_yticks([0, 100, 200, 300, 400])
    ax2.set_yticks([0, 30, 60, 90, 120, 150])
    ax2.set_yticklabels([])
    ax2.hist(Phi, bins=90)
    ax2.set_xticks([0, np.pi, 2*np.pi], ["$0$", r"$\pi$", r"$2\pi$"])
    ax1.set_xlabel(r"Frequencies $\omega_n$", fontsize=12)
    ax1.set_ylabel("counts", fontsize=12)
    ax2.set_xlabel(r"Phase difference $\Delta\phi$", fontsize=12)
    ax2.set_ylabel("distribution", fontsize=12)
    plt.savefig(FIGURE_APDX_SAVE_DIR.joinpath(f"final_freq_phase_dis.svg"), bbox_inches="tight", dpi=800)
    plt.savefig(FIGURE_APDX_SAVE_DIR.joinpath(f"final_freq_phase_dis.pdf"), bbox_inches="tight", dpi=800)
    plt.show()


def early_lr(ax = None, p = 59, N = 512, epochs = 400, interval = 5, E_window=(0.4, 0.6)):
    self_plot = (ax is None)
    if self_plot:
        fig, ax = plt.subplots(1, 1)
    ax2 = ax.twinx()
    # ax.set_zorder(2)
    # ax.patch.set_visible(False)
    # ax2.set_zorder(1)
    filename = f"checkpoints_sgd_thm3/early/{p}(0.50)-{N}-2"
    Epochs = np.arange(0, epochs + 1, interval)
    E = np.zeros((len(Epochs), N))
    cn = p**2 / 2 * (1/6 - 1/8)
    data = mod_tensors.ModAddDataTrainValid(p=p, select=29)
    for i, ep in enumerate(Epochs):
        m = model_network.ModelNetworkCE(p=p, N=N, sigma=model.ReLU(), from_file=filename + f"/model_epoch_{ep}.pt")
        F = m.activate(data.X)
        E[i, :] = 0.5 * np.linalg.norm(data.Y @ F.T, axis=0)**2 / np.linalg.norm(m.W, axis=-1)**2
    
    loss = np.loadtxt(ENV_DIR / filename / "results.txt", delimiter=',')[:len(Epochs), 0]
    ax2.plot(Epochs, loss, '--', color="indianred")
    ax2.set_yticks([0, 1, 2, 3, 4])
    
    E /= cn # normalize to 1
    selected = np.flatnonzero(E[-1, :] >= 0.9)
    unselected = np.flatnonzero(E[-1, :] < 0.9)
    ref_E = E[:, selected]
    un_E = E[:, unselected]
    ts = np.array([np.searchsorted(ref_E[:, i], E_window) for i in range(ref_E.shape[1])])
    t_avg = np.mean(ts) * interval
    r_avg = np.mean((E_window[1] - E_window[0]) / ((ts[:, 1] - ts[:, 0]) * interval))
    colormap = plt.get_cmap("viridis")
    for e in ref_E.T:
        ax.plot(Epochs, e, color=colormap(1 - e[-1]))
    for e in un_E.T:
        ax.plot(Epochs, e, color=colormap(1 - e[-1]), alpha=0.2)
    ax.set_xlabel("epochs", fontsize=12)
    ax.set_ylabel("normalized energy", fontsize=12)
    ax2.set_ylabel("loss", fontsize=12)

    if self_plot:
        plt.show()
    return t_avg, r_avg

def late_lr(ax=None, p = 59, N= 512, epochs = 10000, ref_epoch=2000, interval = 50, 
            ref_crit=(0.3, 0.6), E_window=(0.4, 0.6)):
    self_plot = (ax is None)
    if self_plot:
        fig, ax = plt.subplots(1, 1)
    ax2 = ax.twinx()
    filename = f"checkpoints_sgd_thm3/{p}(0.50)-{N}-2"
    Epochs = np.arange(0, epochs + 1, interval)
    E = np.zeros((len(Epochs), N))
    cn = p**2 / 2 * (1/6 - 1/8)
    data = mod_tensors.ModAddDataTrainValid(p=p, select=29)
    for i, ep in enumerate(Epochs):
        m = model_network.ModelNetworkCE(p=p, N=N, sigma=model.ReLU(), from_file=filename + f"/model_epoch_{ep}.pt")
        F = m.activate(data.X)
        E[i, :] = 0.5 * np.linalg.norm(data.Y @ F.T, axis=0)**2 / np.linalg.norm(m.W, axis=-1)**2

    loss = np.loadtxt(ENV_DIR / filename / "results.txt", delimiter=',')[:len(Epochs), 0]
    ax2.plot(Epochs, loss, '--', color="indianred")
    ax2.set_ylim(-0.005, 0.125)
    
    E /= cn # normalize to 1
    ref_index = ref_epoch // interval
    selected = np.flatnonzero((E[-1, :] >= ref_crit[1]) & (E[ref_index, :] <= ref_crit[0]))
    unselected = np.flatnonzero((E[-1, :] < ref_crit[1]) | (E[ref_index, :] > ref_crit[0]))
    ref_E = E[:, selected]
    un_E = E[:, unselected]
    
    colormap = plt.get_cmap("viridis")
    for e in un_E.T:
        ax.plot(Epochs, e, color=colormap(e[ref_epoch // interval]), alpha=0.1)
    for e in ref_E.T:
        ax.plot(Epochs, e, color=colormap(e[ref_epoch // interval]))
    ax.set_xlabel("epochs", fontsize=12)
    ax.set_ylabel("normalized energy", fontsize=12)
    ax2.set_ylabel("loss", fontsize=12)
    
    if self_plot:
        plt.show()

def sci_latex(x, sig=2):
    s = f"{x:.{sig}e}"           # 如 '1.23e+05'
    mantissa, exp = s.split('e')
    exp = int(exp)               # 去掉 + 号,转成整数
    return rf"{mantissa}\times10^{{{exp}}}"

def early_change_rate(ax=None, p=59, N=512, epochs=400, interval=5):
    # feom dynamics/early_late.py
    self_plot = (ax is None)
    if self_plot:
        fig, ax = plt.subplots(1, 1)
    filename = f"checkpoints_sgd_thm3/early/{p}(0.50)-{N}-2"
    Epochs = np.arange(interval, epochs + 1, interval)
    Epochs_mid = np.arange(interval, epochs + 1, interval) - interval / 2
    Diff = np.zeros((len(Epochs), N))

    # data = ModAddDataTrainValid(p=p, select=29)
    m0 = model_network.ModelNetworkCE(p=p, N=N, sigma=model.ReLU(), from_file=filename + f"/model_epoch_{0}.pt")
    for i, ep in enumerate(Epochs):
        m = model_network.ModelNetworkCE(p=p, N=N, sigma=model.ReLU(), from_file=filename + f"/model_epoch_{ep}.pt")
        coss = model.cos_similarity_ax(m0.W, m.W, axis=-1)
        coss[coss > 1] = 1
        Diff[i, :] = np.arcsin(np.sqrt(1 - coss))
        m0 = m
    Diff /=  interval * 8
    Times = Epochs_mid[:9] * 8
    slope = np.sum(Diff[:9, :].T * Times) / N / np.sum(Times * Times)
    
    # for d, e in zip(Diff.T, Ef):
    #     ax.plot(Epochs, d, color=colormap(1 - e))
    
    Diff = np.insert(Diff, 0, 0, axis=0)
    Epochs0 = np.insert(Epochs_mid, 0, 0)
    TimesR = np.array([0, 45 * 8])
    TimesExt = np.array([45 * 8, 55 * 8])
    y_scale = 1e-4
    ax.plot(Epochs0, Diff / y_scale, color=plt.get_cmap("viridis")(0.15), alpha=0.5)
    ax.plot(TimesR / 8, slope * TimesR / y_scale, color='orange', label=rf"$\bar{{v}}={sci_latex(slope)}\times t$")
    ax.plot(TimesExt / 8, slope * TimesExt / y_scale, '--', color='orange')
    ax.legend(loc=1, frameon=False, fontsize=12)
    ax.set_xlabel("epochs", fontsize=12)
    ax.set_ylabel(r"changing rate $(\times 10^{-4})$", fontsize=12)
    if self_plot:
        plt.show()

def late_change_rate(ax=None, p=59, N=512, epochs = 10000, start_epoch=4000, interval=50):
    self_plot = (ax is None)
    if self_plot:
        fig, ax = plt.subplots(1, 1)
    filename = f"checkpoints_sgd_thm3/{p}(0.50)-{N}-2"
    Epochs = np.arange(start_epoch + interval, epochs + 1, interval)
    Epochs_mid = Epochs - interval / 2
    def ep_map(i):
        return i * interval + interval / 2 + start_epoch
    Diff = np.zeros((len(Epochs), N))
    
    data = mod_tensors.ModAddDataTrainValid(p=p, select=29)
    m0 = model_network.ModelNetworkCE(p=p, N=N, sigma=model.ReLU(), from_file=filename + f"/model_epoch_{start_epoch}.pt")
    data = model_network.ModAddDataTrainValid(p=p, select=29)
    F = m0.activate(data.X)
    cn = p**2 / 2 * (1/6 - 1/8)
    E = 0.5 * np.linalg.norm(data.Y @ F.T, axis=0)**2 / np.linalg.norm(m0.W, axis=-1)**2 / cn
    # selected = np.flatnonzero(E <= 0.3)
    # final ref
    mf = model_network.ModelNetworkCE(p=p, N=N, sigma=model.ReLU(), from_file=filename + f"/model_epoch_{epochs}.pt")
    Ff = mf.activate(data.X)
    Ef = 0.5 * np.linalg.norm(data.Y @ Ff.T, axis=0)**2 / np.linalg.norm(mf.W, axis=-1)**2 / cn
    selected = np.flatnonzero((E <= 0.3) & (Ef >= 0.6))
    # selected = np.flatnonzero((E <= 0.3))
    for i, ep in enumerate(Epochs):
        m = model_network.ModelNetworkCE(p=p, N=N, sigma=model.ReLU(), from_file=filename + f"/model_epoch_{ep}.pt")
        coss = model.cos_similarity_ax(m0.W, m.W, axis=-1)
        coss[coss > 1] = 1
        Diff[i, :] = np.arcsin(np.sqrt(1 - coss))
        m0 = m
    Diff /= interval * 8
    Diff_Ref = Diff[:, selected]
    # print(np.mean(np.max(Diff, axis=0)))
    y_scale = 1e-4
    max_Diff_Ref = np.max(Diff_Ref, axis=0)
    mean_mdf = np.mean(max_Diff_Ref)
    ax.plot(Epochs_mid, Diff_Ref / y_scale, color=plt.get_cmap("viridis")(0.15), alpha=0.5)
    ax.plot([Epochs[0] - interval, Epochs[-1]], [mean_mdf / y_scale, mean_mdf / y_scale], color='orange', label=rf"$\bar{{v}}={sci_latex(mean_mdf)}$")
    ax.plot(ep_map(np.argmax(Diff_Ref, axis=0)), max_Diff_Ref / y_scale, 's', color="midnightblue", markerfacecolor='none')
    ax.legend(loc=1, frameon=False, fontsize=12)
    ax.set_xlabel("epochs", fontsize=12)
    ax.set_ylabel(r"changing rate $(\times 10^{-4})$", fontsize=12)
    if self_plot:
        plt.show()

def early_late_change_rate():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    early_change_rate(ax1)
    late_change_rate(ax2)
    ax1.set_title("(a) Early stage", fontsize=12)
    ax2.set_title("(b) Late stage", fontsize=12)
    plt.savefig(FIGURE_APDX_SAVE_DIR.joinpath("early_late_change_rate.svg"), bbox_inches="tight")
    plt.savefig(FIGURE_APDX_SAVE_DIR.joinpath("early_late_change_rate.pdf"), bbox_inches="tight")
    plt.show()

def early_late_lr():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"wspace": 0.3})
    early_lr(ax1)
    late_lr(ax2)
    ax1.set_title("(a) Early stage", fontsize=12)
    ax2.set_title("(b) Late stage", fontsize=12)
    plt.savefig(FIGURE_APDX_SAVE_DIR.joinpath("early_late_lr.svg"), bbox_inches="tight")
    plt.savefig(FIGURE_APDX_SAVE_DIR.joinpath("early_late_lr.pdf"), bbox_inches="tight")
    plt.show()

def kernel_acc_compare(ax=None, p=59, N=512, epochs=30001, interval=50, hide_y=False, show_label=False, title=""):
    self_plot = (ax is None)
    if self_plot:
        fig, ax = plt.subplots(1, 1)
    filename = f"checkpoints_sgd_thm3/aligns/{p}({p//2})-{N}-2"
    data = mod_tensors.ModAddDataTrainValid(p, p//2)
    d = np.loadtxt(ENV_DIR.joinpath(filename + "/loss_acc.txt"))
    k = np.loadtxt(ENV_DIR.joinpath(filename + "/kernel_accs_S.txt"))
    Epochs = np.arange(0, epochs, interval)
    Epochs0 = np.insert(Epochs + 1, 0, 0)
    k = np.insert(k, 0, 1 / p, axis=0)
    # color1 = "#201E8F"
    # color2 = "#DB2E28"
    color1 = "C0"
    color2 = "C1"
    select = slice(0, epochs // interval + 2)
    ax.plot(Epochs0, d[select, 2], color=color1, label="train (model)" if show_label else None)
    ax.plot(Epochs0, k[select, 1], '--', color=color1, label="train (kernel)" if show_label else None)
    ax.plot(Epochs0, k[select, 3], '-.', color=color1, label="train ($SK_{tt}$)" if show_label else None)
    ax.plot(Epochs0, d[select, 3], color=color2, label="test (model)" if show_label else None)
    ax.plot(Epochs0, k[select, 2], '--', color=color2, label="test (kernel)" if show_label else None)
    ax.plot(Epochs0, k[select, 4], '-.', color=color2, label="test ($SK_{tv}$)" if show_label else None)
    ax.set_title(title)
    ax.set_xlabel("epochs", fontsize=12)
    ax.set_xticks([0, 10000, 20000, 30000] if epochs > 30000 else [0, 2500, 5000, 7500, 10000])
    if show_label:
        ax.legend(loc=4, frameon=False, fontsize=12)
    if hide_y:
        ax.tick_params(axis="y", labelleft=False)
    else:
        ax.set_ylabel("accuracy", fontsize=12)
    if self_plot:
        plt.show()

def kernel_acc_compare_plot():
    fig, axes = plt.subplots(2, 2, figsize=(9, 9), gridspec_kw={"wspace": 0.03, "hspace": 0.25})
    kernel_acc_compare(axes[0, 0], N=512, epochs=10001, title="(a) $N=512$ in 10000 epochs")
    kernel_acc_compare(axes[0, 1], N=2048, epochs=10001, title="(b) $N=2048$ in 10000 epochs", hide_y=True)
    kernel_acc_compare(axes[1, 0], N=512, title="(c) $N=512$ in 30000 epochs")
    kernel_acc_compare(axes[1, 1], N=2048, title="(d) $N=2048$ in 30000 epochs", hide_y=True, show_label=True)
    plt.savefig(FIGURE_APDX_SAVE_DIR.joinpath("kernel_acc_compare.svg"), bbox_inches="tight")
    plt.savefig(FIGURE_APDX_SAVE_DIR.joinpath("kernel_acc_compare.pdf"), bbox_inches="tight")
    plt.show()

def balance_random_split_comparison_acc_loss():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))
    pos = ax2.get_position()
    ax2.set_position([pos.x0 - 0.01, pos.y0, pos.width, pos.height])
    epochs=10001
    interval=50
    Epochs = np.arange(0, epochs, interval)
    Epochs0 = np.insert(Epochs + 1, 0, 0)
    d_balc = np.loadtxt(ENV_DIR.joinpath("checkpoints_sgd_thm3/aligns/59(29)-512-2/loss_acc.txt"), delimiter=' ')[:epochs // interval + 2]
    d_rand = np.loadtxt(ENV_DIR.joinpath("checkpoints_sgd_thm3/random/59(0.49)-512-2/loss_acc.txt"), delimiter=' ')[:epochs // interval + 2]
    color1 = "C0"
    color2 = "C1"
    ax1.plot(Epochs0, d_balc[:, 2],       color=color1) # train acc
    ax1.plot(Epochs0, d_rand[:, 2], '--', color=color1)
    ax1.plot(Epochs0, d_balc[:, 3],       color=color2) # test acc
    ax1.plot(Epochs0, d_rand[:, 3], '--', color=color2)
    ax2.plot(Epochs0, d_balc[:, 0],       color=color1, label="train (balance)") # train loss
    ax2.plot(Epochs0, d_rand[:, 0], '--', color=color1, label="train (random)")
    ax2.plot(Epochs0, d_balc[:, 1],       color=color2, label="test (balance)") # test loss
    ax2.plot(Epochs0, d_rand[:, 1], '--', color=color2, label="test (random)")

    s_balc = np.loadtxt(ENV_DIR.joinpath("checkpoints_sgd_thm3/aligns/59(29)-512-2/cos_sim.txt"), delimiter=' ')
    s_rand = np.loadtxt(ENV_DIR.joinpath("checkpoints_sgd_thm3/random/59(0.49)-512-2/cos_sim.txt"), delimiter=' ')
    ax3.plot(Epochs0[1:], s_balc[:epochs // interval + 1, 1], color=color1, label="balance")
    ax3.plot(Epochs0[1:], s_rand[:, 1], '--', color=color1, label="random")

    ax2.legend(loc=1, frameon=False, fontsize=12)
    ax3.legend(loc=4, frameon=False, fontsize=12)

    # ax1.set_xticks([0, 10000, 20000, 30000])
    # ax2.set_xticks([0, 10000, 20000, 30000])
    ax1.set_xticks([0, 2500, 5000, 7500, 10000])
    ax2.set_xticks([0, 2500, 5000, 7500, 10000])
    ax3.set_xticks([0, 2500, 5000, 7500, 10000])
    ax3.set_ylim(0.916, 1.004)
    ax3.set_yticks([0.92, 0.94, 0.96, 0.98, 1.0])

    ax1.set_title("(a)", fontsize=12)
    ax2.set_title("(b)", fontsize=12)
    ax3.set_title("(c)", fontsize=12)

    ax1.set_xlabel("epochs", fontsize=12)
    ax2.set_xlabel("epochs", fontsize=12)
    ax3.set_xlabel("epochs", fontsize=12)
    ax1.set_ylabel("accuracy", fontsize=12)
    ax2.set_ylabel("loss", fontsize=12)
    ax3.set_ylabel("cosine similarity", fontsize=12)
    plt.savefig(FIGURE_APDX_SAVE_DIR.joinpath("balance_random_split_comparison_acc_loss.svg"), bbox_inches="tight")
    plt.savefig(FIGURE_APDX_SAVE_DIR.joinpath("balance_random_split_comparison_acc_loss.pdf"), bbox_inches="tight")
    plt.show()

def random_split_loss_dis(ax=None, title_label="", p=59, N=512):
    self_plot = (ax is None)
    if self_plot:
        fig, ax = plt.subplots(1, 1)
    data = mod_tensors.ModAddDataRandom(p=p, fraction=0.49)
    hs = np.array(data.get_as_pairs())[data.train_selection, 2]
    filename = f"checkpoints_sgd_thm3/random/{p}(0.49)-{N}-2/model_epoch_{30000}.pt"
    m = model_network.ModelNetworkCE(p=p, N=N, sigma=model.ReLU(), from_file=filename)
    loss = m.forward_loss_foreach(data.X_train, data.Y_train)
    counts = dict.fromkeys(np.arange(0, p), 0)
    for h in hs:
        counts[h] += 1
    hs_c = [counts[h] for h in hs]
    ax.plot(hs_c, loss, 's', alpha=0.5)

    ax.set_xticks(np.arange(18, 38.1, 4))
    ax.set_xlabel("group size", fontsize=12)
    ax.set_ylabel("loss on each sample", fontsize=12)
    ax.set_title(title_label, fontsize=12)

    if self_plot:
        plt.show()

def random_balance_loss_std(ax=None, title_label="", p=59):
    self_plot = (ax is None)
    if self_plot:
        fig, ax = plt.subplots(1, 1)
    data_rand = mod_tensors.ModAddDataRandom(p=p, fraction=0.49)
    data_balc = mod_tensors.ModAddDataTrainValid(p=p, select=29)
    groups = data_rand.group_selection_train()
    N_lists = [256, 512, 1024, 2048]
    loss_std_rand = []
    loss_std_rddm = []
    loss_std_balc = []
    for N in N_lists:
        filename = f"checkpoints_sgd_thm3/random/59(0.49)-{N}-2/model_epoch_{30000}.pt"
        m = model_network.ModelNetworkCE(p=p, N=N, sigma=model.ReLU(), from_file=filename)
    
        # sizes = []
        # means = []
        loss_dm = []
        loss_rd = []
        for key, select in groups.items():
            X = data_rand.X[:, select]
            Y = data_rand.Y[:, select]
        
            loss = m.forward_loss_foreach(X, Y)
            # sizes.append(len(select))
            # means.append(np.mean(loss))
            loss_dm.append(loss - np.mean(loss))
            loss_rd.append(loss)
        # sizes = np.array(sizes)
        # means = np.array(means)
        loss_std_rddm.append(np.std(np.concat(loss_dm)))
        loss_std_rand.append(np.std(np.concat(loss_rd)))

        filename = f"checkpoints_sgd_thm3/59(0.50)-{N}-2/model_epoch_{30000}.pt"
        m = model_network.ModelNetworkCE(p=p, N=N, sigma=model.ReLU(), from_file=filename)
        loss = m.forward_loss_foreach(data_balc.X_train, data_balc.Y_train)
        loss_std_balc.append(np.std(loss))
    loss_std_balc = np.array(loss_std_balc)
    loss_std_rand = np.array(loss_std_rand)
    loss_std_rddm = np.array(loss_std_rddm)

    y_scale = 1e-3
    ax.plot(N_lists, loss_std_balc / y_scale, 'o--', color='C0', label="balance")
    ax.plot(N_lists, loss_std_rand / y_scale, 's-',  color='C1', label="random (total)")
    ax.plot(N_lists, loss_std_rddm / y_scale, 's-',  color='C0', label="random (intra)")

    ax.legend(loc=1, frameon=False, fontsize=12)
    ax.set_xticks([256, 512, 1024, 2048])

    ax.set_title(title_label, fontsize=12)
    ax.set_xlabel("$N$", fontsize=12)
    ax.set_ylabel(r"standard deviation $(\times10^{-3})$", fontsize=12)
    
    if self_plot:
        plt.show()

def random_split_loss_analysis():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    random_split_loss_dis(ax1, "(a)")
    random_balance_loss_std(ax2, "(b)")
    plt.savefig(FIGURE_APDX_SAVE_DIR.joinpath("random_split_loss_analysis.svg"), bbox_inches="tight")
    plt.savefig(FIGURE_APDX_SAVE_DIR.joinpath("random_split_loss_analysis.pdf"), bbox_inches="tight")
    plt.show()

def early_late_feature_ratio():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    cmap = plt.get_cmap('Reds')
    Ns = [256, 512, 1024, 2048]
    cmap_2 = plt.get_cmap('plasma')
    colors = cmap(np.linspace(0.3, 0.8, len(Ns), endpoint=True)[::-1]) * 0.7 + cmap_2(np.linspace(0, 0.8, len(Ns), endpoint=True)) * 0.3
    for i, N in enumerate(Ns):
        filename_e = f"checkpoints_sgd_thm3/early/59(29)-{N}-2/feature_num_E.txt"
        filename_l = f"checkpoints_sgd_thm3/aligns/59(29)-{N}-2/feature_num_E.txt"
        d_e = np.loadtxt(ENV_DIR.joinpath(filename_e), delimiter=' ')
        d_l = np.loadtxt(ENV_DIR.joinpath(filename_l), delimiter=' ')
        ax1.plot(d_e[:, 0], d_e[:, 1] / N, color=colors[i], zorder=10-i)
        ax2.plot(d_l[:, 0], d_l[:, 1] / N, color=colors[i], zorder=10-i, label=f"$N={N}$")
    ax2.legend(loc=4, frameon=False, fontsize=12)
    ax2.set_ylim(-0.05, 1.05)
    
    ax2.set_xticks([0, 10000, 20000, 30000])
    ax1.set_xlabel("epochs", fontsize=12)
    ax2.set_xlabel("epochs", fontsize=12)
    ax1.set_ylabel("Feature ratio", fontsize=12)
    ax2.set_ylabel("Feature ratio", fontsize=12)
    ax1.set_title("(a)", fontsize=12)
    ax2.set_title("(b)", fontsize=12)

    plt.savefig(FIGURE_APDX_SAVE_DIR.joinpath("early_late_feature_ratio.svg"), bbox_inches="tight")
    plt.savefig(FIGURE_APDX_SAVE_DIR.joinpath("early_late_feature_ratio.pdf"), bbox_inches="tight")
    plt.show()

def get_gernel_gaps(p=113, q=56, Ns = np.array([256, 512, 1024, 2048, 4096])):
    atvs = []
    N_effs = []
    N_eff_fs = []
    accit_size = 10
    seeds = list(range(10))
    Ns = np.array([256, 512, 1024, 2048, 4096])
    # seeds mean
    for N in Ns:
        gap_sum = np.zeros(accit_size)
        n_eff_sum = np.zeros(accit_size)
        n_eff_f_sum = np.zeros(accit_size)
        for seed in seeds:
            data = np.loadtxt(ENV_DIR / f"checkpoints_sgd_thm3/gap_seed/{p}({q})-{N}-2/{seed}/kernel_gaps.txt", delimiter=" ", skiprows=1)
            gap_sum += data[:, 2] / data[:, 3]
            n_eff_sum += data[:, 4]
            n_eff_f_sum += data[:, 5]
        atvs.append(gap_sum / len(seeds))
        N_effs.append(n_eff_sum / len(seeds))
        N_eff_fs.append(n_eff_f_sum / len(seeds))
    atvs = np.array(atvs)
    N_effs = np.array(N_effs)
    N_eff_fs = np.array(N_eff_fs)
    return Ns, N_effs, N_eff_fs, atvs

def kernel_atv0_width(ax=None, p=113, q=56):
    self_plot = (ax is None)
    if self_plot:
        fig, ax = plt.subplots(1, 1)
    Ns, N_effs, N_eff_fs, gaps = get_gernel_gaps(p, q)

    a2 = 21.1
    ax.plot(Ns, gaps[:, 0], 's-')
    ax.plot([Ns.min(), Ns.max()], [a2, a2], '-', label=f"$a_2={a2:.1f}$")
    # ax.legend(loc=1, frameon=False, fontsize=12)

    # plt.plot(Ns, gaps[:, accit], 's-')
    # plt.plot(N_effs, gaps, marker='o')
    ax.set_xlabel(r"$N$", fontsize=12)
    ax.set_xticks(Ns, [256, "", 1024, 2048, 4096])
    ax.set_ylabel(r"$a_{tv}(t_0)$", fontsize=12)
    ax.set_title("(b)", fontsize=12)
    if self_plot:
        plt.show()

def kernel_collision_Ctv_dis(ax=None, p=113, q=56, N=2048, accit=0, seed=0):
    self_plot = (ax is None)
    if self_plot:
        fig, ax = plt.subplots(1, 1)
    ax2 = ax.twinx()
    N_eff, N_eff_f = np.loadtxt(ENV_DIR / f"checkpoints_sgd_thm3/gap_seed/{p}({q})-{N}-2/{seed}/kernel_gaps.txt", delimiter=" ", skiprows=1)[accit, 4:6]
    m = model_network.ModelNetworkCE(p=p, N=N, sigma=model.ReLU(), from_model=None, from_file=f"checkpoints_sgd_thm3/gap_seed/{p}({q})-{N}-2/{seed}/model_accit_{accit}.pt")
    data = mod_tensors.ModAddDataTrainValid(p, select=q, rand_seed=seed)
    F_t = m.sigma(m.W @ data.X_train)
    F_v = m.sigma(m.W @ data.X_valid)
    K_tv = F_t.T @ F_v
    K_tv /= (np.linalg.norm(m.W)**2 / p)
    C_tv = data.X_train.T @ data.X_valid
    V_tv = K_tv * C_tv
    Coll = {0:[], 1:[], 2:[]}
    Coll_m = {}
    Coll_s = {}
    vf = []
    def take_column(mat, h, j):
        return mat[h * q:(h + 1) * q, j]
    from itertools import product
    for i, j in product(range(p), range(p * q)):
        if i == j // q:
            vf.append(np.sum(take_column(K_tv, i, j)))
            continue
        c = round(np.sum(take_column(C_tv, i, j)))
        v = np.sum(take_column(V_tv, i, j))
        # k = (np.sum(take_column(K_tv, i, j)) - v) / (q - c) * q
        k = (np.sum(take_column(K_tv, i, j)))
        Coll[c].append((v, k))
    # v1 = np.mean(np.array(Coll[1]), axis=0)
    for c, vk in Coll.items():
        vk = np.array(vk)
        sigma = np.std(vk[:, 1])
        Coll_s[c] = sigma
        # ax.hist(vk[:, 1], bins=100, alpha=0.5, label=f"collision {c} v sigma = {sigma:.4f}", color=f"C{c}") # k
        ax.hist(vk[:, 1], bins=100, alpha=0.5, label=f"collision {c}", color=f"C{c}") # k
        v1_m = np.mean(vk[:, 1])
        Coll_m[c] = v1_m
        ax.plot([v1_m, v1_m], [0, 1.5 * p * q], color=f"C{c}", linestyle="-", label=f"collision {c} mean")
        # plt.plot([v1_m + sigma, v1_m + sigma], [0, 0.5 *p * q], color=f"C{c}", linestyle="--")
        # plt.plot([v1_m - sigma, v1_m - sigma], [0, 0.5 *p * q], color=f"C{c}", linestyle="--")
        # print(sigma * np.sqrt(N), sigma * np.sqrt(N_eff_f), sigma * np.sqrt(N_eff))

    sigma = np.std(vf)
    # ax2.hist(vf, bins=100, alpha=0.5, label=f"feature sigma = {sigma:.4f}", color=f"C3")
    ax2.hist(vf, bins=100, alpha=0.5, label=f"feature", color=f"C3")
    ax2.set_ylim(0, 1100)
    ax2.tick_params(axis='y', colors='C3')
    ax2.spines['right'].set_color('C3')
    vf_m = np.mean(vf)
    ax.plot([vf_m, vf_m], [0, 1.5 * p * q], color=f"C3", linestyle="-", label=f"feature mean")
    # ax.plot([vf_m + sigma, vf_m + sigma], [0, 0.5 *p * q], color=f"C3", linestyle="--")
    # ax.plot([vf_m - sigma, vf_m - sigma], [0, 0.5 *p * q], color=f"C3", linestyle="--")
    # print(sigma * np.sqrt(N), sigma * np.sqrt(N_eff_f), sigma * np.sqrt(N_eff))
    # print(Coll_m[0], Coll_m[1], Coll_m[2], vf_m)

    ax.legend(*(lambda a, b: (a[0][:-1] + b[0] +a[0][-1:], a[1][:-1] + b[1] + a[1][-1:]))(ax.get_legend_handles_labels(), ax2.get_legend_handles_labels()),
              loc=1, frameon=False)
    ax.set_xlabel(r"$\tilde{R}_g(\nu)$", fontsize=12)
    ax.set_ylabel("distribution", fontsize=12)
    ax.set_yticklabels([])
    ax2.set_yticklabels([])
    # rho_str = str(accit * 10) + "\%" if accit !=0 else "1/p"
    ax.set_title("(a)", fontsize=12)
    if self_plot:
        plt.show()

def onset_grokking():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    kernel_collision_Ctv_dis(ax1)
    kernel_atv0_width(ax2)
    plt.savefig(FIGURE_APDX_SAVE_DIR.joinpath("onset_grokking.svg"), bbox_inches="tight")
    plt.savefig(FIGURE_APDX_SAVE_DIR.joinpath("onset_grokking.pdf"), bbox_inches="tight")
    plt.show()

def gap_width_scaling(ax=None, title_label="", p=59, q=29, Ns = [256, 512, 1024, 2048, 4096], yticks=[]):
    self_plot = (ax is None)
    if self_plot:
        fig, ax = plt.subplots(1, 1)
    # color_0 = ["#1134a5", "#86164C"]
    # color_2 = ["#485a96", "#A42D67"]
    color_0 = ["C0", "C1"]
    color_2 = ["C0", "C1"]

    Acc = [5, 9]
    Ns, N_effs, N_eff_fs, atvs = get_gernel_gaps(p, q)

    # log linear fit
    slopes = []
    for i, acc in enumerate(Acc):
        gaps = atvs[:, acc] - atvs[:, 0]
        slope, intercept = np.polyfit(np.log(N_eff_fs[:, acc]), np.log(gaps), 1)
        slopes.append(slope)
        print(slope)

        ax.loglog(N_eff_fs[:, acc], gaps, 's', color=color_0[i])
        # rf"$\log\Delta_\rho={slope:.2f}\log N_{{\rm eff}}^F+{intercept:.2f}$"
        ax.loglog(N_eff_fs[:, acc], np.exp(slope * np.log(N_eff_fs[:, acc]) + intercept), '-', label=rf"$\gamma={-slope:.2f}, \rho={acc * 0.1:.1f}$", color=color_2[i])
    ax.legend(loc=1, frameon=False, fontsize=12)

    # ax.set_xticks([120, 200, 300, 400, 500, 600], [120, 200, 300, 400, 500, 600])
    # ax.set_yticks(np.arange(6, 12.1), np.arange(6, 13, dtype=int))
    ax.set_yticks(yticks, yticks)
    ax.set_xlabel(r"$N_{\rm eff}^F$", fontsize=12)
    ax.set_ylabel(r"$\Delta_\rho$", fontsize=12)
    ax.set_title(title_label + f"$p={p},q={q}$")
    # ax.set_title(f'Gap Scaling vs N for p={p}, q={q}')
    if self_plot:
        plt.show()

def gap_width_scaling_for_different_test():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    gap_width_scaling(ax1, p=59,  q=29, title_label="(a) ", yticks=[6, 10, 20])
    gap_width_scaling(ax2, p=113, q=34, title_label="(b) ", yticks=[12, 20, 30])
    gap_width_scaling(ax3, p=113, q=56, title_label="(c) ", yticks=[20, 30, 40, 60])
    plt.savefig(FIGURE_APDX_SAVE_DIR.joinpath("gap_width_scaling_for_different_test.svg"), bbox_inches="tight")
    plt.savefig(FIGURE_APDX_SAVE_DIR.joinpath("gap_width_scaling_for_different_test.pdf"), bbox_inches="tight")
    plt.show()

def kernel_gh_corr(ax=None, p=113, q=56, N=2048, accit=0, seed=0):
    self_plot = (ax is None)
    if self_plot:
        fig, ax = plt.subplots(1, 1)
    # N_eff, N_eff_f = np.loadtxt(ENV_DIR / f"checkpoints_sgd_thm3/gap_seed/{p}({q})-{N}-2/{seed}/kernel_gaps.txt", delimiter=" ", skiprows=1)[accit, 4:6]
    m = model_network.ModelNetworkCE(p=p, N=N, sigma=model.ReLU(), from_model=None, from_file=f"checkpoints_sgd_thm3/gap_seed/{p}({q})-{N}-2/{seed}/model_accit_{accit}.pt")
    data = mod_tensors.ModAddDataTrainValid(p, select=q, rand_seed=seed)
    F_t = m.activate(data.X_train)
    F_v = m.activate(data.X_valid)
    K_tv = F_t.T @ F_v
    K_tv /= (np.linalg.norm(m.W)**2 / p)
    # C_tv = data.X_train.T @ data.X_valid
    R_g = np.zeros(p * q)
    R_h = np.zeros(p * q)
    def take_column(mat, h, j):
        return mat[h * q:(h + 1) * q, j]
    
    for j in range(p * q):
        total_column_sum = np.sum(K_tv[:, j])
        R_h_column_sum = np.sum(take_column(K_tv, j // q, j))
        R_h[j] = R_h_column_sum
        R_g[j] = (total_column_sum - R_h_column_sum) / (p - 1)
    
    ax.plot(R_h, R_g, 's', alpha=0.5)
    ax.set_xlabel(r"$\tilde{R}_h(\nu)$", fontsize=12)
    ax.set_ylabel(r"average $\tilde{R}_g(\nu),g\neq h$", fontsize=12)
    ax.set_title("(b)", fontsize=12)

    if self_plot:
        plt.show()

def kernel_dis_and_corr():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"wspace": 0.25})
    kernel_collision_Ctv_dis(ax1, accit=5)
    kernel_gh_corr(ax2, accit=5)
    plt.savefig(FIGURE_APDX_SAVE_DIR.joinpath("kernel_dis_and_corr.svg"), bbox_inches="tight")
    plt.savefig(FIGURE_APDX_SAVE_DIR.joinpath("kernel_dis_and_corr.pdf"), bbox_inches="tight")
    plt.show()

def N_eff_with_time(ax=None, p=59, q=29, Ns = [512, 1024, 2048, 4096]):
    self_plot = (ax is None)
    if self_plot:
        fig, ax = plt.subplots(1, 1)
    Ns = np.array(Ns)
    cmap = plt.get_cmap('inferno')
    colors = [cmap(i) for i in np.linspace(0.1, 0.4, 5, endpoint=True)]
    line_handles = {}
    # epochs alignment W_norm_sqr alignment_tt alignment_tv std effective_width effective_feature_width
    for i, N in enumerate(Ns):
        d = np.loadtxt(ENV_DIR / f"checkpoints_sgd_thm3/aligns/{p}({q})-{N}-2/align_init.txt", delimiter=" ", skiprows=1)
        line_handles[f"{i}w"], = ax.plot(d[:, 0] + 1, d[:, 6] / N, color=colors[i], zorder=10-i, alpha=0.4, linestyle='-')
        line_handles[f"{i}f"], = ax.plot(d[:, 0] + 1, d[:, 7] / N, color=colors[i], zorder=10-i, label=rf"$N={N}$")
    
    lg_h, lg_l = ax.get_legend_handles_labels()
    ax.legend([(line_handles["0f"], line_handles["0w"])] + lg_h, [r"$N_{\rm eff}^F(N_{\rm eff})$"] + lg_l,
                handler_map={tuple: HandlerTuple(ndivide=None)},
        loc=1, frameon=False, fontsize=12)

    ax.set_xticks([0, 10000, 20000, 30000])
    ax.set_xlabel("epochs", fontsize=12)
    ax.set_ylabel(r"$N_{\rm eff}^F/N$ and $N_{\rm eff}/N$", fontsize=12)
        
    if self_plot:
        plt.show()

def N_eff_with_N(ax=None, p=59, q=29, Ns = [512, 1024, 2048, 4096]):
    self_plot = (ax is None)
    if self_plot:
        fig, ax = plt.subplots(1, 1)
    Ns = np.array(Ns)
    pairs = [(59, 29), (113, 34), (113, 56)]
    cmap = plt.get_cmap('magma')
    # colors = [cmap(i) for i in np.linspace(0.1, 0.4, 5, endpoint=True)]
    N_eff_fs = []
    N_eff_fs_0 = []
    N_eff_fs_5 = []
    N_eff_fs_9 = []
    for i, N in enumerate(Ns):
        # epochs alignment W_norm_sqr alignment_tt alignment_tv std effective_width effective_feature_width
        d = np.loadtxt(ENV_DIR / f"checkpoints_sgd_thm3/aligns/{p}({q})-{N}-2/align_init.txt", delimiter=" ", skiprows=1)
        N_eff_fs.append(d[-1, 7])
        # # epochs base_acc total_gap W_norm_sqr N_eff N_eff_f
        k = np.loadtxt(ENV_DIR / f"checkpoints_sgd_thm3/aligns/{p}({q})-{N}-2/kernel_gaps.txt", delimiter=" ", skiprows=1)
        N_eff_fs_0.append(k[0, 5])
        N_eff_fs_5.append(k[5, 5])
        N_eff_fs_9.append(k[9, 5])

    ax.plot(Ns, N_eff_fs,   's-', color=cmap(0.8),  label="final")
    ax.plot(Ns, N_eff_fs_0, 's-', color=cmap(0.15), label=r"$\rho=1/p$")
    ax.plot(Ns, N_eff_fs_5, 's-', color=cmap(0.3),  label=r"$\rho=0.5$")
    ax.plot(Ns, N_eff_fs_9, 's-', color=cmap(0.45), label=r"$\rho=0.9$")
    ax.set_xticks([256, 1024, 2048, 4096])

    ax.legend(loc=4, frameon=False, fontsize=12)
    ax.set_xlabel("$N$", fontsize=12)
    ax.set_ylabel(r"$N_{\rm eff}^F$", fontsize=12)
        
    if self_plot:
        plt.show()

def gap_width_scaling_comp(ax=None, title_label="", p=59, q=29, Ns = [256, 512, 1024, 2048, 4096]):
    self_plot = (ax is None)
    if self_plot:
        fig, ax = plt.subplots(1, 1)
    color_0 = ["C0", "C1"]
    color_2 = ["C0", "C1"]

    acc = 5
    Ns, N_effs, N_eff_fs, atvs = get_gernel_gaps(p, q)

    # log linear fit
    
    gaps = atvs[:, acc] - atvs[:, 0]
    slope, intercept = np.polyfit(np.log(N_eff_fs[:, acc]), np.log(gaps), 1)
    slope_n, intercept_n = np.polyfit(np.log(Ns), np.log(gaps), 1)
    
    ax.loglog(N_eff_fs[:, acc], gaps, 's', color=color_0[0])
    ax.loglog(N_eff_fs[:, acc], np.exp(slope * np.log(N_eff_fs[:, acc]) + intercept), '-', label=rf"$N_{{\rm eff}}^F,\gamma={-slope:.2f}$", color=color_2[0])
    ax.loglog(Ns, gaps, 's', color=color_0[1])
    ax.loglog(Ns, np.exp(slope_n * np.log(Ns) + intercept_n), '-', label=rf"$N,\gamma={-slope_n:.2f}$", color=color_2[1])
    ax.legend(loc=1, frameon=False, fontsize=12)

    ax.set_xticks([100, 500, 1000, 2000, 4000], [100, 500, 1000, 2000, 4000])
    ax.set_yticks([6, 7, 8, 9, 12], [6, 7, 8, 9, 12])
    ax.set_xlabel(r"$N_{\rm eff}^F$ and $N$", fontsize=12)
    ax.set_ylabel(r"$\Delta_\rho$", fontsize=12)
    if self_plot:
        plt.show()

def loss_std_scaling(ax=None, title_label="", p=59, q=29, y_ticks=None):
    self_plot = (ax is None)
    if self_plot:
        fig, ax = plt.subplots(1, 1)
    data = mod_tensors.ModAddDataTrainValid(p=p, select=29)
    N_lists = np.array([256, 512, 1024, 2048, 4096])
    N_effs = []
    loss_std = []
    # epochs alignment W_norm_sqr alignment_tt alignment_tv std effective_width effective_feature_width
    for N in N_lists:
        a = np.loadtxt(ENV_DIR / f"checkpoints_sgd_thm3/aligns/{p}({q})-{N}-2/align.txt", delimiter=' ')
        d = np.loadtxt(ENV_DIR / f"checkpoints_sgd_thm3/aligns/{p}({q})-{N}-2/results.txt", delimiter=',')
        loss_std.append(a[-1, 5] / d[-1, 0])
        N_effs.append(a[-1, 6])
    loss_std = np.array(loss_std)
    N_effs = np.array(N_effs)

    slope, intercept = np.polyfit(np.log(N_effs), np.log(loss_std), 1)
    print(slope)

    ax.loglog(N_effs, loss_std, 's', color='C0')

    ax.loglog(N_effs, N_effs ** slope * np.exp(intercept), '-', color='C0', label=rf"$\gamma={-slope:.2f}$")
    ax.legend(loc=1, frameon=False, fontsize=12)
    # ax.set_xticks(N_lists)

    ax.set_title(title_label + rf"$p={p},q={q}$", fontsize=12)
    ax.set_xlabel(r"$N_{\rm eff}$", fontsize=12)
    ax.set_ylabel(r"$\tilde{\sigma}_{\rm loss}$", fontsize=12)
    if y_ticks is not None:
        ax.set_yticks(y_ticks, y_ticks)
    if N_effs.max() > 600:
        ax.set_xticks([200, 300, 400, 600], [200, 300, 400, 600])
    else:
        ax.set_xticks([200, 300, 400], [200, 300, 400])
    
    if self_plot:
        plt.show()

def loss_std_scaling_for_settings():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4), gridspec_kw={"wspace": 0.25})
    loss_std_scaling(ax1, p=59,  q=29, title_label="(a) ", y_ticks=[0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16])
    loss_std_scaling(ax2, p=113, q=34, title_label="(b) ", y_ticks=[0.14, 0.16, 0.18, 0.2, 0.22, 0.24])
    loss_std_scaling(ax3, p=113, q=56, title_label="(c) ", y_ticks=[0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26])
    plt.savefig(FIGURE_APDX_SAVE_DIR.joinpath("loss_std_scaling_for_settings.svg"), bbox_inches="tight")
    plt.savefig(FIGURE_APDX_SAVE_DIR.joinpath("loss_std_scaling_for_settings.pdf"), bbox_inches="tight")
    plt.show()

def acc_multilayer_34():
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8), gridspec_kw={"wspace": 0.05, "hspace": 0.05})
    p, q, N = 59, 29, 512
    Epochs = np.arange(0, 30001, 50)
    ep_s = slice(0, 10000 // 50 + 2)
    Epochs0 = np.insert(Epochs + 1, 0, 0)
    d3 = np.loadtxt(ENV_DIR.joinpath(f"checkpoints_sgd_thm3/multilayer/{p}({q})-{N}-3/loss_acc.txt"))
    d4 = np.loadtxt(ENV_DIR.joinpath(f"checkpoints_sgd_thm3/multilayer/{p}({q})-{N}-4/loss_acc.txt"))
    for ax, d in zip((ax1, ax2), (d3, d4)):
        ax.plot(Epochs0[ep_s], d[ep_s, 2], label="train")
        ax.plot(Epochs0[ep_s], d[ep_s, 3], label="test")
        ax.plot([-250, 1500], [1/p, 1/p], '--', color="gray", alpha=0.5)
        ax.set_xlim(-400, 10400)
        ax.set_xticklabels([])
    ax1.set_title("3 layers")
    ax2.set_title("4 layers")
    ax1.set_ylabel("accuracy", fontsize=12)
    ax2.set_yticklabels([])
    for ax, d in zip((ax3, ax4), (d3, d4)):
        ax.plot(Epochs0[ep_s], d[ep_s, 0])
        ax.plot(Epochs0[ep_s], d[ep_s, 1])
        ax.plot([-250, 3000], [np.log(p)] * 2, '--', color="gray", alpha=0.5)
        ax.set_xlim(-400, 10400)
        ax.set_ylim(-0.5, 10.2)
        ax.set_xlabel("epochs", fontsize=12)
    ax3.set_ylabel("loss", fontsize=12)
    ax4.set_yticklabels([])
    ax1.legend(loc=4, frameon=False, fontsize=12)
    plt.savefig(FIGURE_APDX_SAVE_DIR.joinpath("acc_multilayer_34.svg"), bbox_inches="tight")
    plt.savefig(FIGURE_APDX_SAVE_DIR.joinpath("acc_multilayer_34.pdf"), bbox_inches="tight")
    plt.show()

def other_acc(ax, dirname):
    size = 59 - 1
    self_plot = (ax is None)
    if self_plot:
        fig, ax = plt.subplots(1, 1)
    d = np.loadtxt(ENV_DIR.joinpath(dirname + "/loss_acc.txt"))
    ep_s = 10000 // 50 + 2
    Epochs = np.arange(0, 30001, 50)
    Epochs0 = np.insert(Epochs + 1, 0, 0)
    ax.plot(Epochs0[:ep_s], d[:ep_s,2], label="train")
    ax.plot(Epochs0[:ep_s], d[:ep_s,3], label="test")
    ax.plot([-250, 1500], [1/size, 1/size], '--', color="gray", alpha=0.5)
    ax.set_xlim(-400, 10400)
    ax.legend(loc=4, frameon=False, fontsize=12)
    ax.set_xlabel("epochs", fontsize=12)
    ax.set_ylabel("accuracy", fontsize=12)
    if self_plot:
        plt.show()

def other_cos_sim(ax, dirname):
    self_plot = (ax is None)
    if self_plot:
        fig, ax = plt.subplots(1, 1)
    d = np.loadtxt(ENV_DIR.joinpath(dirname + "/cos_sim.txt"))
    Epochs0 = d[:, 0]
    ax.plot(Epochs0, d[:, 1], label=r"$S, \tilde{Y}$", zorder=2)
    legend_y1, = ax.plot(Epochs0, d[:, 2], label=r"$V, \tilde{{Y}}F^T$", zorder=1, color="C1")
    legend_s1, = ax.plot(Epochs0, d[:, 3], label=r"$V, SF^T$", zorder=1, color="C2")
    
    ax.legend(loc=4, frameon=False, fontsize=12)
    ax.set_xlabel("epochs", fontsize=12)
    ax.set_xticks([0, 10000, 20000, 30000])
    ax.set_ylim(0.89, 1.01)
    ax.set_ylabel("cosine similarity", fontsize=12)
    # plt.savefig(RAW_FIGURE_SAVE_DIR.joinpath(f"dyn_cos_similarity_of_ansatz_with_time.svg"), bbox_inches="tight")
    if self_plot:
        plt.show()

def other_align(ax, dirname):
    self_plot = (ax is None)
    if self_plot:
        fig, ax = plt.subplots(1, 1)
    d = np.loadtxt(ENV_DIR.joinpath(dirname + "/align_init.txt"))
    a = np.loadtxt(ENV_DIR.joinpath(dirname + "/kernel_gaps.txt"))
    Epochs0 = d[:, 0]
    ep_s = 10000 // 50 + 2
    # epochs alignment W_norm_sqr alignment_tt alignment_tv std effective_width effective_feature_width
    ax.plot(Epochs0[:ep_s], d[:ep_s, 4] / d[:ep_s, 2], color="C3")
    for accit, m, l in zip([0, 5, 9], ['s', 'o', 'D'], [r"$\rho=1/p$", r"$\rho=0.5$", r"$\rho=0.9$"]):
        ax.plot(a[accit, 0] + 1, a[accit, 2] / a[accit, 3], m, color="C3", label=l)
    ax.legend(loc=4, frameon=False, fontsize=12)
    ax.set_xlabel("epochs", fontsize=12)
    ax.set_ylabel(r"$a_{tv}$", fontsize=12)
    if self_plot:
        plt.show()

def other_plot(axes, dirname):
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    other_acc(axes[0], dirname)
    other_cos_sim(axes[1], dirname)
    # ax2.set_ylim(0.85, 1.01)
    other_align(axes[2], dirname)
    # ax1.set_title("(a)", fontsize=12)
    # ax2.set_title("(b)", fontsize=12)
    # ax3.set_title("(c)", fontsize=12)
    # plt.savefig(FIGURE_APDX_SAVE_DIR.joinpath(f"other_plot_{label}.svg"), bbox_inches="tight")
    # plt.savefig(FIGURE_APDX_SAVE_DIR.joinpath(f"other_plot_{label}.pdf"), bbox_inches="tight")
    # plt.show()

def other_plots_2():
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    other_plot(axes[0], "checkpoints_sgd_thm3/tasks/mul/59(29)-512-2")
    other_plot(axes[1], "checkpoints_sgd_thm3/tasks/sqr_sum/59(20)-512-2")
    axes[1, 1].set_ylim(0.85, 1.01)
    ls = [["a", "b", "c"], ["d", "e", "f"]]
    for a, al in zip(axes, ls):
        for b, bl in zip(a, al):
            b.set_title(f"({bl})", fontsize=12)
    plt.savefig(FIGURE_APDX_SAVE_DIR.joinpath(f"other_plots_2.svg"), bbox_inches="tight")
    plt.savefig(FIGURE_APDX_SAVE_DIR.joinpath(f"other_plots_2.pdf"), bbox_inches="tight")
    plt.show()

def other_plot_chirality():
    # change to the correct dir
    filename_1 = "/chirality/runs/xxx/results.csv"
    filename_2 = "/chirality_2/runs/xxx/results.csv"
    d_1 = np.loadtxt(CODE_DIR.joinpath(filename_1), skiprows=1, delimiter=',')
    d_2 = np.loadtxt(CODE_DIR.joinpath(filename_2), skiprows=1, delimiter=',')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 9), gridspec_kw={"wspace": 0.05, "hspace": 0.05})
    Epochs0 = d_1[:, 0] + 1
    
    for ax, d in zip((ax1, ax2), (d_1, d_2)):
        ax.plot(Epochs0, d[:, 3], label="train")
        ax.plot(Epochs0, d[:, 4], label="test")
        ax.plot([0, 1000], [0.5, 0.5], '--', color="gray", alpha=0.5)
        ax.set_xscale("symlog", linthresh=100)
        ax.set_xticklabels([])
        ax.set_ylim(0.36, 1.03)
    ax1.set_title("Setting 1")
    ax2.set_title("Setting 2")
    ax1.set_ylabel("accuracy", fontsize=12)
    ax2.set_yticklabels([])
    for ax, d in zip((ax3, ax4), (d_1, d_2)):
        ax.plot(Epochs0, d[:, 1])
        ax.plot(Epochs0, d[:, 2])
        ax.set_xscale("symlog", linthresh=100)
        ax.set_xlabel("epochs", fontsize=12)
        ax.set_ylim(-0.03, 1.03)
    ax3.set_ylabel("loss", fontsize=12)
    ax4.set_yticklabels([])
    ax1.legend(loc=4, frameon=False, fontsize=12)
    plt.savefig(FIGURE_APDX_SAVE_DIR.joinpath("other_plot_chirality.svg"), bbox_inches="tight")
    plt.savefig(FIGURE_APDX_SAVE_DIR.joinpath("other_plot_chirality.pdf"), bbox_inches="tight")
    plt.show()
    plt.show()

def add_fourier_init_acc_loss():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    Epochs = np.arange(0, 301, 2)
    Epochs0 = np.insert(Epochs, 0, 0) + 1
    d = np.loadtxt(ENV_DIR / f"checkpoints_sgd_thm3/tasks/add_fourier_init/59(29)-512-2/loss_acc.txt")
    ax1.plot(d[:, 2], label="train")
    ax1.plot(d[:, 3], label="test")
    ax2.plot(d[:, 0])
    ax2.plot(d[:, 1])
    ax1.legend(loc=4, frameon=False, fontsize=12)
    ax1.set_xlabel("epochs", fontsize=12)
    ax2.set_xlabel("epochs", fontsize=12)
    ax1.set_ylabel("accuracy", fontsize=12)
    ax2.set_ylabel("loss", fontsize=12)
    plt.savefig(FIGURE_APDX_SAVE_DIR.joinpath("add_fourier_init_acc_loss.svg"), bbox_inches="tight")
    plt.savefig(FIGURE_APDX_SAVE_DIR.joinpath("add_fourier_init_acc_loss.pdf"), bbox_inches="tight")
    plt.show()

def N_eff_vs_N_t():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    pos = ax1.get_position()
    ax1.set_position([pos.x0 - 0.01, pos.y0, pos.width, pos.height])
    N_eff_with_time(ax1)
    N_eff_with_N(ax2)
    gap_width_scaling_comp(ax3)
    for ax, t in zip([ax1, ax2, ax3], ["(a)", "(b)", "(c)"]):
        ax.set_title(t, fontsize=12)
    plt.savefig(FIGURE_APDX_SAVE_DIR.joinpath("N_eff_vs_N_t.svg"), bbox_inches="tight")
    plt.savefig(FIGURE_APDX_SAVE_DIR.joinpath("N_eff_vs_N_t.pdf"), bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    # final_freq_phase_dis()
    # late_lr()
    # early_change_rate()
    # late_change_rate()
    # early_late_change_rate()
    # early_late_lr()
    # balance_random_split_comparison_acc_loss()
    # kernel_acc_compare_plot()
    # random_split_loss_analysis()
    # onset_grokking()
    # kernel_collision_Ctv_dis(accit=0)
    # loss_std_scaling_for_settings()
    # other_plots_2()
    # gap_width_scaling_for_different_test()
    # other_plot_chirality()
    # add_fourier_init_acc_loss()
    N_eff_vs_N_t()