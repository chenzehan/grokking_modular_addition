from model_network import ModelNetworkCE, ModAddDataTrainValid, ModAddDataRandom
from matplotlib import pyplot as plt
import numpy as np
from model import ReLU

from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent

# 0: epochs 1: base_acc 2: total_gap 3: W_norm_sqr 4: N_eff 5: N_eff_f
def plot_loss_scaling_single(p=59, q=29, Ns = [256, 512, 1024]):
    for s in range(10):
        gaps = [(lambda data: data[0, 2] / data[0, 3])(np.loadtxt(SCRIPT_DIR / f"checkpoints_sgd_thm3/gap_seed/{p}({q})-{N}-2/{s}/kernel_gaps.txt", delimiter=" ", skiprows=1)) for N in Ns]
        plt.plot(Ns, gaps, label=f"seed {s}")
    plt.xlabel("hidden size")
    plt.ylabel("loss scaling")
    plt.legend()
    plt.show()

def kernel_at_acc_0(p=113, q=56, N=1024, seed=0):
    m = ModelNetworkCE(p=p, N=N, sigma=ReLU(), from_model=None, from_file=f"checkpoints_sgd_thm3/gap_seed/{p}({q})-{N}-2/{seed}/model_accit_0.pt")
    data = ModAddDataTrainValid(p, select=q, rand_seed=seed)
    F_t = m.sigma(m.W @ data.X_train)
    F_v = m.sigma(m.W @ data.X_valid)
    K_tv = F_t.T @ F_v
    h, i = 0, 2
    x = K_tv[:, h * q + i]
    x = x.reshape((p, q))
    x = np.sum(x, axis=1)
    plt.plot([0, p], [x[h], x[h]], 'r--')
    plt.bar(range(p), x)
    plt.show()

def kernel_at_acc_0_acc_on_labels(p=113, q=56, N=1024, seed=0):
    m = ModelNetworkCE(p=p, N=N, sigma=ReLU(), from_model=None, from_file=f"checkpoints_sgd_thm3/gap_seed/{p}({q})-{N}-2/{seed}/model_accit_0.pt")
    data = ModAddDataTrainValid(p, select=q, rand_seed=seed)
    P = m.forward(data.X_valid)
    x = np.array([int(np.argmax(a) == np.argmax(b)) for a, b in zip(data.Y_valid.T, P.T)])
    print(x.shape, np.sum(x))
    x = x.reshape((p, q))
    x = np.sum(x, axis=1)
    plt.bar(range(p), x)
    plt.show()

def kernel_collision_Ctv_check(p=113, q=56, N=1024, seed=0):
    m = ModelNetworkCE(p=p, N=N, sigma=ReLU(), from_model=None, from_file=f"checkpoints_sgd_thm3/gap_kernel_acc_seed/{p}({q})-{N}-2/{seed}/model_accit_0.pt")
    data = ModAddDataTrainValid(p, select=q, rand_seed=seed)
    F_t = m.sigma(m.W @ data.X_train)
    F_v = m.sigma(m.W @ data.X_valid)
    K_tv = F_t.T @ F_v
    K_tv /= (np.linalg.norm(m.W)**2 / p)
    C_tv = data.X_train.T @ data.X_valid
    V_tv = K_tv * C_tv
    Coll = {0:[], 1:[], 2:[]}
    def take_column(mat, h, j):
        return mat[h * q:(h + 1) * q, j]
    from itertools import product
    for i, j in product(range(p), range(p * q)):
        if i == j // q:
            continue
        c = round(np.sum(take_column(C_tv, i, j)))
        v = np.sum(take_column(V_tv, i, j))
        k = (np.sum(take_column(K_tv, i, j)) - v) / (q - c) * q
        Coll[c].append((v, k))
    v1 = np.mean(np.array(Coll[1]), axis=0)
    for c, vk in Coll.items():
        vk = np.array(vk)
        plt.scatter(c, np.mean(vk[:, 0]) - v1[0], label=f"collision {c}", color="C0") # v
        plt.scatter(c, np.mean(vk[:, 1]) - v1[1], label=f"collision {c}", color="C1") # k
    plt.show()

def kernel_collision_Ctv_dis(p=113, q=56, N=2048, seed=0):
    accit = 0
    N_eff, N_eff_f = np.loadtxt(SCRIPT_DIR / f"checkpoints_sgd_thm3/gap_seed/{p}({q})-{N}-2/{seed}/kernel_gaps.txt", delimiter=" ", skiprows=1)[accit, 4:6]
    m = ModelNetworkCE(p=p, N=N, sigma=ReLU(), from_model=None, from_file=f"checkpoints_sgd_thm3/gap_seed/{p}({q})-{N}-2/{seed}/model_accit_{accit}.pt")
    data = ModAddDataTrainValid(p, select=q, rand_seed=seed)
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
        # k = (np.sum(take_column(K_tv, i, j)) - v) / (q - c)
        k = (np.sum(take_column(K_tv, i, j)))
        Coll[c].append((v, k))
    v1 = np.mean(np.array(Coll[1]), axis=0)
    for c, vk in Coll.items():
        vk = np.array(vk)
        sigma = np.std(vk[:, 1])
        Coll_s[c] = sigma
        plt.hist(vk[:, 1], bins=100, alpha=0.5, label=f"collision {c} v sigma = {sigma:.4f}", color=f"C{c}") # v
        v1_m = np.mean(vk[:, 1])
        Coll_m[c] = v1_m
        print("v0_m = ", np.mean(vk[:, 0]), " dk_m = ", (v1_m - Coll_m[0]))
        plt.plot([v1_m, v1_m], [0, 1.5 * p * q], color=f"C{c}", linestyle="-", label=f"collision {c} mean v")
        # plt.plot([v1_m + sigma, v1_m + sigma], [0, 0.5 *p * q], color=f"C{c}", linestyle="--")
        # plt.plot([v1_m - sigma, v1_m - sigma], [0, 0.5 *p * q], color=f"C{c}", linestyle="--")
        print(sigma * np.sqrt(N), sigma * np.sqrt(N_eff_f), sigma * np.sqrt(N_eff))

    sigma = np.std(vf)
    plt.hist(vf, bins=100, alpha=0.5, label=f"feature sigma = {sigma:.4f}", color=f"C3")
    vf_m = np.mean(vf)
    plt.plot([vf_m, vf_m], [0, 1.5 * p * q], color=f"C3", linestyle="-", label=f"feature mean v")
    plt.plot([vf_m + sigma, vf_m + sigma], [0, 0.5 *p * q], color=f"C3", linestyle="--")
    plt.plot([vf_m - sigma, vf_m - sigma], [0, 0.5 *p * q], color=f"C3", linestyle="--")
    # print(sigma * np.sqrt(N), sigma * np.sqrt(N_eff_f), sigma * np.sqrt(N_eff))
    print(Coll_m[0], Coll_m[1], Coll_m[2], vf_m)
    print("a_2real = ", q*(p-1)/p * (Coll_m[2] - Coll_m[1]), " a_2th = ", q*(p-1)/p * (0.304-1/(2*np.pi)))

    plt.legend()
    plt.show()

def kernel_collision_Ctv_corr(p=113, q=56, N=1024, seed=0):
    accit = 0
    N_eff, N_eff_f = np.loadtxt(SCRIPT_DIR / f"checkpoints_sgd_thm3/gap_seed/{p}({q})-{N}-2/{seed}/kernel_gaps.txt", delimiter=" ", skiprows=1)[accit, 4:6]
    m = ModelNetworkCE(p=p, N=N, sigma=ReLU(), from_model=None, from_file=f"checkpoints_sgd_thm3/gap_seed/{p}({q})-{N}-2/{seed}/model_accit_{accit}.pt")
    data = ModAddDataTrainValid(p, select=q, rand_seed=seed)
    F_t = m.sigma(m.W @ data.X_train)
    F_v = m.sigma(m.W @ data.X_valid)
    K_tv = F_t.T @ F_v
    K_tv /= (np.linalg.norm(m.W)**2 / p)
    C_tv = data.X_train.T @ data.X_valid
    V_tv = K_tv * C_tv
    Coll = {0:np.zeros(p * q), 1:np.zeros(p * q), 2:np.zeros(p * q)}
    vf = np.zeros(p * q)
    def take_column(mat, h, j):
        return mat[h * q:(h + 1) * q, j]
    
    for j in range(p * q):
        Cor = {0:[], 1:[], 2:[]}
        for i in range(p):
            if i == j // q:
                vf[j] = np.sum(take_column(K_tv, i, j))
                continue
            c = round(np.sum(take_column(C_tv, i, j)))
            # v = np.sum(take_column(V_tv, i, j))
            # k = (np.sum(take_column(K_tv, i, j)) - v) / (q - c) * q
            k = (np.sum(take_column(K_tv, i, j)))
            Cor[c].append(k)
        for c in range(3):
            Coll[c][j] = np.mean(Cor[c])
    for c in range(3):
        plt.plot(vf, Coll[c], "s", color=f"C{c}", label=f"collision {c}", alpha=0.5)
    plt.legend()
    plt.show()
    
    plt.plot(Coll[2], Coll[0], "s", color=f"C{c}", label=f"collision {c}", alpha=0.5)
    plt.legend()
    plt.show()

def kernel_acc_test_59():
    gaps = []
    N_list = [256, 512, 1024, 2048, 4096]
    for N in N_list:
        filename = f"checkpoints_sgd_thm3/kernel_acc/{59}({29})-{N}-2"
        data = np.loadtxt(SCRIPT_DIR / f"{filename}/kernel_gaps.txt", delimiter=" ", skiprows=1)
        gaps.append(data[0, 2] / data[0, 3])

    plt.plot(N_list, gaps, marker='o')
    plt.xlabel('N')
    plt.ylabel('Gap')
    plt.title('Kernel Accuracy vs N')
    plt.show()

def gap_scaling_plot(p=59, q=29, Ns = [256, 512, 1024, 2048, 4096]):
    Ns = np.array(Ns)
    gaps = []
    N_effs = []
    N_eff_fs = []
    accit = 5
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # seeds mean
    for N in Ns:
        gap_sum = 0
        n_eff_sum = 0
        n_eff_f_sum = 0
        for seed in seeds:
            data = np.loadtxt(SCRIPT_DIR / f"checkpoints_sgd_thm3/gap_seed/{p}({q})-{N}-2/{seed}/kernel_gaps.txt", delimiter=" ", skiprows=1)
            # gap_sum += data[accit, 2] / data[accit, 3] - data[0, 2] / data[0, 3]
            gap_sum += data[accit, 0] - data[0, 0]
            n_eff_sum += data[accit, 4]
            n_eff_f_sum += data[accit, 5]
        gaps.append(gap_sum / len(seeds))
        N_effs.append(n_eff_sum / len(seeds))
        N_eff_fs.append(n_eff_f_sum / len(seeds))
    gaps = np.array(gaps)
    N_effs = np.array(N_effs)
    N_eff_fs = np.array(N_eff_fs)

    # log linear fit
    slope, intercept = np.polyfit(np.log(N_eff_fs), np.log(gaps), 1)
    print(slope)

    plt.loglog(N_eff_fs, gaps, 's-')
    # plt.plot(N_effs, gaps, marker='o')
    plt.xlabel('N')
    plt.ylabel('Gap Scaling')
    plt.title(f'Gap Scaling vs N for p={p}, q={q}')
    plt.show()

def gap_accit_0_plot(p=59, q=29, Ns = [256, 512, 1024, 2048, 4096]):
    gaps = []
    N_effs = []
    N_eff_fs = []
    accit = 5
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # seeds mean
    for N in Ns:
        gap_sum = 0
        n_eff_sum = 0
        n_eff_f_sum = 0
        for seed in seeds:
            data = np.loadtxt(SCRIPT_DIR / f"checkpoints_sgd_thm3/gap_seed/{p}({q})-{N}-2/{seed}/kernel_gaps.txt", delimiter=" ", skiprows=1)
            gap_sum += data[accit, 2] / data[accit, 3] - data[0, 2] / data[0, 3]
            n_eff_sum += data[accit, 4]
            n_eff_f_sum += data[accit, 5]
        gaps.append(gap_sum / len(seeds))
        N_effs.append(n_eff_sum / len(seeds))
        N_eff_fs.append(n_eff_f_sum / len(seeds))
    gaps = np.array(gaps)
    N_effs = np.array(N_effs)
    N_eff_fs = np.array(N_eff_fs)

    # log linear fit
    slope, intercept = np.polyfit(np.log(N_eff_fs), np.log(gaps), 1)
    print(slope)

    plt.loglog(N_eff_fs, gaps, 's-')
    # plt.plot(N_effs, gaps, marker='o')
    plt.xlabel('N')
    plt.ylabel('Gap Scaling')
    plt.title(f'Gap Scaling vs N for p={p}, q={q}')
    plt.show()

def plot_aligns(p=113, q=56, Ns = [256, 512, 1024, 2048, 4096]):
    for N in Ns:
        data = np.loadtxt(SCRIPT_DIR / f"checkpoints_sgd_thm3/aligns/{p}({q})-{N}-2/align.txt", delimiter=" ", skiprows=1)
        # plt.plot(data[:, 0], data[:, 4] / data[:, 2], label=f"{N}")
        plt.plot(data[:, 0], data[:, 1] / data[:, 2], label=f"{N}")
    plt.legend()
    plt.show()

def kernel_bg_shift(p=113, q=56, N=2048):
    seed = 0
    data = ModAddDataTrainValid(p, select=q, rand_seed=seed)
    Accits = [1/p] + list(range(1, 10))
    bgs = []
    for accit in range(10):
        m = ModelNetworkCE(p=p, N=N, sigma=ReLU(), from_model=None, from_file=f"checkpoints_sgd_thm3/gap_seed/{p}({q})-{N}-2/{seed}/model_accit_{accit}.pt")
        # F = m.activate(data.X)
        F_t = m.activate(data.X_train)
        F_v = m.activate(data.X_valid)
        K = F_t.T @ F_v
        bgs.append(np.sum(K) / (np.linalg.norm(m.W) ** 2))
    plt.plot(Accits, bgs)
    plt.show()

if __name__ == "__main__":
    # kernel_at_acc_0_acc_on_labels(p=113, q=56, N=1024, seed=0)
    # kernel_collision_Ctv_dis(p=113, q=56, N=2048, seed=0)
    # kernel_collision_Ctv_corr(p=113, q=56, N=256, seed=0)
    # gap_scaling_plot(p=59, q=29, Ns = [256, 512, 1024, 2048, 4096])
    # plot_aligns()
    # gap_accit_0_plot()
    kernel_bg_shift()