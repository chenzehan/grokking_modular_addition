from model_network import ModelNetworkCE, ModAddDataTrainValid, ModAddDataRandom
from matplotlib import pyplot as plt
import numpy as np
from model import ReLU

if __name__ == "__main__":
    p = 59
    at_epoch = 30000
    N = 512
    data = ModAddDataRandom(p, 29 / 59)
    filename = f"checkpoints_sgd_thm3/random/59(0.49)-{N}-2/model_epoch_{at_epoch}.pt"
    m = ModelNetworkCE(p=p, N=N, sigma=ReLU(), from_file=filename)

    groups = data.group_selection_train()
    sizes = []
    means = []
    for key, select in groups.items():
        X = data.X[:, select]
        Y = data.Y[:, select]
    
        loss = m.forward_loss_foreach(X, Y)
        sizes.append(len(select))
        means.append(np.mean(loss))
    sizes = np.array(sizes)
    means = np.array(means)

    plt.plot(sizes, means, 's')
    plt.show()

    sizes_w = sizes / np.sum(sizes)
    print(np.sum(sizes_w * means**2) - np.sum(sizes_w * means)**2)

    # rand scaling plot with correction to label-sample imbalance
    # the imbalance constributes variance 2.6e-5
    # p = 59
    # at_epoch = 30000
    # N_list = np.array([256, 512, 1024, 2048])
    # stds = []
    # N_eff = []
    # # vars = []
    # data = ModAddDataRandom(p, (p // 2) / p) # rand
    # for N in N_list:
    #     filename = f"checkpoints_sgd_thm3/random/59(0.49)-{N}-2/model_epoch_{at_epoch}.pt"
    #     m = ModelNetworkCE(p=p, N=N, sigma=ReLU(), from_file=filename)
    #     N_eff.append(m.effective_width())
    #     # F_t = m.activate(data.X_train)
    #     # F_v = m.activate(data.X_valid)
    #     loss = m.forward_loss_foreach(data.X_train, data.Y_train)
    #     stds.append(np.sqrt(np.var(loss) - 2.6e-5) / np.mean(loss)) # rand correction
    #     print(np.mean(loss), np.std(loss), N_eff[-1])
    #     # vars.append(np.var(loss))
    # N_eff = np.array(N_eff)
    # stds = np.array(stds)
    # # vars = np.array(vars)

    # slope, intercept = np.polyfit(np.log(N_eff), np.log(stds), 1)
    # print(slope)

    # plt.loglog(N_eff, stds, 's-')
    # plt.show()
    