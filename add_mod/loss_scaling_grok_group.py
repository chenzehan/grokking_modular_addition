from model_network import ModelNetworkCE, ModAddDataTrainValid, ModAddDataRandom
from matplotlib import pyplot as plt
import numpy as np
from model import ReLU

if __name__ == "__main__":
    p = 59
    at_epoch = 3000
    accit = 0
    data = ModAddDataTrainValid(p, p // 2)

    N_list = np.array([256, 512, 1024, 2048, 4096])
    stds = []
    N_eff = []
    # vars = []
    data = ModAddDataTrainValid(p, p // 2)
    # data = ModAddDataRandom(p, (p // 2) / p) # rand
    for N in N_list:
        # filename = f"checkpoints_sgd_thm3/59(0.50)-{N}-2/model_epoch_{at_epoch}.pt"
        filename = f"checkpoints_sgd_thm3/59(0.50)-{N}-2/model_accit_{accit}.pt"
        m = ModelNetworkCE(p=p, N=N, sigma=ReLU(), from_file=filename)
        # N_eff.append(m.effective_width())
        N_eff.append(m.effective_feature_width(data.X_train, data.Y_train))
        var = 0
        mn = 0
        for h in range(p):
            selc = data.select
            X = data.X_train[:, h * selc: (h + 1) * selc]
            Y = data.Y_train[:, h * selc: (h + 1) * selc]
        
            loss = m.forward_loss_foreach(X, Y)
            var += np.var(loss)
            # var += np.var(loss) * p / np.mean(loss)**2
            mn += np.mean(loss)
        stds.append(np.sqrt(var / p) / (mn / p))
        # stds.append(np.sqrt(var))

    N_eff = np.array(N_eff)
    stds = np.array(stds)

    slope, intercept = np.polyfit(np.log(N_eff), np.log(stds), 1)
    print(slope)

    plt.loglog(N_eff, stds, 's-')
    plt.show()