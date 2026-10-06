from add_mod_2_sgd_thm3 import *
from model_network import ModelNetworkCE
from model import ModAddDataTrainValid
from model import ReLU

class MonitorEarlyRate:
    def __init__(self, filename, p=59, N=512, interval=5, save_model=False, data_seed=42):
        self.filename = filename
        self.save_model = save_model
        self.data = ModAddDataTrainValid(p=p, select=train_select, rand_seed=data_seed)
        self.p = p
        self.N = N
        self.interval = interval
        self.ph = train_select
        self.cn = p**2 / 2 * (1/6 - 1/8)
        self.Y_TY = self.data.Y_train.T @ self.data.Y_train

        self.Epochs = []
        self.E = []
    def __call__(self, epoch, _0, _1, _2, test_acc, model):
        m = ModelNetworkCE(p=self.p, N=self.N, sigma=ReLU(), from_model=model, from_file=None)
        F = m.activate(self.data.X)
        self.E.append(0.5 * np.linalg.norm(self.data.Y @ F.T, axis=0)**2 / np.linalg.norm(m.W, axis=-1)**2 / self.cn)
        return True
    
    def end(self):
        E = np.array(self.E)
        selected = np.flatnonzero(E[-1, :] >= 0.9)
        ref_E = E[:, selected]
        E_window = (0.4, 0.6)
        ts = np.array([np.searchsorted(ref_E[:, i], E_window) for i in range(ref_E.shape[1])])
        # t_avg = np.mean(ts) * self.interval
        # r_avg = np.mean((E_window[1] - E_window[0]) / ((ts[:, 1] - ts[:, 0]) * self.interval))
        header_str = "# Time Rate"
        np.savetxt(f"{self.filename}/early_rate.txt", np.column_stack((ts * self.interval, (E_window[1] - E_window[0]) / ((ts[:, 1] - ts[:, 0]) * self.interval))), header=header_str)

if __name__ == "__main__":

    N = 512
    for seed in range(42, 43):
        filename = f"checkpoints_sgd_thm3/early/{p}({train_select})-{N}-2/{seed}"
            
        os.makedirs(filename, exist_ok=True)
        mntr = MonitorEarlyRate(filename, data_seed=seed)

        train_losses, train_accuracies, test_losses, test_accuracies = train_with_balanced_data(bias=False, save_model=False, show_plot=None, monitor=mntr, save_final=False, hidden_size=N, data_seed=seed, checkpoint_interval=5, epochs=401)
        np.savetxt(f"{filename}/results.txt", np.array([train_losses, test_losses, train_accuracies, test_accuracies]).T, header=f"#Train Loss, Test Loss, Train Accuracy, Test Accuracy, fraction={fraction_train}, interval={checkpoint_interval}", delimiter=",")

