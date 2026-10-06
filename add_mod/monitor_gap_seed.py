from add_mod_2_sgd_thm3_gpu import *
from model_network import ModelNetworkCE
from model import ModAddDataTrainValid
from model import relu

class MonitorAccKernelGapEffWidth:
    def __init__(self, filename, save_model=False, auto_end=True, data_seed=42):
        self.filename = filename
        self.save_model = save_model
        self.auto_end = auto_end
        self.data = ModAddDataTrainValid(p, select=train_select, rand_seed=data_seed)
        self.p = p
        self.ph = train_select
        self.Y_TY = self.data.Y_train.T @ self.data.Y_train
        self.test_acc = 0
        self.base_acc_list = np.array([1 / p] + [0.1 * i for i in range(1, 10)])
        self.iter_acc = 0
        self.kernel_gap = np.zeros(len(self.base_acc_list))
        self.epochs = np.zeros(len(self.base_acc_list))
        self.W_normsqr = np.zeros(len(self.base_acc_list))
        self.N_eff = np.zeros(len(self.base_acc_list))
        self.N_eff_f = np.zeros(len(self.base_acc_list))
    def __call__(self, epoch, _0, _1, _2, test_acc, model):
        if self.iter_acc >= len(self.base_acc_list):
            return True
        if self.base_acc_list[self.iter_acc] > test_acc:
            self.test_acc = test_acc
            return True

        self.test_acc = test_acc

        print(f"Monitor reaches {self.iter_acc}-checkpoint, base acc = {self.base_acc_list[self.iter_acc]}, acc = {test_acc}.")
        self.epochs[self.iter_acc] = epoch
        m = ModelNetworkCE(p=self.p, N=None, sigma=relu, from_model=model, from_file=None)
        W = m.W
        F_t = relu(W @ self.data.X_train)
        F_v = relu(W @ self.data.X_valid)
        K_tv = F_t.T @ F_v
        self.kernel_gap[self.iter_acc] = np.sum(K_tv * self.Y_TY)
        self.W_normsqr[self.iter_acc] = np.linalg.norm(W)**2
        self.N_eff[self.iter_acc] = m.effective_width()
        self.N_eff_f[self.iter_acc] = m.effective_feature_width(self.data.X_train, self.data.Y_train)

        if self.save_model:
            checkpoint_path = f"{self.filename}/model_accit_{self.iter_acc}.pt"
            torch.save(model.state_dict(), checkpoint_path)

        self.iter_acc += 1
        if self.iter_acc < len(self.base_acc_list):
            return True
        self.save_file()
        print("Monitor ends.")
        return not self.auto_end
    
    def save_file(self):
        header_str = "# epochs base_acc total_gap W_norm_sqr N_eff N_eff_f"
        np.savetxt(f"{self.filename}/kernel_gaps.txt", np.column_stack((self.epochs, self.base_acc_list, self.kernel_gap, self.W_normsqr, self.N_eff, self.N_eff_f)), header=header_str)

class MonitorKernelAccKernelGapEffWidth:
    def __init__(self, filename, save_model=False, auto_end=True, data_seed=42):
        self.filename = filename
        self.save_model = save_model
        self.auto_end = auto_end
        self.data = ModAddDataTrainValid(p, select=train_select, rand_seed=data_seed)
        self.p = p
        self.ph = train_select
        self.valid_labels = np.argmax(self.data.Y_valid, axis=0)
        self.test_acc = 0
        self.network_acc = 0
        self.base_acc_list = np.array([1 / p, 2.5 / p, 4 / p] + [0.1 * i for i in range(1, 6)])
        self.iter_acc = 0
        self.kernel_gap = np.zeros(len(self.base_acc_list))
        self.epochs = np.zeros(len(self.base_acc_list))
        self.W_normsqr = np.zeros(len(self.base_acc_list))
        self.N_eff = np.zeros(len(self.base_acc_list))
        self.N_eff_f = np.zeros(len(self.base_acc_list))
        self.kernel_accs = np.zeros(len(self.base_acc_list))
        self.network_accs = np.zeros(len(self.base_acc_list))
    def __call__(self, epoch, _0, _1, _2, test_acc, model):
        if self.iter_acc >= len(self.base_acc_list):
            return not self.auto_end
        self.network_acc = float(test_acc)
        m = ModelNetworkCE(p=self.p, N=model[0].weight.shape[0], sigma=relu, from_model=model, from_file=None)
        W = m.W
        F_t = relu(W @ self.data.X_train)
        F_v = relu(W @ self.data.X_valid)
        # Y_train is already centered. Associativity avoids materializing the
        # M x M cross-kernel at every evaluation of the training loop.
        class_features = self.data.Y_train @ F_t.T
        kernel_scores = class_features @ F_v
        self.test_acc = float(np.mean(np.argmax(kernel_scores, axis=0) == self.valid_labels))
        if self.test_acc < self.base_acc_list[self.iter_acc]:
            return True

        # <Y_train K_tv, Y_valid> equals <K_tv, Y_train.T Y_train>
        # for this balanced, equally ordered train/validation split.
        total_gap = np.sum(kernel_scores * self.data.Y_valid)
        W_normsqr = np.linalg.norm(W)**2
        N_eff = m.effective_width()
        feature_energy = np.sum(class_features**2, axis=0)
        feature_energy_sqr_sum = np.sum(feature_energy**2)
        N_eff_f = (np.sum(feature_energy)**2 / feature_energy_sqr_sum
                   if feature_energy_sqr_sum > 0 else np.nan)

        # An evaluation can cross several milestones; they all belong to the
        # same epoch and model, rather than later evaluations.
        while (self.iter_acc < len(self.base_acc_list)
               and self.test_acc >= self.base_acc_list[self.iter_acc]):
            i = self.iter_acc
            print(f"Monitor reaches {i}-checkpoint, base kernel acc = {self.base_acc_list[i]}, "
                  f"kernel acc = {self.test_acc}, network acc = {self.network_acc}.")
            self.epochs[i] = epoch
            self.kernel_gap[i] = total_gap
            self.W_normsqr[i] = W_normsqr
            self.N_eff[i] = N_eff
            self.N_eff_f[i] = N_eff_f
            self.kernel_accs[i] = self.test_acc
            self.network_accs[i] = self.network_acc
            if self.save_model:
                checkpoint_path = f"{self.filename}/model_accit_{i}.pt"
                torch.save(model.state_dict(), checkpoint_path)
            self.iter_acc += 1

        self.save_file()
        if self.iter_acc < len(self.base_acc_list):
            return True
        print("Monitor ends.")
        return not self.auto_end
    
    def save_file(self):
        # Keep the original six column positions, append observed accuracies,
        # and save only reached milestones so unfinished runs remain usable.
        header_str = "epochs base_acc total_gap W_norm_sqr N_eff N_eff_f kernel_acc network_acc\naccuracy_source=kernel"
        values = np.column_stack((self.epochs, self.base_acc_list, self.kernel_gap,
                                  self.W_normsqr, self.N_eff, self.N_eff_f,
                                  self.kernel_accs, self.network_accs))
        np.savetxt(f"{self.filename}/kernel_gaps.txt", values[:self.iter_acc], header=header_str)

class MonitorKernelAcc:
    def __init__(self, filename):
        self.filename = filename
        self.p = p
        self.data = ModAddDataTrainValid(p, select=train_select)
        # self.Y_TY = self.data.Y_train.T @ self.data.Y_train
        self.valid_labels = np.argmax(self.data.Y_valid, axis=0)
        self.epochs = []
        self.kernel_t_accs = []
        self.kernel_v_accs = []
        self.kernel_st_accs = []
        self.kernel_sv_accs = []

    def __call__(self, epoch, _0, _1, _2, _3, model):
        m = ModelNetworkCE(p=self.p, N=hidden_size, sigma=relu, from_model=model, from_file=None)
        F_t = m.activate(self.data.X_train)
        S = (self.data.Y_train + 1 / p) - m.forward(self.data.X_train)
        self.epochs.append(epoch)
        m = ModelNetworkCE(p=self.p, N=model[0].weight.shape[0], sigma=relu, from_model=model, from_file=None)
        W = m.W
        F_t = relu(W @ self.data.X_train)
        F_v = relu(W @ self.data.X_valid)
        p_tt = self.data.Y_train @ F_t.T @ F_t
        p_tv = self.data.Y_train @ F_t.T @ F_v
        acc_t = np.mean(np.argmax(p_tt, axis=0) == self.valid_labels)
        acc_v = np.mean(np.argmax(p_tv, axis=0) == self.valid_labels)
        s_tt = S @ F_t.T @ F_t
        s_tv = S @ F_t.T @ F_v
        acc_st = np.mean(np.argmax(s_tt, axis=0) == self.valid_labels)
        acc_sv = np.mean(np.argmax(s_tv, axis=0) == self.valid_labels)
        self.kernel_t_accs.append(acc_t)
        self.kernel_v_accs.append(acc_v)
        self.kernel_st_accs.append(acc_st)
        self.kernel_sv_accs.append(acc_sv)
        print(self.kernel_t_accs[-1], self.kernel_v_accs[-1], self.kernel_st_accs[-1], self.kernel_sv_accs[-1])
        
        return True
    def end(self):
        print("MonitorKernelAcc ends.")
        np.savetxt(f"{self.filename}/kernel_accs_S.txt", np.column_stack((self.epochs, self.kernel_t_accs, self.kernel_v_accs, self.kernel_st_accs, self.kernel_sv_accs)), fmt="%d %f %f %f %f", header="# epochs kernel_t_accs kernel_v_accs kernel_st_accs kernel_sv_accs")

if __name__ == "__main__":
        
    # for N in [2048, 4096]:
    #     for seed in range(1, 10):
    #         filename = f"checkpoints_sgd_thm3/gap_seed/{p}({train_select})-{N}-2/{seed}"
                
    #         os.makedirs(filename, exist_ok=True)
    #         mntr = MonitorAccKernelGapEffWidth(filename, save_model=True if seed == 0 else False, auto_end=True, data_seed=seed)

    #         train_losses, train_accuracies, test_losses, test_accuracies = train_with_balanced_data(bias=False, save_model=False, show_plot=None, monitor=mntr, save_final=False, hidden_size=N, data_seed=seed)
    #         np.savetxt(f"{filename}/results.txt", np.array([train_losses, test_losses, train_accuracies, test_accuracies]).T, header=f"#Train Loss, Test Loss, Train Accuracy, Test Accuracy, fraction={fraction_train}, interval={checkpoint_interval}", delimiter=",")

    
    # for N in [256, 512, 1024, 2048, 4096]:
    for N in [512, 2048]:
        filename = f"checkpoints_sgd_thm3/aligns/{p}({train_select})-{N}-2/"
        os.makedirs(filename, exist_ok=True)
        # mntr = MonitorAccKernelGapEffWidth(filename)
        mntr = MonitorKernelAcc(filename)

        train_losses, train_accuracies, test_losses, test_accuracies = train_with_balanced_data(bias=False, save_model=False, show_plot=None, monitor=mntr, save_final=False, hidden_size=N)
        # np.savetxt(f"{filename}/results.txt", np.array([train_losses, test_losses, train_accuracies, test_accuracies]).T, header=f"#Train Loss, Test Loss, Train Accuracy, Test Accuracy, fraction={fraction_train}, interval={checkpoint_interval}", delimiter=",")

