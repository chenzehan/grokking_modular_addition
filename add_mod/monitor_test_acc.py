from add_mod_2_sgd_thm3_gpu import *
# from add_mod_2_sgd_thm3 import *
from model_network import ModelNetworkCE
from model import ModAddDataTrainValid
from model import relu
from model import cos_similarity

class MonitorGetKernelGap:
    def __init__(self, filename, save_model=False, auto_end=True):
        self.filename = filename
        self.save_model = save_model
        self.auto_end = auto_end
        self.data = ModAddDataTrainValid(p, select=int(fraction_train * p))
        self.p = p
        self.ph = int(fraction_train * p)
        self.Y_TY = self.data.Y_train.T @ self.data.Y_train
        self.test_acc = 0
        self.base_acc_list = np.array([1 / p] + [0.1 * i for i in range(1, 10)])
        self.iter_acc = 0
        self.kernel_gap = np.zeros(len(self.base_acc_list))
        self.epochs = np.zeros(len(self.base_acc_list))
        self.W_normsqr = np.zeros(len(self.base_acc_list))
    def __call__(self, epoch, _0, _1, _2, test_acc, model):
        if self.iter_acc >= len(self.base_acc_list):
            return True
        if self.base_acc_list[self.iter_acc] > test_acc:
            self.test_acc = test_acc
            return True

        self.test_acc = test_acc

        print(f"Monitor reaches {self.iter_acc}-checkpoint, base acc = {self.base_acc_list[self.iter_acc]}, acc = {test_acc}.")
        self.epochs[self.iter_acc] = epoch
        W = model[0].weight.data.numpy()
        F_t = relu(W @ self.data.X_train)
        F_v = relu(W @ self.data.X_valid)
        K_tv = F_t.T @ F_v
        self.kernel_gap[self.iter_acc] = np.sum(K_tv * self.Y_TY)
        self.W_normsqr[self.iter_acc] = np.linalg.norm(W)**2

        if self.save_model:
            checkpoint_path = f"{filename}/model_accit_{self.iter_acc}.pt"
            torch.save(model.state_dict(), checkpoint_path)

        self.iter_acc += 1
        if self.iter_acc < len(self.base_acc_list):
            return True
        self.save_file()
        print("Monitor ends.")
        return not self.auto_end
    
    def save_file(self):
        header_str = "# epochs base_acc total_gap W_norm_sqr"
        np.savetxt(self.filename, np.column_stack((self.epochs, self.base_acc_list, self.kernel_gap, self.W_normsqr)), header=header_str)

class MonitorFeatureNumberFreq:
    def __init__(self, filename):
        self.filename = filename
        self.p = p
        self.epochs = []
        self.f_num_05 = []
        self.f_num_09 = []
    def __call__(self, epoch, _0, _1, _2, _3, model):
        W = model[0].weight.data.numpy()
        n_05 = 0
        n_09 = 0
        for w in W:
            wfabs = np.abs(np.fft.fft(w[:self.p])[:self.p // 2])
            c = wfabs ** 2
            c = c / np.sum(c)
            if np.max(c) > 0.5:
                n_05 += 1
            if np.max(c) > 0.9:
                n_09 += 1
        print(f"Feature counts {n_05}, {n_09}.")
        self.epochs.append(epoch)
        self.f_num_05.append(n_05)
        self.f_num_09.append(n_09)
        return True
    def end(self):
        np.savetxt(self.filename, np.column_stack((self.epochs, self.f_num_05, self.f_num_09)).astype(int), fmt="%d")

class MonitorFeatureNumber:
    def __init__(self, filename):
        self.filename = filename
        self.p = p
        self.data = ModAddDataTrainValid(p, select=train_select)
        self.epochs = []
        self.f_num_05 = []
        self.f_num_09 = []
    def __call__(self, epoch, _0, _1, _2, _3, model):
        m = ModelNetworkCE(p=self.p, N=None, sigma=relu, from_model=model, from_file=None)
        W = m.W
        F = m.activate(self.data.X)
        cn = self.p**2 / 2 * (1/6 - 1/8)
        E = 0.5 * np.linalg.norm(self.data.Y @ F.T, axis=0)**2 / np.linalg.norm(m.W, axis=-1)**2 / cn
        n_05 = np.sum(E >= 0.5)
        n_09 = np.sum(E >= 0.9)
        print(f"Feature counts {n_05}, {n_09}.")
        self.epochs.append(epoch)
        self.f_num_05.append(n_05)
        self.f_num_09.append(n_09)
        return True
    def end(self):
        np.savetxt(self.filename, np.column_stack((self.epochs, self.f_num_05, self.f_num_09)).astype(int), fmt="%d")

class MonitorAlignment:
    def __init__(self, filename, show=False):
        self.filename = filename
        self.p = p
        self.show = show
        self.data = ModAddDataTrainValid(p, select=train_select)
        self.Y_TY = self.data.Y_train.T @ self.data.Y_train
        self.epochs = []
        self.aligns = []
        self.W_normsqr = []
        self.aligns_tt = []
        self.aligns_tv = []
        self.stds = []
        self.n_eff = []
        self.n_eff_f = []

    def __call__(self, epoch, _0, _1, _2, _3, model):
        m = ModelNetworkCE(p=self.p, N=None, sigma=relu, from_model=model, from_file=None)
        W = m.W
        F = m.activate(self.data.X)
        E = np.linalg.norm(F @ self.data.Y.T)**2
        F_t = m.activate(self.data.X_train)
        F_v = m.activate(self.data.X_valid)
        E_t = np.linalg.norm(F_t @ self.data.Y_train.T)**2
        K_tv = F_t.T @ F_v
        E_v = np.sum(K_tv * self.Y_TY)
        Wnsqr = np.linalg.norm(W)**2
        loss = m.forward_loss_foreach(self.data.X_train, self.data.Y_train)
        N_eff = m.effective_width()
        N_eff_f = m.effective_feature_width(self.data.X_train, self.data.Y_train)
        if self.show:
            print(f"E = {E}, Wnsqr = {Wnsqr}, Et = {E_t}, Ev = {E_v}, std = {np.std(loss)}, N_eff = {N_eff}, N_eff_f = {N_eff_f}")

        self.epochs.append(epoch)
        self.aligns.append(E)
        self.W_normsqr.append(Wnsqr)
        self.aligns_tt.append(E_t)
        self.aligns_tv.append(E_v)
        self.stds.append(np.std(loss))
        self.n_eff.append(N_eff)
        self.n_eff_f.append(N_eff_f)
        return True
    def end(self):
        print("MonitorAlignment ends.")
        np.savetxt(self.filename, np.column_stack((self.epochs, self.aligns, self.W_normsqr, self.aligns_tt, self.aligns_tv, self.stds, self.n_eff, self.n_eff_f)), fmt="%d %f %f %f %f %f %f %f", header="# epochs alignment W_norm_sqr alignment_tt alignment_tv std effective_width effective_feature_width")

class MonitorCosSimilarity:
    def __init__(self, filename):
        self.filename = filename
        self.p = p
        self.data = ModAddDataTrainValid(p, select=train_select)
        # self.data = ModAddDataRandom(p, fraction=0.49)
        self.Y_TY = self.data.Y_train.T @ self.data.Y_train
        self.epochs = []
        self.S_Y_sim = []
        self.V_SF_sim = []
        self.V_YF_sim = []
        self.S_dS_rat = []

    def __call__(self, epoch, _0, _1, _2, _3, model):
        m = ModelNetworkCE(p=self.p, N=hidden_size, sigma=relu, from_model=model, from_file=None)
        F_t = m.activate(self.data.X_train)
        S = (self.data.Y_train + 1 / p) - m.forward(self.data.X_train)
        self.epochs.append(epoch)
        self.S_Y_sim.append(cos_similarity(S, self.data.Y_train))
        self.V_YF_sim.append(cos_similarity(m.V, self.data.Y_train @ F_t.T))
        self.V_SF_sim.append(cos_similarity(m.V, S @ F_t.T))
        
        return True
    def end(self):
        print("MonitorCosSimilarity ends.")
        np.savetxt(self.filename, np.column_stack((self.epochs, self.S_Y_sim, self.V_YF_sim, self.V_SF_sim)), fmt="%d %f %f %f", header="# epochs S_Y_sim V_YF_sim V_SF_sim")

# 运行实验
if __name__ == "__main__":
    
    # os.makedirs(filename, exist_ok=True)
    # # mntr = MonitorGetKernelGap(f"{filename}/kernel_gaps.txt", save_model=True, auto_end=False)
    # mntr = MonitorAlignment(f"{filename}/align.txt")

    # train_losses, train_accuracies, test_losses, test_accuracies = train_with_balanced_data(bias=False, save_model=False, show_plot="acc", monitor=mntr, save_final=True)
    # np.savetxt(f"{filename}/results.txt", np.array([train_losses, test_losses, train_accuracies, test_accuracies]).T, header=f"#Train Loss, Test Loss, Train Accuracy, Test Accuracy, fraction={fraction_train}, interval={checkpoint_interval}", delimiter=",")
    
    # aligns
    # for N in [256, 512, 1024, 2048, 4096]:
    #     filename = f"checkpoints_sgd_thm3/aligns/{p}({train_select})-{N}-2/"
    #     os.makedirs(filename, exist_ok=True)
    #     mntr = MonitorAlignment(f"{filename}/align.txt")

    #     train_losses, train_accuracies, test_losses, test_accuracies = train_with_balanced_data(bias=False, save_model=False, show_plot=None, monitor=mntr, save_final=False, hidden_size=N)
    #     np.savetxt(f"{filename}/results.txt", np.array([train_losses, test_losses, train_accuracies, test_accuracies]).T, header=f"#Train Loss, Test Loss, Train Accuracy, Test Accuracy, fraction={fraction_train}, interval={checkpoint_interval}", delimiter=",")

    # cos sims
    # for N in [2048]:
    #     filename = f"checkpoints_sgd_thm3/aligns/{p}({train_select})-{N}-2/"
    #     # filename = f"checkpoints_sgd_thm3/random/{p}({fraction_train:.2f})-{N}-2/"
    #     os.makedirs(filename, exist_ok=True)
    #     # mntr = MonitorAlignment(f"{filename}/align.txt")
    #     mntr = MonitorCosSimilarity(f"{filename}/cos_sim.txt")

    #     train_losses, train_accuracies, test_losses, test_accuracies = train_with_balanced_data(bias=False, save_model=False, show_plot=None, monitor=mntr, save_final=False, p=59, hidden_size=N)
    #     # np.savetxt(f"{filename}/results.txt", np.array([train_losses, test_losses, train_accuracies, test_accuracies]).T, header=f"#Train Loss, Test Loss, Train Accuracy, Test Accuracy, fraction={fraction_train}, interval={checkpoint_interval}", delimiter=",")

    # for N in [256, 512, 1024, 2048, 4096]:
    #     filename = f"checkpoints_sgd_thm3/aligns/{p}({train_select})-{N}-2/"
    #     os.makedirs(filename, exist_ok=True)
    #     mntr = MonitorAlignment(f"{filename}/align.txt", show=True)

    #     train_losses, train_accuracies, test_losses, test_accuracies = train_with_balanced_data(bias=False, save_model=False, show_plot=None, monitor=mntr, save_final=False, hidden_size=N)
    #     np.savetxt(f"{filename}/results.txt", np.array([train_losses, test_losses, train_accuracies, test_accuracies]).T, header=f"#Train Loss, Test Loss, Train Accuracy, Test Accuracy, fraction={fraction_train}, interval={checkpoint_interval}", delimiter=",")
    
    # feature num
    for N in [256, 512, 1024, 2048]:
        filename = f"checkpoints_sgd_thm3/early/{p}({train_select})-{N}-2/"
        os.makedirs(filename, exist_ok=True)
        mntr = MonitorFeatureNumber(f"{filename}/feature_num_E.txt")

        train_losses, train_accuracies, test_losses, test_accuracies = train_with_balanced_data(bias=False, save_model=False, show_plot=None, monitor=mntr, save_final=False, hidden_size=N, epochs=501, checkpoint_interval=5)
        # np.savetxt(f"{filename}/results.txt", np.array([train_losses, test_losses, train_accuracies, test_accuracies]).T, header=f"#Train Loss, Test Loss, Train Accuracy, Test Accuracy, fraction={fraction_train}, interval={checkpoint_interval}", delimiter=",")
    