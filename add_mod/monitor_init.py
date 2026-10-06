from add_mod_2_sgd_init import *
# from add_mod_2_sgd_thm3 import *
from model_network import ModelNetworkCE
from model import ModAddDataTrainValid
from model import relu
from model import cos_similarity

class MonitorAlignment:
    def __init__(self, filename, show=False):
        self.filename = filename
        self.p = p
        self.show = show
        self.data = ModAddDataTrainValid(p, select=train_select)
        self.Y_TY = self.data.Y_train.T @ self.data.Y_train

    def __call__(self, epoch, _0, _1, _2, _3, model):
        m = ModelNetworkCE(p=self.p, N=hidden_size, sigma=relu, from_model=model, from_file=None)
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

        d = np.loadtxt(f"{self.filename}/align.txt", delimiter=' ')
        d = np.row_stack((np.array([epoch, E, Wnsqr, E_t, E_v, np.std(loss), N_eff, N_eff_f]), d))
        np.savetxt(f"{self.filename}/align_init.txt", d, fmt="%d %f %f %f %f %f %f %f", header="epochs alignment W_norm_sqr alignment_tt alignment_tv std effective_width effective_feature_width")
        np.savetxt(f"checkpoints_sgd_thm3/aligns/align_init.txt", d, fmt="%d %f %f %f %f %f %f %f", header="epochs alignment W_norm_sqr alignment_tt alignment_tv std effective_width effective_feature_width")
        return True

if __name__ == "__main__":
    for N in [256, 512, 1024, 2048, 4096]:
        mntr = MonitorAlignment(f"checkpoints_sgd_thm3/aligns/{p}({train_select})-{N}-2/")
        train_loss, test_loss, train_accuracy, test_accuracy = train_with_balanced_data(bias=False, save_model=True, show_plot="loss", monitor=mntr, save_final=True, hidden_size=N)