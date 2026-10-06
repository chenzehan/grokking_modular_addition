from mod_tensors import *
import numpy as np

from pathlib import Path
from matplotlib import pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent

def get_model_from(filename, p, N, bias=False):
    model_path = Path(filename).expanduser()
    if not model_path.is_absolute():
        model_path = (SCRIPT_DIR / model_path).resolve()

    model = nn.Sequential(
        nn.Linear(2 * p, N, bias=bias),
        nn.ReLU(),
        nn.Linear(N, p, bias=bias)
    )

    if not model_path.exists():
        print(f"Don't exist: {model_path}")
        return None
    
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    return model

class ModelNetworkCE():
    def __init__(self, p, N, from_file, sigma=relu, from_model=None):
        self.p = p
        self.N = N
        self.sigma = sigma
        if from_model is not None:
            self.m = from_model
        else:
            self.m = get_model_from(from_file, p, N)
        self.W = self.m[0].weight.data.cpu().detach().numpy()
        self.V = self.m[2].weight.data.cpu().detach().numpy()
    def activate(self, X):
        return self.sigma(self.W @ X)
    def forward_logits(self, X):
        return self.V @ self.activate(X)
    def forward(self, X):
        return column_softmax(self.forward_logits(X))
    def error_s(self, X, Y):
        return (Y + 1 / self.p) - self.forward(X)
    def forward_loss_foreach(self, X, Y): # zero-mean Y
        P = self.forward(X)
        return np.sum(-(Y + 1 / self.p) * np.log(P + 1e-12), axis=0)
    def forward_loss_mean(self, X, Y):
        return np.sum(self.forward_loss_foreach(X, Y)) / Y.shape[1]
    def forward_accuray(self, X, Y):
        P = self.forward(X)
        acc = np.sum([int(np.argmax(a) == np.argmax(b)) for a, b in zip(Y.T, P.T)]) / Y.shape[1]
        return acc
    def sieve_freq(self, threshold=0.5):
        indices = []
        for i, w in enumerate(self.W):
            wfabs = np.abs(np.fft.fft(w[:self.p])[:self.p // 2])
            c = wfabs ** 2
            c = c / np.sum(c)
            if np.max(c) > threshold:
                indices.append(i)
        return indices
    def clear_freq(self, threshold=0.5):
        indices = self.sieve_freq(threshold=threshold)
        for i in indices:
            self.W[i, :] = np.zeros(2*self.p)
    def clear_nonfreq(self, threshold=0.5):
        indices = self.sieve_freq(threshold=threshold)
        save_indices = np.zeros(self.W.shape[0], dtype=int)
        for i in indices:
            save_indices[i] = 1
        for i in range(self.W.shape[0]):
            if save_indices[i] == 0:
                self.W[i, :] = np.zeros(2*self.p)

    def effective_width(self):
        return np.linalg.norm(self.W)**4 / np.sum(np.linalg.norm(self.W, axis=-1)**4)
    def effective_feature_width(self, X, Y):
        F = self.activate(X)
        p = np.linalg.norm(F @ Y.T, axis=-1)**2
        p = p / np.sum(p)
        return 1 / np.sum(p**2)
    def effective_feature_tv_width(self, X_t, X_v, Y):
        F_t = self.activate(X_t)
        F_v = self.activate(X_v)
        Y_TY = Y.T @ Y
        p = np.array([np.trace(F_v[i:i+1] @ Y_TY @ F_t[i:i+1].T) for i in range(F_t.shape[0])])
        p = p / np.sum(p)
        return 1 / np.sum(p**2)

def fourier_feature_amount(w):
    wf = np.fft.fft(w)
    wf_sqr = np.abs(wf)**2
    c = wf_sqr / np.sum(wf_sqr)
    return np.sum(c**2)
def fourier_int(W):
    p = W.shape[1] // 2
    sum = 0
    for w in W:
        c1 = fourier_feature_amount(w[:p])
        c2 = fourier_feature_amount(w[p:])
        sum += c1 + c2
    return sum

def matrix_alignment(A, B):
    if A.shape != B.shape:
        return None
    return np.sum(A * B) / (np.linalg.norm(A) * np.linalg.norm(B))

def matrix_alignment_zm(A_, B_):
    A = A_ - np.mean(A_)
    B = B_ - np.mean(B_)
    if A.shape != B.shape:
        return None
    return np.sum(A * B) / (np.linalg.norm(A) * np.linalg.norm(B))

if __name__ == "__main__":
    filename = "checkpoints_sgd_thm3/59(0.50)-512-2/model_epoch_1000.pt"
    p = 59
    m = ModelNetworkCE(p=p, N=512, from_file=filename)
    data = ModAddDataTrainValid(p, p // 2)
    # print(m.forward_loss_mean(data.X_train, data.Y_train))
    # print(len(m.sieve_freq(0.2)))
    # m.clear_freq(0.5)
    # print(m.forward_accuray(data.X_train, data.Y_train))
    # print(m.forward_loss_mean(data.X_valid, data.Y_valid))
    # print(m.forward_accuray(data.X_valid, data.Y_valid))

    # error = loss = m.forward_loss_foreach(data.X_train, data.Y_train)
    # print(np.mean(loss), np.std(loss))
    # plt.hist(loss, bins=50)
    # plt.show()

    # F = m.activate(data.X)
    # K = F.T @ F
    # s = (data.Y + 1 / p) - m.forward(data.X)
    # plt.imshow(s.T @ s)
    # plt.show() 

    F_t = m.activate(data.X_train)
    F_v = m.activate(data.X_valid)
    # K = F_t.T @ F_t
    # K = K - np.mean(K)
    # # P = column_softmax((error * data.Y_train) @ K)
    s = (data.Y_train + 1 / p) - m.forward(data.X_train)
    # P = column_softmax(s @ K)
    
    # acc = np.sum([int(np.argmax(a) == np.argmax(b)) for a, b in zip(data.Y_train.T, P.T)]) / data.Y_train.shape[1]
    # print(acc)

    print(matrix_alignment(s @ F_t.T, m.V))
    print(matrix_alignment(data.Y_train @ F_t.T, m.V))
    print(matrix_alignment(s @ F_t.T, data.Y_train @ F_t.T))

    filename = "checkpoints_sgd_thm3/ReverseTrainValid59(0.50)-512-2/model_epoch_3000.pt"
    mr = ModelNetworkCE(p=p, N=512, from_file=filename)
    s = (data.Y_train + 1 / p) - mr.forward(data.X_valid)

    print(matrix_alignment(s @ F_v.T, mr.V))
    print(matrix_alignment(data.Y_train @ F_v.T, mr.V))
    print(matrix_alignment(s @ F_v.T, data.Y_train @ F_v.T))
