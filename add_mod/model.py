from mod_tensors import *
import numpy as np
from matplotlib import pyplot as plt
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

class rand_init():
    def __init__(self, p, mean=0, std=1, rand_seed=42):
        self.p = p
        self.mean = mean
        self.std = std
        self.rand_seed = rand_seed
        self.rng = np.random.default_rng(self.rand_seed)
    def __call__(self):
        return self.rng.normal(loc=self.mean, scale=self.std, size=2 * self.p)

class zero_init():
    def __init__(self, p):
        self.p = p
    def __call__(self):
        return np.zeros((2 * p))

class cos_init():
    def __init__(self, p, rand_seed=42):
        self.p = p
        self.rand_seed = rand_seed
        self.rng = np.random.default_rng(self.rand_seed)
    def __call__(self):
        n = self.rng.integers(low=1, high=self.p)
        c = self.rng.choice([0, 1])
        cs = self.rng.choice([0, 1])
        s = self.rng.choice([-1, 1])
        if c > 0:
            w_a = np.cos(2 * np.pi * n * np.arange(self.p) / self.p)
        else:
            w_a = np.sin(2 * np.pi * n * np.arange(self.p) / self.p)
        if cs > 0:
            w_b = np.cos(2 * np.pi * n * np.arange(self.p) / self.p)
        else:
            w_b = np.sin(2 * np.pi * n * np.arange(self.p) / self.p)
        return np.concatenate([w_a, s * w_b])

class cos_init_mix():
    def __init__(self, p, rand_seed=42):
        self.p = p
        self.rand_seed = rand_seed
        self.rng = np.random.default_rng(self.rand_seed)
    def __call__(self):
        n1, n2 = self.rng.permutation(self.p)[:2]
        c = self.rng.choice([0, 1])
        cs = self.rng.choice([0, 1])
        s = self.rng.choice([-1, 1])
        if c > 0:
            w_a = np.cos(2 * np.pi * n1 * np.arange(self.p) / self.p)
        else:
            w_a = np.sin(2 * np.pi * n1 * np.arange(self.p) / self.p)
        if cs > 0:
            w_b = np.cos(2 * np.pi * n1 * np.arange(self.p) / self.p)
        else:
            w_b = np.sin(2 * np.pi * n1 * np.arange(self.p) / self.p)

        w1 = np.concatenate([w_a, s * w_b])
        c = self.rng.choice([0, 1])
        s = self.rng.choice([-1, 1])
        if c > 0:
            w_a = np.cos(2 * np.pi * n2 * np.arange(self.p) / self.p)
        else:
            w_a = np.sin(2 * np.pi * n2 * np.arange(self.p) / self.p)
        if cs > 0:
            w_b = np.cos(2 * np.pi * n2 * np.arange(self.p) / self.p)
        else:
            w_b = np.sin(2 * np.pi * n2 * np.arange(self.p) / self.p)
        w2 = np.concatenate([w_a, s * w_b])

        f = self.rng.uniform(0, 1)
        return f * w1 + (1 - f) * w2

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
        print(f"模型检查点不存在: {model_path}")
        return None
    
    model.load_state_dict(torch.load(model_path))
    
    return model

class ModelCE():
    def __init__(self, p, N, sigma=relu, w_init=None, from_file=None):
        self.p = p
        self.N = N
        self.sigma = sigma
        if from_file is not None:
            m = get_model_from(from_file, p, N)
            self.W = m[0].weight.data.numpy()
            return
        if w_init is None:
            self.w_init = rand_init(p)
        else:
            self.w_init = w_init
        self.W = np.column_stack([self.w_init() for _ in range(N)]).T
    def activate(self, X):
        return self.sigma(self.W @ X)
    def forward_logits(self, X, Y):
        F = self.activate(X)
        return Y @ F.T @ F
    def forward(self, X, Y):
        Z = self.forward_logits(X, Y)
        return column_softmax(Z)
    def forward_loss(self, X, Y): # zero-mean Y
        P = self.forward(X, Y)
        return -np.sum((Y + 1 / self.p) * np.log(P + 1e-12)) / Y.shape[1]

# class cos_half():
#     def __init__(self, p, rand_seed=42):
#         self.p = p
#         self.rand_seed = rand_seed
#         self.n = 0
#     def __call__(self):
#         c = (self.n % 4) // 2
#         s = self.n % 2 * 2 - 1
#         n = (self.n // 4) + 1
#         self.n += 1
#         if c > 0:
#             w = np.cos(2 * np.pi * n * np.arange(self.p) / self.p)
#             return np.concatenate([w, s * w])
#         else:
#             w = np.sin(2 * np.pi * n * np.arange(self.p) / self.p)
#             return np.concatenate([w, -s * w])

class fourier_complete():
    def __init__(self, p, rand_seed=42):
        self.p = p
        self.rand_seed = rand_seed
        self.n = 0
    def __call__(self):
        c = (self.n % 8) // 2
        s = self.n % 2 * 2 - 1
        cs = c % 2
        c = c // 2
        n = (self.n // 8) + 1
        self.n += 1
        # w1 = np.cos(2 * np.pi * n * np.arange(self.p) / self.p + np.pi * c / 2)
        # w2 = np.cos(2 * np.pi * n * np.arange(self.p) / self.p - np.pi * c / 2)
        wc = np.cos(2 * np.pi * n * np.arange(self.p) / self.p)
        ws = np.sin(2 * np.pi * n * np.arange(self.p) / self.p)
        if cs > 0:
            if c > 0:
                w1 = wc
                w2 = ws
            else:
                w1 = ws
                w2 = wc
        else:
            if c > 0:
                w1 = wc
                w2 = -ws
            else:
                w1 = -ws
                w2 = wc
        
        return np.concatenate([w1, s * w2])

class fourier_init_rand_phase():
    def __init__(self, p, rand_seed=42):
        self.p = p
        self.rand_seed = rand_seed
        self.rng = np.random.default_rng(self.rand_seed)
    def __call__(self):
        n = self.rng.integers(low=1, high=self.p)
        
        phi1 = self.rng.uniform(0, 2 * np.pi)
        phi2 = self.rng.uniform(0, 2 * np.pi)

        return np.concatenate([
            np.cos(2 * np.pi * n * np.arange(self.p) / self.p + phi1),
            np.cos(2 * np.pi * n * np.arange(self.p) / self.p + phi2)
        ])

class fourier_rand_phase_paired():
    def __init__(self, p, rand_seed=42):
        self.p = p
        self.rand_seed = rand_seed
        self.rng = np.random.default_rng(self.rand_seed)
    def __call__(self):
        n = self.rng.integers(low=1, high=self.p)
        
        phi1 = self.rng.uniform(0, 2 * np.pi)
        s2 = self.rng.integers(0, 2)
        return np.concatenate([
            np.cos(2 * np.pi * n * np.arange(self.p) / self.p + phi1),
            np.cos(2 * np.pi * n * np.arange(self.p) / self.p + phi1 + s2 * np.pi / 2)
        ])

class fourier_init_rand_phase_mix():
    def __init__(self, p, rand_seed=42):
        self.p = p
        self.rand_seed = rand_seed
        self.rng = np.random.default_rng(self.rand_seed)
    def __call__(self):
        nx = self.rng.permutation(self.p)[:5]
        n1 = nx[0]
        # c = self.rng.choice([0, 1])
        # cs = self.rng.choice([0, 1])
        # s = self.rng.choice([-1, 1])
        # if c > 0:
        #     w_a = np.cos(2 * np.pi * n1 * np.arange(self.p) / self.p)
        # else:
        #     w_a = np.sin(2 * np.pi * n1 * np.arange(self.p) / self.p)
        # if cs > 0:
        #     w_b = np.cos(2 * np.pi * n1 * np.arange(self.p) / self.p)
        # else:
        #     w_b = np.sin(2 * np.pi * n1 * np.arange(self.p) / self.p)
        # w1 = np.concatenate([w_a, s * w_b])
        
        phi1 = self.rng.uniform(0, 2 * np.pi)
        phi2 = self.rng.uniform(0, 2 * np.pi)
        w1 = np.concatenate([
            np.cos(2 * np.pi * n1 * np.arange(self.p) / self.p + phi1),
            np.cos(2 * np.pi * n1 * np.arange(self.p) / self.p + phi2)
        ])
        w2 = self.rng.normal(loc=0, scale=1, size=2 * self.p)

        d = self.rng.uniform(0, 1)
        f = 1 if d > 0.7 else 0
        # f = d
        return f * w1 + (1 - f) * w2

class rand_double_init():
    def __init__(self, p, bias=[-1,1], std=0.5, rand_seed=42):
        self.p = p
        self.bias = bias
        self.std = std
        self.rand_seed = rand_seed
        self.rng = np.random.default_rng(self.rand_seed)
    def __call__(self):
        return self.rng.normal(loc=0, scale=self.std, size=2 * self.p) + self.rng.choice(self.bias, size=2*p)

class multiple_fourier_init_mix():
    def __init__(self, p, n=5, rand_seed=42):
        self.p = p
        self.n = n
        self.rand_seed = rand_seed
        self.rng = np.random.default_rng(self.rand_seed)
        self.d_list = []
    def __call__(self):
        nx = (self.rng.permutation(self.p - 1) + 1)[:self.n]
        
        w_list = []
        
        for n in nx:
            phi1 = self.rng.uniform(0, 2 * np.pi)
            phi2 = self.rng.uniform(0, 2 * np.pi)
            w1 = np.concatenate([
                np.cos(2 * np.pi * n * np.arange(self.p) / self.p + phi1),
                np.cos(2 * np.pi * n * np.arange(self.p) / self.p + phi2)
            ])
            w_list.append(w1)

        d = self.rng.uniform(0, 1, size=self.n)
        d = d / np.linalg.norm(d)
        self.d_list.append(d)
        return np.sum(np.multiply(d, np.array(w_list).T), axis=-1)
    def count_freq(self, threshold=0.5):
        count = 0
        for d in self.d_list:
            c = d**2 / np.sum(d**2)
            if np.max(c) > threshold:
                count += 1
        return count

class decay_fourier_init_mix():
    def __init__(self, p, eta=0.6, rand_seed=42):
        self.p = p
        self.lb = eta
        self.rand_seed = rand_seed
        self.rng = np.random.default_rng(self.rand_seed)
        self.d_list = []
    def __call__(self):
        m = self.p // 2
        nx = (self.rng.permutation(m) + 1)
        
        w_list = []
        
        for n in nx:
            phi1 = self.rng.uniform(0, 2 * np.pi)
            phi2 = self.rng.uniform(0, 2 * np.pi)
            w1 = np.concatenate([
                np.cos(2 * np.pi * n * np.arange(self.p) / self.p + phi1),
                np.cos(2 * np.pi * n * np.arange(self.p) / self.p + phi2)
            ])
            w_list.append(w1)

        d = self.rng.uniform(0, 1, size=m)
        # d = np.exp(-self.lb * np.arange(m)) * d
        d = 1 / ((np.arange(m) + 1) ** self.lb) * d
        d = d / np.linalg.norm(d)
        self.d_list.append(d)
        return np.sum(np.multiply(d, np.array(w_list).T), axis=-1)
    def count_freq(self, threshold=0.5):
        count = 0
        for d in self.d_list:
            c = d**2 / np.sum(d**2)
            if np.max(c) > threshold:
                count += 1
        return count

def Lanczos_factor(n, N):
    return np.sin(n*np.pi/N) / (n*np.pi/N)
def sqrwave_expansion(x, omega, phi, n=6):
    y = 0
    for i in range(n):
        k = 2*i + 1
        y += Lanczos_factor(k, n) * np.sin(k * (omega * x + phi)) / k
    return y * (4 / np.pi)
def sqrwave(x, omega, phi):
    return np.sign(np.cos(omega * x + phi))

class sqrwave_init_rand_phase():
    def __init__(self, p, order=4, rand_seed=42):
        self.p = p
        self.n = order
        self.rand_seed = rand_seed
        self.rng = np.random.default_rng(self.rand_seed)
    def __call__(self):
        n = self.rng.integers(low=1, high=self.p)
        
        phi1 = self.rng.uniform(0, 2 * np.pi)
        phi2 = self.rng.uniform(0, 2 * np.pi)

        return np.concatenate([
            sqrwave(np.arange(self.p), 2 * np.pi * n / self.p, phi1) / np.sqrt(2),
            sqrwave(np.arange(self.p), 2 * np.pi * n / self.p, phi2) / np.sqrt(2)
        ])

class multiple_sqrwave_init_mix():
    def __init__(self, p, n=5, order=4, rand_seed=42):
        self.p = p
        self.n = n
        self.order = order
        self.rand_seed = rand_seed
        self.rng = np.random.default_rng(self.rand_seed)
    def __call__(self):
        nx = (self.rng.permutation(self.p - 1) + 1)[:self.n]
        
        w_list = []
        
        for n in nx:
            phi1 = self.rng.uniform(0, 2 * np.pi)
            phi2 = self.rng.uniform(0, 2 * np.pi)
            w1 = np.concatenate([
                sqrwave(np.arange(self.p), 2 * np.pi * n / self.p, phi1, n=self.order),
                sqrwave(np.arange(self.p), 2 * np.pi * n / self.p, phi2, n=self.order)
            ])
            w_list.append(w1)

        d = self.rng.uniform(0, 1, size=self.n)
        d = d / np.linalg.norm(d)
        # self.d_list.append(d)
        return np.sum(np.multiply(d, np.array(w_list).T), axis=-1)

def cos_similarity(A, B, zero_as_aligned=True):
    A_n = np.linalg.norm(A)
    B_n = np.linalg.norm(B)
    if zero_as_aligned:
        if A_n < 1e-8 or B_n < 1e-8:
            return 1
    return np.sum(A * B) / A_n / B_n
def cos_similarity_ax(A, B, axis=None):
    return np.sum(A * B, axis=axis) / np.linalg.norm(A, axis=axis) / np.linalg.norm(B, axis=axis)

from numpy.polynomial.hermite import Hermite
def f(x):
    # return x*x / 2 + x
    return Hermite([0, 0, 1])(x)
    h_coeff = [0.39894228, 0.39894228, 0.19947114, -0.06649038]
    return Hermite(h_coeff)(x)

if __name__ == "__main__":
    filename = "checkpoints_sgd_thm3/59(0.50)-512-2/model_epoch_300.pt"
    p = 59
    data = ModAddDataTrainValid(p, select=p//2)
    m = ModelCE(p, N=256, w_init=fourier_init_rand_phase(p), sigma=relu, from_file=None)
    
    # for w in m.W:
    #     f1 = np.abs(np.fft.fft(w[:p])[:p//2])
    #     f2 = np.abs(np.fft.fft(w[p:])[:p//2])
    #     plt.plot(f1)
    #     plt.plot(f2)
    #     plt.show()

    # print(np.sum(m.W))
    # from grokking_simulation import grad_update_W, grad_update_W_init
    # G_W = grad_update_W_init(m, data.X_train, data.Y_train, relu_derivative)
    # m.W += G_W * 0.01
    F_t = m.activate(data.X_train)
    F_v = m.activate(data.X_valid)
    # (lambda x, y: print(x, y, y / x))(np.linalg.norm(m.W), np.linalg.norm(F_t @ data.Y_train.T))
    F = m.activate(data.X)
    # print(np.sum(np.abs(F_t))/np.prod(F_t.shape))
    # G_ft = F_t @ data.Y_train.T @ data.Y_train
    # print(np.sum(np.abs(G_ft))/np.prod(F_t.shape))
    # G_ft[F_t <= 0] = 0
    # F_t += G_ft * 0.1
    # plt.imshow(G_ft)
    # plt.colorbar()
    # plt.show()
    # plt.hist(m.W.flatten(), bins=100)
    # plt.show()
    K = F.T @ F
    K = K - np.mean(K)
    # plt.hist(np.diag(K), bins=50)
    # plt.show()
    # K[0, 0] = 0
    plt.imshow(K)
    plt.colorbar()
    plt.show()
    # P = column_softmax(data.Y @ K)
    # loss = np.sum(-(data.Y + 1 / p) * np.log(P + 1e-12)) / data.Y.shape[1]
    # acc = np.sum([int(np.argmax(a) == np.argmax(b)) for a, b in zip(data.Y.T, P.T)]) / data.Y.shape[1]
    # print(acc, loss)

    # G_inv = np.diag(np.diag(np.linalg.inv(F_t @ F_t.T + 1e-6)))
    # K = F_t.T @ G_inv @ F_t
    # K = F_t.T @ (np.linalg.inv(F_t @ F_t.T + 1e-6)) @ F_t
    K = F_t.T @ F_t
    K = K - np.mean(K) #+ np.eye(*K.shape) * 250
    plt.imshow(K)
    plt.colorbar()
    plt.show()
    P = column_softmax(data.Y_train @ K)
    loss = np.sum(-(data.Y_train + 1 / p) * np.log(P + 1e-12)) / data.Y_train.shape[1]
    acc = np.sum([int(np.argmax(a) == np.argmax(b)) for a, b in zip(data.Y_train.T, P.T)]) / data.Y_train.shape[1]
    print(acc, loss)

    # K = F_t.T @ np.linalg.inv(F_t @ F_t.T + 1e-6) @ F_v
    K = F_t.T @ F_v
    K = K - np.mean(K)
    plt.imshow(K)
    plt.colorbar()
    plt.show()
    P = column_softmax(data.Y_train @ K)
    loss = np.sum(-(data.Y_train + 1 / p) * np.log(P + 1e-12)) / data.Y_train.shape[1]
    acc = np.sum([int(np.argmax(a) == np.argmax(b)) for a, b in zip(data.Y_train.T, P.T)]) / data.Y_train.shape[1]
    print(acc, loss)

    K = F_v.T @ F_v
    K = K - np.mean(K)
    plt.imshow(K)
    plt.colorbar()
    plt.show()
    P = column_softmax(data.Y_train @ K)
    loss = np.sum(-(data.Y_train + 1 / p) * np.log(P + 1e-12)) / data.Y_train.shape[1]
    acc = np.sum([int(np.argmax(a) == np.argmax(b)) for a, b in zip(data.Y_train.T, P.T)]) / data.Y_train.shape[1]
    print(acc, loss)
