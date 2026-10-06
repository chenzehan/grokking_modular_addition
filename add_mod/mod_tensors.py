import numpy as np
import torch
import torch.nn as nn
import itertools

class ModAddDataComplete():
    def __init__(self, p):
        self.p = p
        self.M = p * p
        # Y is zero-mean target
        self.Y = np.kron(np.eye(p), np.ones(p)) - 1 / p
        self.gen_X()
    def gen_X(self):
        self._P = np.roll(np.fliplr(np.eye(self.p)), 1, axis=1)
        self.X = np.concatenate(
            [np.vstack((np.roll(np.eye(self.p), i, axis=1).T, self._P)) for i in range(self.p)], 
            axis=1
        )
    def get_as_pairs(self):
        if hasattr(self, "pairs"):
            return self.pairs
        self.pairs = [((a + h) % self.p, (self.p - a) % self.p, h) for h, a in itertools.product(range(self.p), range(self.p))]
        return self.pairs

class ModOpDataComplete():
    def __init__(self, p, H, f):
        self.H = H
        self.p = p
        self.size = len(H)
        self.M = p * p
        self.f = f
        self.id = dict(zip(H, list(range(len(H)))))
        self.gen_XY()
    def gen_XY(self):
        H_H = np.stack([g.ravel() for g in np.meshgrid(self.H, self.H, indexing='ij')], axis=-1)
        from collections import defaultdict
        self.d = defaultdict(list)
        for a, b in H_H:
            h = self.f(a, b) % self.p
            if h not in self.H:
                raise ValueError("results not in H")
            self.d[h].append((a, b))

        def set_ones(l, h):
            m = np.zeros((self.size, l))
            m[self.id[h], :] = 1
            return m
        def set_double(a, b):
            m = np.zeros(self.size * 2)
            m[self.id[a]] = 1
            m[self.id[b] + self.size] = 1
            return m
        # Y is zero-mean target
        self.Y = np.hstack([set_ones(len(self.d[h]), h) for h in self.H]) - 1 / self.size
        self.X = np.stack([set_double(a, b) for h in self.H for a, b in self.d[h]]).T

        self.lens = [len(self.d[h]) for h in self.H]
        self.offset = np.cumsum([0] + self.lens)
        
    def get_as_pairs(self):
        if hasattr(self, "pairs"):
            return self.pairs
        self.pairs = [(self.id[a], self.id[b], self.id[h]) for h in self.H for a, b in self.d[h]]
        return self.pairs

class ModAddDataTrainValid(ModAddDataComplete):
    def __init__(self, p, select, rand_seed=42):
        super().__init__(p)
        if select > p // 2:
            print("WARNING: select exceeds p // 2")
            # raise ValueError("select must be less than p")
        self.select = select
        self.rand_seed = rand_seed
        self.gen_train_valid()
    def gen_train_valid(self):
        self.train_size = self.p * self.select
        rng = np.random.default_rng(self.rand_seed)
        self.selection = np.vstack([(rng.permutation(self.p) + i * self.p) for i in range(self.p)])
        self.train_selection = self.selection[:, :self.select].flatten()
        if self.select <= self.p // 2:
            self.valid_selection = self.selection[:, self.select: self.select*2].flatten()
        else:
            self.valid_selection = self.selection[:, -self.select:].flatten()
        self.X_train = self.X[:, self.train_selection]
        self.Y_train = self.Y[:, self.train_selection]
        self.X_valid = self.X[:, self.valid_selection]
        self.Y_valid = self.Y[:, self.valid_selection]
    def get_as_pairs_tv(self):
        if hasattr(self, "train_pairs"):
            return self.train_pairs, self.valid_pairs
        self.get_as_pairs()
        self.train_pairs = [self.pairs[i] for i in self.train_selection]
        self.valid_pairs = [self.pairs[i] for i in self.valid_selection]
        return self.train_pairs, self.valid_pairs

class ModAddDataRandom(ModAddDataComplete):
    def __init__(self, p, fraction, rand_seed=42):
        super().__init__(p)
        self.fraction = fraction
        self.train_size = int(fraction * (p * p))
        self.rand_seed = rand_seed
        self.gen_train_valid()
    def gen_train_valid(self):
        rng = np.random.default_rng(self.rand_seed)
        self.selection = rng.permutation(self.p * self.p)
        self.train_selection = self.selection[:self.train_size]
        self.valid_selection = self.selection[self.train_size:]
        self.X_train = self.X[:, self.train_selection]
        self.Y_train = self.Y[:, self.train_selection]
        self.X_valid = self.X[:, self.valid_selection]
        self.Y_valid = self.Y[:, self.valid_selection]
    def get_as_pairs_tv(self):
        if hasattr(self, "train_pairs"):
            return self.train_pairs, self.valid_pairs
        self.get_as_pairs()
        self.train_pairs = [self.pairs[i] for i in self.train_selection]
        self.valid_pairs = [self.pairs[i] for i in self.valid_selection]
        return self.train_pairs, self.valid_pairs
    def group_selection(self, selection):
        groups = {}
        for i in selection:
            key = i // self.p
            groups.setdefault(key, []).append(i)
        return groups
    def group_selection_train(self):
        return self.group_selection(self.selection[:self.train_size])
    def group_selection_valid(self):
        return self.group_selection(self.selection[self.train_size:])

class ModOpDataTrainValid(ModOpDataComplete):
    def __init__(self, p, H, f, select, rand_seed=42):
        super().__init__(p, H, f)
        self.select = select
        if select > np.min(self.lens) // 2:
            print("WARNING: select exceeds min(lens) // 2")
        if select > np.min(self.lens):
            print("WARNING: select exceeds min(lens)")
        self.rand_seed = rand_seed
        self.gen_train_valid()
    def gen_train_valid(self):
        self.train_size = self.size * self.select
        rng = np.random.default_rng(self.rand_seed)
        def split(permu, off):
            train_selection = permu[:self.select] + off
            valid_selection = permu[-self.select:] + off
            return [train_selection, valid_selection]
        self.selections = np.array([split(rng.permutation(l), off) for l, off in zip(self.lens, self.offset[:-1])])
        self.train_selection = np.concat(self.selections[:, 0])
        self.valid_selection = np.concat(self.selections[:, 1])
        self.X_train = self.X[:, self.train_selection]
        self.Y_train = self.Y[:, self.train_selection]
        self.X_valid = self.X[:, self.valid_selection]
        self.Y_valid = self.Y[:, self.valid_selection]
    def get_as_pairs_tv(self):
        if hasattr(self, "train_pairs"):
            return self.train_pairs, self.valid_pairs
        self.get_as_pairs()
        self.train_pairs = [self.pairs[i] for i in self.train_selection]
        self.valid_pairs = [self.pairs[i] for i in self.valid_selection]
        return self.train_pairs, self.valid_pairs

class ModOpDataRandom(ModOpDataComplete):
    def __init__(self, p, H, f, fraction, rand_seed=42):
        super().__init__(p, H, f)
        self.fraction = fraction
        self.train_size = int(fraction * (p * p))
        self.rand_seed = rand_seed
        self.gen_train_valid()
    def gen_train_valid(self):
        rng = np.random.default_rng(self.rand_seed)
        self.selection = rng.permutation(self.p * self.p)
        self.train_selection = self.selection[:self.train_size]
        self.valid_selection = self.selection[self.train_size:]
        self.X_train = self.X[:, self.train_selection]
        self.Y_train = self.Y[:, self.train_selection]
        self.X_valid = self.X[:, self.valid_selection]
        self.Y_valid = self.Y[:, self.valid_selection]
    def get_as_pairs_tv(self):
        if hasattr(self, "train_pairs"):
            return self.train_pairs, self.valid_pairs
        self.get_as_pairs()
        self.train_pairs = [self.pairs[i] for i in self.train_selection]
        self.valid_pairs = [self.pairs[i] for i in self.valid_selection]
        return self.train_pairs, self.valid_pairs
    def group_selection(self, selection):
        groups = {}
        for i in selection:
            key = i // self.p
            groups.setdefault(key, []).append(i)
        return groups
    def group_selection_train(self):
        return self.group_selection(self.selection[:self.train_size])
    def group_selection_valid(self):
        return self.group_selection(self.selection[self.train_size:])

class ReLU:
    def __call__(self, x):
        return np.maximum(0, x)
    def derivative(self, x):
        return (x > 0).astype(float)

class Sqr:
    def __call__(self, x):
        return x * x
    def derivative(self, x):
        return 2 * x

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def sqr(x):
    return x * x

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def column_softmax(x):
    e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return e_x / e_x.sum(axis=0, keepdims=True)

def f_normalize(w):
    y = w**2
    return y / np.sum(y)

if __name__ == "__main__":
    p = 59
    d = ModOpDataTrainValid(p, np.arange(1, p), lambda x, y: x**2 + y**2, 20)
    # d = ModOpDataTrainValid(p, np.arange(1, p), lambda x, y: x*y, 29)
    from matplotlib import pyplot as plt

    plt.imshow(d.Y_train + 1 / d.size)
    # plt.imshow(d.Y + 1 / d.size)
    plt.show()
    plt.imshow(d.X_train)
    plt.show()