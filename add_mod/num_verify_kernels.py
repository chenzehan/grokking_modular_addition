from model import *
from matplotlib import pyplot as plt

from pathlib import Path
ENV_DIR = Path(__file__).resolve().parent

def kernel_values(p, N, type="fourier"):
    data = ModAddDataComplete(p)
    X = data.X
    Y = data.Y
    m = ModelCE(p, N, w_init=fourier_init_rand_phase(p) if type == "fourier" 
                else sqrwave_init_rand_phase(p))
    F = m.activate(X)
    K = F.T @ F / N
    C = X.T @ X
    pairs = np.array(data.get_as_pairs())
    d = (pairs[:,0] - pairs[:,1]) % p
    D = np.eye(p)[d].T
    C_mod = D.T @ D
    Mask_diag = np.eye(p*p)
    Mask_h = np.kron(np.eye(p), np.ones((p, p)))
    Mash_feat = Mask_h - np.eye(p * p)
    Mask_coll = C - np.diag(np.diag(C))
    Mask_cmod = C_mod - C_mod * Mask_h
    Mask_bg = np.ones((p*p, p*p)) - Mask_h - Mask_coll - Mask_cmod
    Masks = [Mask_diag, Mash_feat, Mask_coll, Mask_cmod, Mask_bg]
    return [np.sum(K * mask) / np.sum(mask) for mask in Masks]

if __name__ == "__main__":

    names = ["diag", "feat", "coll", "cmod", "bg"]
    tests = [(59, 2048), (113, 4096), (151, 8192)]
    # tests = [(59, 2048)]
    kappas_f = [1/2, 2/np.pi**2, 1/6+5/(4*np.pi**2), 2/np.pi**2, 16/np.pi**4]
    kappas_s = [1/2, 1/6, 1/4, 1/6, 1/8]

    results = [kappas_s]
    for t in tests:
        results.append(kernel_values(*t, type="sqrwave"))
    results = np.array(results)
    print(results.T)
    np.savetxt(ENV_DIR.joinpath("num_verify_kernels_s.txt"), results.T, header="theory " + str(tests), delimiter=" ")
    np.savetxt(ENV_DIR.joinpath("num_verify_kernels_s_tab.txt"), results.T, fmt="%.3f", header="theory " + str(tests), delimiter=" | ")