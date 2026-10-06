from add_mod_2_sgd_thm3_gpu import *

if __name__ == "__main__":
    # N = 512
    # p = 59
    # for lmb in np.arange(0.5e-4, 4.1e-4, 0.5e-4):
    #     q = p // 2
    #     filename = f"checkpoints_sgd_thm3/scaling/[lambda={lmb:.2e}]{p}({q})-{N}-2"
                    
    #     os.makedirs(filename, exist_ok=True)
    #     Losses = []
    #     for seed in range(10):
    #         train_losses, train_accuracies, test_losses, test_accuracies = train_with_balanced_data(bias=False, save_model=False, show_plot=None, save_final=False, hidden_size=N, p=p, train_select=q, weight_decay=lmb, checkpoint_interval=1000, data_seed=seed)
    #         final_loss = train_losses[-1]
    #         Losses.append(final_loss)
    #     np.savetxt(f"{filename}/seeded_final_loss.txt", Losses)
        
    # N = 512
    # p = 59
    # for p in [37, 47, 59, 67, 89, 97, 113, 131]:
    #     q = p // 2
    #     filename = f"checkpoints_sgd_thm3/scaling/{p}({q})-{N}-2"
                    
    #     os.makedirs(filename, exist_ok=True)
    #     Losses = []
    #     for seed in range(10):
    #         train_losses, train_accuracies, test_losses, test_accuracies = train_with_balanced_data(bias=False, save_model=False, show_plot=None, save_final=False, hidden_size=N, p=p, train_select=q, weight_decay=2e-4, checkpoint_interval=1000, data_seed=seed)
    #         final_loss = train_losses[-1]
    #         Losses.append(final_loss)
    #     np.savetxt(f"{filename}/seeded_final_loss.txt", Losses)

    N = 512
    p = 113
    for alpha in np.arange(0.3, 0.71, 0.1):
        q = int(p * alpha)
        if q == 56:
            continue
        filename = f"checkpoints_sgd_thm3/scaling/{p}({q})-{N}-2"
                    
        os.makedirs(filename, exist_ok=True)
        Losses = []
        for seed in range(10):
            train_losses, train_accuracies, test_losses, test_accuracies = train_with_balanced_data(bias=False, save_model=False, show_plot=None, save_final=False, hidden_size=N, p=p, train_select=q, weight_decay=2e-4, checkpoint_interval=1000, data_seed=seed)
            final_loss = train_losses[-1]
            Losses.append(final_loss)
        np.savetxt(f"{filename}/seeded_final_loss.txt", Losses)

    N = 2048
    p = 113
    for alpha in np.arange(0.3, 0.71, 0.1):
        q = int(p * alpha)
        filename = f"checkpoints_sgd_thm3/scaling/{p}({q})-{N}-2"
                    
        os.makedirs(filename, exist_ok=True)
        Losses = []
        for seed in range(10):
            train_losses, train_accuracies, test_losses, test_accuracies = train_with_balanced_data(bias=False, save_model=False, show_plot=None, save_final=False, hidden_size=N, p=p, train_select=q, weight_decay=2e-4, checkpoint_interval=1000, data_seed=seed)
            final_loss = train_losses[-1]
            Losses.append(final_loss)
        np.savetxt(f"{filename}/seeded_final_loss.txt", Losses)
    