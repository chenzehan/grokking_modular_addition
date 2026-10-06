# using gradient descent
# using random generated mod data

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os
from model_plot_dis import DynamicMultiPlot
from mod_tensors import ModAddDataTrainValid
from mod_tensors import ModAddDataRandom

torch.manual_seed(42)

p = 59
hidden_size = 512
fraction_train = 0.49
train_select = 29
# fraction_train = train_select / p
lr = 2
weight_decay = 2e-4
# betas = (0.9, 0.98)
epochs = 30001
checkpoint_interval = 50
batch_number = 4

os.chdir(os.path.dirname(__file__))
# filename = f"checkpoints_sgd_thm3/multilayer/{p}({train_select})-{hidden_size}-4"
filename = f"checkpoints_sgd_thm3/random/{p}({fraction_train:.2f})-{hidden_size}-2"
# filename = f"checkpoints_sgd_thm3/aligns/{p}({train_select})-{hidden_size}-2"
# filename = f"checkpoints_sgd_thm3/sqrs/ref{p}({fraction_train:.2f})-{hidden_size}-2"
# filename = f"checkpoints_special_value/target{p}({fraction_train:.2f})-{hidden_size}-2"
os.makedirs(filename, exist_ok=True)

def pairs_to_tensor(pairs, p):
    inputs = []
    targets = []
    
    for a, b, result in pairs:
        a_one_hot = nn.functional.one_hot(torch.tensor(a), num_classes=p).float()
        b_one_hot = nn.functional.one_hot(torch.tensor(b), num_classes=p).float()
        input_tensor = torch.cat([a_one_hot, b_one_hot])
        inputs.append(input_tensor)
        targets.append(result)
    
    return torch.stack(inputs), torch.tensor(targets)

def compute_weight_norms(model):
    total_norm = 0.0
    for param in model.parameters():
        if param.requires_grad:
            total_norm += param.norm(2).item() ** 2
    return total_norm ** 0.5

def compute_layerwise_norms(model):
    layer_norms = {}
    for name, param in model.named_parameters():
        if param.requires_grad and 'weight' in name:
            layer_norms[name] = param.norm(2).item()
    return layer_norms

class SqrActivation(nn.Module):
    def __init__(self):
        super(SqrActivation, self).__init__()
    def forward(self, x):
        return x * x
    
class SqrActivation_rl(nn.Module):
    def __init__(self, t):
        super(SqrActivation_rl, self).__init__()
        self.t = t
    def forward(self, x):
        return (x + self.t) * (x + self.t) / (4 * self.t)

class TanhReLU(nn.Module):
    def forward(self, x):
        return torch.tanh(torch.relu(x))

def plot_grokking_with_norms(train_accuracies, test_accuracies, weight_norms, layer_norms_history):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs_acc = np.arange(len(train_accuracies)) * 100
    axes[0, 0].plot(epochs_acc, train_accuracies, label='Train Accuracy', linewidth=2)
    axes[0, 0].plot(epochs_acc, test_accuracies, label='Test Accuracy', linewidth=2)
    axes[0, 0].set_title('Grokking Phenomenon')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    epochs_norm = np.arange(len(weight_norms)) * 100
    axes[0, 1].plot(epochs_norm, weight_norms, color='red', linewidth=2)
    axes[0, 1].set_title('Total Weight L2 Norm')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('L2 Norm')
    axes[0, 1].grid(True, alpha=0.3)
    
    for name, norms in layer_norms_history.items():
        short_name = name.replace('.weight', '').replace('network.', '')
        axes[1, 0].plot(epochs_norm, norms, label=short_name, alpha=0.8)
    axes[1, 0].set_title('Layer-wise Weight Norms')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('L2 Norm')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].scatter(weight_norms, test_accuracies, c=epochs_norm, 
                       cmap='viridis', alpha=0.6, s=20)
    axes[1, 1].set_xlabel('Weight L2 Norm')
    axes[1, 1].set_ylabel('Test Accuracy')
    axes[1, 1].set_title('Test Accuracy vs Weight Norm')
    axes[1, 1].grid(True, alpha=0.3)
    plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1], label='Epoch')
    
    plt.tight_layout()
    plt.show()

def train_with_balanced_data(bias=True, show_plot="acc", save_model=False, save_final=True, restore_from=None, data_seed=42, monitor=None, nn_seed=42, **kwargs):
    if True:
        p = kwargs.get('p', globals()['p'])
        hidden_size = kwargs.get('hidden_size', globals()['hidden_size'])
        fraction_train = kwargs.get('fraction_train', globals()['fraction_train'])
        train_select = kwargs.get('train_select', globals()['train_select'])
        lr = kwargs.get('lr', globals()['lr'])
        weight_decay = kwargs.get('weight_decay', globals()['weight_decay'])
        epochs = kwargs.get('epochs', globals()['epochs'])
        checkpoint_interval = kwargs.get('checkpoint_interval', globals()['checkpoint_interval'])
        batch_number = kwargs.get('batch_number', globals()['batch_number'])
        print(f"Using parameters: p={p}, hidden_size={hidden_size}, fraction_train={fraction_train}, train_select={train_select}, lr={lr}, weight_decay={weight_decay}, epochs={epochs}, checkpoint_interval={checkpoint_interval}, batch_number={batch_number}")

    input_size = 2 * p
    output_size = p
    
    data = ModAddDataTrainValid(p=p, select=train_select, rand_seed=data_seed)
    # data = ModAddDataRandom(p=p, fraction=fraction_train, rand_seed=data_seed)
    train_pairs, test_pairs = data.get_as_pairs_tv()
    X_train, y_train = pairs_to_tensor(train_pairs, p)
    X_test, y_test = pairs_to_tensor(test_pairs, p)
    # set special value
    # X_train[0] /= 2
    # y_train[0] = 1
    
    # 模型
    if nn_seed is not None:
        torch.manual_seed(nn_seed)
    model = nn.Sequential(
        nn.Linear(input_size, hidden_size, bias=bias),
        # SqrActivation(),
        nn.ReLU(),
        # TanhReLU(),
        nn.Linear(hidden_size, output_size, bias=bias)
    )
    nn.init.zeros_(model[2].weight)
    if model[2].bias is not None:
        nn.init.zeros_(model[2].bias)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 训练
    batch_size = int(np.ceil(X_train.size(0) / batch_number))
    
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    test_losses = []
    weight_norms = []
    layer_norms_history = defaultdict(list)
    
    print("Start...")
    start_epoch = 0

    model.eval()
    with torch.no_grad():
        train_outputs = model(X_train)
        train_loss = criterion(train_outputs, y_train)
        _, train_pred = torch.max(train_outputs, 1)
        train_acc = (train_pred == y_train).float().mean()
        
        test_outputs = model(X_test)
        _, test_pred = torch.max(test_outputs, 1)
        test_acc = (test_pred == y_test).float().mean()
        test_loss = criterion(test_outputs, y_test)
        
        train_losses.append(train_loss.item())
        train_accuracies.append(train_acc.item())
        test_accuracies.append(test_acc.item())
        test_losses.append(test_loss.item())
        total_norm = compute_weight_norms(model)
        weight_norms.append(total_norm)

        layer_norms = compute_layerwise_norms(model)
        for name, norm in layer_norms.items():
            layer_norms_history[name].append(norm)

        if monitor is not None:
            monitor(-1, train_loss.item(), test_loss.item(), train_acc, test_acc, model)

        print(f'Epoch {-1:5d}: Train Loss={train_loss.item():.4f}, Test Loss={test_loss.item():.4f}, '
            f'Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}, '
            f'Weight Norm: {total_norm:.4f}')
        print(f'{train_loss.item():.4f}, {test_loss.item():.4f}, {train_acc:.4f}, {test_acc:.4f}, {total_norm:.4f}')
                
    return train_loss.item(), test_loss.item(), train_acc, test_acc

if __name__ == "__main__":
    
    os.makedirs(filename, exist_ok=True)

    train_loss, test_loss, train_accuracy, test_accuracy = train_with_balanced_data(bias=False, save_model=True, show_plot="loss", save_final=True)
    d = np.loadtxt(f"{filename}/results.txt", delimiter=',')
    d = np.vstack((np.array([train_loss, test_loss, train_accuracy, test_accuracy]), d))
    np.savetxt(f"{filename}/loss_acc.txt", d, header=f"Train Loss, Test Loss, Train Accuracy, Test Accuracy, fraction={fraction_train}, interval={checkpoint_interval}", delimiter=" ")
