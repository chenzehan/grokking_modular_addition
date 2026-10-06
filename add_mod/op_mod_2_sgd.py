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
from mod_tensors import ModOpDataTrainValid

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
# filename = f"checkpoints_sgd_thm3/tasks/mul/{p}({train_select})-{hidden_size}-2"
filename = f"checkpoints_sgd_thm3/tasks/add_fourier_init/{p}({train_select})-{hidden_size}-2"
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

def train_with_balanced_data(data: ModOpDataTrainValid, init_from_model_ce=None, bias=True, show_plot="acc", save_model=False, save_final=True, restore_from=None, data_seed=42, nn_seed=42, monitor=None, **kwargs):
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

    input_size = 2 * data.size
    output_size = data.size
    
    train_pairs, test_pairs = data.get_as_pairs_tv()
    X_train, y_train = pairs_to_tensor(train_pairs, data.size)
    X_test, y_test = pairs_to_tensor(test_pairs, data.size)
    # set special value
    # X_train[0] /= 2
    # y_train[0] = 1
    
    if nn_seed is not None:
        torch.manual_seed(nn_seed)
    model = nn.Sequential(
        nn.Linear(input_size, hidden_size, bias=bias),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size, bias=bias)
    )
    nn.init.zeros_(model[2].weight)
    if model[2].bias is not None:
        nn.init.zeros_(model[2].bias)
    if init_from_model_ce is not None:
        with torch.no_grad():
            model[0].weight.copy_(torch.from_numpy(init_from_model_ce.W))
            if model[0].bias is not None:
                nn.init.zeros_(model[0].weight)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    batch_size = int(np.ceil(X_train.size(0) / batch_number))
    
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    test_losses = []
    weight_norms = []
    layer_norms_history = defaultdict(list)

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
            if isinstance(monitor, list):
                for mntr in monitor:
                    mntr(-1, train_loss.item(), test_loss.item(), train_acc, test_acc, model)
            else:
                monitor(-1, train_loss.item(), test_loss.item(), train_acc, test_acc, model)

        print(f'Epoch {-1:5d}: Train Loss={train_loss.item():.4f}, Test Loss={test_loss.item():.4f}, '
            f'Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}, '
            f'Weight Norm: {total_norm:.4f}')
        print(f'{train_loss.item():.4f}, {test_loss.item():.4f}, {train_acc:.4f}, {test_acc:.4f}, {total_norm:.4f}')
    
    print("Start...")
    start_epoch = 0
    if restore_from is not None:
        start_epoch = restore_from + 1
        os.makedirs(filename, exist_ok=True)
        if os.path.exists(f"{filename}/model_epoch_{restore_from}.pt"):
            model.load_state_dict(torch.load(f"{filename}/model_epoch_{restore_from}.pt"))
            print(f"recover {restore_from}")
    
    if show_plot is not None:
        dashboard = DynamicMultiPlot((1, 1), figsize=(10, 6))
        if show_plot == "loss":
            Loss_plot_id = dashboard.add_curve(title="loss", labels=["train_loss", "val_loss"], position=(0, 0))
        elif show_plot == "acc":
            Loss_plot_id = dashboard.add_curve(title="acc", labels=["train_acc", "val_acc"], position=(0, 0))

    for epoch in range(start_epoch, epochs):
        model.train()
        
        indices = torch.randperm(X_train.size(0))
        for i in range(0, X_train.size(0), batch_size):
            batch_indices = indices[i:i+batch_size]
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if epoch % checkpoint_interval == 0:
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
                    if isinstance(monitor, list):
                        for mntr in monitor:
                            mntr(epoch, train_loss.item(), test_loss.item(), train_acc, test_acc, model)
                    elif not monitor(epoch, train_loss.item(), test_loss.item(), train_acc, test_acc, model):
                        break

                if save_model:
                    checkpoint_path = f"{filename}/model_epoch_{epoch}.pt"
                    torch.save(model.state_dict(), checkpoint_path)
                
                if save_final and not save_model and epoch == epochs - 1:
                    checkpoint_path = f"{filename}/model_epoch_{epoch}.pt"
                    torch.save(model.state_dict(), checkpoint_path)
                
                print(f'Epoch {epoch:5d}: Train Loss={train_loss.item():.4f}, Test Loss={test_loss.item():.4f}, '
                  f'Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}, '
                  f'Weight Norm: {total_norm:.4f}')
                
                if show_plot is not None:
                    if show_plot == "loss":
                        dashboard.update_curve(Loss_plot_id, {'train_loss': train_loss.item(), 'val_loss': test_loss.item()})
                    elif show_plot == "acc":
                        dashboard.update_curve(Loss_plot_id, {'train_acc': train_acc, 'val_acc': test_acc})
                    dashboard.refresh()

    if monitor is not None:
        if isinstance(monitor, list):
            for mntr in monitor:
                if hasattr(mntr, "end"):
                    mntr.end()
        elif hasattr(monitor, "end"):
            monitor.end()
    if show_plot is not None:
        dashboard.close()

    return train_losses, train_accuracies, test_losses, test_accuracies

if __name__ == "__main__":
    
    os.makedirs(filename, exist_ok=True)
    data = ModOpDataTrainValid(p, np.arange(p), lambda x, y: x + y, train_select)
    from model import ModelCE, fourier_init_rand_phase
    m = ModelCE(p, hidden_size, w_init=fourier_init_rand_phase(p))

    train_losses, train_accuracies, test_losses, test_accuracies = train_with_balanced_data(data, init_from_model_ce=m, bias=False, save_model=False, show_plot="acc", save_final=False, checkpoint_interval=2, epochs=301)
    np.savetxt(f"{filename}/loss_acc.txt", np.array([train_losses, test_losses, train_accuracies, test_accuracies]).T, header=f"#Train Loss, Test Loss, Train Accuracy, Test Accuracy, fraction={fraction_train}, interval={checkpoint_interval}", delimiter=" ")
