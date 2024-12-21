import os
import torch
import numpy as np

def list_of_distances(X, Y):
    return torch.sum((torch.unsqueeze(X, dim=2) - torch.unsqueeze(Y.t(), dim=0)) ** 2, dim=1)

def make_one_hot(target, target_one_hot):
    target = target.view(-1,1)
    target_one_hot.zero_()
    target_one_hot.scatter_(dim=1, index=target, value=1.)

def makedir(path):
    '''
    if path does not exist in the file system, create it
    '''
    if not os.path.exists(path):
        os.makedirs(path)

def print_and_write(str, file):
    print(str)
    file.write(str + '\n')

def find_high_activation_crop(activation_map, percentile=95):
    threshold = np.percentile(activation_map, percentile)
    mask = np.ones(activation_map.shape)
    mask[activation_map < threshold] = 0
    lower_y, upper_y, lower_x, upper_x = 0, 0, 0, 0
    for i in range(mask.shape[0]):
        if np.amax(mask[i]) > 0.5:
            lower_y = i
            break
    for i in reversed(range(mask.shape[0])):
        if np.amax(mask[i]) > 0.5:
            upper_y = i
            break
    for j in range(mask.shape[1]):
        if np.amax(mask[:,j]) > 0.5:
            lower_x = j
            break
    for j in reversed(range(mask.shape[1])):
        if np.amax(mask[:,j]) > 0.5:
            upper_x = j
            break
    return lower_y, upper_y+1, lower_x, upper_x+1

import matplotlib.pyplot as plt

def plot_training_curves(metrics, model_dir):
    # Use the minimum length across all metrics to determine the range
    epochs = list(range(min(len(metrics["train_loss"]), len(metrics["train_accuracy"]), len(metrics["test_accuracy"]))))
    
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Plot Loss
    axs[0].plot(epochs, metrics["train_loss"][:len(epochs)], label="Loss", color='tab:blue', marker='o')
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].set_title("Loss Over Epochs")
    axs[0].set_xticks(epochs)
    axs[0].legend()
    axs[0].grid(True)

    # Plot Accuracy
    axs[1].plot(epochs, metrics["train_accuracy"][:len(epochs)], label="Train Accuracy", color='tab:green', marker='o')
    # Handle None values that might be in test_accuracy by replacing them with a marker
    axs[1].plot(epochs, [acc if acc is not None else 0 for acc in metrics["test_accuracy"][:len(epochs)]], label="Test Accuracy", color='tab:red', marker='o')
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy (%)")
    axs[1].set_title("Training and Testing Accuracy Over Epochs")
    axs[1].set_xticks(epochs)
    axs[1].legend()
    axs[1].grid(True)

    # Adjust layout and save the plot
    plt.tight_layout()
    plot_path = os.path.join(model_dir, "train_report.png")
    plt.savefig(plot_path)
    plt.close()
