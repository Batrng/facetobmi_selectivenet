import torch
import torch.nn as nn
from torchvision import models
import wandb

class UncertaintyLoss(nn.Module):
    def __init__(self):
        super(UncertaintyLoss, self).__init__()

    def forward(self, y_true, y_pred, log_sigma):
        sigma = torch.exp(log_sigma)  # Convert log(sigma) back to sigma
        loss = (1 / (2 * sigma**2)) * (y_true - y_pred)**2 + log_sigma

        accuracy = get_accuracy(y_pred, y_true)
        wandb.log({"loss_sigma": loss, "accuracy": accuracy})
        return torch.mean(loss)

def get_accuracy(pred, target): 

    accuracy = (torch.abs(pred - target) < 0.05).float()
    return torch.mean(accuracy)