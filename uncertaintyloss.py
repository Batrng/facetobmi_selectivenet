import torch
import torch.nn as nn
from torchvision import models

class UncertaintyLoss(nn.Module):
    def __init__(self):
        super(UncertaintyLoss, self).__init__()

    def forward(self, y_true, y_pred, log_sigma):
        sigma = torch.exp(log_sigma)  # Convert log(sigma) back to sigma
        loss = (1 / (2 * sigma**2)) * (y_true - y_pred)**2 + log_sigma
        return torch.mean(loss)