import torch
import torch.nn as nn
from dataloader import get_dataloaders
from model import HeightEstimationNet

import numpy as np
import argparse
import wandb


# train one epoch
def train(train_loader, model, loss_fn, optimizer):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    train_loss_mse = 0
    train_loss_mae = 0
    # Train
    model.train()
    for batch, (X_fullbody, X_face, y) in enumerate(train_loader):
        X_face = X_face.to(device)
        X_fullbody = X_fullbody.to(device)
        y = y.to(device)

        # Compute prediction error
        pred = model(X_fullbody, X_face)
        y = y.unsqueeze(1).float()
        loss_mse = nn.MSELoss()(pred, y)
        train_loss_mse += loss_mse.item()
        loss_mae = nn.L1Loss()(pred, y)
        train_loss_mae += loss_mae.item()

        # Backpropagation
        optimizer.zero_grad()
        loss_mse.backward()
        optimizer.step()
        wandb.log({"loss_train": loss_mse})
    loss_mse /= len(train_loader)
    loss_mae /= len(train_loader)



# validate and return mae loss
def validate(val_loader, model):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # Validation
    model.eval()
    val_loss_mse = 0
    val_loss_mae = 0
    with torch.no_grad():
        for batch_idx, (X_fullbody, X_face, y) in enumerate(val_loader):
            X_face = X_face.to(device)
            X_fullbody = X_fullbody.to(device)
            y = y.to(device)

            pred = model(X_fullbody, X_face)
            y = y.unsqueeze(1)

            loss_mse = nn.MSELoss()(pred, y)
            val_loss_mse += loss_mse.item()
            loss_mae = nn.L1Loss()(pred, y)
            val_loss_mae += loss_mae.item()
            wandb.log({"loss_val": loss_mse})

    val_loss_mse /= len(val_loader)
    val_loss_mae /= len(val_loader)


    return val_loss_mae



# test and return mse and mae loss
def test(test_loader, model):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # Test
    model.eval()
    test_loss_mse = 0
    test_loss_mae = 0
    with torch.no_grad():
        for batch_idx_, (X_face, X_fullbody, y) in enumerate(test_loader):
            X_face = X_face.to(device)
            X_fullbody = X_fullbody.to(device)
            y = y.to(device)

            pred = model(X_fullbody, X_face)
            y = y.unsqueeze(1)

            loss_mse = nn.MSELoss()(pred, y)
            test_loss_mse += loss_mse.item()
            loss_mae = nn.L1Loss()(pred, y)
            test_loss_mae += loss_mae.item()
            wandb.log({"loss_val": test_loss_mse})

    test_loss_mse /= len(test_loader)
    test_loss_mae /= len(test_loader)

    return test_loss_mse, test_loss_mae



# helper class for early stopping
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), '../weights/aug_epoch_7.pt')  # save checkpoint
        self.val_loss_min = val_loss



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument('--augmented', type=bool, default=False, help='set to True to use augmented dataset')
    parser.add_argument('--batchsize', type=int, default=8, help='set to True to use augmented dataset')
    parser.add_argument('--lr', type=float, default=0.001, help='set to True to use augmented dataset')
    parser.add_argument('--wandbproject', type=str, default="height", help='set to True to use augmented dataset')
    args = parser.parse_args()

    train_loader, val_loader, test_loader = get_dataloaders(args.batchsize, augmented=args.augmented, vit_transformed=False, show_sample=False)
    model = HeightEstimationNet().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    epochs = 1
    early_stopping = EarlyStopping(patience=5, verbose=True)

    with wandb.init(project=args.wandbproject):
        config = wandb.config
        config.lr = args.lr
        wandb.watch(model)
        for t in range(epochs):
            print(f"Epoch {t + 1}\n-------------------------------")
            train(train_loader, model, loss_fn, optimizer)
            val_loss = validate(test_loader, model)
            early_stopping(val_loss, model)

            if early_stopping.early_stop:
                print("Early stopping")
                break

        model.load_state_dict(torch.load('../weights/checkpoint.pt'))
        test(test_loader, model)
        wandb.finish()
        #print("Done!")

