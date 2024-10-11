import torch
import torch.nn as nn
from loader import get_dataloaders
from models import get_model

import numpy as np
import argparse
import wandb

# train one epoch
def train(train_loader, model, loss_fn, optimizer):
    wandb.init(project=args.wandbname, config={"learning_rate":args.lr, "architecture": "Resnet", "dataset": "testdataset100", "epochs": args.epochs, "batch":args.batch_size, "dataset": args.dataset}, resume=False)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    correct = 0
    total = 0
    true_positive = 0
    false_positive = 0
    mae_loss_fn = nn.L1Loss()

    # Train
    model.train()
    for batch, (X, y) in enumerate(train_loader):
        X = X.to(device)
        y = y.to(device)
        # Compute prediction error
        pred = model(X)
        y = y.unsqueeze(1).float()
        loss = loss_fn(pred, y) #mse

        #mae
        mae_loss = mae_loss_fn(pred, y).item() 

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #accuracy
        total += y.size(0)
        correct += (pred.round() == y).sum().item()
        accuracy = correct/total

        #precision
        true_positive += (pred * y).sum().item()  # TP: Both pred and actual are 1
        false_positive += (pred * (1 - y)).sum().item()  # FP: Pred is 1, actual is 0

        if (true_positive + false_positive) > 0:
            precision = true_positive / (true_positive + false_positive)
        else:
            precision = 0  # Avoid division by zero

        # Show progress
        #if batch % 2 == 0:
        #    loss, current = loss.item(), batch * len(X)
        #    print(f"train loss: {loss:>7f} [{current:>5d}/{len(train_loader.dataset):>5d}]")

        wandb.log({"loss_train_mse":loss, "precision_train":precision, "accuracy_train":accuracy, "loss_train_mae": mae_loss}, step=batch)
    wandb.finish()
# validate and return mae loss
def validate(val_loader, model):
    wandb.init(project=args.wandbname, config={"learning_rate":args.lr, "architecture": "Resnet", "dataset": "testdataset100", "epochs": args.epochs, "batch":args.batch_size, "dataset": args.dataset}, resume=False)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # Validation
    model.eval()
    val_loss_mse = 0
    val_loss_mae = 0
    total = 0
    correct = 0
    true_positive = 0
    false_positive = 0
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(val_loader):
            X = X.to(device)
            y = y.to(device)

            pred = model(X)
            y = y.unsqueeze(1)

            loss_mse = nn.MSELoss()(pred, y)
            loss_mae = nn.L1Loss()(pred, y)

            #accuracy
            total += y.size(0)
            correct += (pred.round() == y).sum().item()
            accuracy = 100.*correct/total

            #precision
            true_positive += (pred * y).sum().item()  # TP: Both pred and actual are 1
            false_positive += (pred * (1 - y)).sum().item()  # FP: Pred is 1, actual is 0

            if (true_positive + false_positive) > 0:
                precision = true_positive / (true_positive + false_positive)
            else:
                precision = 0  # Avoid division by zero
            wandb.log({"loss_val_mse":loss_mse, "precision_val":precision, "accuracy_val":accuracy, "loss_val_mae": loss_mae}, step=batch_idx)

    val_loss_mse /= len(val_loader)
    val_loss_mae /= len(val_loader)
    wandb.finish()
    

    #print(f"val mse loss: {val_loss_mse:>7f}, val mae loss: {val_loss_mae}")
    return val_loss_mse



# test and return mse and mae loss
def test(test_loader, model):
    wandb.init(project=args.wandbname, config={"learning_rate":args.lr, "architecture": "Resnet", "dataset": "testdataset100", "epochs": args.epochs, "batch":args.batch_size, "dataset": args.dataset}, resume=False)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # Test
    model.eval()
    test_loss_mse = 0
    test_loss_mae = 0
    total = 0
    correct = 0
    true_positive = 0
    false_positive = 0
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(test_loader):
            X = X.to(device)
            y = y.to(device)

            pred = model(X)
            y = y.unsqueeze(1)

            loss_mse = nn.MSELoss()(pred, y)
            loss_mae = nn.L1Loss()(pred, y)
            #accuracy
            total += y.size(0)
            correct += (pred.round() == y).sum().item()
            accuracy = 100.*correct/total

            #precision
            true_positive += (pred * y).sum().item()  # TP: Both pred and actual are 1
            false_positive += (pred * (1 - y)).sum().item()  # FP: Pred is 1, actual is 0

            if (true_positive + false_positive) > 0:
                precision = true_positive / (true_positive + false_positive)
            else:
                precision = 0  # Avoid division by zero
            wandb.log({"loss_test_mse":loss_mse, "precision_test":precision, "accuracy_test":accuracy, "loss_test_mae": loss_mae},step=batch_idx)


    test_loss_mse /= len(test_loader)
    test_loss_mae /= len(test_loader)
    wandb.finish()

    #print(f"test mse loss: {test_loss_mse:>7f}, test mae loss: {test_loss_mae}")
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

        score = val_loss

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
        #if self.verbose:
         #   print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), '/home/nguyenbt/nobackup/weights/aug_epoch_7.pt')  # save checkpoint
        self.val_loss_min = val_loss

def get_args():
    parser = argparse.ArgumentParser(description="Training script for a ResNet model.")
    
    # Add arguments
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for data loader')    
    parser.add_argument('--dataset', type=str, default=test, help='dataset') 
    parser.add_argument('--augmented', type=bool, default=False, help='set to True to use augmented dataset')
    parser.add_argument('--wandbname', type=str, default="Test", help='for wandblogging')
    
    # Parse arguments
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    train_loader, val_loader, test_loader = get_dataloaders(args.batch_size, augmented=args.augmented, vit_transformed=True, show_sample=True)
    model = get_model().float().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    early_stopping = EarlyStopping(patience=5, verbose=True)

    for t in range(args.epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_loader, model, loss_fn, optimizer)
        val_loss = validate(test_loader, model)
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    model.load_state_dict(torch.load('/home/nguyenbt/nobackup/weights/aug_epoch_7.pt'))
    test(test_loader, model)
    
    print("Done!")

