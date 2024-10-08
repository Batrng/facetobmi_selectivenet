import torch
import torch.nn as nn
from loader import get_dataloaders
from models import get_model
from SelectiveNet import *
import numpy as np
import argparse
from models import get_model
from selectiveLoss import SelectiveLoss
from collections import OrderedDict
from metric import MetricDict
import wandb

alpha = 0.5
# train one epoch
def train(train_loader, model, loss_fn, loss_selective, optimizer):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    true_positive = 0
    false_positive = 0
    mae_loss_fn = nn.L1Loss()
    # Train
    model.train()
    for batch, (X, y) in enumerate(train_loader):
        X = X.to(device).float()
        y = y.to(device).float()
        
        # Compute prediction
        pred, pred_select, pred_aux = model(X)
        y = y.unsqueeze(1).float()
        selective_loss = loss_selective(pred, pred_select, y)
        selective_loss *= alpha

        #mae for pred
        mae_loss = mae_loss_fn(pred, y).item() 
        mse_loss = loss_fn(pred, y).item() 

        #precision
        true_positive += (pred * y).sum().item()  # TP: Both pred and actual are 1
        false_positive += (pred * (1 - y)).sum().item()  # FP: Pred is 1, actual is 0

        #aux loss
        ce_loss = loss_fn(pred_aux, y)
        ce_loss *= (1.0 - alpha)

        #total loss
        loss = selective_loss + ce_loss
        loss = loss.float()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (true_positive + false_positive) > 0:
            precision = true_positive / (true_positive + false_positive)
        else:
            precision = 0  # Avoid division by zero

        # Show progress
        wandb.log({"losstotal_train_mse":loss, "precision_train":precision, "loss_train_mae": mae_loss, "Loss_train_mse":mse_loss},step=batch)


# validate and return mae loss
def validate(val_loader, model, loss_fn):
    true_positive = 0
    false_positive = 0
    mae_loss_fn = nn.L1Loss()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # Validation
    model.eval()
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(val_loader):
            X = X.to(device).float()
            y = y.to(device).float()

            pred, pred_select, pred_aux = model(X)
            y = y.unsqueeze(1)
            selective_loss = loss_selective(pred, pred_select, y)
            selective_loss *= alpha #for total loss
            
            #precision
            true_positive += (pred * y).sum().item()  # TP: Both pred and actual are 1
            false_positive += (pred * (1 - y)).sum().item()  # FP: Pred is 1, actual is 0

            if (true_positive + false_positive) > 0:
                precision = true_positive / (true_positive + false_positive)
            else:
                precision = 0  # Avoid division by zero
            #mae/mse for pred
            mae_loss = mae_loss_fn(pred, y).item() 
            mse_loss = loss_fn(pred, y).item() 

            #total loss prepare
            ce_loss = loss_fn(pred_aux, y)
            ce_loss *= (1.0 - alpha)

            # total loss
            loss = selective_loss + ce_loss
            loss = loss.float() 
            wandb.log({"losstotal_train_mse":loss, "precision_train":precision, "loss_train_mse":mse_loss, "loss_train_mae": mae_loss},step=batch_idx)

    print(f"val mse loss: {loss.item():>7f}")
    return loss #val_loss_mae



# test and return mse and mae loss
def test(test_loader, model, loss_fn):
    true_positive = 0
    false_positive = 0
    mae_loss_fn = nn.L1Loss()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # Test
    model.eval()
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(test_loader):
            X = X.to(device)
            y = y.to(device)

            pred, pred_select, pred_aux = model(X)
            y = y.unsqueeze(1)
            selective_loss = loss_selective(pred, pred_select, y, test=True)
            selective_loss *= alpha
            #precision
            true_positive += (pred * y).sum().item()  # TP: Both pred and actual are 1
            false_positive += (pred * (1 - y)).sum().item()  # FP: Pred is 1, actual is 0

            if (true_positive + false_positive) > 0:
                precision = true_positive / (true_positive + false_positive)
            else:
                precision = 0  # Avoid division by zero
            #mae/mse for pred
            mae_loss = mae_loss_fn(pred, y).item() 
            mse_loss = loss_fn(pred, y).item()
            
            #total loss prepare
            ce_loss = loss_fn(pred_aux, y)
            ce_loss *= (1.0 - alpha)

            #total loss
            loss = selective_loss + ce_loss
            loss = loss.float() 
            wandb.log({"losstotal_test_mse":loss, "precision_test":precision, "loss_test_mse":mse_loss, "loss_test_mae": mae_loss}, step=batch_idx)

    print(f"test mse loss: {loss.item():>7f}")
    return loss #test_loss_mse, test_loss_mae



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
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss.item():.6f}).  Saving model ...')
        torch.save(model.state_dict(), '/home/nguyenbt/nobackup/weights/aug_epoch_8.pt')  # save checkpoint Bao
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
    wandb.init(project=args.wandbname, config={"learning_rate":args.lr, "architecture": "Resnet", "dataset": "testdataset100", "epochs": args.epochs, "batch":args.batch_size, "dataset": args.dataset})

    train_loader, val_loader, test_loader = get_dataloaders(args.batch_size, augmented=args.augmented, vit_transformed=True, show_sample=False)
    
    # model for normal and selectivenet
    features = get_model().float().to(device)
    model = SelectiveNet(features, 80).float().to(device)

    # optimizer and loss

    loss_fn = nn.MSELoss()
    loss_selective = SelectiveLoss(loss_fn, 0.7) #edit coverage
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    epochs = 1
    early_stopping = EarlyStopping(patience=5, verbose=True)

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_loader, model, loss_fn, loss_selective, optimizer)
        val_loss = validate(test_loader, model, loss_fn=nn.MSELoss())
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    model.load_state_dict(torch.load('/home/nguyenbt/nobackup/weights/aug_epoch_8.pt'))
    test(test_loader, model, loss_fn)

    print("Done!")


