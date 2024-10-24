import torch
import torch.nn as nn
from dataloader import get_dataloaders, AugmentedBMIDataset
from model import HeightEstimationNet
from selectiveloss import SelectiveLoss
import numpy as np
from selectivenet import SelectiveNet
import argparse
import wandb

alpha = 0.5
# train one epoch
def train(train_loader, features, model, loss_selective, optimizer):
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    train_loss_mse = 0
    train_loss_mae = 0
    train_counter = 0
    # Train
    model.train()
    for batch, (X_fullbody, X_face, y) in enumerate(train_loader):
        X_face = X_face.to(device)
        X_fullbody = X_fullbody.to(device)
        y = y.to(device)

        # Compute prediction error
        pred, pred_select, pred_aux  = model(X_fullbody, X_face)
        #y = y.unsqueeze(1).float()
        y = y.float()
        selective_loss = loss_selective(pred, pred_select, y, pred_aux, train=True)
        selective_loss *= alpha

        # MSE MAE
        loss_mse_train = nn.MSELoss()(pred, y)
        #train_loss_mse += loss_mse_train.item()
        loss_mae = nn.L1Loss()(pred, y)
        #train_loss_mae += loss_mae.item()


        #aux loss
        ce_loss = nn.MSELoss()(pred_aux, y)
        ce_loss *= (1.0 - alpha)

        #total loss
        loss_total_train = selective_loss + ce_loss
        loss_total_train = loss_total_train.float()

        # Backpropagation
        optimizer.zero_grad()
        loss_total_train.backward()
        optimizer.step()
        train_counter += 1
        wandb.log({"loss_mse_train": loss_total_train.item(), "train_log_cnt": train_counter, "loss_mae_train": loss_mae, "loss_mse_train": loss_mse_train})
    #loss_mse_train /= len(train_loader)
    #loss_mae /= len(train_loader)



# validate and return mae loss
def validate(val_loader, model, loss_selective):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # Validation
    model.eval()
    val_loss_mse = 0
    val_loss_mae = 0
    val_counter = 0
    with torch.no_grad():
        for batch_idx, (X_fullbody, X_face, y) in enumerate(val_loader):
            X_face = X_face.to(device)
            X_fullbody = X_fullbody.to(device)
            y = y.to(device)

            # Compute prediction error
            pred, pred_select, pred_aux  = model(X_fullbody, X_face)
            #y = y.squeeze(1).float()
            y = y.float()
            selective_loss = loss_selective(pred, pred_select, y, pred_aux,val=True)
            selective_loss *= alpha

            pred, pred_select, pred_aux = model(X_fullbody, X_face)
            #y = y.unsqueeze(1)

            # MSE MAE
            loss_mse_val = nn.MSELoss()(pred, y)
            #val_loss_mse += loss_mse_val.item()
            loss_mae = nn.L1Loss()(pred, y)
            #val_loss_mae += loss_mae.item()
            val_counter += 1

            #aux loss
            ce_loss = nn.MSELoss()(pred_aux, y)
            ce_loss *= (1.0 - alpha)

            #total loss
            loss_total_val = selective_loss + ce_loss
            loss_total_val = loss_total_val.float()
            wandb.log({"loss_val": loss_total_val.item(), "val_log_cnt": val_counter, "loss_mae_val": loss_mae, "loss_mse_val": loss_mse_val})

    #val_loss_mse /= len(val_loader)
    #val_loss_mae /= len(val_loader)


    return val_loss_mae



# test and return mse and mae loss
def test(test_loader, model, loss_selective):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # Test
    model.eval()
    test_loss_mse = 0
    test_loss_mae = 0
    test_counter = 0
    with torch.no_grad():
        for batch_idx_, (X_face, X_fullbody, y) in enumerate(test_loader):
            X_face = X_face.to(device)
            X_fullbody = X_fullbody.to(device)
            y = y.to(device)

            pred, pred_select, pred_aux = model(X_fullbody, X_face)
            #y = y.unsqueeze(1)
            y = y.float()
            # Compute prediction error
            pred, pred_select, pred_aux  = model(X_fullbody, X_face)
            #y = y.unsqueeze(1).float()
            selective_loss = loss_selective(pred, pred_select, y, pred_aux, test=True)
            selective_loss *= alpha

            #MSE MAE
            loss_mse_test = nn.MSELoss()(pred, y)
            #test_loss_mse += loss_mse_test.item()
            loss_mae = nn.L1Loss()(pred, y)
            #test_loss_mae += loss_mae.item()
            test_counter +=1

            #aux loss
            ce_loss = nn.MSELoss()(pred_aux, y)
            ce_loss *= (1.0 - alpha)

            #total loss
            loss_total_test = selective_loss + ce_loss
            loss_total_test = loss_total_test.float()
    
            wandb.log({"loss_test": loss_total_test.item(), "test_log_cnt": test_counter, "loss_mae_test": loss_mae, "loss_mse_test": loss_mse_test})

    #test_loss_mse /= len(test_loader)
    #test_loss_mae /= len(test_loader)

    return test_loss_mse, test_loss_mae


def post_calibrate(model, data_loader, coverage):
    out_select_all = []
    with torch.autograd.no_grad():
        for i, (X_face, X_fullbody, y) in enumerate(data_loader):
            model.eval()
            x = x.to('cuda', non_blocking=True)
            t = t.to('cuda', non_blocking=True)
            # forward
            out_class, out_select, out_aux = model(X_fullbody, X_face)
            out_select_all.append(out_select.cpu().detach().numpy())
    out_select_all = np.concatenate(out_select_all, axis=0)
    threshold = np.percentile(out_select.cpu().detach().numpy(), 100 - 100 * coverage)
    print('>>> Threshold found is : ', threshold)
    return threshold
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
        #torch.save(model.state_dict(), '/home/nguyenbt/nobackup/face-to-bmi-vit/weights/aug_epoch_7.pt')  # save checkpoint
        self.val_loss_min = val_loss



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument('--augmented', type=bool, default=False, help='set to True to use augmented dataset')
    parser.add_argument('--batchsize', type=int, default=32, help='set to True to use augmented dataset')
    parser.add_argument('--lr', type=float, default=0.001, help='set to True to use augmented dataset')
    parser.add_argument('--wandbproject', type=str, default="height", help='set to True to use augmented dataset')
    parser.add_argument('--epochs', type=int, default=5, help='set to True to use augmented dataset')
    args = parser.parse_args()

    train_loader, val_loader, test_loader, calibration_loader = get_dataloaders(args.batchsize, augmented=args.augmented, vit_transformed=False, show_sample=False)
    features = HeightEstimationNet().to(device)
    model = SelectiveNet(features=features).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    epochs = args.epochs
    early_stopping = EarlyStopping(patience=5, verbose=True)
    loss_selective = SelectiveLoss(loss_fn, 0.8) #edit coverage
    with wandb.init(project=args.wandbproject):

        config = wandb.config
        wandb.define_metric("custom_step")
        wandb.watch(model,nn.MSELoss(),log='all',log_freq=1)
        config.lr = args.lr
        #wandb.watch(model)
        for t in range(epochs):
            print(f"Epoch {t + 1}\n-------------------------------")
            train(train_loader, features, model, loss_selective, optimizer)
            val_loss = validate(test_loader, model, loss_selective)
            early_stopping(val_loss, features)

            if early_stopping.early_stop:
                print("Early stopping")
                break

        #model.load_state_dict(torch.load('/home/nguyenbt/nobackup/face-to-bmi-vit/weights/checkpoint.pt'))
        torch.save(model.state_dict(), '../weights/checkpoint_selective.pt')
        test(test_loader, model,loss_selective)
        post_calibrate(model, calibration_loader, 0.7)
        wandb.finish()
        #print("Done!")

