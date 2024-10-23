import torch
from torch.nn.modules.loss import _Loss
import wandb
import numpy as np 

class SelectiveLoss(torch.nn.Module):
    def __init__(self, loss_func, coverage:float, lm:float=10.0):
        """
        Args:
            loss_func: base loss function. the shape of loss_func(x, target) shoud be (B). 
                       e.g.) torch.nn.CrossEntropyLoss(reduction=none) : classification
            coverage: target coverage.
            lm: Lagrange multiplier for coverage constraint. original experiment's value is 32. 
        """
        super(SelectiveLoss, self).__init__()
        assert 0.0 < coverage <= 1.0
        assert 0.0 < lm

        self.loss_func = loss_func
        self.coverage = coverage
        self.lm = lm

    def forward(self, prediction_out, selection_out, target, auxiliary_out, train=False, val=False, test=False):
        """
        Args:
            prediction_out: (B,num_classes)
            selection_out:  (B, 1)
        """
        # compute emprical coverage (=phi^)
        empirical_coverage = selection_out.mean() 

        # compute emprical risk (=r^)
        empirical_risk = (self.loss_func(prediction_out, target)*selection_out.view(-1)).mean()
        empirical_risk = empirical_risk / empirical_coverage

        # Compute incorrect predictions (where the prediction is wrong)
        wrong_predictions = (torch.abs(prediction_out - target) > 0.05).float()  # Adjust threshold as necessary

        # Penalize high selection scores for wrong predictions
        penalty_for_confident_wrong = torch.mean(wrong_predictions * selection_out.view(-1))        

        # compute penalty (=psi)
        coverage = torch.tensor([self.coverage], dtype=torch.float32, requires_grad=True, device='cuda')
        penalty = torch.max(coverage-empirical_coverage, torch.tensor([0.0], dtype=torch.float32, requires_grad=True, device='cuda'))**2
        #penalty *= self.lm
        penalty += penalty_for_confident_wrong * self.lm

        selective_loss = empirical_risk + penalty

        # compute coverage based on source implementation
        selective_head_coverage = self.get_coverage(selection_out, threshold=0.7)

        # compute selective accuracy based on source implementation
        selective_head_selective_acc = self.get_selective_acc(prediction_out, selection_out, target)

        # compute accuracy based on source implementation
        prediction_head_acc = self.get_accuracy(auxiliary_out, target)
        
        # compute selective loss (=selective_head_loss) based on source implementation
        selective_head_loss = self.get_selective_loss(prediction_out, selection_out, target)

        if (train):
           wandb.log({"train/empirical coverage":empirical_coverage, "train/selectiveloss":selective_loss, "train/empirical_risk": empirical_risk, "train/coverage": selective_head_coverage, "train/selective_accuracy": selective_head_selective_acc, "accuracy_train": prediction_head_acc, "train/selectiveheadloss": selective_head_loss})
        if (val):
           wandb.log({"val/empirical coverage":empirical_coverage, "val/selectiveloss":selective_loss, "val/empirical_risk": empirical_risk, "val/coverage": selective_head_coverage, "val/selective_accuracy": selective_head_selective_acc, "accuracy_val": prediction_head_acc, "val/selectiveheadloss": selective_head_loss})
        if (test):
           wandb.log({"test/empirical coverage":empirical_coverage, "test/selectiveloss":selective_loss, "test/empirical_risk": empirical_risk, "test/coverage": selective_head_coverage, "test/selective_accuracy": selective_head_selective_acc, "accuracy_test": prediction_head_acc, "test/selectiveheadloss": selective_head_loss})

        return selective_loss#, loss_dict
    
    '''def selective_acc(y_true, y_pred):
    # g represents whether the model is confident (i.e., didn't abstain) based on threshold of 0.5
        g = (y_pred[:, -1] > 0.5).float()

        # Get predicted and true labels (ignoring the last column used for selection)
        y_pred_classes = torch.argmax(y_pred[:, :-1], dim=-1)
        y_true_classes = torch.argmax(y_true[:, :-1], dim=-1)

        # Correct predictions where the model did not abstain
        correct_predictions = (y_pred_classes == y_true_classes).float() * g

        # Sum correct predictions and divide by number of non-abstained predictions
        selective_accuracy = correct_predictions.sum() / g.sum()

        return selective_accuracy
'''

    def get_selective_acc(self, prediction_out, selection_out, target):
        """
        Equivalent to selective_acc function of source implementation
        Args:
            prediction_out: (B,num_classes)
            selection_out:  (B, 1)
        """
        g = (selection_out.squeeze(-1) > 0.7).float()
        prediction_out = prediction_out.squeeze(-1)
        #target = target.squeeze(-1)
        print(target)

        accuracy = (torch.abs(prediction_out - target) < 0.05).float()
        num = torch.dot(g, accuracy)
        return num / torch.sum(g)

    # based on source implementation
    def get_coverage(self, selection_out, threshold):
        """
        Equivalent to coverage function of source implementation
        Args:
            selection_out:  (B, 1)
        """
        g = (selection_out.squeeze(-1) >= threshold).float()
        return torch.mean(g)

    # based on source implementation
    def get_accuracy(self, auxiliary_out, target): #TODO: Check implementation with Lili
        """
        Equivalent to "accuracy" in Tensorflow
        Args:
            selection_out:  (B, 1)
        """ 
        #num = torch.sum((torch.argmax(auxiliary_out, dim=-1) == target).float())
        accuracy = (torch.abs(auxiliary_out - target) < 0.05).float()
        return torch.mean(accuracy)
    
    # based on source implementation
    def get_selective_loss(self, prediction_out, selection_out, target):
        """
        Equivalent to selective_loss function of source implementation
        Args:
            prediction_out: (B,num_classes)
            selection_out:  (B, 1)
        """
        ce = self.loss_func(prediction_out, target)
        empirical_risk_variant = torch.mean(ce * selection_out.view(-1))
        empirical_coverage = selection_out.mean() 
        penalty = torch.max(self.coverage - empirical_coverage, torch.tensor([0.0], dtype=torch.float32, requires_grad=True, device='cuda'))**2
        loss = empirical_risk_variant + self.lm * penalty
        return loss

    # selective risk in test mode
    def get_selective_risk(self, prediction_out, selection_out, target, threshold):
        g = (selection_out.squeeze(-1) >= threshold).float()
        empirical_coverage_rjc = torch.mean(g)
        empirical_risk_rjc = torch.mean(self.loss_func(prediction_out, target) * g.view(-1))
        empirical_risk_rjc /= empirical_coverage_rjc
        return empirical_risk_rjc
