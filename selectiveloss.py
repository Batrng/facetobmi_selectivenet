import torch
from torch.nn.modules.loss import _Loss
import wandb

class SelectiveLoss(torch.nn.Module):
    def __init__(self, loss_func, coverage:float, lm:float=32.0):
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

    def forward(self, prediction_out, selection_out, target, test=False):
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

        # compute penulty (=psi)
        coverage = torch.tensor([self.coverage], dtype=torch.float32, requires_grad=True, device='cuda')
        penulty = torch.max(coverage-empirical_coverage, torch.tensor([0.0], dtype=torch.float32, requires_grad=True, device='cuda'))**2
        penulty *= self.lm

        selective_loss = empirical_risk + penulty

        if (test):
           wandb.log({"test/empirical coverage":empirical_coverage, "test_selectiveloss":selective_loss, "test-empirical_risk": empirical_risk})

        return selective_loss#, loss_dict