import torch
import torch.nn as nn

import torch

class SelectiveNet(torch.nn.Module):
    """
    SelectiveNet for classification with rejection option.
    In the experiments of original papaer, variant of VGG-16 is used as body block for feature extraction.  
    """
    def __init__(self, features):
        """
        Args
            features: feature extractor network (called body block in the paper).
            dim_featues: dimension of feature from body block.  
            num_classes: number of classification class.
        """
        super(SelectiveNet, self).__init__()
        self.features = features
        self.dim_features = dim_features #512 for final output layer

        # represented as f()
        self.bmi_final_layer = nn.Linear(512, 1) 
        self.gelu = nn.GELU()     
  
        # represented as g() in the original paper
        self.selector = torch.nn.Sequential(
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm1d(512),
            torch.nn.Linear(512, 1),
            torch.nn.Sigmoid()
        )

        # represented as h() in the original paper
        self.bmi_final_layer = nn.Linear(512, 1)
        #self.gelu = nn.GELU()   

    def forward(self, x):
        x = self.features(x)
        #x = x.view(x.size(0), -1)

        # for f
        bmi_out = self.bmi_final_layer(x)
        #bmi_out = self.gelu(bmi_out)
        prediction_out = bmi_out

        # for g
        selection_out  = self.selector(x)

        # for h same as for f in this case
        auxiliary_out  = bmi_out

        return prediction_out, selection_out, auxiliary_out


if __name__ == '__main__':
    import os
    import sys

    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
    sys.path.append(base)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    from model import HeightEstimationNet

    features = HeightEstimationNet().cuda()
    #model = SelectiveNet(features, 80).to(device)