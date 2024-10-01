
import torch
import torch.nn as nn
from torchvision.models import vit_h_14, ViT_H_14_Weights
from torchvision.models import resnet50, ResNet50_Weights

class BMIHead(nn.Module):
    def __init__(self):
        super(BMIHead, self).__init__()
        self.linear1 = nn.Linear(2048, 640)
        self.linear2 = nn.Linear(640, 320)
        self.linear3 = nn.Linear(320, 160)
        self.linear4 = nn.Linear(160, 80)
        self.linear5 = nn.Linear(80, 1)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.gelu(x)
        x = self.linear3(x)
        x = self.gelu(x)
        x = self.linear4(x)
        out = self.gelu(x)
        return out



def get_model():

    model = resnet50(weights='IMAGENET1K_V2')
    for param in model.parameters():
        param.requires_grad = True

    #model = nn.Sequential(*list(model.children())[:-1])
    model.fc = BMIHead()

    return model


if __name__ == "__main__":
    model = get_model().float()
    print(model)