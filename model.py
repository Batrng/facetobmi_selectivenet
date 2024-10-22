import torch
import torch.nn as nn
from torchvision import models

class HeightEstimationNet(nn.Module):
    def __init__(self):
        super(HeightEstimationNet, self).__init__()

        # ResNet50 for full-body crop (Coarse stream)
        self.resnet_body = models.resnet50(pretrained=True)
        
        # Remove final classification layers from ResNet for body crop
        self.resnet_body = nn.Sequential(*list(self.resnet_body.children())[:-1])

        # EfficientNet-B0 for face crop (Fine stream)
        self.efficientnet_face = models.efficientnet_b0(pretrained=True)
        
        # Modify the classifier layer of EfficientNet to remove final FC layer
        self.efficientnet_face = nn.Sequential(*list(self.efficientnet_face.children())[:-1])

        # Fully connected layers after concatenation of body and face features
        self.fc1 = nn.Linear(2048 + 1280, 1024)  # 2048 from ResNet, 1280 from EfficientNet-B0
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 1)  # Predicting a single value (height)   

        # Two separate outputs: one for height prediction (mean) and one for log variance
        self.fc_mu = nn.Linear(512, 1)  # Predicting height (mean)
        self.fc_log_sigma = nn.Linear(512, 1)  # Predicting log(variance)
    
    def forward(self, body_crop, face_crop):
        # Coarse stream: full-body crop through ResNet50
        body_feat = self.resnet_body(body_crop)
        body_feat = torch.flatten(body_feat, 1)  # Flatten the output

        # Fine stream: face crop through EfficientNet-B0
        face_feat = self.efficientnet_face(face_crop)
        face_feat = torch.flatten(face_feat, 1)  # Flatten the output

        # Concatenate both features
        combined_feat = torch.cat((body_feat, face_feat), dim=1)

        # Fully connected layers
        x = self.fc1(combined_feat)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)  # Output predicted height
        # Predict height (mean) and log variance
        mu = self.fc_mu(x)  # Predicted mean (height)
        log_sigma = self.fc_log_sigma(x)  # Predicted log(variance)
        return mu, log_sigma
        #return x
