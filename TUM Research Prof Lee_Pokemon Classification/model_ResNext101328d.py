import torchvision
import torch.nn as nn
# !! Also submit this py file to tutor
class Model(nn.Module):
    def __init__(self, feat_dim = 2048, output_dim =10): # feat_dim needs to be changed for different models, output dimension as well
        super(Model, self).__init__()

        self.feat_dim = feat_dim
        self.output_dim = output_dim

        self.backbone = torchvision.models.resnext101_32x8d(pretrained=True) # name needs to be changed for different models

        self.backbone.fc = nn.Linear(feat_dim, output_dim)

    def forward(self, img):
        out = self.backbone(img) 
        return out