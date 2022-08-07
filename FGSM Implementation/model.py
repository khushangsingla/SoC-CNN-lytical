import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
class Propagator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784,512)
        self.bn1 = nn.BatchNorm1d(512)
        self.layer2 = nn.Linear(512,512)
        self.layer3 = nn.Linear(512,256)
        self.layer4 = nn.Linear(256,64)
        self.bn2 = nn.BatchNorm1d(64)
        self.layer5 = nn.Linear(64,64)
        self.final_layer = nn.Linear(64,10)
    def forward(self,x):
        x = F.relu(self.bn1(self.layer1(x)))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.bn2(self.layer4(x)))
        x = F.relu(self.layer5(x))
        x = self.final_layer(x)
        x = F.log_softmax(x,dim=1)
        return x
