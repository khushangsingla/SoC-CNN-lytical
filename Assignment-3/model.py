import torch
device='cpu'
if(torch.cuda.is_available()):	device='cuda'
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=torch.nn.Conv2d(3,8,3,padding='same') # 8,32,32
        self.bn2 = torch.nn.BatchNorm2d(8)
        self.pool3 = torch.nn.MaxPool2d(2,2) # 8,16,16
        self.conv4 = torch.nn.Conv2d(8,16,3,padding=1) # 16,16,16
        self.bn5 = torch.nn.BatchNorm2d(16)
        self.conv6 = torch.nn.Conv2d(16,16,3,padding=1) # 16,16,16
        self.bn7 = torch.nn.BatchNorm2d(16)
        self.pool8 = torch.nn.MaxPool2d(2,2) # 16,8,8
        self.conv9 = torch.nn.Conv2d(16,16,3,padding=1) # 16,8,8
        self.bn10 = torch.nn.BatchNorm2d(16)
        self.pool11 = torch.nn.MaxPool2d(2,2) # 16,4,4
        self.conv12 = torch.nn.Conv2d(16,32,3,padding=1) # 32,4,4
        self.bn13 = torch.nn.BatchNorm2d(32)
        self.conv14 = torch.nn.Conv2d(32,32,3,padding=1) # 32,4,4
        self.bn15 = torch.nn.BatchNorm2d(32)
        self.pool16 = torch.nn.AvgPool2d(4,4) # 32,1,1
        # 32,1
        self.fc17 = torch.nn.Linear(32,32) # 16,32
        self.bn18 = torch.nn.BatchNorm1d(32)
        self.fc19 = torch.nn.Linear(32,16) # 16,16
        self.bn20 = torch.nn.BatchNorm1d(16)
        self.fc21 = torch.nn.Linear(16,10)
        self.relu = torch.nn.ReLU()
    def forward(self,x):
        x.to(device)
        out = self.relu(self.bn2(self.conv1(x)))
        out = self.pool3(out)
        out = self.relu(self.bn5(self.conv4(out)))
        out = self.relu(self.bn7(self.conv6(out)))
        out = self.pool8(out)
        out = self.relu(self.bn10(self.conv9(out)))
        out = self.pool11(out)
        out = self.relu(self.bn13(self.conv12(out)))
        out = self.relu(self.bn15(self.conv14(out)))
        out = self.pool16(out)
        out = torch.reshape(out,(-1,32))
        out = self.relu(self.bn18(self.fc17(out)))
        out = self.relu(self.bn20(self.fc19(out)))
        out = self.fc21(out)
        out = torch.nn.functional.log_softmax(out,dim=1)
        return out
