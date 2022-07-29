from torch.utils.data import Dataset
import torch
import torchvision
transform_norm = torchvision.transforms.Compose([
    torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
# define your dataset class
class ImageDataSet(Dataset):
    def __init__(self,X,y):
        self.X = transform_norm(X.float())
        self.y=torch.from_numpy(y)
    def __len__(self):
        return len(self.y)
    def __getitem__(self,idx):
        return self.X[idx],self.y[idx]
