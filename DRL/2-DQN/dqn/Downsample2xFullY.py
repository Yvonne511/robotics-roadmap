import torch.nn as nn
import torchvision.transforms as transforms

class Downsample2xFullY(nn.Module):
    def __init__(self):
        super(Downsample2xFullY, self).__init__()
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((42, 42), interpolation=transforms.InterpolationMode.BILINEAR)
        ])
    
    def forward(self, x):
        return self.transform(x)