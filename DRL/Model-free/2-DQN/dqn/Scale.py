import torch.nn as nn
import torchvision.transforms as transforms

class Scale(nn.Module):
    def __init__(self, height, width):
        super(Scale, self).__init__()
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((height, width), interpolation=transforms.InterpolationMode.BILINEAR)
        ])
    
    def forward(self, x):
        return self.transform(x)