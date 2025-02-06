import torch
import torch.nn as nn

class Rectifier(nn.Module):
    def __init__(self):
        super(Rectifier, self).__init__()
    
    def forward(self, input):
        return torch.maximum(input, torch.tensor(0.0, device=input.device))