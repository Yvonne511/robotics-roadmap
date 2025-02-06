import torch
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self, input_dims, hist_len, ncols, n_units, filter_size, filter_stride, n_hid, n_actions, device):
        super(ConvNet, self).__init__()
        
        self.device = device
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(hist_len * ncols, n_units[0], kernel_size=filter_size[0], stride=filter_stride[0], padding=1),
            nn.ReLU(),
            nn.Conv2d(n_units[0], n_units[1], kernel_size=filter_size[1], stride=filter_stride[1], padding=1),
            nn.ReLU(),
            nn.Conv2d(n_units[1], n_units[2], kernel_size=filter_size[2], stride=filter_stride[2], padding=1),
            nn.ReLU()
        )
        
        # Compute feature size after convolution
        with torch.no_grad():
            sample_input = torch.zeros(1, hist_len * ncols, *input_dims)
            nel = self.conv_layers(sample_input).numel()
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(nel, n_hid[0]),
            nn.ReLU(),
            nn.Linear(n_hid[0], n_hid[1]),
            nn.ReLU(),
            nn.Linear(n_hid[1], n_actions)
        )
        
        self.to(self.device)
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x