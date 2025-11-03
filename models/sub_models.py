import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, dropout_rate=0.0):
        super().__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        self.dropout_rate = dropout_rate

    def forward(self, x):
        out = self.conv(self.relu(self.norm(x)))
        if self.dropout_rate > 0:
            out = F.dropout(out, p=self.dropout_rate, training=self.training)
        out = torch.cat([x, out], 1)
        return out
    
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, n_layers, dropout_rate=0.0):
        super().__init__()
        layers = []
        channels = in_channels
        for _ in range(n_layers):
            layers.append(DenseLayer(channels, growth_rate, dropout_rate))
            channels += growth_rate
        self.block = nn.Sequential(*layers)
        self.out_channels = channels

    def forward(self, x):
        return self.block(x)

class TransitionDown(nn.Module):
    def __init__(self, in_channels, dropout_rate=0.0):
        super().__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.dropout_rate = dropout_rate
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(self.relu(self.norm(x)))
        if self.dropout_rate > 0:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.pool(x)
        return x

class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.transposed_conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
        )

    def forward(self, x):
        return self.transposed_conv(x)