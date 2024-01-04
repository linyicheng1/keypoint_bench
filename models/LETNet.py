import torch
from torch import nn
from torchvision.models import resnet
from typing import Optional, Callable
import torch.nn.functional as F
import utils


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 gate: Optional[Callable[..., nn.Module]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        super().__init__()
        if gate is None:
            self.gate = nn.ReLU(inplace=True)
        else:
            self.gate = gate
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = resnet.conv3x3(in_channels, out_channels)
        self.bn1 = norm_layer(out_channels)
        self.conv2 = resnet.conv3x3(out_channels, out_channels)
        self.bn2 = norm_layer(out_channels)

    def forward(self, x):
        x = self.gate(self.bn1(self.conv1(x)))  # B x in_channels x H x W
        x = self.gate(self.bn2(self.conv2(x)))  # B x out_channels x H x W
        return x


class LETNet(nn.Module):
    def __init__(self, c1: int = 8, c2: int = 16, grayscale: bool = False):
        super().__init__()
        self.gate = nn.ReLU(inplace=True)
        # ================================== feature encoder
        if grayscale:
            self.block1 = ConvBlock(1, c1, self.gate, nn.BatchNorm2d)
        else:
            self.block1 = ConvBlock(3, c1, self.gate, nn.BatchNorm2d)
        self.conv1 = resnet.conv1x1(c1, c2)
        # ================================== detector and descriptor head
        self.conv_head = resnet.conv1x1(c2, 4)

    def forward(self, x: torch.Tensor):
        # ================================== feature encoder
        x = self.block1(x)
        x = self.gate(self.conv1(x))
        # ================================== detector and descriptor head
        x = self.conv_head(x)
        scores_map = torch.sigmoid(x[:, 3, :, :]).unsqueeze(1)
        local_descriptor = torch.sigmoid(x[:, :-1, :, :])
        return scores_map, local_descriptor


if __name__ == '__main__':
    from thop import profile
    import numpy as np
    net = LETNet(c1=8, c2=16, grayscale=False)
    weight = torch.load('../weights/letnet.pth')
    net.load_state_dict(weight)
    net.eval()

    image = torch.tensor(np.random.random((1, 3, 512, 512)), dtype=torch.float32)
    flops, params = profile(net, inputs=(image,))
    print('{:<30}  {:<8} GFLops'.format('Computational complexity: ', flops / 1e9))
    print('{:<30}  {:<8} KB'.format('Number of parameters: ', params / 1e3))

