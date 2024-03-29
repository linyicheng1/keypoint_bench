import torch
from torch import nn
from typing import Optional, Callable
import torch.nn.functional as F
from torch.nn import Module
from torchvision.models import resnet


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


class ResBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            gate: Optional[Callable[..., nn.Module]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResBlock, self).__init__()
        if gate is None:
            self.gate = nn.ReLU(inplace=True)
        else:
            self.gate = gate
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('ResBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in ResBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = resnet.conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.conv2 = resnet.conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> Module:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gate(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.gate(out)

        return out


class GoodPoint(nn.Module):
    def __init__(self, params):
        super().__init__()
        c0 = params['c0']
        c1 = params['c1']
        self.gate = nn.ReLU(inplace=True)
        # first layer
        self.block = ConvBlock(c0, c1)
        self.conv_head1 = resnet.conv1x1(c1, 3)
        self.conv_head2 = resnet.conv3x3(c1, 1)

    def forward(self, x):
        """
        :param x: [B, C, H, W] C = 3, H, W % 64 == 0
        :return:
        score map        [B, 1, H, W]
        local desc map 0 [B, 3, H, W]
        """
        # backbone feature
        layer1 = self.block(x)

        # head
        x1 = self.conv_head1(layer1)
        x2 = self.conv_head2(layer1)
        scores_map = torch.sigmoid(x2)
        desc_map = torch.sigmoid(x1)

        return scores_map, desc_map


if __name__ == '__main__':
    from thop import profile
    params = {
        'c0': 3,
        'c1': 8,
        'h0': 4,
    }
    model = GoodPoint(params)
    weight = torch.load("../weights/goodpoint.pth")
    model.load_state_dict(weight)

    x = torch.randn(1, 3, 512, 512)
    flops, params = profile(model, inputs=(x,))
    print('{:<30}  {:<8} GFLops'.format('Computational complexity: ', flops / 1e9))
    print('{:<30}  {:<8} KB'.format('Number of parameters: ', params / 1e3))

    # inference time
    x = x.to('cuda')
    model = model.to('cuda')
    import time

    for i in range(100):
        scores, _ = model(x)

    start = time.time()
    for i in range(1000):
        scores, _ = model(x)
    end = time.time()

    print('Inference time: ', (end - start) / 1000 * 1000, 'ms')
