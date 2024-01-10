import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.filters import spatial_gradient
from kornia.filters import filter2d


class handcrafted_block(nn.Module):
    '''
        It defines the handcrafted filters within the Key.Net handcrafted block
    '''
    def __init__(self):
        super(handcrafted_block, self).__init__()

    def forward(self, x):

        sobel = spatial_gradient(x)
        dx, dy = sobel[:, :, 0, :, :], sobel[:, :, 1, :, :]

        sobel_dx = spatial_gradient(dx)
        dxx, dxy = sobel_dx[:, :, 0, :, :], sobel_dx[:, :, 1, :, :]

        sobel_dy = spatial_gradient(dy)
        dyy = sobel_dy[:, :, 1, :, :]

        hc_feats = torch.cat([dx, dy, dx**2., dy**2., dx*dy, dxy, dxy**2., dxx, dyy, dxx*dyy], dim=1)

        return hc_feats


class learnable_block(nn.Module):
    '''
        It defines the learnable blocks within the Key.Net
    '''
    def __init__(self, in_channels=10):
        super(learnable_block, self).__init__()

        self.conv0 = conv_blck(in_channels)
        self.conv1 = conv_blck()
        self.conv2 = conv_blck()

    def forward(self, x):
        x = self.conv2(self.conv1(self.conv0(x)))
        return x


def conv_blck(in_channels=8, out_channels=8, kernel_size=5,
              stride=1, padding=2, dilation=1):
    '''
    Default learnable convolutional block.
    '''
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation),
                         nn.BatchNorm2d(out_channels),
                         nn.ReLU(inplace=True))


class feature_extractor(nn.Module):
    '''
        It loads both, the handcrafted and learnable blocks
    '''
    def __init__(self):
        super(feature_extractor, self).__init__()

        self.hc_block = handcrafted_block()
        self.lb_block = learnable_block()

    def forward(self, x):
        x_hc = self.hc_block(x)
        x_lb = self.lb_block(x_hc)
        return x_lb


def _get_pyramid_gaussian_kernel() -> torch.Tensor:
    """Utility function that return a pre-computed gaussian kernel."""
    return torch.tensor([[
        [1., 4., 6., 4., 1.],
        [4., 16., 24., 16., 4.],
        [6., 24., 36., 24., 6.],
        [4., 16., 24., 16., 4.],
        [1., 4., 6., 4., 1.]
    ]]) / 256.


def custom_pyrdown(input: torch.Tensor, factor: float = 2., border_type: str = 'reflect', align_corners: bool = False) -> torch.Tensor:
    if not len(input.shape) == 4:
        raise ValueError(f"Invalid input shape, we expect BxCxHxW. Got: {input.shape}")
    kernel: torch.Tensor = _get_pyramid_gaussian_kernel()
    b, c, height, width = input.shape
    # blur image
    x_blur: torch.Tensor = filter2d(input, kernel, border_type)

    # downsample.
    out: torch.Tensor = F.interpolate(x_blur, size=(int(height // factor), int(width // factor)), mode='bilinear',
                                      align_corners=align_corners)
    return out


class KeyNet(nn.Module):
    '''
    Key.Net model definition
    '''
    def __init__(self, keynet_conf):
        super(KeyNet, self).__init__()

        num_filters = keynet_conf['num_filters']
        self.num_levels = keynet_conf['num_levels']
        kernel_size = keynet_conf['kernel_size']
        padding = kernel_size // 2

        self.feature_extractor = feature_extractor()
        self.last_conv = nn.Sequential(nn.Conv2d(in_channels=num_filters*self.num_levels,
                                                 out_channels=1, kernel_size=kernel_size, padding=padding),
                                       nn.ReLU(inplace=True))

    def forward(self, x):
        """
        x - input image
        """
        x = torch.sum(x, dim=1, keepdim=True)
        shape_im = x.shape
        for i in range(self.num_levels):
            if i == 0:
                feats = self.feature_extractor(x)
            else:
                x = custom_pyrdown(x, factor=1.2)
                feats_i = self.feature_extractor(x)
                feats_i = F.interpolate(feats_i, size=(shape_im[2], shape_im[3]), mode='bilinear')
                feats = torch.cat([feats, feats_i], dim=1)

        scores = self.last_conv(feats)
        return scores, None


if __name__ == '__main__':
    from thop import profile
    # Test the model
    keynet_conf = {'num_filters': 8, 'num_levels': 3, 'kernel_size': 5}
    model = KeyNet(keynet_conf)
    print(model)
    # load the pretrained model
    checkpoint = torch.load('/home/server/linyicheng/py_proj/keypoint_bench/keypoint_bench/weights/keynet_pytorch.pth')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    # Test the model
    x = torch.randn(1, 1, 512, 512)
    flops, params = profile(model, inputs=(x,))
    print('{:<30}  {:<8} GFLops'.format('Computational complexity: ', flops / 1e9))
    print('{:<30}  {:<8} KB'.format('Number of parameters: ', params / 1e3))
