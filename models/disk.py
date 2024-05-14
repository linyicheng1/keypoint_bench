import torch
import torch.nn as nn
import numpy as np
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
from utils.export import export_model


def size_is_pow2(t):
    return all(s % 2 == 0 for s in t.size()[-2:])


def cut_to_match(reference, t, n_pref=2):
    '''
    Slice tensor `t` along spatial dimensions to match `reference`, by
    picking the central region. Ignores first `n_pref` axes
    '''

    if reference.shape[n_pref:] == t.shape[n_pref:]:
        # sizes match, no slicing necessary
        return t

    # compute the difference along all spatial axes
    diffs = [s - r for s, r in zip(t.shape[n_pref:], reference.shape[n_pref:])]

    # check if diffs are even, which is necessary if we want a truly centered crop
    if not all(d % 2 == 0 for d in diffs) and all(d >= 0 for d in diffs):
        fmt = "Tried to slice `t` of size {} to match `reference` of size {}"
        msg = fmt.format(t.shape, reference.shape)
        raise RuntimeError(msg)

    # pick the full extent of `batch` and `feature` axes
    slices = [slice(None, None)] * n_pref

    # for all remaining pick between diff//2 and size-diff//2
    for d in diffs:
        if d > 0:
            slices.append(slice(d // 2, -(d // 2)))
        elif d == 0:
            slices.append(slice(None, None))

    if slices == []:
        return t
    else:
        return t[slices]


class TrivialUpsample(nn.Module):
    def __init__(self, *args, **kwargs):
        super(TrivialUpsample, self).__init__()

    def forward(self, x):
        r = F.interpolate(
            x, scale_factor=2, mode='bilinear', align_corners=False
        )
        return r


class TrivialDownsample(nn.Module):
    def __init__(self, *args, **kwargs):
        super(TrivialDownsample, self).__init__()

    def forward(self, x):
        if not size_is_pow2(x):
            msg = f"Trying to downsample feature map of size {x.size()}"
            raise RuntimeError(msg)

        return F.avg_pool2d(x, 2)


class NoOp(nn.Module):
    def __init__(self, *args, **kwargs):
        super(NoOp, self).__init__()

    def forward(self, x):
        return x


class Conv(nn.Sequential):
    def __init__(self, in_, out_, size, setup=None):
        norm = setup['norm'](in_)
        nonl = setup['gate'](in_)
        dropout = setup['dropout']()

        if setup['padding']:
            padding = size // 2
        else:
            padding = 0

        if 'bias' in setup:
            bias = setup['bias']
        else:
            bias = True

        conv = nn.Conv2d(in_, out_, size, padding=padding, bias=bias)

        super(Conv, self).__init__(norm, nonl, dropout, conv)


class ThinUnetDownBlock(nn.Sequential):
    def __init__(self, in_, out_, size=5, name=None, is_first=False,
                 setup=None):

        self.name = name
        self.in_ = in_
        self.out_ = out_

        if is_first:
            downsample = NoOp()
            conv = Conv(in_, out_, size, setup={**setup, 'gate': NoOp, 'norm': NoOp})
        else:
            downsample = setup['downsample'](in_, size, setup=setup)
            conv = Conv(in_, out_, size, setup=setup)

        super(ThinUnetDownBlock, self).__init__(downsample, conv)


class ThinUnetUpBlock(nn.Module):
    def __init__(self, bottom_, horizontal_, out_,
                 size=5, name=None, setup=None):

        super(ThinUnetUpBlock, self).__init__()

        self.name = name
        self.bottom_ = bottom_
        self.horizontal_ = horizontal_
        self.cat_ = bottom_ + horizontal_
        self.out_ = out_

        self.upsample = setup['upsample'](bottom_, size, setup=setup)
        self.conv = Conv(self.cat_, self.out_, size, setup=setup)

    def forward(self, bot: ['b', 'fb', 'hb', 'wb'],
                      hor: ['b', 'fh', 'hh', 'wh']
               ) -> ['b', 'fo', 'ho', 'wo']:

        bot_big = self.upsample(bot)
        hor = cut_to_match(bot_big, hor, n_pref=2)
        combined = torch.cat([bot_big, hor], dim=1)

        return self.conv(combined)


class UnetDownBlock(nn.Sequential):
    def __init__(self, in_, out_, size=5, name=None, is_first=False,
                 setup=None):

        self.name = name
        self.in_ = in_
        self.out_ = out_

        if is_first:
            downsample = NoOp()
            conv1 = Conv(in_, out_, size, setup={**setup, 'gate': NoOp, 'norm': NoOp})
        else:
            downsample = setup['downsample'](in_, size, setup=setup)
            conv1 = Conv(in_, out_, size, setup=setup)

        conv2 = Conv(out_, out_, size, setup=setup)
        super(UnetDownBlock, self).__init__(downsample, conv1, conv2)


class UnetUpBlock(nn.Module):
    def __init__(self, bottom_, horizontal_, out_,
                 size=5, name=None, setup=None):

        super(UnetUpBlock, self).__init__()

        self.name = name
        self.bottom_ = bottom_
        self.horizontal_ = horizontal_
        self.cat_ = bottom_ + horizontal_
        self.out_ = out_

        self.upsample = setup['upsample'](bottom_, size, setup=setup)

        conv1 = Conv(self.cat_, self.cat_, size, setup=setup)
        conv2 = Conv(self.cat_, self.out_, size, setup=setup,)
        self.seq = nn.Sequential(conv1, conv2)

    def forward(self, bot: ['b', 'fb', 'hb', 'wb'],
                      hor: ['b', 'fh', 'hh', 'wh']
               ) -> ['b', 'fo', 'ho', 'wo']:

        bot_big = self.upsample(bot)
        hor = cut_to_match(bot_big, hor, n_pref=2)
        combined = torch.cat([bot_big, hor], dim=1)

        return self.seq(combined)


thin_setup = {
    'gate': nn.PReLU,
    'norm': nn.InstanceNorm2d,
    'upsample': TrivialUpsample,
    'downsample': TrivialDownsample,
    'down_block': ThinUnetDownBlock,
    'up_block': ThinUnetUpBlock,
    'dropout': NoOp,
    'padding': False,
    'bias': True
}

fat_setup = {
    'gate': nn.PReLU,
    'norm': nn.InstanceNorm2d,
    'upsample': TrivialUpsample,
    'downsample': TrivialDownsample,
    'down_block': UnetDownBlock,
    'up_block': UnetUpBlock,
    'dropout': NoOp,
    'padding': False,
    'bias': True
}

DEFAULT_SETUP = {**thin_setup, 'bias': True, 'padding': True}


def checkpointed(cls):
    assert issubclass(cls, torch.nn.Module)

    class Checkpointed(cls):
        def forward(self, *args, **kwargs):
            super_fwd = super(Checkpointed, self).forward
            if any((torch.is_tensor(arg) and arg.requires_grad) for arg in args):
                return checkpoint(super_fwd, *args, **kwargs)
            else:
                return super_fwd(*args, **kwargs)

    return Checkpointed


class Unet(nn.Module):
    def __init__(self, in_features=1, up=None, down=None,
                 size=5, setup=fat_setup):

        super(Unet, self).__init__()

        if not len(down) == len(up) + 1:
            raise ValueError("`down` must be 1 item longer than `up`")

        self.up = up
        self.down = down
        self.in_features = in_features

        DownBlock = setup['down_block']
        UpBlock = setup['up_block']

        if 'checkpointed' in setup and setup['checkpointed']:
            UpBlock = checkpointed(UpBlock)
            DownBlock = checkpointed(DownBlock)

        down_dims = [in_features] + down
        self.path_down = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(zip(down_dims[:-1], down_dims[1:])):
            block = DownBlock(
                d_in, d_out, size=size, name=f'down_{i}', setup=setup, is_first=i == 0,
            )
            self.path_down.append(block)

        bot_dims = [down[-1]] + up
        hor_dims = down_dims[-2::-1]
        self.path_up = nn.ModuleList()
        for i, (d_bot, d_hor, d_out) in enumerate(zip(bot_dims, hor_dims, up)):
            block = UpBlock(
                d_bot, d_hor, d_out, size=size, name=f'up_{i}', setup=setup
            )
            self.path_up.append(block)

        self.n_params = 0
        for param in self.parameters():
            self.n_params += param.numel()

    def forward(self, inp: ['b', 'fi', 'hi', 'wi']) -> ['b', 'fo', 'ho', 'wo']:
        if inp.size(1) != self.in_features:
            fmt = "Expected {} feature channels in input, got {}"
            msg = fmt.format(self.in_features, inp.size(1))
            raise ValueError(msg)

        features = [inp]
        for i, layer in enumerate(self.path_down):
            features.append(layer(features[-1]))

        f_bot = features[-1]
        features_horizontal = features[-2::-1]

        for layer, f_hor in zip(self.path_up, features_horizontal):
            f_bot = layer(f_bot, f_hor)

        return f_bot


class DISK(torch.nn.Module):
    def __init__(
            self,
            desc_dim=128,
            setup=DEFAULT_SETUP,
            kernel_size=5,
    ):
        super(DISK, self).__init__()
        self.desc_dim = desc_dim
        self.unet = Unet(
            in_features=3, size=kernel_size,
            down=[16, 32, 64, 64, 64],
            up=[64, 64, 64, desc_dim + 1],
            setup=setup,
        )

    def forward(self, x):
        feature_map = self.unet(x)
        desc_map = feature_map[:, :self.desc_dim]
        score_map = feature_map[:, self.desc_dim:]
        return score_map, desc_map


if __name__ == '__main__':
    from thop import profile
    net = DISK()
    weight = torch.load('../weights/disk.pth')
    net.load_state_dict(weight['extractor'])
    image = torch.randn(1, 3, 512, 512)
    flops, params = profile(net, inputs=(image,))
    print('{:<30}  {:<8} GFLops'.format('Computational complexity: ', flops / 1e9))
    print('{:<30}  {:<8} KB'.format('Number of parameters: ', params / 1e3))
    # export_model(net, '/home/server/linyicheng/py_proj/keypoint_bench/keypoint_bench/weights/disk')
