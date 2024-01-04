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


def optical_flow(model):
    import cv2
    import numpy as np

    img1 = cv2.imread('/home/server/linyicheng/py_proj/MLPoint-Feature/MLPoint/data/1.png')
    img2 = cv2.imread('/home/server/linyicheng/py_proj/MLPoint-Feature/MLPoint/data/2.png')

    # resize
    img1 = cv2.resize(img1, (640, 480))
    img2 = cv2.resize(img2, (640, 480))

    # convert to tensor
    img1_t = torch.tensor(img1, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255
    img2_t = torch.tensor(img2, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255

    scores_map1, local_desc1 = model(img1_t)
    scores_map2, local_desc2 = model(img2_t)

    kps_0 = utils.detection(scores_map1, nms_dist=10, threshold=0.01,
                            max_pts=100)
    # to numpy
    _, C, H, W = img1_t.shape
    kps_0 = kps_0[:, :-1].squeeze().numpy() * np.array([W - 1, H - 1])
    kps_0 = kps_0.astype(np.float32)
    local_desc1 = local_desc1.detach().squeeze().permute(1, 2, 0).numpy() * 255
    local_desc2 = local_desc2.detach().squeeze().permute(1, 2, 0).numpy() * 255
    local_desc1 = local_desc1.astype(np.uint8)
    local_desc2 = local_desc2.astype(np.uint8)

    # optical flow
    kps_1, st, err = cv2.calcOpticalFlowPyrLK(local_desc1, local_desc2, kps_0, None, winSize=(15, 15), maxLevel=3,
                                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # draw corners
    show = np.concatenate((img1, img2), axis=1)
    for i in range(len(kps_0)):
        cv2.circle(show, (int(kps_0[i][0]), int(kps_0[i][1])), 3, (0, 0, 255), -1)

    # for i in range(len(kps_1)):
    #     cv2.line(show, (int(kps_0[i][0]), int(kps_0[i][1])), (int(kps_1[i][0] + W), int(kps_1[i][1])), (0, 255, 0), 1)

    cv2.imshow('optical flow', show)
    cv2.waitKey(0)

    # to gray
    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    #
    # # goodFeaturesToTrack
    # corners = cv2.goodFeaturesToTrack(img1, 100, 0.01, 10)
    #
    # # calculate optical flow
    # p1, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, corners, None, winSize=(21, 21), maxLevel=3,
    #                                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    #
    # # show optical flow
    # show = np.concatenate((img1, img2), axis=1)
    # show = cv2.cvtColor(show, cv2.COLOR_GRAY2BGR)
    # # draw corners
    # for i in range(len(corners)):
    #     cv2.circle(show, (int(corners[i][0][0]), int(corners[i][0][1])), 3, (0, 0, 255), -1)
    # # draw optical flow
    # for i in range(len(corners)):
    #     cv2.line(show, (int(corners[i][0][0]), int(corners[i][0][1])),
    #              (int(p1[i][0][0] + 640), int(p1[i][0][1])), (0, 255, 0), 2)
    # cv2.imshow('optical flow', show)
    # cv2.waitKey(0)


if __name__ == '__main__':
    from thop import profile
    import cv2
    import numpy as np
    net = LETNet(c1=8, c2=16, grayscale=False)
    weight = torch.load('/home/server/wangshuo/ALIKE_code/training/model.pt')
    # load pretrained model
    net_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in weight.items() if k in net_dict}
    net_dict.update(pretrained_dict)
    net.load_state_dict(net_dict)

    # image = torch.tensor(np.random.random((1, 3, 240, 320)), dtype=torch.float32)
    # flops, params = profile(net, inputs=(image,))
    # print('{:<30}  {:<8} GFLops'.format('Computational complexity: ', flops / 1e9))
    # print('{:<30}  {:<8} KB'.format('Number of parameters: ', params / 1e3))
    optical_flow(net)
