import torch
from torch import nn
import cv2
import numpy as np


class Harris(nn.Module):

    def __init__(self, params):
        super().__init__()
        self.params = params

    def forward(self, x):
        # 1. to numpy
        img0 = (torch.sum(x[0].cpu().permute(1, 2, 0), dim=2) * 255).numpy().astype('uint8')
        block_size = self.params['block_size']
        ksize = self.params['ksize']
        k = self.params['k']
        harris_map_0 = torch.from_numpy(cv2.cornerHarris(img0, block_size, ksize, k)).unsqueeze(0).unsqueeze(0)
        harris_map_0 = harris_map_0.to(x.device)
        return harris_map_0, None




