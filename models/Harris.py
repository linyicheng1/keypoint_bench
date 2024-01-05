import torch
from torch import nn
import cv2
import numpy as np


class Harris(nn.Module):

    def __init__(self, params):
        super().__init__()
        self.params = params

    def forward(self, x):
        return x




