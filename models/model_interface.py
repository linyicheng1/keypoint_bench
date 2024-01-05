import inspect
import logging
import cv2
import torch
import importlib
import numpy as np
from torch import Tensor
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
import utils
# import models
from models.ALike import ALNet
from models.GoodPoint import GoodPoint
from models.KeyNet import KeyNet
from models.LETNet import LETNet
from models.SuperPoint import SuperPoint

# import tasks
from tasks.VisualizeTrackingError import visualize_tracking_error


class MInterface(pl.LightningModule):
    def __init__(self, params) -> None:
        super().__init__()
        self.params = params

        # model choice
        if params['model_type'] == 'Alike':
            self.model = ALNet(params['Alike_params'])
        elif params['model_type'] == 'GoodPoint':
            self.model = GoodPoint(params['GoodPoint_params'])
        elif params['model_type'] == 'KeyNet':
            self.model = KeyNet(params['KeyNet_params'])
        elif params['model_type'] == 'LETNet':
            self.model = LETNet(params['LETNet_params'])
        elif params['model_type'] == 'SuperPoint':
            self.model = SuperPoint(params['SuperPoint_params'])
        else:
            raise NotImplementedError
        self.model.eval()
        self.num_feat = None
        self.repeatability = None
        self.accuracy = None
        self.matching_score = None

    def on_test_start(self) -> None:
        self.num_feat = []
        self.repeatability = []
        self.accuracy = []
        self.matching_score = []

    def on_test_end(self) -> None:
        self.num_feat = np.mean(self.num_feat)
        self.repeatability = np.mean(self.repeatability)
        self.accuracy = np.mean(self.accuracy)
        self.matching_score = np.mean(self.matching_score)

    def test_step(self, batch: Tensor, batch_idx: int) -> STEP_OUTPUT:

        warp01_params = {}
        warp10_params = {}
        if 'warp01_params' in batch:
            for k, v in batch['warp01_params'].items():
                warp01_params[k] = v[0]
            for k, v in batch['warp10_params'].items():
                warp10_params[k] = v[0]

        # pairs dataset
        score_map_0 = None
        score_map_1 = None
        desc_map_0 = None
        desc_map_1 = None

        if batch['dataset'] == 'HPatches' or batch['dataset'] == 'megaDepth':
            result0 = self.model(batch['image0'])
            result1 = self.model(batch['image1'])
            score_map_0 = result0[0].detach().cpu()
            score_map_1 = result1[0].detach().cpu()
            desc_map_0 = result0[1].detach().cpu()
            desc_map_1 = result1[1].detach().cpu()

        # task
        if self.params['task_type'] == 'VisualizeTrackingError':
            visualize_tracking_error(score_map_0, score_map_1, desc_map_0, desc_map_1, self.params)

        return {}
