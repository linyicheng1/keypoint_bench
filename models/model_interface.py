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
from models.Harris import Harris

# import tasks
from tasks.VisualizeTrackingError import visualize_tracking_error, plot_tracking_error
from tasks.repeatability import repeatability, plot_repeatability


class MInterface(pl.LightningModule):
    def __init__(self, params) -> None:
        super().__init__()
        self.params = params

        # model choice
        if params['model_type'] == 'Alike':
            self.model = ALNet(params['Alike_params'])
            self.model.load_state_dict(torch.load(params['Alike_params']['weight']))
        elif params['model_type'] == 'GoodPoint':
            self.model = GoodPoint(params['GoodPoint_params'])
            self.model.load_state_dict(torch.load(params['GoodPoint_params']['weight']))
        elif params['model_type'] == 'KeyNet':
            self.model = KeyNet(params['KeyNet_params'])
            checkpoint = torch.load(params['KeyNet_params']['weight'])
            self.model.load_state_dict(checkpoint['state_dict'])
        elif params['model_type'] == 'LETNet':
            self.model = LETNet(params['LETNet_params'])
        elif params['model_type'] == 'SuperPoint':
            self.model = SuperPoint(params['SuperPoint_params'])
        elif params['model_type'] == 'Harris':
            self.model = Harris(params['Harris_params'])
        else:
            raise NotImplementedError
        self.model.eval()
        self.num_feat = None
        self.repeatability = None
        self.accuracy = None
        self.matching_score = None
        self.track_error = None

    def on_test_start(self) -> None:
        self.num_feat = []
        self.repeatability = []
        self.accuracy = []
        self.matching_score = []
        self.track_error = []

    def on_test_end(self) -> None:
        self.num_feat = np.mean(self.num_feat)
        self.accuracy = np.mean(self.accuracy)
        self.matching_score = np.mean(self.matching_score)

        if self.params['task_type'] == 'repeatability':
            rep = torch.as_tensor(self.repeatability).cpu().numpy()
            plot_repeatability(rep, self.params['repeatability_params']['save_path'])
            rep = np.mean(rep)
            print('repeatability', rep)
        elif self.params['task_type'] == 'VisualizeTrackingError':
            error = torch.as_tensor(self.track_error).cpu().numpy()
            plot_tracking_error(error, self.params['VisualizeTrackingError_params']['save_path'])
            error = np.mean(error)
            print('track_error', error)

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

        if batch['dataset'][0] == 'HPatches' or batch['dataset'][0] == 'megaDepth':
            result0 = self.model(batch['image0'])
            result1 = self.model(batch['image1'])
            score_map_0 = result0[0].detach()
            score_map_1 = result1[0].detach()
            if result0[1] is not None:
                desc_map_0 = result0[1].detach()
                desc_map_1 = result1[1].detach()

        # task
        result = None
        if self.params['task_type'] == 'VisualizeTrackingError':
            if self.params['model_type'] == 'LETNet':
            # if self.params['model_type'] == 'LETNet' or self.params['model_type'] == 'GoodPoint':
                result = visualize_tracking_error(batch_idx, batch['image0'], score_map_0,
                                                  desc_map_0, desc_map_1,
                                                  self.params, warp01_params)
            else:
                result = visualize_tracking_error(batch_idx, batch['image0'], score_map_0,
                                                  batch['image0'], batch['image1'],
                                                  self.params, warp01_params)
            self.track_error.append(result['track_error'])
        elif self.params['task_type'] == 'repeatability':
            result = repeatability(batch_idx, batch['image0'], score_map_0,
                                   batch['image1'], score_map_1,
                                   warp01_params, warp10_params, self.params)
            self.num_feat.append(result['num_feat'])
            self.repeatability.append(result['repeatability'])
        return result
