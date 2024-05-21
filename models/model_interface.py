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
from models.SuperPoint import SuperPointNet
from models.Harris import Harris
from models.D2_Net import D2Net
from models.XFeat import XFeatModel
from models.disk import DISK
from models.r2d2 import *
from models.sfd2 import ResSegNetV2
from models.lightglue import LightGlue


# import tasks
from tasks.VisualizeTrackingError import visualize_tracking_error, plot_tracking_error
from tasks.repeatability import repeatability, plot_repeatability
from tasks.FundamentalMatrix import fundamental_matrix, plot_fundamental_matrix, fundamental_matrix_ransac
from tasks.visual_odometer import visual_odometry, plot_visual_odometry


class MInterface(pl.LightningModule):
    def __init__(self, params) -> None:
        super().__init__()
        self.params = params
        self.matcher = None
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
            self.model = SuperPointNet()
            self.model.load_state_dict(torch.load(params['SuperPoint_params']['weight']))
            if params['matcher_params']['type'] == 'light_glue':
                self.matcher = LightGlue(features="superpoint", weight_path=params['matcher_params']['light_glue_params']['weight'])
        elif params['model_type'] == 'D2Net':
            self.model = D2Net(model_file=params['D2Net_params']['weight'])
        elif params['model_type'] == 'XFeat':
            self.model = XFeatModel()
            self.model.load_state_dict(torch.load(params['XFeat_params']['weight']))
        elif params['model_type'] == 'r2d2':
            checkpoint = torch.load(params['r2d2_params']['weight'])
            self.model = eval(checkpoint['net'])
            weights = checkpoint['state_dict']
            self.model.load_state_dict({k.replace('module.', ''): v for k, v in weights.items()})
        elif params['model_type'] == 'sfd2':
            self.model = ResSegNetV2(outdim=128, require_stability=True).eval()
            self.model.load_state_dict(torch.load(params['sfd2_params']['weight'])['model'], strict=False)
        elif params['model_type'] == 'DISK':
            self.model = DISK()
            self.model.load_state_dict(torch.load(params['DISK_params']['weight'])['extractor'])
        elif params['model_type'] == 'Harris':
            self.model = Harris(params['Harris_params'])
        else:
            raise NotImplementedError
        self.model.eval()
        self.num_feat = None
        self.repeatability = None
        self.rep_mean_err = None
        self.accuracy = None
        self.matching_score = None
        self.track_error = None
        self.last_batch = None
        self.fundamental_error = None
        self.fundamental_radio = None
        self.fundamental_num = None
        self.r_est = None
        self.t_est = None

    def on_test_start(self) -> None:
        self.num_feat = []
        self.repeatability = []
        self.rep_mean_err = []
        self.accuracy = []
        self.matching_score = []
        self.track_error = []
        self.fundamental_error = []
        self.fundamental_radio = []
        self.fundamental_num = []
        self.r_est = [np.eye(3)]
        self.t_est = [np.zeros([3, 1])]

    def on_test_end(self) -> None:
        self.num_feat = np.mean(self.num_feat)
        self.accuracy = np.mean(self.accuracy)
        self.matching_score = np.mean(self.matching_score)
        print('task: ', self.params['task_type'])
        if self.params['task_type'] == 'repeatability':
            rep = torch.as_tensor(self.repeatability).cpu().numpy()
            plot_repeatability(rep, self.params['repeatability_params']['save_path'])
            rep = np.mean(rep)

            error = torch.as_tensor(self.rep_mean_err).cpu().numpy()
            error = error[~np.isnan(error)]
            plot_repeatability(error, self.params['repeatability_params']['save_path'].replace('.png', '_error.png'))
            error = np.mean(error)
            print('repeatability', rep, ' rep_mean_err', error)
        elif self.params['task_type'] == 'VisualizeTrackingError':
            error = torch.as_tensor(self.track_error).cpu().numpy()
            plot_tracking_error(error, self.params['VisualizeTrackingError_params']['save_path'])
            error = np.mean(error)
            print('track_error', error)
        elif self.params['task_type'] == 'FundamentalMatrix':
            error = torch.as_tensor(self.fundamental_error).cpu().numpy()
            plot_fundamental_matrix(error, self.params['FundamentalMatrix_params']['save_path'])
            error = np.mean(error)

            radio = torch.as_tensor(self.fundamental_radio).cpu().numpy()
            plot_fundamental_matrix(radio, self.params['FundamentalMatrix_params']['save_path'].replace('.png', '_radio.png'))
            radio = np.mean(radio)

            num = torch.as_tensor(self.fundamental_num).cpu().numpy()
            plot_fundamental_matrix(num, self.params['FundamentalMatrix_params']['save_path'].replace('.png', '_num.png'))
            num = np.mean(num)

            print('fundamental_error', error, ' fundamental_radio', radio, ' fundamental_num', num)
        elif self.params['task_type'] == 'visual_odometry':
            self.r_est = torch.as_tensor(self.r_est).cpu().numpy()
            self.t_est = torch.as_tensor(self.t_est).cpu().numpy()
            np.save(self.params['visual_odometer_params']['save_path'].replace('.png', '_r_est.npy'), self.r_est)
            np.save(self.params['visual_odometer_params']['save_path'].replace('.png', '_t_est.npy'), self.t_est)
            plot_visual_odometry(self.r_est, self.t_est, self.params['visual_odometer_params']['save_path'])

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

        # image pair dataset
        if batch['dataset'][0] == 'HPatches' or \
           batch['dataset'][0] == 'megaDepth' or \
           batch['dataset'][0] == 'image_pair':
            result0 = self.model(batch['image0'])
            result1 = self.model(batch['image1'])
            score_map_0 = result0[0].detach()
            score_map_1 = result1[0].detach()
            if result0[1] is not None:
                desc_map_0 = result0[1].detach()
                desc_map_1 = result1[1].detach()

        # sequence dataset
        last_img = None

        if batch['dataset'][0] == 'Kitti' or batch['dataset'][0] == 'Euroc' or batch['dataset'][0] == 'TartanAir':
            if self.last_batch is None:
                self.last_batch = batch
            result0 = self.model(self.last_batch['image0'])
            result1 = self.model(batch['image0'])
            score_map_0 = result0[0].detach()
            score_map_1 = result1[0].detach()
            if result0[1] is not None:
                desc_map_0 = result0[1].detach()
                desc_map_1 = result1[1].detach()
            last_img = self.last_batch['image0']
            self.last_batch = batch
        # task
        result = None
        if self.params['task_type'] == 'VisualizeTrackingError':
            # if self.params['model_type'] == 'LETNet':
            if self.params['model_type'] == 'LETNet' or self.params['model_type'] == 'GoodPoint':
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
            self.rep_mean_err.append(result['mean_error'])
        elif self.params['task_type'] == 'FundamentalMatrix':
            if self.params['matcher_params']['type'] == 'optical_flow' and \
                    (self.params['model_type'] != 'LETNet' and self.params['model_type'] != 'GoodPoint'):
                # self.params['model_type'] != 'LETNet' and self.params['model_type'] != 'GoodPoint':
                result = fundamental_matrix(batch_idx, last_img, batch,
                                            score_map_0, score_map_1,
                                            last_img, batch['image0'], self.params)
                # print(result['fundamental_error'], result['fundamental_radio'])
            else:
                result = fundamental_matrix(batch_idx, last_img, batch,
                                            score_map_0, score_map_1,
                                            desc_map_0, desc_map_1, self.params)
                # print(result['fundamental_error'], result['fundamental_radio'])
            self.fundamental_error.append(result['fundamental_error'])
            self.fundamental_radio.append(result['fundamental_radio'])
            self.fundamental_num.append(result['fundamental_num'])
        elif self.params['task_type'] == 'FundamentalMatrixRansac':
            result = fundamental_matrix_ransac(batch_idx, batch['image0'], batch['image1'],
                                               score_map_0, score_map_1,
                                               desc_map_0, desc_map_1,
                                               self.matcher, self.params)
            self.fundamental_radio.append(result['fundamental_radio'])
        elif self.params['task_type'] == 'visual_odometry':
            if self.params['matcher_params']['type'] == 'optical_flow' and \
                    (self.params['model_type'] != 'LETNet' or self.params['model_type'] != 'GoodPoint'):
                result = visual_odometry(batch_idx, self.r_est[-1], self.t_est[-1],
                                         last_img, batch,
                                         score_map_0, score_map_1,
                                         last_img, batch['image0'], self.params)
            else:
                result = visual_odometry(batch_idx, self.r_est[-1], self.t_est[-1],
                                         last_img, batch,
                                         score_map_0, score_map_1,
                                         desc_map_0, desc_map_1, self.params)
            self.r_est.append(result['R'])
            self.t_est.append(result['t'])
        return result
