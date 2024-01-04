import inspect
import logging
import cv2
from utils.logger import board
import torch
import importlib
import numpy as np
from torch import Tensor
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from .MLPoint import ML_Point
import utils
from model.losses import projection_loss, local_loss, OpticalFlowLoss, PeakyLoss
from model.SuperPoint import SuperPoint
from model.LETNet import LETNet


class MInterface(pl.LightningModule):
    def __init__(self, params) -> None:
        super().__init__()
        self.params = params
        self.model = ML_Point(params['model_size'])
        # self.sp = SuperPoint(params['weight'])
        self.model = self.model.to(self.device)
        if params['pretrained']:
            self.model.load_state_dict(torch.load(params['weight']))
        self.board = None
        self.losses = {}
        self.op_loss = OpticalFlowLoss()
        self.pk_loss = PeakyLoss()
        # vals
        self.num_feat = None
        self.repeatability = None
        self.accuracy = None
        self.matching_score = None
        self.max_repeatability = 0

        self.letnet = LETNet(c1=8, c2=16, grayscale=False)
        weight = torch.load('/home/server/wangshuo/ALIKE_code/weight/model.pt')
        # load pretrained model
        net_dict = self.letnet.state_dict()
        pretrained_dict = {k: v for k, v in weight.items() if k in net_dict}
        net_dict.update(pretrained_dict)
        self.letnet.load_state_dict(net_dict)

    def on_train_start(self) -> None:
        self.board = board(self.logger)

    def forward(self, img: Tensor):
        return self.model(img)

    def loss(self, scores_map_0: Tensor, scores_map_1: Tensor,
             desc_map_00: Tensor, desc_map_01: Tensor, batch) -> Tensor:

        # 1. gt values
        # warp
        warp01_params = {}
        for k, v in batch['warp01_params'].items():
            warp01_params[k] = v[0]
        warp10_params = {}
        for k, v in batch['warp10_params'].items():
            warp10_params[k] = v[0]

        # 2. detection and description
        kps_0 = utils.detection(scores_map_0, nms_dist=8, threshold=0.1, max_pts=500)
        kps_1 = utils.detection(scores_map_1, nms_dist=8, threshold=0.1, max_pts=500)

        # 2.1 random add 30 points
        # random_kps_0 = (torch.rand(500, 3).to(self.device) - 0.5) * 0.01
        # random_kps_1 = (torch.rand(500, 3).to(self.device) - 0.5) * 0.01
        # kps_0 = kps_0 + random_kps_0
        # kps_1 = kps_1 + random_kps_1

        # random_kps_0 = torch.rand(100, 3).to(self.device)
        # random_kps_1 = torch.rand(100, 3).to(self.device)
        # kps_0 = torch.cat([kps_0, random_kps_0], dim=0)
        # kps_1 = torch.cat([kps_1, random_kps_1], dim=0)

        kps0_valid, kps01_valid, ids01, ids01_out = utils.warp(kps_0, warp01_params)
        kps1_valid, kps10_valid, ids10, ids10_out = utils.warp(kps_1, warp10_params)

        # desc
        sample_pts_0 = kps0_valid * 2. - 1.
        sample_pts_1 = kps1_valid * 2. - 1.
        desc_00 = F.grid_sample(desc_map_00, sample_pts_0.unsqueeze(0).unsqueeze(0),
                                mode='bilinear', align_corners=True, padding_mode='zeros')
        desc_01 = F.grid_sample(desc_map_01, sample_pts_1.unsqueeze(0).unsqueeze(0),
                                mode='bilinear', align_corners=True, padding_mode='zeros')

        sample_pts_0_out = kps_0[ids01_out][:, :-1] * 2. - 1.
        sample_pts_1_out = kps_1[ids10_out][:, :-1] * 2. - 1.
        desc_00_out = F.grid_sample(desc_map_00, sample_pts_0_out.unsqueeze(0).unsqueeze(0),
                                    mode='bilinear', align_corners=True, padding_mode='zeros')
        desc_01_out = F.grid_sample(desc_map_01, sample_pts_1_out.unsqueeze(0).unsqueeze(0),
                                    mode='bilinear', align_corners=True, padding_mode='zeros')

        score0 = F.grid_sample(scores_map_0, sample_pts_0.unsqueeze(0).unsqueeze(0),
                               mode='bilinear', align_corners=True, padding_mode='zeros')
        score1 = F.grid_sample(scores_map_1, sample_pts_1.unsqueeze(0).unsqueeze(0),
                               mode='bilinear', align_corners=True, padding_mode='zeros')
        # 3. loss
        # 3.1 projection loss
        # loss_kps_0 = projection_loss(kps01_valid.unsqueeze(0), score0.view(1, -1, 1), scores_map_1, 4)
        # loss_kps_1 = projection_loss(kps10_valid.unsqueeze(0), score1.view(1, -1, 1), scores_map_0, 4)

        # 3.1 optical flow loss
        loss_kps_0, error_01 = self.op_loss(desc_map_00.detach(), desc_map_01.detach(),
                                            kps0_valid, kps01_valid, score0)
        loss_kps_1, error_10 = self.op_loss(desc_map_01.detach(), desc_map_00.detach(),
                                            kps1_valid, kps10_valid, score1)
        # 3.1 peaky loss
        loss_kps_0 += self.pk_loss(scores_map_0, kps_0[:, :-1]) * 0.1
        loss_kps_1 += self.pk_loss(scores_map_1, kps_1[:, :-1]) * 0.1

        if torch.mean(scores_map_0.detach()) > 0.3:
            loss_kps_0 += torch.mean(scores_map_0)
        if torch.mean(scores_map_1.detach()) > 0.3:
            loss_kps_1 += torch.mean(scores_map_1)
        # 3.2 local consistency loss
        loss_desc_00 = local_loss(desc_00.squeeze(2).transpose(1, 2),
                                  desc_00_out.squeeze(2).transpose(1, 2),
                                  desc_map_01, kps01_valid.unsqueeze(0), win_size=32)
        loss_desc_01 = local_loss(desc_01.squeeze(2).transpose(1, 2),
                                  desc_01_out.squeeze(2).transpose(1, 2),
                                  desc_map_00, kps10_valid.unsqueeze(0), win_size=32)

        # 3.4 total loss
        proj_weight = self.params['loss']['projection_loss_weight']
        cons_weight = self.params['loss']['local_consistency_loss_weight']

        self.losses = {
            'image0': batch['image0'],
            'image1': batch['image1'],
            'scores_map_0': scores_map_0,
            'scores_map_1': scores_map_1,
            'loss_kps_0': loss_kps_0,
            'loss_kps_1': loss_kps_1,
            'kps_0': kps_0[:, :2],
            'kps_1': kps_1[:, :2],
            'kps_0_v': kps0_valid,
            'kps_1_v': kps1_valid,
            'match_01': error_01[0],
            'match_10': error_10[0],
            'loss_desc_00': loss_desc_00,
            'loss_desc_01': loss_desc_01,
            'desc_map_00': desc_map_00,
            'desc_map_01': desc_map_01,
        }

        return proj_weight * (loss_kps_0 + loss_kps_1) + cons_weight * (loss_desc_00 + loss_desc_01)

    def step(self, batch: Tensor) -> Tensor:
        if self.params['model_size']['c0'] == 1:
            result0 = self(batch['gray0'])
            result1 = self(batch['gray1'])
        else:
            result0 = self(batch['image0'])
            result1 = self(batch['image1'])
        loss = self.loss(result0[0], result1[0], result0[1], result1[1], batch)
        return loss

    def training_step(self, batch: Tensor, batch_idx: int) -> STEP_OUTPUT:
        return {"loss": self.step(batch)}

    def configure_optimizers(self):
        if 'weight_decay' in self.params:
            weight_decay = self.params['weight_decay']
        else:
            weight_decay = 0
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.params['lr'], weight_decay=weight_decay)

        if 'lr_scheduler' not in self.params:
            return optimizer
        else:
            if self.params['lr_scheduler']['type'] == 'step':
                scheduler = lrs.StepLR(optimizer,
                                       step_size=self.params['lr_scheduler']['step_size'],
                                       gamma=self.params['lr_scheduler']['gamma'])
            elif self.params['lr_scheduler']['type'] == 'cosine':
                scheduler = lrs.CosineAnnealingLR(optimizer,
                                                  T_max=self.params['lr_scheduler']['lr_decay_steps'],
                                                  eta_min=self.params['lr_scheduler']['lr_decay_min_lr'])
            else:
                raise ValueError('Invalid lr_scheduler type!')
            return [optimizer], [scheduler]

    def backward(self, loss: Tensor, *args, **kwargs) -> None:
        if loss.requires_grad:
            loss.backward()
        else:
            logging.debug('loss does not require grad, skipping backward')

    def on_train_batch_end(self,  *args, **kwargs):
        if self.global_step % 20 == 0:
            # log
            self.board.add_local_loss0(self.losses['loss_desc_00'], self.losses['loss_desc_01'], self.global_step)
            self.board.add_projection_loss(self.losses['loss_kps_0'], self.losses['loss_kps_1'], self.global_step)

            # score map
            self.board.add_score_map(self.losses['scores_map_0'], self.losses['scores_map_1'], self.global_step)

            # images
            show_0 = (self.losses['image0'][0].cpu().permute(1, 2, 0).numpy() * 255).astype('uint8')
            show_1 = (self.losses['image1'][0].cpu().permute(1, 2, 0).numpy() * 255).astype('uint8')

            # kps image
            show_3 = utils.plot_keypoints(show_0, self.losses['kps_0'].cpu())
            show_7 = utils.plot_keypoints(show_1, self.losses['kps_1'].cpu())

            show_8 = utils.plot_keypoints(show_0, self.losses['kps_0_v'].cpu(), self.losses['match_01'].cpu(), radius=4)
            show_9 = utils.plot_keypoints(show_1, self.losses['kps_1_v'].cpu(), self.losses['match_10'].cpu(), radius=4)

            # draw images
            self.board.add_local_desc_map(self.losses['desc_map_00'], self.losses['desc_map_01'], self.global_step)
            self.board.add_keypoint_image(show_3, show_7, show_8, show_9, self.global_step)

        # 100 times save model
        if self.global_step % 100 == 0:
            torch.save(self.model.state_dict(),
                       "/home/server/linyicheng/py_proj/MLPoint/weight/last_{}.pth".format(self.global_step))

    def validation_step(self, batch: Tensor, batch_idx: int) -> STEP_OUTPUT:
        warp01_params = {}
        for k, v in batch['warp01_params'].items():
            warp01_params[k] = v[0]
        warp10_params = {}
        for k, v in batch['warp10_params'].items():
            warp10_params[k] = v[0]
        self.model.eval()
        result0 = self(batch['image0'])
        result1 = self(batch['image1'])
        nms_dist = self.params['nms_dist']
        threshold = self.params['threshold']
        top_k = self.params['top_k']
        min_score = self.params['min_score']
        # sp
        # result0 = self.sp(batch['image0'].cpu())
        # result1 = self.sp(batch['image1'].cpu())
        # harris
        # img0 = (torch.sum(batch['image0'][0].cpu().permute(1, 2, 0), dim=2) * 255).numpy().astype('uint8')
        # img1 = (torch.sum(batch['image1'][0].cpu().permute(1, 2, 0), dim=2) * 255).numpy().astype('uint8')
        #
        # harris_map_0 = torch.from_numpy(cv2.cornerHarris(img0, 2, 3, 0.04)).unsqueeze(0).unsqueeze(0)
        # harris_map_1 = torch.from_numpy(cv2.cornerHarris(img1, 2, 3, 0.04)).unsqueeze(0).unsqueeze(0)
        # kps_0 = utils.detection(harris_map_0.to(batch['image0'].device), nms_dist=4, threshold=0.0, max_pts=1000)
        # kps_1 = utils.detection(harris_map_1.to(batch['image0'].device), nms_dist=4, threshold=0.0, max_pts=1000)

        kps_0 = utils.detection(result0[0].to(batch['image0'].device), nms_dist=nms_dist, threshold=min_score,
                                max_pts=top_k)
        kps_1 = utils.detection(result1[0].to(batch['image0'].device), nms_dist=nms_dist, threshold=min_score,
                                max_pts=top_k)

        val_pts = utils.val_key_points(kps_0, kps_1, warp01_params, warp10_params, th=threshold)
        # val_matches = utils.val_matches(kps_0, kps_1, result0[1], result1[1],
        #                                 warp01_params, warp10_params)

        if val_pts['num_feat'] > 0:
            self.num_feat.append(val_pts['num_feat'])
            self.repeatability.append(val_pts['repeatability'])
            # self.matching_score.append(val_matches['tracking_error'])
        return {"num_feat_mean": 0,
                "repeatability_mean": 0,
                }

    def on_validation_start(self):
        self.num_feat = []
        self.repeatability = []
        self.accuracy = []
        self.matching_score = []
        # add optical flow
        self.model.eval()
        # 1. key points sample
        img0 = cv2.imread('test/socre_map.png')
        img0 = torch.from_numpy(img0).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        result = self(img0.to(self.device))
        kps_0 = utils.detection(result[0].to(self.device), nms_dist=4, threshold=0.0, max_pts=100)
        show_0 = (img0[0].cpu().permute(1, 2, 0).numpy() * 255).astype('uint8')
        show_0 = utils.plot_keypoints(show_0, kps_0.cpu())
        cv2.imwrite('test/kps.png', show_0)
        # 2. optical flow tow image
        # img0_raw = cv2.imread('test/1.JPG')
        # img1_raw = cv2.imread('test/2.JPG')
        # img0 = torch.from_numpy(img0_raw).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        # img1 = torch.from_numpy(img1_raw).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        # result0 = self(img0.to(self.device))
        # result1 = self(img1.to(self.device))
        # kps_0 = utils.detection(result0[0].detach().to(self.device), nms_dist=4, threshold=0.0, max_pts=100)
        # kps_1 = utils.detection(result1[0].detach().to(self.device), nms_dist=4, threshold=0.0, max_pts=100)
        # show_0 = (img0[0].cpu().permute(1, 2, 0).numpy() * 255).astype('uint8')
        # show_1 = (img1[0].cpu().permute(1, 2, 0).numpy() * 255).astype('uint8')
        # show_0 = utils.plot_keypoints(show_0, kps_0.cpu())
        # show_1 = utils.plot_keypoints(show_1, kps_1.cpu())
        # desc_0 = (result0[0].detach().cpu().permute(1, 2, 0).numpy() * 255).astype('uint8')
        # desc_1 = (result1[0].detach().cpu().permute(1, 2, 0).numpy() * 255).astype('uint8')

        # 3. optical flow video


    def on_validation_end(self) -> None:
        if self.global_step < 10:  # first epoch is warming up, so skip it
            return
        num_feat_mean = np.mean(np.array(self.num_feat))
        repeatability_mean = np.mean(np.array(self.repeatability))
        if self.max_repeatability < repeatability_mean:
            self.max_repeatability = repeatability_mean
            torch.save(self.model.state_dict(),
                       "/home/server/linyicheng/py_proj/MLPoint-LET/MLPoint/weight/best_{}.pth".format(repeatability_mean))

        print('num_feat_mean: ', num_feat_mean)
        print('repeatability_mean: ', repeatability_mean)
        self.logger.experiment.add_scalar('val/num_feat_mean', num_feat_mean, self.global_step)
        self.logger.experiment.add_scalar('val/repeatability_mean', repeatability_mean, self.global_step)

    def on_test_start(self) -> None:
        self.num_feat = []
        self.repeatability = []
        self.accuracy = []
        self.matching_score = []

    def on_test_end(self) -> None:
        num_feat_mean = np.mean(np.array(self.num_feat))
        repeatability_mean = np.mean(np.array(self.repeatability))

        matching_score = np.array(self.matching_score)
        matching_1 = np.mean(matching_score[:, 0])
        matching_2 = np.mean(matching_score[:, 1])
        matching_3 = np.mean(matching_score[:, 2])
        matching_4 = np.mean(matching_score[:, 3])
        matching_5 = np.mean(matching_score[:, 4])
        print('num_feat_mean: ', num_feat_mean)
        print('repeatability_mean: ', repeatability_mean)
        print('matching_mean 1: ', matching_1)
        print('matching_mean 2: ', matching_2)
        print('matching_mean 3: ', matching_3)
        print('matching_mean 4: ', matching_4)
        print('matching_mean 5: ', matching_5)

    def test_step(self, batch: Tensor, batch_idx: int) -> STEP_OUTPUT:
        warp01_params = {}
        for k, v in batch['warp01_params'].items():
            warp01_params[k] = v[0]
        warp10_params = {}
        for k, v in batch['warp10_params'].items():
            warp10_params[k] = v[0]
        self.model.eval()
        result0 = self(batch['image0'])
        result1 = self(batch['image1'])
        score_map_0 = result0[0].detach().cpu()
        score_map_1 = result1[0].detach().cpu()
        desc_map_0 = result0[1].detach().cpu()
        desc_map_1 = result1[1].detach().cpu()

        nms_dist = self.params['nms_dist']
        threshold = self.params['threshold']
        top_k = self.params['top_k']
        min_score = self.params['min_score']

        # temp
        img1 = cv2.imread('/home/server/linyicheng/py_proj/MLPoint-Feature/MLPoint/data/1.png')
        img2 = cv2.imread('/home/server/linyicheng/py_proj/MLPoint-Feature/MLPoint/data/2.png')

        # resize
        img1 = cv2.resize(img1, (640, 480))
        img2 = cv2.resize(img2, (640, 480))

        # convert to tensor
        img1_t = torch.tensor(img1, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255
        img2_t = torch.tensor(img2, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255

        result0 = self(img1_t.to(self.device))
        result1 = self(img2_t.to(self.device))

        scores_map1 = result0[0].detach().cpu()
        local_desc1 = result0[1].detach().cpu()
        scores_map2 = result1[0].detach().cpu()
        local_desc2 = result1[1].detach().cpu()

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
        #     cv2.line(show, (int(kps_0[i][0]), int(kps_0[i][1])), (int(kps_1[i][0] + W), int(kps_1[i][1])), (0, 255, 0),
        #              1)

        cv2.imshow('optical flow', show)
        cv2.waitKey(0)


        # sp
        # result0 = self.sp(batch['image0'].cpu())
        # result1 = self.sp(batch['image1'].cpu())
        # harris
        # img0 = (torch.sum(batch['image0'][0].cpu().permute(1, 2, 0), dim=2) * 255).numpy().astype('uint8')
        # img1 = (torch.sum(batch['image1'][0].cpu().permute(1, 2, 0), dim=2) * 255).numpy().astype('uint8')
        #
        # harris_map_0 = torch.from_numpy(cv2.cornerHarris(img0, 2, 3, 0.04)).unsqueeze(0).unsqueeze(0)
        # harris_map_1 = torch.from_numpy(cv2.cornerHarris(img1, 2, 3, 0.04)).unsqueeze(0).unsqueeze(0)
        # kps_0 = utils.detection(harris_map_0.to(batch['image0'].device), nms_dist=nms_dist, threshold=min_score,
        #                         max_pts=top_k)
        # kps_1 = utils.detection(harris_map_1.to(batch['image0'].device), nms_dist=nms_dist, threshold=min_score,
        #                         max_pts=top_k)
        # let net
        # to gray
        # gray0 = torch.sum(batch['image0'], dim=1).unsqueeze(0)
        # gray1 = torch.sum(batch['image1'], dim=1).unsqueeze(0)
        score_map_0, desc_map_0 = self.letnet(batch['image0'])
        score_map_1, desc_map_1 = self.letnet(batch['image0'])

        kps_0 = utils.detection(score_map_0.to(batch['image0'].device), nms_dist=nms_dist, threshold=min_score,
                                max_pts=top_k)
        kps_1 = utils.detection(score_map_1.to(batch['image0'].device), nms_dist=nms_dist, threshold=min_score,
                                max_pts=top_k)

        val_pts = utils.val_key_points(kps_0, kps_1, warp01_params, warp10_params, th=threshold)
        val_matches = utils.val_matches(kps_0, kps_1, batch['image0'], batch['image1'],
                                        warp01_params, warp10_params)
        # val_matches = utils.val_matches(kps_0, kps_1, desc_map_0, desc_map_1,
        #                                 warp01_params, warp10_params)
        if val_pts['num_feat'] > 0:
            self.num_feat.append(val_pts['num_feat'])
            self.repeatability.append(val_pts['repeatability'])
            self.matching_score.append(val_matches['tracking_error'])

        print('step: ', batch_idx, 'num_feat: ', val_pts['num_feat'], 'repeatability: ', val_pts['repeatability'],
              'matching: ', val_matches['tracking_error'])

        return {"num_feat_mean": 0,
                "repeatability_mean": 0,
                }
