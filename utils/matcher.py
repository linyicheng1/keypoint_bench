import cv2
import numpy as np
import torch


class OpticalFlow(object):
    def __init__(self, params=None):
        if params is None:
            params = {
                'distance': 3,
                'win_size': 3,
                'levels': 1,
                'interation': 40,
            }
        self.distance = params['distance']
        self.win_size = params['win_size']
        self.levels = params['levels']
        self.interation = params['interation']
        dx = torch.tensor([[[[1, 0, -1], [2, 0, -2], [1, 0, -1]]]])
        dy = torch.tensor([[[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]])
        zero = torch.zeros_like(dx)
        dx0 = torch.cat([dx, zero, zero], dim=1)
        dx1 = torch.cat([zero, dx, zero], dim=1)
        dx2 = torch.cat([zero, zero, dx], dim=1)
        dx = torch.cat([dx0, dx1, dx2], dim=0)
        dy0 = torch.cat([dy, zero, zero], dim=1)
        dy1 = torch.cat([zero, dy, zero], dim=1)
        dy2 = torch.cat([zero, zero, dy], dim=1)
        dy = torch.cat([dy0, dy1, dy2], dim=0)
        self.dx = dx
        self.dy = dy

    def build_pyramid(self, img, levels):
        """
        :param img: [B, C, H, W]
        :return:
        """
        img_pyr = [img]
        for i in range(levels - 1):
            img_pyr.append(torch.nn.functional.avg_pool2d(img, kernel_size=2*(i+1), stride=2*(i+1)))
        return img_pyr

    def __call__(self, img1, img2, pts1, pts2):
        # pts to [H - 1, W - 1]
        N, C, H, W = img1.shape
        pts1 = pts1.unsqueeze(0) * torch.tensor([W - 1, H - 1], dtype=torch.float32, device=img1.device)
        pts2 = pts2.unsqueeze(0) * torch.tensor([W - 1, H - 1], dtype=torch.float32, device=img1.device)

        # random pts
        random_angle = torch.randn(pts1[0, :, 0].shape, device=img1.device) * 6.28
        random_pts = torch.stack([torch.cos(random_angle), torch.sin(random_angle)], dim=1)

        # add random pts as initial pts for optical flow
        pts2_randon = pts2 + random_pts * self.distance
        pts2_randon[0, :, 0] = torch.clamp(pts2_randon[0, :, 0], min=10, max=W - 10)
        pts2_randon[0, :, 1] = torch.clamp(pts2_randon[0, :, 1], min=10, max=H - 10)

        # build pyramid
        img1_pyr = self.build_pyramid(img1, self.levels)
        img2_pyr = self.build_pyramid(img2, self.levels)

        scale_all = []
        for i in range(self.levels):
            scale_all.append(2 ** (self.levels - i - 1))

        pts_all_level = []
        pts_pre = pts2_randon

        # track optical flow
        for i in range(self.levels):
            # scale up
            pts1_level = pts1 / (2 ** (self.levels - i - 1))
            pts2_level = pts_pre / (2 ** (self.levels - i - 1))
            # calculate optical flow
            pts_pre, pts_all = self.optical_flow_level(
                img1_pyr[self.levels - i - 1], img2_pyr[self.levels - i - 1],
                pts1_level, pts2_level)
            pts_pre = pts_pre * scale_all[i]
            pts_all = pts_all * scale_all[i]
            pts_all_level.append(pts_all)

        # calculate error
        error = torch.zeros_like(pts2[:, :, 0])
        error = torch.clamp(torch.norm(pts_pre - pts2, dim=2), max=8)

        return pts_pre, error

    def optical_flow_level(self, img1, img2, pts1, pts2):
        """
        :param img1: [B, C, H, W]
        :param img2: [B, C, H, W]
        :param pts1: [B, N, 2]
        :param pts2: [B, N, 2]
        :return:
        """
        # calculate gradient
        dx = self.dx.to(img1.device).to(img1.dtype)
        dy = self.dy.to(img1.device).to(img1.dtype)
        img1_dx = torch.nn.functional.conv2d(img1, dx, padding=1)
        img1_dy = torch.nn.functional.conv2d(img1, dy, padding=1)
        img2_dx = torch.nn.functional.conv2d(img2, dx, padding=1)
        img2_dy = torch.nn.functional.conv2d(img2, dy, padding=1)

        # to patch
        N, C, H, W = img1.shape
        patch_img1 = torch.nn.functional.unfold(img1, kernel_size=self.win_size, padding=self.win_size//2)
        patch_img2 = torch.nn.functional.unfold(img2, kernel_size=self.win_size, padding=self.win_size//2)
        patch_img1_dx = torch.nn.functional.unfold(img1_dx, kernel_size=self.win_size, padding=self.win_size//2)
        patch_img1_dy = torch.nn.functional.unfold(img1_dy, kernel_size=self.win_size, padding=self.win_size//2)
        patch_img2_dx = torch.nn.functional.unfold(img2_dx, kernel_size=self.win_size, padding=self.win_size//2)
        patch_img2_dy = torch.nn.functional.unfold(img2_dy, kernel_size=self.win_size, padding=self.win_size//2)
        patch_img1 = patch_img1.view(N, -1, H, W).contiguous()
        patch_img2 = patch_img2.view(N, -1, H, W).contiguous()
        patch_img1_dx = patch_img1_dx.view(N, -1, H, W).contiguous()
        patch_img1_dy = patch_img1_dy.view(N, -1, H, W).contiguous()
        patch_img2_dx = patch_img2_dx.view(N, -1, H, W).contiguous()
        patch_img2_dy = patch_img2_dy.view(N, -1, H, W).contiguous()

        sample_pts1 = pts1 / torch.tensor([W - 1, H - 1], dtype=torch.float32, device=pts1.device) * 2 - 1
        patch_pts1 = torch.nn.functional.grid_sample(patch_img1, sample_pts1.unsqueeze(1), align_corners=True).squeeze(2)
        pts_pre = pts2
        pts_all = []
        for i in range(self.interation):
            # calculate optical flow
            sample_pts2 = pts_pre / torch.tensor([W - 1, H - 1], dtype=torch.float32, device=pts1.device) * 2 - 1
            patch_pts2 = torch.nn.functional.grid_sample(patch_img2, sample_pts2.unsqueeze(1), align_corners=True).squeeze(2)
            dI = (patch_pts1 - patch_pts2).squeeze(0)
            patch_pts2_dx = torch.nn.functional.grid_sample(patch_img2_dx, sample_pts2.unsqueeze(1), align_corners=True).squeeze(2)
            patch_pts2_dy = torch.nn.functional.grid_sample(patch_img2_dy, sample_pts2.unsqueeze(1), align_corners=True).squeeze(2)
            J = torch.cat([patch_pts2_dx, patch_pts2_dy], dim=0)
            G = torch.einsum('bik,cik->bck', [J, J])
            b = torch.einsum('ik,bik->bk', dI, J)
            valid = torch.where(torch.det(G.permute(2, 0, 1)) > 1e-6)[0]
            G_inverse = G.permute(2, 0, 1)[valid].inverse().permute(1, 2, 0)
            pts_pre[:, valid, :] = pts_pre[:, valid, :] - torch.einsum('bik,bk->bk', [G_inverse, b[:, valid]]).transpose(0, 1).unsqueeze(0)
            pts_all.append(pts_pre)
        return pts_pre, pts_all


def optical_flow_cv(pts0: torch.tensor,
                    pts1: torch.tensor,
                    img0: torch.tensor,
                    img1: torch.tensor,
                    params=None):
    """
    :param pts0: (n, 2) in [0, 1]
    :param pts1: (n, 2) in [0, 1]
    :param img0: (b, c, h, w)
    :param img1: (b, c, h, w)
    :param params: None
    :return:
    """
    if params is None:
        params = {
            "winSize": (5, 5),
            "maxLevel": 3,
            "flags": cv2.OPTFLOW_USE_INITIAL_FLOW,
        }
    # pts to pixels & numpy
    B, C, H, W = img0.shape
    assert B == 1 and pts0.shape[0] == pts1.shape[0]
    pts0 = (pts0 * torch.tensor([W - 1, H - 1]).to(pts0)).detach().cpu().numpy().astype(np.float32)
    pts1 = (pts1 * torch.tensor([W - 1, H - 1]).to(pts1)).detach().cpu().numpy().astype(np.float32)

    # convert to numpy
    img0 = (img0 * 255.).detach().cpu().numpy().astype(np.uint8)[0]
    img1 = (img1 * 255.).detach().cpu().numpy().astype(np.uint8)[0]
    img0 = np.transpose(img0, (1, 2, 0))
    img1 = np.transpose(img1, (1, 2, 0))

    # optical flow
    lk_params = dict(winSize=params["winSize"],
                     maxLevel=params["maxLevel"],
                     flags=params["flags"],
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    pts1_, status, _ = cv2.calcOpticalFlowPyrLK(img0, img1, pts0, pts1, **lk_params)

    # to tensor
    pts1_ = torch.from_numpy(pts1_).to(pts0.device) / torch.tensor([W - 1, H - 1]).to(pts0.device)
    return pts1_


def optical_flow_tensor(pts0: torch.tensor,
                        pts1: torch.tensor,
                        img0: torch.tensor,
                        img1: torch.tensor,
                        params=None):
    """
    :param pts0: (n, 2) in [0, 1]
    :param pts1: (n, 2) in [0, 1]
    :param img0: (b, c, h, w)
    :param img1: (b, c, h, w)
    :param params: None
    :return:
    """
    optical_flow = OpticalFlow(params)
    pts1_, error_ = optical_flow(img0, img1, pts0, pts1)
    return pts1_



