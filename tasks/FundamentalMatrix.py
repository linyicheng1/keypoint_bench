import cv2
import torch
from utils.extracter import detection
from utils.matcher import optical_flow_tensor, optical_flow_cv, brute_force_matcher
from utils.projection import warp
from utils.visualization import plot_epipolar_lines, write_txt, plot_kps_error
from utils.mvg import fundamental_estimate
import numpy as np
from models.lightglue import LightGlue


def fundamental_matrix_ransac(step: int,
                              image_0: torch.Tensor,
                              image_1: torch.Tensor,
                              score_map_0: torch.Tensor,
                              score_map_1: torch.Tensor,
                              desc_map_0: torch.Tensor,
                              desc_map_1: torch.Tensor,
                              matcher: LightGlue,
                              params: dict):
    show = None
    show0 = None
    show1 = None
    b, c, h, w = score_map_0.shape
    if params['matcher_params']['save_result']:
        show0 = image_0[0].detach().cpu().numpy().transpose(1, 2, 0) * 255
        show0 = cv2.cvtColor(show0, cv2.COLOR_RGB2BGR)
        show1 = image_1[0].detach().cpu().numpy().transpose(1, 2, 0) * 255
        show1 = cv2.cvtColor(show1, cv2.COLOR_RGB2BGR)
        show = cv2.hconcat([show0, show1])
    # 1. extract key points
    kps0 = detection(score_map_0, params['extractor_params'])
    # tmp = cv2.imread('/home/server/linyicheng/py_proj/keypoint_bench/keypoint_bench/score_map.jpg', cv2.IMREAD_GRAYSCALE)
    # score_map_1 = torch.tensor(tmp).unsqueeze(0).unsqueeze(0).float().to(score_map_0.device) / 255.
    kps1 = detection(score_map_1, params['extractor_params'])
    if params['extractor_params']['save_result']:
        score_map_show_0 = score_map_0[0, 0].detach().cpu().numpy() * 255
        score_map_show_1 = score_map_1[0, 0].detach().cpu().numpy() * 255
        score_map_show_0 = cv2.cvtColor(score_map_show_0, cv2.COLOR_GRAY2BGR)
        score_map_show_1 = cv2.cvtColor(score_map_show_1, cv2.COLOR_GRAY2BGR)
        for i in range(kps0.shape[0]):
            cv2.circle(show0, (int(kps0[i, 0] * (w-1)), int(kps0[i, 1] * (h - 1))), 1, (0, 0, 255), -1)
        for i in range(kps1.shape[0]):
            cv2.circle(show1, (int(kps1[i, 0] * (w-1)), int(kps1[i, 1] * (h - 1))), 1, (0, 0, 255), -1)
        cv2.imwrite(params['FundamentalMatrixRansac']['output']+'kps0_'+str(step)+".png", show0)
        cv2.imwrite(params['FundamentalMatrixRansac']['output']+'kps1_'+str(step)+".png", show1)
        cv2.imwrite(params['FundamentalMatrixRansac']['output']+'score_map_0_'+str(step)+".png", score_map_show_0)
        cv2.imwrite(params['FundamentalMatrixRansac']['output']+'score_map_1_'+str(step)+".png", score_map_show_1)

    total_size = (kps0.shape[0] + kps1.shape[0])
    # 2. match key points
    if params['matcher_params']['type'] == 'optical_flow':
        kps1 = optical_flow_tensor(kps0[:, 0:2], kps0[:, 0:2], desc_map_0,
                                   desc_map_1, params['matcher_params']['optical_flow_params'])
        kps1 = torch.cat([kps1[0], torch.ones(kps1.shape[1], 1).to(kps1.device)], dim=1)
    elif params['matcher_params']['type'] == 'brute_force':
        kps0, kps1 = brute_force_matcher(kps0, kps1, desc_map_0, desc_map_1,
                                         params['matcher_params']['brute_force_params'])
    elif params['matcher_params']['type'] == 'light_glue':
        if matcher is None:
            kps0, kps1 = brute_force_matcher(kps0, kps1, desc_map_0, desc_map_1,
                                             params['matcher_params']['brute_force_params'])
        else:
            param = {
                'w': w,
                'h': h,
            }
            kps0, kps1 = matcher.match(kps0, kps1, desc_map_0,
                                       desc_map_1, param)
    if params['matcher_params']['save_result']:
        for i in range(kps0.shape[0]):
            cv2.line(show, (int(kps0[i, 0] * (w-1)), int(kps0[i, 1] * (h - 1))),
                     (int(kps1[i, 0] * (w-1) + w), int(kps1[i, 1] * (h - 1))), (0, 0, 255), 2)

    # 3. calculate fundamental matrix by ransac
    kps0_wh = kps0[:, :-1] * torch.tensor([w - 1, h - 1], dtype=torch.float32).to(score_map_0.device)
    kps1_wh = kps1[:, :-1] * torch.tensor([w - 1, h - 1], dtype=torch.float32).to(score_map_0.device)
    F, kps0_, kps1_ = fundamental_estimate(kps0_wh, kps1_wh)
    valid_size = (kps0_.shape[0] + kps1_.shape[0])
    if params['matcher_params']['save_result']:
        for i in range(kps0_.shape[0]):
            cv2.line(show, (int(kps0_[i, 0]), int(kps0_[i, 1])),
                     (int(kps1_[i, 0] + w), int(kps1_[i, 1])), (0, 255, 0), 2)
        cv2.imwrite(params['FundamentalMatrixRansac']['output']+str(step)+".png", show)

    return {"fundamental_error": 0, "fundamental_radio": valid_size / total_size, "fundamental_num": valid_size}


def fundamental_matrix(step: int,
                       last_img: torch.Tensor,
                       batch,
                       score_map_0: torch.Tensor,
                       score_map_1: torch.Tensor,
                       desc_map_0: torch.Tensor,
                       desc_map_1: torch.Tensor,
                       matcher: LightGlue,
                       params: dict):
    """
    :param step:
    :param last_img:
    :param batch:
    :param score_map_0:
    :param score_map_1:
    :param desc_map_0:
    :param desc_map_1:
    :param matcher:
    :param params:
    :return:
    """

    # 1. extract key points
    kps0 = detection(score_map_0, params['extractor_params'])
    kps1 = detection(score_map_1, params['extractor_params'])
    b, c, h, w = score_map_0.shape
    # 2. match key points
    if params['matcher_params']['type'] == 'optical_flow':
        kps1 = optical_flow_tensor(kps0[:, 0:2], kps0[:, 0:2], desc_map_0,
                                   desc_map_1, params['matcher_params']['optical_flow_params'])
        kps1 = torch.cat([kps1[0], torch.ones(kps1.shape[1], 1).to(kps1.device)], dim=1)
    elif params['matcher_params']['type'] == 'brute_force':
        kps0, kps1 = brute_force_matcher(kps0, kps1, desc_map_0, desc_map_1,
                                         params['matcher_params']['brute_force_params'])
    elif params['matcher_params']['type'] == 'light_glue':
        if matcher is None:
            kps0, kps1 = brute_force_matcher(kps0, kps1, desc_map_0, desc_map_1,
                                             params['matcher_params']['brute_force_params'])
        else:
            param = {
                'w': w,
                'h': h,
            }
            kps0, kps1 = matcher.match(kps0, kps1, desc_map_0,
                                       desc_map_1, param)
            kps1 = kps1[:, :-1] * torch.tensor([w - 1, h - 1], dtype=torch.float32).to(score_map_0.device)
            kps1 = torch.cat([kps1, torch.ones(kps1.shape[0], 1).to(kps1.device)], dim=1)
    # 3. calculate error
    kps0_wh = kps0[:, :-1] * torch.tensor([score_map_0.shape[3] - 1, score_map_0.shape[2] - 1], dtype=torch.float32).to(score_map_0.device)
    kps0_wh = torch.cat([kps0_wh, torch.ones(kps0_wh.shape[0], 1).to(kps0_wh.device)], dim=1)

    I = batch['fundamental'][0] @ kps0_wh.transpose(0, 1)
    error = kps1 @ I
    error = torch.abs(torch.diag(error))
    I = torch.norm(I[:-1, :], dim=0).clamp(min=1e-6)
    error = error / I

    th = params['FundamentalMatrix_params']['th']

    # kps1[:, :-1] = kps1[:, :-1] / torch.tensor([score_map_0.shape[3] - 1, score_map_0.shape[2] - 1], dtype=torch.float32).to(score_map_0.device)
    # show = plot_epipolar_lines(batch['image0'], kps0[:, :-1], kps1[:, :-1], batch['fundamental'][0])
    # root = params['repeatability_params']['output']
    # cv2.imwrite(root + str(step) + '_epipolar.png', show)

    # show = plot_epipolar_lines(batch['image0'], kps0[:, :-1], kps1[:, :-1], f)
    # cv2.imwrite(root + str(step) + '_epipolar_2.png', show)

    # show = plot_kps_error(last_img, kps0)
    # cv2.imwrite(root + str(step) + '_epipolar_0.png', show)

    radio = (error < th).sum().item() / error.shape[0]
    num = (error < th).sum().item()
    return {"fundamental_error": torch.mean(error), "fundamental_radio": radio, "fundamental_num": num}


def plot_fundamental_matrix(error, save_path):
    import matplotlib.pyplot as plt
    plt.plot(error)
    plt.savefig(save_path)
    plt.close()
    write_txt(save_path.replace('.png', '.txt'), error)
