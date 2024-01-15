import cv2
import torch
from utils.extracter import detection
from utils.matcher import optical_flow_tensor, optical_flow_cv
from utils.projection import warp
from utils.visualization import plot_epipolar_lines, write_txt, plot_kps_error
from utils.mvg import fundamental_estimate


def fundamental_matrix(step: int,
                       last_img: torch.Tensor,
                       batch,
                       score_map_0: torch.Tensor,
                       score_map_1: torch.Tensor,
                       desc_map_0: torch.Tensor,
                       desc_map_1: torch.Tensor,
                       params: dict):
    """
    :param step:
    :param last_img:
    :param batch:
    :param score_map_0:
    :param score_map_1:
    :param desc_map_0:
    :param desc_map_1:
    :param params:
    :return:
    """

    # 1. extract key points
    kps0 = detection(score_map_0, params['extractor_params'])
    kps1 = detection(score_map_1, params['extractor_params'])

    # 2. match key points
    if params['matcher_params']['type'] == 'optical_flow':
        kps1 = optical_flow_tensor(kps0[:, 0:2], kps0[:, 0:2], desc_map_0,
                                   desc_map_1, params['matcher_params']['optical_flow_params'])
    elif params['matcher_params']['type'] == 'bf_matcher':
        pass  # TODO

    # 3. calculate error
    kps0_wh = kps0[:, :-1] * torch.tensor([score_map_0.shape[3] - 1, score_map_0.shape[2] - 1], dtype=torch.float32).to(score_map_0.device)
    kps1 = torch.cat([kps1[0], torch.ones(kps1.shape[1], 1).to(kps1.device)], dim=1)
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

    return {"fundamental_error": torch.mean(error), "fundamental_radio": radio}


def plot_fundamental_matrix(error, save_path):
    import matplotlib.pyplot as plt
    plt.plot(error)
    plt.savefig(save_path)
    plt.close()
    write_txt(save_path.replace('.png', '.txt'), error)
