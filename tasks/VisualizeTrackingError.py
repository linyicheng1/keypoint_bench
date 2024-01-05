import cv2
import torch
from utils.extracter import detection
from utils.matcher import optical_flow_tensor, optical_flow_cv
from utils.projection import warp
from utils.visualization import plot_kps_error


def visualize_tracking_error(step: int,
                             img0: torch.Tensor,
                             score_map_0: torch.Tensor,
                             desc_map_0: torch.Tensor,
                             desc_map_1: torch.Tensor,
                             params: dict,
                             warp01=None):
    """
    :param step:
    :param img0:
    :param score_map_0:
    :param desc_map_0:
    :param desc_map_1:
    :param params:
    :param warp01: warp from img0 to img1
    :return:
    """

    # extract key points
    kps0 = detection(score_map_0, params['extractor_params'])

    # warp to kps01
    kps1, error, kps01 = None, None, None
    if params['matcher_params']['type'] == 'optical_flow':
        if warp01 is None:
            kps1 = optical_flow_tensor(kps0[:, 0:2], kps0[:, 0:2], desc_map_0,
                                       desc_map_1, params['matcher_params']['optical_flow_params'])
        else:
            kps0, kps01, ids, ids_out = warp(kps0[:, 0:2], warp01)
            kps1 = optical_flow_tensor(kps0, kps01, desc_map_0,
                                       desc_map_1, params['matcher_params']['optical_flow_params'])

    # calculate error
    if kps01 is None:
        error = torch.zeros(kps0.shape[0], dtype=torch.float32).to(kps0.device)
    else:
        kps01_wh = kps01 * torch.tensor([img0.shape[3] - 1, img0.shape[2] - 1], dtype=torch.float32).to(img0.device)
        error = torch.norm(kps01_wh - kps1, dim=2)[0]
    # show tracking error
    show = plot_kps_error(img0, kps0, error, params['VisualizeTrackingError_params']['image'])
    root = params['VisualizeTrackingError_params']['output']
    cv2.imwrite(root + str(step) + '_tracking_error.png', show)
    return {"track_error": torch.mean(error), "kps0": kps0, "kps1": kps1, "kps01": kps01}

