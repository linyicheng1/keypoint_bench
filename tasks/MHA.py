import torch
import numpy as np
import cv2
from utils.projection import warp
from utils.extracter import detection
from utils.visualization import plot_kps_error, write_txt
from utils.matcher import optical_flow_tensor, optical_flow_cv, brute_force_matcher



def mha(idx, img_0, score_map_0, desc_map_0, img_1, score_map_1, desc_map_1, warp01, warp10, params):
    """
    Args:
        idx: int
        img_0: torch.tensor [H,W,3]
        score_map_0: torch.tensor [H,W]
        img_1: torch.tensor [H,W,3]
        score_map_1: torch.tensor [H,W]
        warp01: dict
        warp10: dict
        params: int
    Returns:
        dict
    """
    th = params['MHA_params']['th']
    mha_result = []
    for i in th:
        mha_result.append(0)
    # 1. detection
    kps0 = detection(score_map_0, params['extractor_params'])
    kps1 = detection(score_map_1, params['extractor_params'])
    # validation
    kps0_cov, kps01_cov, _, _ = warp(kps0, warp01)
    kps1_cov, kps10_cov, _, _ = warp(kps1, warp10)
    if kps0_cov.shape[0] == 0 or kps1_cov.shape[0] == 0:
        return mha_result
    # 2. matching
    m_pts0, m_pts1 = brute_force_matcher(kps0_cov, kps1_cov, desc_map_0, desc_map_1,
                                         params['matcher_params']['brute_force_params'])
    h, w = warp01['height'].cpu().numpy(), warp01['width'].cpu().numpy()
    m_pts0_wh = m_pts0[:, 0:2] * torch.tensor([w - 1, h - 1], dtype=torch.float32, device=img_0.device)
    m_pts1_wh = m_pts1[:, 0:2] * torch.tensor([w - 1, h - 1], dtype=torch.float32, device=img_1.device)
    m_pts0_wh = m_pts0_wh.cpu().numpy()
    m_pts1_wh = m_pts1_wh.cpu().numpy()
    H, inliers = cv2.findHomography(m_pts0_wh,
                                    m_pts1_wh,
                                    cv2.RANSAC)
    if H is None:
        return mha_result

    real_H = warp01['homography_matrix'].cpu().numpy()
    corners = np.array([[0, 0, 1],
                        [h - 1, 0, 1],
                        [0, w - 1, 1],
                        [h - 1, w - 1, 1]])
    real_warped_corners = np.dot(corners, np.transpose(real_H))
    real_warped_corners = real_warped_corners[:, :2] / real_warped_corners[:, 2:]
    warped_corners = np.dot(corners, np.transpose(H))
    warped_corners = warped_corners[:, :2] / warped_corners[:, 2:]
    resize_h = img_0.shape[2]
    resize_w = img_0.shape[3]

    real_warped_corners = real_warped_corners * np.array([resize_h/h, resize_w/w])
    warped_corners = warped_corners * np.array([resize_h/h, resize_w/w])

    mean_dist = np.mean(np.linalg.norm(real_warped_corners - warped_corners, axis=1))

    mha_result = []
    for i in th:
        mha_result.append(float(mean_dist <= i))

    return mha_result
