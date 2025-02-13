import torch
import numpy as np
import cv2
from utils.projection import warp
from utils.extracter import detection
from utils.visualization import plot_kps_error, write_txt, plot_matches
from utils.matcher import optical_flow_tensor, optical_flow_cv, brute_force_matcher


class PID_controller(object):
    def __init__(self, kp_threshold=3, kp_max_distance=10, kp_max_angle=30, kp_max_scale=0.2):
        self.kp_threshold = kp_threshold
        self.kp_max_distance = kp_max_distance
        self.kp_max_angle = kp_max_angle
        self.kp_max_scale = kp_max_scale
    def get_kp_threshold(self):
        return self.kp_threshold
    def get_kp_max_distance(self):
        return self.kp_max_distance
    def get_kp_max_angle(self):
        return self.kp_max_angle
    def get_kp_max_scale(self):
        return self.kp_max_scale
    # 控制函数
    def control(self, target, feedback):
        # 计算 kp 误差
        error = target - feedback
        #  kp 数量控制
        num_kp = len(target)
        kp_error_norm = np.linalg.norm(error, axis=1)
        kp_error_mask = kp_error_norm < self.kp_max_distance
        kp_error_mask &= np.abs(error[:, 0]) < self.kp_max_angle
        kp_error_mask &= np.abs(error[:, 1]) < self.kp_max_angle
        kp_error_mask &= error[:, 2] < self.kp_max_scale
        kp_error_mask &= error[:, 3] < self.kp_max_scale
        kp_error_mask &= error[:, 4] < self.kp_max_scale
        kp_error_mask &= error[:, 5] < self.kp_max_scale
        

def estimate_pose(kpts0, kpts1, K0, K1, thresh, conf=0.99999):
    if len(kpts0) < 5:
        return None

    f_mean = np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])
    norm_thresh = thresh / f_mean

    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=norm_thresh, prob=conf,
        method=cv2.RANSAC)

    assert E is not None

    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, mask_new = cv2.recoverPose(
            _E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            best_num_inliers = n
            ret = (R, t[:, 0], mask.ravel() > 0)
    return ret

def angle_error_mat(R1, R2):
    cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # numercial errors can make it out of bounds
    return np.rad2deg(np.abs(np.arccos(cos)))


def angle_error_vec(v1, v2):
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))


def compute_pose_error(T_0to1, R, t):
    R_gt = T_0to1[:3, :3]
    t_gt = T_0to1[:3, 3]
    error_t = angle_error_vec(t, t_gt)
    error_t = np.minimum(error_t, 180 - error_t)  # ambiguity of E estimation
    error_R = angle_error_mat(R, R_gt)
    return error_t, error_R


def pose_auc(errors, thresholds):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0., errors]
    recall = np.r_[0., recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index-1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.trapz(r, x=e)/t)
    return aucs


def auc(idx, img_0, score_map_0, desc_map_0, img_1, score_map_1, desc_map_1, warp01, warp10, params):
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
    # 1. detection
    kps0 = detection(score_map_0, params['extractor_params'])
    kps1 = detection(score_map_1, params['extractor_params'])
    # 2. matching
    m_pts0, m_pts1 = brute_force_matcher(kps0, kps1, desc_map_0, desc_map_1,
                                         params['matcher_params']['brute_force_params'])

    h0, w0 = score_map_0.shape[2], score_map_0.shape[3]
    h1, w1 = score_map_1.shape[2], score_map_1.shape[3]

    m_pts0_wh = m_pts0[:, 0:2] * torch.tensor([w0 - 1, h0 - 1], dtype=torch.float32, device=img_0.device)
    m_pts1_wh = m_pts1[:, 0:2] * torch.tensor([w1 - 1, h1 - 1], dtype=torch.float32, device=img_1.device)
    m_pts0_wh = m_pts0_wh.cpu().numpy()
    m_pts1_wh = m_pts1_wh.cpu().numpy()

    # 3. estimate pose
    thresh = 1.
    K0 = warp01['intrinsics0'].cpu().numpy()
    K1 = warp01['intrinsics1'].cpu().numpy()
    result = estimate_pose(m_pts0_wh, m_pts1_wh, K0, K1, thresh)
    if result is None:
        return {
            'AUC': 180,
            'inliers': 0
        }
    R, t, inliers = result
    # 4. compute pose error
    T_0to1 = warp01['pose01'].cpu().numpy()
    err_t, err_R = compute_pose_error(T_0to1, R, t)

    # 5. visualization
    out_dir = params['AUC_params']['output']
    show = plot_matches(img_0, img_1, m_pts0_wh[inliers], m_pts1_wh[inliers])
    cv2.imwrite(out_dir + '/matches_{}.png'.format(idx), show)

    # print('error_t: {:.2f}, error_R: {:.2f}'.format(err_t, err_R))
    return {
        'AUC': np.maximum(err_t, err_R),
        'inliers': np.sum(inliers)
    }
    # 6. compute AUC
