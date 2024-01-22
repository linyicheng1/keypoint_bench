import cv2
import numpy as np
import torch
from utils.extracter import detection
from utils.matcher import optical_flow_tensor, optical_flow_cv
from utils.visualization import plot_matches, write_position, plot_kps_error


def visual_odometry(step: int,
                    pose_R,
                    pose_t,
                    last_img: torch.Tensor,
                    batch,
                    score_map_0: torch.Tensor,
                    score_map_1: torch.Tensor,
                    desc_map_0: torch.Tensor,
                    desc_map_1: torch.Tensor,
                    params: dict,
                    ):
    """
    :param step:
    :param pose_R:
    :param pose_t:
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
        kps1, status = optical_flow_cv(kps0[:, 0:2], kps0[:, 0:2], desc_map_0,
                                          desc_map_1, params['matcher_params']['optical_flow_params'])
        kps1 = kps1[np.where(status == 1)[0], :]
        kps0 = kps0[np.where(status == 1)[0], :]
    elif params['matcher_params']['type'] == 'bf_matcher':
        pass  # TODO

    # 2.5. plot matches
    pts0 = kps0.cpu().numpy()
    kps1 = kps1.cpu().numpy()

    kps0_wh = (kps0[:, :-1] * torch.tensor([score_map_0.shape[3] - 1, score_map_0.shape[2] - 1],
                                           dtype=torch.float32).to(score_map_0.device)).cpu().numpy()
    kps1_wh = kps1 * np.array([last_img.shape[3] - 1, last_img.shape[2] - 1], dtype=np.float32)
    show = plot_matches(last_img, batch['image0'], kps0_wh, kps1_wh)
    root = params['visual_odometer_params']['output']
    cv2.imwrite(root + str(step) + '_matches.png', show)
    # 3. compute pose
    pp = (float(batch['cx']), float(batch['cy']))
    E, mask = cv2.findEssentialMat(kps0_wh, kps1_wh,
                                   focal=float(batch['fx']), pp=pp,
                                   method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, mask = cv2.recoverPose(E, kps0_wh, kps1_wh,
                                    focal=float(batch['fx']), pp=pp)
    # 4. compute absolute scale
    ground_truth = batch['ground_truth']
    last_ground_truth = batch['last_ground_truth']
    scale = torch.norm(ground_truth.tensor()[0:3] - last_ground_truth.tensor()[0:3])
    R_est = pose_R
    t_est = pose_t
    # only update pose when scale is large enough
    if scale >= 0.001:
        R_est = pose_R.dot(R)
        t_est = pose_t + float(scale) * pose_R.dot(t)

    return {"R": R_est, "t": t_est}


# evo tools for kitti
#  evo_ape kitti ../traj/good.txt /media/server/4cda377d-28db-4424-921c-6a1e0545ceeb/Dataset/kitti_odom/data_odometry_poses/dataset/poses/00.txt -a
# evo_traj kitti ../traj/good.txt --ref=/media/server/4cda377d-28db-4424-921c-6a1e0545ceeb/Dataset/kitti_odom/data_odometry_poses/dataset/poses/00.txt -p --plot_mode=xz
def plot_visual_odometry(r, t, save_path):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax1 = plt.axes(projection='3d')
    z = t[:, 2, 0]
    x = t[:, 0, 0]
    y = t[:, 1, 0]
    ax1.plot3D(x, y, z)  # 绘制空间曲线
    plt.show()

    write_position(save_path.replace('.png', '.txt'), r, t)


if __name__ == '__main__':
    r = np.load('/home/server/linyicheng/py_proj/keypoint_bench/visual_odometer_r_est.npy')
    t = np.load('/home/server/linyicheng/py_proj/keypoint_bench/visual_odometer_t_est.npy')

    plot_visual_odometry(r, t, 'test.png')

