import cv2
import torch
import numpy as np
from copy import deepcopy


def scale_intrinsics(K, scales):
    scales = np.diag([1. / scales[0], 1. / scales[1], 1.])
    return np.dot(scales, K)


def warp_depth(depth0, intrinsics0, intrinsics1, pose01, shape1):
    points3d0, valid0 = unproject_depth(depth0, intrinsics0)  # [:,3]
    points3d0 = points3d0[valid0]

    points3d01 = warp_points3d(points3d0, pose01)

    uv01, z01 = project(points3d01, intrinsics1)  # uv: [N,2], (h,w); z1: [N]

    depth01 = project_nn(uv01, z01, depth=np.zeros(shape=shape1))

    return depth01


def warp_points2d(uv0, d0, intrinsics0, intrinsics1, pose01):
    points3d0 = unproject(uv0, d0, intrinsics0)
    points3d01 = warp_points3d(points3d0, pose01)
    uv01, z01 = project(points3d01, intrinsics1)
    return uv01, z01


def unproject(uv, d, K):
    """
    unproject pixels uv to 3D points

    Args:
        uv: [N,2]
        d: depth, [N,1]
        K: [3,3]

    Returns:
        3D points, [N,3]
    """
    duv = uv * d  # (u,v) [N,2]
    if type(K) == torch.Tensor:
        duv1 = torch.cat([duv, d], dim=1)  # z*(u,v,1) [N,3]
        K_inv = torch.inverse(K)  # [3,3]
        points3d = torch.einsum('jk,nk->nj', K_inv, duv1)  # [N,3]
    elif type(K) == np.ndarray:
        duv1 = np.concatenate((duv, d), axis=1)  # z*(u,v,1) [N,3]
        K_inv = np.linalg.inv(K)  # [3,3]
        points3d = np.einsum('jk,nk->nj', K_inv, duv1)  # [N,3]
    else:
        raise TypeError("Input type should be 'torch.tensor' or 'numpy.ndarray'")
    return points3d


def project(points3d, K):
    """
    project 3D points to image plane

    Args:
        points3d: [N,3]
        K: [3,3]

    Returns:
        uv, (u,v), [N,2]
    """
    if type(K) == torch.Tensor:
        zuv1 = torch.einsum('jk,nk->nj', K, points3d)  # z*(u,v,1) = K*points3d -> [N,3]
    elif type(K) == np.ndarray:
        zuv1 = np.einsum('jk,nk->nj', K, points3d)
    else:
        raise TypeError("Input type should be 'torch.tensor' or 'numpy.ndarray'")
    uv1 = zuv1 / zuv1[:, -1][:, None]  # (u,v,1) -> [N,3]
    uv = uv1[:, 0:2]  # (u,v) -> [N,2]
    return uv, zuv1[:, -1]


def warp_points3d(points3d0, pose01):
    points3d0_homo = np.concatenate((points3d0, np.ones(points3d0.shape[0])[:, np.newaxis]), axis=1)  # [:,4]

    points3d01_homo = np.einsum('jk,nk->nj', pose01, points3d0_homo)  # [N,4]
    points3d01 = points3d01_homo[:, 0:3]  # [N,3]

    return points3d01


def project_nn(uv, z, depth):
    """
    uv: pixel coordinates [N,2]
    z: projected depth (xyz -> z) [N]
    depth: output depth array: [h,w]
    """
    return project_depth_nn_python(uv, z, depth)


def project_depth_nn_python(uv, z, depth):
    h, w = depth.shape
    # TODO: speed up the for loop
    for idx in range(len(uv)):
        uvi = uv[idx]
        x = int(round(uvi[0]))
        y = int(round(uvi[1]))

        if x < 0 or y < 0 or x >= w or y >= h:
            continue

        if depth[y, x] == 0. or depth[y, x] > z[idx]:
            depth[y, x] = z[idx]
    return depth


def unproject_depth(depth, K):
    h, w = depth.shape

    wh_range = np.mgrid[0:w, 0:h].transpose(2, 1, 0)  # [H,W,2]

    uv = wh_range.reshape(-1, 2)
    d = depth.reshape(-1, 1)
    points3d = unproject(uv, d, K)

    valid = np.logical_and((d[:, 0] > 0), (points3d[:, 2] > 0))

    return points3d, valid


def to_homogeneous(kpts):
    '''
    :param kpts: Nx2
    :return: Nx3
    '''
    ones = kpts.new_ones([kpts.shape[0], 1])
    return torch.cat((kpts, ones), dim=1)


def warp_homography(kpts0, params):
    '''
    :param kpts: Nx2
    :param homography_matrix: 3x3
    :return:
    '''

    homography_matrix = params['homography_matrix']
    w, h = params['width'], params['height']

    kpts0 = kpts0 * torch.tensor([w, h]).to(kpts0.device)
    kpts0_homogeneous = to_homogeneous(kpts0)
    kpts01_homogeneous = torch.einsum('ij,kj->ki', homography_matrix, kpts0_homogeneous)
    kpts01 = kpts01_homogeneous[:, :2] / kpts01_homogeneous[:, 2:]

    kpts01_ = kpts01.detach()
    # due to float coordinates, the upper boundary should be (w-1) and (h-1).
    # For example, if the image size is 480, then the coordinates should in [0~470].
    # 470.5 is not acceptable.
    valid01 = (kpts01_[:, 0] >= 0) * (kpts01_[:, 0] <= w - 1) * (kpts01_[:, 1] >= 0) * (kpts01_[:, 1] <= h - 1)
    kpts0_valid = kpts0[valid01]
    kpts01_valid = kpts01[valid01]
    ids = torch.nonzero(valid01, as_tuple=False)[:, 0]
    ids_out = torch.nonzero(~valid01, as_tuple=False)[:, 0]

    # kpts0_valid: valid keypoints0, the invalid and inconsistance keypoints are removed
    # kpts01_valid: the warped valid keypoints0
    # ids: the valid indices
    kpts0_valid = kpts0_valid / torch.tensor([w, h]).to(kpts0.device)
    kpts01_valid = kpts01_valid / torch.tensor([w, h]).to(kpts0.device)
    return kpts0_valid, kpts01_valid, ids, ids_out


def warp_dense(h: int, w: int, params: dict):
    mode = params['mode']
    pts_x = torch.range(1/w/2, 1 - 1/w/2, 1/w)
    pts_y = torch.range(1/h/2, 1 - 1/h/2, 1/h)
    pts = torch.stack(torch.meshgrid(pts_y, pts_x), dim=-1).reshape(-1, 2)[..., [1, 0]]
    pts = pts.to(params['depth0'].device)
    if mode == 'homo':
        kps0_valid, kps01_valid, ids01, ids01_out = warp_homography(pts, params)
    elif mode == 'se3':
        kps0_valid, kps01_valid, ids01, ids01_out = warp_se3(pts, params)
    else:
        raise ValueError('unknown mode!')
    return [kps0_valid, kps01_valid, ids01, ids01_out]


def warp(kpts0, params: dict):
    mode = params['mode']
    if mode == 'homo':
        return warp_homography(kpts0[:, 0:2], params)
    elif mode == 'se3':
        return warp_se3(kpts0[:, 0:2], params)
    else:
        raise ValueError('unknown mode!')

def warp_se3(kpts0, params):
    pose01 = params['pose01']  # relative motion
    bbox0 = params['bbox0']  # row, col
    bbox1 = params['bbox1']
    depth0 = params['depth0']
    depth1 = params['depth1']
    intrinsics0 = params['intrinsics0']
    intrinsics1 = params['intrinsics1']
    kpts0 = kpts0 * torch.from_numpy(np.array([depth0.shape[1], depth0.shape[0]])).to(torch.float32).to(kpts0.device)

    # kpts0_valid: valid kpts0
    # z0_valid: depth of valid kpts0
    # ids0: the indices of valid kpts0 ( valid corners and valid depth)
    # ids0_valid_corners: the valid indices of kpts0 in image ( 0<=x<w, 0<=y<h )
    # ids0_valid_depth: the valid indices of kpts0 with valid depth ( depth > 0 )
    z0_valid, kpts0_valid, ids0, ids0_valid_corners, ids0_valid_depth = interpolate_depth(kpts0, depth0)

    # COLMAP convention
    bkpts0_valid = kpts0_valid + bbox0[[1, 0]][None, :] + 0.5

    # unproject pixel coordinate to 3D points (camera coordinate system)
    bpoints3d0 = unproject(bkpts0_valid, z0_valid.unsqueeze(1), intrinsics0)  # [:,3]
    bpoints3d0_homo = to_homogeneous(bpoints3d0)  # [:,4]

    # warp 3D point (camera 0 coordinate system) to 3D point (camera 1 coordinate system)
    bpoints3d01_homo = torch.einsum('jk,nk->nj', pose01, bpoints3d0_homo)  # [:,4]
    bpoints3d01 = bpoints3d01_homo[:, 0:3]  # [:,3]

    # project 3D point (camera coordinate system) to pixel coordinate
    buv01, z01 = project(bpoints3d01, intrinsics1)  # uv: [:,2], (h,w); z1: [N]

    uv01 = buv01 - bbox1[None, [1, 0]] - .5

    # kpts01_valid: valid kpts01
    # z01_valid: depth of valid kpts01
    # ids01: the indices of valid kpts01 ( valid corners and valid depth)
    # ids01_valid_corners: the valid indices of kpts01 in image ( 0<=x<w, 0<=y<h )
    # ids01_valid_depth: the valid indices of kpts01 with valid depth ( depth > 0 )
    z01_interpolate, kpts01_valid, ids01, ids01_valid_corners, ids01_valid_depth = interpolate_depth(uv01, depth1)

    outimage_mask = torch.ones(ids0.shape[0], device=ids0.device).bool()
    outimage_mask[ids01_valid_corners] = 0
    ids01_invalid_corners = torch.arange(0, ids0.shape[0], device=ids0.device)[outimage_mask]
    ids_outside = ids0[ids01_invalid_corners]

    # ids_valid: matched kpts01 without occlusion
    ids_valid = ids0[ids01]
    kpts0_valid = kpts0_valid[ids01]
    z01_proj = z01[ids01]

    inlier_mask = torch.abs(z01_proj - z01_interpolate) < 0.05

    # indices of kpts01 with occlusion
    ids_occlude = ids_valid[~inlier_mask]

    ids_valid = ids_valid[inlier_mask]
    if ids_valid.size(0) == 0:
        # raise ValueError('empty tensor: ids')
        print('empty tensor: ids')

    kpts01_valid = kpts01_valid[inlier_mask]
    kpts0_valid = kpts0_valid[inlier_mask]

    # indices of kpts01 which are no matches in image1 for sure,
    # other projected kpts01 are not sure because of no depth in image0 or imgae1
    ids_out = torch.cat([ids_outside, ids_occlude])

    # kpts0_valid: valid keypoints0, the invalid and inconsistance keypoints are removed
    # kpts01_valid: the warped valid keypoints0
    # ids: the valid indices
    kpts0_valid = kpts0_valid / torch.from_numpy(np.array([depth0.shape[1], depth0.shape[0]])).to(torch.float32).to(kpts0.device)
    kpts01_valid = kpts01_valid / torch.from_numpy(np.array([depth1.shape[1], depth1.shape[0]])).to(torch.float32).to(kpts0.device)
    return kpts0_valid, kpts01_valid, ids_valid, ids_out


def interpolate_depth(pos, depth):
    border = 10
    pos = pos.t()[[1, 0]]  # Nx2 -> 2xN; w,h -> h,w(i,j)

    # =============================================== from d2-net
    device = pos.device

    ids = torch.arange(0, pos.size(1), device=device)

    h, w = depth.size()

    i = pos[0, :].detach()  # TODO: changed here
    j = pos[1, :].detach()  # TODO: changed here

    # Valid corners
    i_top_left = torch.floor(i).long()
    j_top_left = torch.floor(j).long()
    valid_top_left = torch.min(i_top_left >= border, j_top_left >= border)

    i_top_right = torch.floor(i).long()
    j_top_right = torch.ceil(j).long()
    valid_top_right = torch.min(i_top_right >= border, j_top_right < w-border)

    i_bottom_left = torch.ceil(i).long()
    j_bottom_left = torch.floor(j).long()
    valid_bottom_left = torch.min(i_bottom_left < h - border, j_bottom_left >= border)

    i_bottom_right = torch.ceil(i).long()
    j_bottom_right = torch.ceil(j).long()
    valid_bottom_right = torch.min(i_bottom_right < h - border, j_bottom_right < w - border)

    valid_corners = torch.min(torch.min(valid_top_left, valid_top_right),
                              torch.min(valid_bottom_left, valid_bottom_right))

    i_top_left = i_top_left[valid_corners]
    j_top_left = j_top_left[valid_corners]

    i_top_right = i_top_right[valid_corners]
    j_top_right = j_top_right[valid_corners]

    i_bottom_left = i_bottom_left[valid_corners]
    j_bottom_left = j_bottom_left[valid_corners]

    i_bottom_right = i_bottom_right[valid_corners]
    j_bottom_right = j_bottom_right[valid_corners]

    ids = ids[valid_corners]
    ids_valid_corners = deepcopy(ids)
    if ids.size(0) == 0:
        # raise ValueError('empty tensor: ids')
        print('empty tensor: ids')

    # Valid depth
    valid_depth = torch.min(torch.min(depth[i_top_left, j_top_left] > 0,
                                      depth[i_top_right, j_top_right] > 0),
                            torch.min(depth[i_bottom_left, j_bottom_left] > 0,
                                      depth[i_bottom_right, j_bottom_right] > 0))

    i_top_left = i_top_left[valid_depth]
    j_top_left = j_top_left[valid_depth]

    i_top_right = i_top_right[valid_depth]
    j_top_right = j_top_right[valid_depth]

    i_bottom_left = i_bottom_left[valid_depth]
    j_bottom_left = j_bottom_left[valid_depth]

    i_bottom_right = i_bottom_right[valid_depth]
    j_bottom_right = j_bottom_right[valid_depth]

    ids = ids[valid_depth]
    ids_valid_depth = deepcopy(ids)
    if ids.size(0) == 0:
        # raise ValueError('empty tensor: ids')
        print('empty tensor: ids')

    # Interpolation
    i = i[ids]
    j = j[ids]
    dist_i_top_left = i - i_top_left.float()
    dist_j_top_left = j - j_top_left.float()
    w_top_left = (1 - dist_i_top_left) * (1 - dist_j_top_left)
    w_top_right = (1 - dist_i_top_left) * dist_j_top_left
    w_bottom_left = dist_i_top_left * (1 - dist_j_top_left)
    w_bottom_right = dist_i_top_left * dist_j_top_left

    interpolated_depth = (w_top_left * depth[i_top_left, j_top_left] +
                          w_top_right * depth[i_top_right, j_top_right] +
                          w_bottom_left * depth[i_bottom_left, j_bottom_left] +
                          w_bottom_right * depth[i_bottom_right, j_bottom_right])

    # pos = torch.cat([i.view(1, -1), j.view(1, -1)], dim=0)

    pos = pos[:, ids]

    # =============================================== from d2-net
    pos = pos[[1, 0]].t()  # 2xN -> Nx2;  h,w(i,j) -> w,h

    # interpolated_depth: valid interpolated depth
    # pos: valid position (keypoint)
    # ids: indices of valid position (keypoint)

    return [interpolated_depth, pos, ids, ids_valid_corners, ids_valid_depth]
