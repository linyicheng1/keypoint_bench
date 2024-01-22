import cv2
import numpy as np
import torch
from copy import deepcopy


def plot_kps_error(image: torch.tensor,
                   kps: torch.tensor,
                   error: torch.tensor = None,
                   params: dict = None) -> np.ndarray:
    """ visualize keypoints on image
    :param image: [H, W, 3] in range [0, 1]
    :param kps:  [N, 2] in range [0, 1]
    :param error:
    :param params: None or dict with keys:
    :return: image with key points
    """
    if params is None:
        params = {
            'radius': 2,
            'max_error': 8,
            'color': '255,0,0'
        }
    image = image[0].cpu().detach().numpy().transpose(1, 2, 0) if isinstance(image, torch.Tensor) else image
    kps = kps.cpu().detach().numpy() if isinstance(kps, torch.Tensor) else kps

    if image.dtype is not np.dtype('uint8'):
        image = image * 255
        image = image.astype(np.uint8)

    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    H, W, _ = image.shape
    out = np.ascontiguousarray(deepcopy(image))
    pts = kps[:, 0:2] * np.array([W - 1, H - 1])
    pts = np.round(pts).astype(int)
    if error is not None:
        error = torch.clamp(error, max=4, min=0)
    for i in range(pts.shape[0]):
        x0, y0 = pts[i]
        color = params['color'].split(',')
        color = (int(color[0]), int(color[1]), int(color[2]))
        # print(error)
        if error is not None:
            if i >= error.shape[0] or error[i] > 3:
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)
            # e = int(error[i] * 255. / params['max_error'])
            # if e < 127:
            #     color = (0, 255 - 2*e, e*2)
            # else:
            #     e = e - 127
            #     color = (2*e, 0, 255 - e*2)
            # color = (e, 0, 255 - e)
        cv2.drawMarker(out, (x0, y0), color, cv2.MARKER_CROSS, params['radius'])
    return out


def plot_epipolar_lines(image0: torch.tensor,
                        kps0: torch.tensor,
                        kps1: torch.tensor,
                        f: torch.tensor,
                        params: dict = None) -> np.ndarray:
    """ visualize epipolar lines
    :param image0: [H, W, 3] in range [0, 1]
    :param kps0:  [N, 2] in range [0, 1]
    :param kps1:  [N, 2] in range [0, 1]
    :param f: [3, 3]
    :param params: None or dict with keys:
    :return: image with key points
    """
    if params is None:
        params = {
            'radius': 2,
            'max_error': 8,
            'color': (255, 0, 0)
        }
    image0 = image0[0].cpu().detach().numpy().transpose(1, 2, 0) if isinstance(image0, torch.Tensor) else image0
    kps0 = kps0.cpu().detach().numpy() if isinstance(kps0, torch.Tensor) else kps0
    kps1 = kps1.cpu().detach().numpy() if isinstance(kps1, torch.Tensor) else kps1

    if image0.dtype is not np.dtype('uint8'):
        image = image0 * 255
        image = image.astype(np.uint8)

    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    H, W, _ = image.shape
    out = np.ascontiguousarray(deepcopy(image))
    pts0_wh = kps0[:, 0:2] * np.array([W - 1, H - 1])
    pts0_wh = np.round(pts0_wh).astype(int)
    pts1_wh = kps1[:, 0:2] * np.array([W - 1, H - 1])
    pts1_wh = np.round(pts1_wh).astype(int)
    # draw pts on image
    for i in range(pts1_wh.shape[0]):
        x0, y0 = pts1_wh[i]
        color = (255, 0, 0)
        cv2.drawMarker(out, (x0, y0), color, cv2.MARKER_CROSS, params['radius'])

    for i in range(pts0_wh.shape[0]):
        x0, y0 = pts0_wh[i]
        color = (0, 255, 0)
        cv2.drawMarker(out, (x0, y0), color, cv2.MARKER_CROSS, params['radius'])

    # draw epipolar lines
    sum_error = 0
    for i in range(pts0_wh.shape[0]):
        x0, y0 = pts0_wh[i]
        color = (0, 0, 255)
        # draw epipolar lines
        f = f.cpu().detach().numpy() if isinstance(f, torch.Tensor) else f
        a = f[0, 0] * x0 + f[0, 1] * y0 + f[0, 2]
        b = f[1, 0] * x0 + f[1, 1] * y0 + f[1, 2]
        c = f[2, 0] * x0 + f[2, 1] * y0 + f[2, 2]
        if b == 0 or np.isnan(a) or np.isnan(b) or np.isnan(c):
            continue
        # y = -a/b * x - c/b
        x1 = 0
        y1 = int(-c/b)
        x2 = W - 1
        y2 = int(-(a*x2 + c)/b)
        cv2.line(out, (x1, y1), (x2, y2), color, 1)
        sum_error += np.abs(a * pts1_wh[i, 0] + b * pts1_wh[i, 1] + c) / np.sqrt(a * a + b * b)
    print('sum_error', sum_error / pts0_wh.shape[0])
    return out


def plot_matches(image0: torch.tensor,
                 image1: torch.tensor,
                 kps0: torch.tensor,
                 kps1: torch.tensor):
    """ visualize matches
    :param image0: [H, W, 3] in range [0, 1]
    :param image1: [H, W, 3] in range [0, 1]
    :param kps0:  [N, 2] in range [H, W]
    :param kps1:  [N, 2] in range [H, W]
    :return: image with key points
    """
    image0 = image0[0].cpu().detach().numpy().transpose(1, 2, 0) if isinstance(image0, torch.Tensor) else image0
    image1 = image1[0].cpu().detach().numpy().transpose(1, 2, 0) if isinstance(image1, torch.Tensor) else image1
    kps0 = kps0.cpu().detach().numpy() if isinstance(kps0, torch.Tensor) else kps0
    kps1 = kps1.cpu().detach().numpy() if isinstance(kps1, torch.Tensor) else kps1

    if image0.dtype is not np.dtype('uint8'):
        image0 = image0 * 255
        image0 = image0.astype(np.uint8)

    if image1.dtype is not np.dtype('uint8'):
        image1 = image1 * 255
        image1 = image1.astype(np.uint8)

    if len(image0.shape) == 2 or image0.shape[2] == 1:
        image0 = cv2.cvtColor(image0, cv2.COLOR_GRAY2RGB)

    if len(image1.shape) == 2 or image1.shape[2] == 1:
        image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2RGB)

    H, W, _ = image0.shape
    show = np.zeros((H * 2, W, 3), dtype=np.uint8)
    show[0:H, :, :] = image0
    show[H:2*H, :, :] = image1
    # draw pts on image
    for i in range(kps0.shape[0]):
        x0, y0 = kps0[i]
        color = (0, 255, 0)
        cv2.drawMarker(show, (int(x0), int(y0)), color, cv2.MARKER_CROSS, 5)

    for i in range(kps1.shape[0]):
        x0, y0 = kps1[i]
        color = (255, 0, 0)
        cv2.drawMarker(show, (int(x0), int(y0) + H), color, cv2.MARKER_CROSS, 5)
    # draw lines
    for i in range(kps0.shape[0]):
        x0, y0 = kps0[i]
        x1, y1 = kps1[i]
        color = (0, 0, 255)
        cv2.line(show, (int(x0), int(y0)), (int(x1), int(y1) + H), color, 1)
    return show


def write_txt(filename: str,
              data: np.ndarray,
              mode: str = 'w'):
    """ write data to txt file
    :param filename: filename
    :param data: data to write
    :param mode: 'w' or 'a'
    :return:
    """
    with open(filename, mode) as f:
        for i in range(data.shape[0]):
            f.write(str(data[i]) + '\n')


def write_position(filename: str,
                   r, t,
                   mode: str = 'w'):
    """ write data to txt file
    :param filename: filename
    :param r: rotation matrix
    :param t: translation vector
    :param mode: 'w' or 'a'
    :return:
    """
    with open(filename, mode) as f:
        for i in range(t.shape[0] - 1):
            # kitti format [R|t]
            f.write(str(r[i + 1, 0, 0]) + ' ' + str(r[i + 1, 0, 1]) + ' ' + str(r[i + 1, 0, 2]) + ' ' + str(t[i + 1, 0, 0]) + ' ' +
                    str(r[i + 1, 1, 0]) + ' ' + str(r[i + 1, 1, 1]) + ' ' + str(r[i + 1, 1, 2]) + ' ' + str(t[i + 1, 1, 0]) + ' ' +
                    str(r[i + 1, 2, 0]) + ' ' + str(r[i + 1, 2, 1]) + ' ' + str(r[i + 1, 2, 2]) + ' ' + str(t[i + 1, 2, 0]) + '\n')
