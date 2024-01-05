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
            'color': (255, 0, 0)
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
        color = params['color']
        if error is not None:
            e = int(error[i] * 255. / params['max_error'])
            if e < 127:
                color = (0, 255 - 2*e, e*2)
            else:
                e = e - 127
                color = (2*e, 0, 255 - e*2)
        cv2.drawMarker(out, (x0, y0), color, cv2.MARKER_CROSS, params['radius'])
    return out
