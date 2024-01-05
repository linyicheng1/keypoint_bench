import numpy as np
import torch
import torch.nn.functional as F


def fast_nms(
        image_probs: torch.Tensor,
        nms_dist: int = 4,
        max_iter: int = -1,
        min_value: float = 0.0,
) -> torch.Tensor:
    """
    The process is slightly different :
      1. Find any local maximum (and count them).
      2. Suppress their neighbors (by setting them to 0).
      3. Repeat 1. and 2. until the number of local maximum stays the same.

    Performance
    -----------
    The original implementation takes about 2-4 seconds on a batch of 32 images of resolution 240x320.
    This fast implementation takes about ~90ms on the same input.

    Parameters
    ----------
    image_probs : torch.Tensor
        Tensor of shape BxCxHxW.
    nms_dist : int, optional
        The minimum distance between two predicted corners after NMS, by default 4
    max_iter : int, optional
        Maximum number of iteration, by default -1.
        Setting this number to a positive integer guarantees execution speed, but not correctness (i.e. good approximation).
    min_value : float
        Minimum value used for suppression.

    Returns
    -------
    torch.Tensor
        Tensor of shape BxCxHxW containing NMS suppressed input.
    """
    if nms_dist == 0:
        return image_probs

    ks = 2 * nms_dist + 1
    midpoint = (ks * ks) // 2
    count = None
    batch_size = image_probs.shape[0]

    i = 0
    while True:
        if i == max_iter:
            break

        # get neighbor probs in last dimension
        unfold_image_probs = F.unfold(
            image_probs,
            kernel_size=(ks, ks),
            dilation=1,
            padding=nms_dist,
            stride=1,
        )
        unfold_image_probs = unfold_image_probs.reshape(
            batch_size,
            ks * ks,
            image_probs.shape[-2],
            image_probs.shape[-1],
        )

        # check if middle point is local maximum
        max_idx = unfold_image_probs.argmax(dim=1, keepdim=True)
        mask = max_idx == midpoint

        # count all local maximum that are found
        new_count = mask.sum()

        # we stop if we din't not find any additional local maximum
        if new_count == count:
            break
        count = new_count

        # propagate local-maximum information to local neighbors (to suppress them)
        mask = mask.float()
        mask = mask.expand(-1, ks * ks, -1, -1)
        mask = mask.view(batch_size, ks * ks, -1)
        mask = mask.contiguous()
        mask[:, midpoint] = 0.0  # make sure we don't suppress the local maximum itself
        fold_ = F.fold(
            mask,
            output_size=image_probs.shape[-2:],
            kernel_size=(ks, ks),
            dilation=1,
            padding=nms_dist,
            stride=1,
        )

        # suppress all points who have a local maximum in their neighboorhood
        image_probs = image_probs.masked_fill(fold_ > 0.0, min_value)

        i += 1

    return image_probs


def predict_positions(desc0: torch.Tensor, desc1: torch.Tensor) -> torch.Tensor:
    # get score_map
    b, d, h, w = desc0.shape
    x = torch.linspace(1 / w / 2, 1 - 1 / w / 2, w)
    y = torch.linspace(1 / h / 2, 1 - 1 / h / 2, h)
    hw_grid = torch.stack(torch.meshgrid([x, y])).view(2, -1).t()[:, [1, 0]].to(desc0.device)
    desc0 = desc0.view(1, desc0.shape[1], -1)
    desc1 = desc1.view(1, desc0.shape[1], -1)
    score_map = torch.einsum('bdn,bdm->bnm', desc0, desc1)
    score_map = torch.cat([score_map, (torch.ones(1, h * w, 1) * 0.01).to(score_map.device)], dim=2)

    # normalize score_map
    max_v = score_map.max(dim=2).values.detach()
    x_exp = torch.exp((score_map - max_v[:, :, None]) / 0.01)[0, :, :-1]

    # predict x, y
    xy_residual = x_exp @ hw_grid / x_exp.sum(dim=1)[:, None]

    # interpolate score_map
    sample_pts = xy_residual * 2. - 1
    scores = F.grid_sample(x_exp.view(b, h * w, h, w), sample_pts.unsqueeze(0).unsqueeze(0),
                           mode='bilinear', align_corners=True, padding_mode='zeros')
    scores = torch.diag(scores.view(h * w, h * w))
    return torch.cat([xy_residual, scores[:, None]], dim=1)


def prob_map_to_positions_with_prob(
        prob_map: torch.Tensor,
        threshold: float = 0.0,
) -> torch.Tensor:
    """Convert probability map to positions with probability associated with each position.

    Parameters
    ----------
    prob_map : torch.Tensor
        Probability map. Tensor of size N x 1 x H x W.
    threshold : float, optional
        Threshold used to discard positions with low probability, by default 0.0

    Returns
    -------
    Tensor
    positions (with probability) tensors of size N x 3 (x, y and prob).
    """
    prob_map = prob_map.squeeze(dim=1)
    positions = tuple(
        torch.nonzero(prob_map[i] > threshold).float() + 0.5
        for i in range(prob_map.shape[0])
    )
    prob = tuple(
        prob_map[i][torch.nonzero(prob_map[i] > threshold, as_tuple=True)][:, None]
        for i in range(prob_map.shape[0])
    )
    positions_with_prob = tuple(
        torch.cat(
            (pos / torch.from_numpy(np.array([prob_map.shape[1], prob_map.shape[2]])).to(positions[0].device), prob),
            dim=1) for pos, prob in zip(positions, prob)
    )
    return positions_with_prob[0][..., [1, 0, 2]]


def remove_border_points(image_nms: torch.Tensor, border_dist: int = 4) -> torch.Tensor:
    """
    Remove predicted points within border_dist pixels of the image border.

    Args:
        image_nms (tensor): the output of the nms function, a tensor of shape
            (img_height, img_width) with corner probability values at each pixel location
        border_dist (int): the distance from the border to remove points

    Returns:
        image_nms (tensor): the image with all probability values equal to 0.0
            for pixel locations within border_dist of the image border
    """
    if border_dist > 0:
        # left columns
        image_nms[..., :, :border_dist] = 0.0

        # right columns
        image_nms[..., :, -border_dist:] = 0.0

        # top rows
        image_nms[..., :border_dist, :] = 0.0

        # bottom rows
        image_nms[..., -border_dist:, :] = 0.0

    return image_nms


def detection(score_map: torch.Tensor,
              params: dict = None):
    """
    :param score_map: Tensor of shape Bx1xHxW
    :param params:  parameters for nms
    :return: Tensor of shape BxNx3 (x, y and prob)
    """
    if params is None:
        nms_dist = 4
        threshold = 0.0
        border_dist = 8
        max_pts = 300
        min_score = 0.0
    else:
        nms_dist = params['nms_dist']
        threshold = params['threshold']
        border_dist = params['border_dist']
        max_pts = params['top_k']
        min_score = params['min_score']
    # fast nms
    score_map_nograd = score_map.detach()
    fast_nms_score_map = fast_nms(score_map_nograd, nms_dist=nms_dist)
    fast_nms_score_map = remove_border_points(fast_nms_score_map, border_dist=border_dist)
    pts = prob_map_to_positions_with_prob(fast_nms_score_map, threshold=threshold)
    if pts.shape[0] > max_pts:
        pts = pts[torch.argsort(pts[:, 2], descending=True)[:max_pts]]
    return pts


if __name__ == '__main__':
    # random generate desc_map
    desc_map_0 = torch.rand(1, 32, 8, 8)
    desc_map_1 = desc_map_0 + 0.3 * torch.rand_like(desc_map_0)
    desc_map_0 = desc_map_0 + 0.3 * torch.rand_like(desc_map_0)

    # normalize desc_map
    desc_map_0 = F.normalize(desc_map_0, dim=1)
    desc_map_1 = F.normalize(desc_map_1, dim=1)

    # predict positions
    positions = predict_positions(desc_map_0, desc_map_1)
    print(positions)
