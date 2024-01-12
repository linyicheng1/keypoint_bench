import cv2


def fundamental_estimate(pts0, pts1):
    """
    :param pts0: (n, 2)
    :param pts1: (n, 2)
    :return:
    """
    # to numpy
    pts0 = pts0.cpu().numpy()
    pts1 = pts1.cpu().numpy()
    F, mask = cv2.findFundamentalMat(pts0, pts1, cv2.FM_RANSAC)
    # pts0 = pts0[mask.ravel() == 1]
    # pts1 = pts1[mask.ravel() == 1]
    return F
