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
    if pts0.shape[0] < 8:
        print("\n too few points to estimate fundamental matrix \n")
        return None, pts0, pts1
    F, mask = cv2.findFundamentalMat(pts0, pts1, cv2.FM_RANSAC)
    pts0 = pts0[mask.ravel() == 1]
    pts1 = pts1[mask.ravel() == 1]
    return F, pts0, pts1
