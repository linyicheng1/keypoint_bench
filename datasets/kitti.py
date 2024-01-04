import torch.utils.data as data
import cv2
import glob
import numpy as np


class KittiDataset(data.Dataset):

    def __init__(self, sequence_path="", gt_path="", gray=False):
        self.fx = 718.856
        self.fy = 718.856
        self.cx = 607.1928
        self.cy = 185.2157
        self.baseline = 0.54
        self.bf = self.baseline * self.fx
        self.sequence_path = sequence_path
        self.gt_path = gt_path
        self.gray = gray
        self.image_0_list = glob.glob(self.sequence_path + 'image_0/*')
        self.image_1_list = glob.glob(self.sequence_path + 'image_1/*')
        self.image_0_list.sort()
        self.image_1_list.sort()

    def __len__(self):
        return len(self.image_0_list)

    def __getitem__(self, item):
        img0 = cv2.imread(self.image_0_list[item], cv2.IMREAD_COLOR)
        img1 = cv2.imread(self.image_1_list[item], cv2.IMREAD_COLOR)
        assert img0 is not None, 'can not load: ' + self.sequence_path

        if self.gray:
            img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY).astype('float32') / 255.
            img0 = np.expand_dims(img0, axis=2)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype('float32') / 255.
            img1 = np.expand_dims(img1, axis=2)
        else:
            # bgr -> rgb
            img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB).astype('float32') / 255.
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB).astype('float32') / 255.

        img0 = img0.transpose((2, 0, 1))
        img1 = img1.transpose((2, 0, 1))
        return {
            'image0': img0,
            'image1': img1,
            'fx': self.fx,
            'fy': self.fy,
            'cx': self.cx,
            'cy': self.cy,
            'bf': self.bf,
        }


if __name__ == '__main__':
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('TkAgg')
    dataset = KittiDataset(sequence_path='/media/server/4cda377d-28db-4424-921c-6a1e0545ceeb/Dataset/kitti_odom/data_odometry_gray/dataset/sequences/00/',
                           gt_path='/media/server/4cda377d-28db-4424-921c-6a1e0545ceeb/Dataset/kitti_odom/data_odometry_poses/dataset/poses/00.txt',
                           gray=True)
    for data in tqdm(dataset):
        image0 = data['image0']
        image1 = data['image1']
        plt.imshow(image0.transpose(1, 2, 0), cmap='gray')
        plt.show()
