import torch.utils.data as data
import cv2
import numpy as np


def read_cam_csv(path, timestamps, image_list):
    with open(path, 'r') as f:
        head = f.readline()
        for line in f.readlines():
            timestamps.append(line.split(',')[0])
            image_list.append(line.split(',')[1].replace('\n',''))


def read_gt_csv(path, ground_truth):
    with open(path, 'r') as f:
        for line in f.readlines():
            ground_truth.append(line.split(',')[0])


class EurocDataset(data.Dataset):

    def __init__(self, root="", gray=False):
        self.root = root
        self.gray = gray
        # euroc camera parameters
        self.fx = 435.2046959714599
        self.fy = 435.2046959714599
        self.cx = 367.4517211914062
        self.cy = 252.2008514404297
        self.baseline = 0.2090607502
        self.bf = self.baseline * self.fx

        self.cam0_csv = self.root + 'cam0/data.csv'
        self.cam1_csv = self.root + 'cam1/data.csv'
        self.ground_truth_csv = self.root + 'state_groundtruth_estimate0/data.csv'
        self.cam0_timestamps = []
        self.cam1_timestamps = []
        self.cam0_image_list = []
        self.cam1_image_list = []
        self.ground_truth = []
        read_cam_csv(self.cam0_csv, self.cam0_timestamps, self.cam0_image_list)
        read_cam_csv(self.cam1_csv, self.cam1_timestamps, self.cam1_image_list)
        read_gt_csv(self.ground_truth_csv, self.ground_truth)

    def __len__(self):
        return len(self.cam0_image_list)

    def __getitem__(self, item):
        # read image
        img0 = cv2.imread(self.root + "cam0/data/" + self.cam0_image_list[item], cv2.IMREAD_COLOR)
        img1 = cv2.imread(self.root + "cam1/data/" + self.cam1_image_list[item], cv2.IMREAD_COLOR)
        assert img0 is not None, 'can not load: ' + self.root
        assert img1 is not None, 'can not load: ' + self.root

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
            'timestamp': self.cam0_timestamps[item],
            'ground_truth': self.ground_truth[item],
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
    dataset = EurocDataset(root='/media/server/4cda377d-28db-4424-921c-6a1e0545ceeb/Dataset/euroc/MH_01_easy/mav0/', gray=True)
    for data in tqdm(dataset):
        image0 = data['image0']
        image1 = data['image1']
        plt.imshow(image0.transpose(1, 2, 0), cmap='gray')
        plt.show()
