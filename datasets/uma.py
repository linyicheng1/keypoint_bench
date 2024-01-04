import torch.utils.data as data
import cv2
import numpy as np


def read_cam_csv(path, timestamps, image_list):
    with open(path, 'r') as f:
        head = f.readline()
        for line in f.readlines():
            timestamps.append(line.split(',')[0])
            image_list.append(line.split(',')[1].replace('\n',''))


class UMADataset(data.Dataset):

    def __init__(self, root="", gray=False):
        self.root = root
        self.gray = gray
        self.cam0_csv = self.root + 'cam0/data.csv'
        self.cam1_csv = self.root + 'cam1/data.csv'
        self.cam2_csv = self.root + 'cam2/data.csv'
        self.cam3_csv = self.root + 'cam3/data.csv'

        self.cam0_timestamps = []
        self.cam1_timestamps = []
        self.cam2_timestamps = []
        self.cam3_timestamps = []
        self.cam0_image_list = []
        self.cam1_image_list = []
        self.cam2_image_list = []
        self.cam3_image_list = []

        read_cam_csv(self.cam0_csv, self.cam0_timestamps, self.cam0_image_list)
        read_cam_csv(self.cam1_csv, self.cam1_timestamps, self.cam1_image_list)
        read_cam_csv(self.cam2_csv, self.cam2_timestamps, self.cam2_image_list)
        read_cam_csv(self.cam3_csv, self.cam3_timestamps, self.cam3_image_list)

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
        }


if __name__ == '__main__':
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('TkAgg')
    dataset = UMADataset(root='/media/server/4cda377d-28db-4424-921c-6a1e0545ceeb/Dataset/uma-vi/two-floors-csc2_2019-02-11-18-33-04_InOut/',
                         gray=False)
    for data in tqdm(dataset):
        image0 = data['image0']
        image1 = data['image1']
        plt.imshow(image0.transpose(1, 2, 0))
        plt.show()
