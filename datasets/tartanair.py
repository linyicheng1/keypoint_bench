import torch.utils.data as data
import cv2
import glob
import numpy as np
import pypose as pp
import torch
import os


def pose_2_fundamental_matrix(pose, fundamental, fx, fy, cx, cy):
    k = torch.tensor([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]])
    fundamental.append(torch.zeros(3, 3).unsqueeze(0))
    for i in range(len(pose) - 1):
        dp = pp.Inv(pose[i + 1]) * pose[i]
        t_hat = torch.tensor([[0, -dp[2], dp[1]],
                              [dp[2], 0, -dp[0]],
                              [-dp[1], dp[0], 0]])
        r = pp.SO3(dp.tensor()[3:7])
        f = torch.inverse(k).t() @ t_hat @ r.matrix() @ torch.inverse(k)
        fundamental.append(f.unsqueeze(0))


def read_gt_csv(path, ground_truth):
    with open(path, 'r') as f:
        for line in f.readlines():
            line_list = line.split(' ')
            # x y z qx qy qz qw
            p = pp.SE3([float(line_list[0]), float(line_list[1]), float(line_list[2]),
                        float(line_list[3]), float(line_list[4]), float(line_list[5]), float(line_list[6])])
            ground_truth.append(p)


class TartanAirDataset(data.Dataset):

    def __init__(self, sequence_path="", gt_path="", gray=False):
        self.fx = 320.0
        self.fy = 320.0
        self.cx = 320.0
        self.cy = 240.0

        self.sequence_path = sequence_path
        self.gt_path = gt_path
        self.fundamental_path = gt_path.replace('.txt', '_fundamental.pt')
        self.gray = gray
        self.image_0_list = glob.glob(self.sequence_path + '*.png')
        self.image_0_list.sort()
        self.ground_truth = []
        self.Fundamentals = []
        read_gt_csv(self.gt_path, self.ground_truth)
        if os.path.exists(self.fundamental_path):
            self.Fundamentals = torch.load(self.fundamental_path)
        else:
            pose_2_fundamental_matrix(self.ground_truth, self.Fundamentals, self.fx, self.fy, self.cx, self.cy)
            fundamentals = torch.cat(self.Fundamentals, dim=0)
            torch.save(fundamentals, self.fundamental_path)

    def __len__(self):
        return len(self.image_0_list)

    def __getitem__(self, item):
        img0 = cv2.imread(self.image_0_list[item], cv2.IMREAD_COLOR)
        assert img0 is not None, 'can not load: ' + self.sequence_path

        if self.gray:
            img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY).astype('float32') / 255.
            img0 = np.expand_dims(img0, axis=2)
        else:
            # bgr -> rgb
            img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB).astype('float32') / 255.

        img0 = img0.transpose((2, 0, 1))
        last_id = max(0, item - 1)
        return {
            'image0': img0,
            'fx': self.fx,
            'fy': self.fy,
            'cx': self.cx,
            'cy': self.cy,
            'ground_truth': self.ground_truth[item],
            'last_ground_truth': self.ground_truth[last_id],
            'fundamental': self.Fundamentals[item],
            'dataset': 'TartanAir'
        }


if __name__ == '__main__':
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('TkAgg')
    dataset = TartanAirDataset(sequence_path='/media/server/4cda377d-28db-4424-921c-6a1e0545ceeb/Dataset/tartanair/mono/tartanair-test-mono-release/mono/ME000/',
                               gt_path='/media/server/4cda377d-28db-4424-921c-6a1e0545ceeb/Dataset/tartanair/tartanair_cvpr_gt/mono_gt/ME000.txt',
                               gray=False)

    for data in tqdm(dataset):
        image0 = data['image0']
        ground_truth = data['ground_truth']
        last_ground_truth = data['last_ground_truth']
        scale = torch.norm(ground_truth.tensor()[0:3] - last_ground_truth.tensor()[0:3])

        plt.imshow(image0.transpose(1, 2, 0))
        plt.show()
