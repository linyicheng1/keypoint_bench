import torch.utils.data as data
import cv2
import numpy as np
import pypose as pp
import torch
import os


def read_cam_csv(path, timestamps, image_list):
    with open(path, 'r') as f:
        head = f.readline()
        for line in f.readlines():
            timestamps.append(float(line.split(',')[0]))
            image_list.append(line.split(',')[1].replace('\n',''))


def read_gt_csv(path, ground_truth):
    with open(path, 'r') as f:
        head = f.readline()
        for line in f.readlines():
            line_list = line.split(',')
            # tx ty tz qx qy qz qw
            p = pp.SE3([float(line_list[1]), float(line_list[2]), float(line_list[3]),
                        float(line_list[5]), float(line_list[6]), float(line_list[7]), float(line_list[4])])
            ground_truth.append(p)


def read_imu_csv(path, imu_timestamps, imu_data):
    with open(path, 'r') as f:
        head = f.readline()
        for line in f.readlines():
            line_list = line.split(',')
            imu_timestamps.append(float(line_list[0]))
            gyro = np.array([float(line_list[1]), float(line_list[2]), float(line_list[3])])
            acc = np.array([float(line_list[4]), float(line_list[5]), float(line_list[6])])
            imu_data.append([torch.from_numpy(gyro).to(torch.float32), torch.from_numpy(acc).to(torch.float32)])


def integrate_imu(imu_timestamps, imu_raw, cam0_timestamps, imu_integrate):

    p = torch.zeros(3)  # Initial Position
    r = pp.identity_SO3()  # Initial rotation
    v = torch.zeros(3)  # Initial Velocity
    integrator = pp.module.IMUPreintegrator(p, r, v,
                                            reset=False)
    # states = []
    # for i in range(len(imu_timestamps) - 1):
    #     state = integrator.forward(dt=torch.tensor([imu_timestamps[i + 1] - imu_timestamps[i]]) / 1e9,
    #                                gyro=imu_raw[i][0],
    #                                acc=imu_raw[i][1])
    #     states.append(state)
    #
    # print(len(states))


def pose_2_fundamental_matrix(pose, fundamental, fx, fy, cx, cy):
    k = torch.tensor([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]])
    fundamental.append(torch.zeros(3, 3).unsqueeze(0))
    for i in range(len(pose) - 1):
        dp = pp.Inv(pose[i]) * pose[i + 1]
        t_hat = torch.tensor([[0, -dp[2], dp[1]],
                              [dp[2], 0, -dp[0]],
                              [-dp[1], dp[0], 0]])
        r = pp.SO3(dp.tensor()[3:7])
        f = torch.inverse(k).t() @ t_hat @ r.matrix() @ torch.inverse(k)
        fundamental.append(f.unsqueeze(0))


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
        self.fundamentals_csv = self.root + 'state_groundtruth_estimate0/f.pt'
        self.imu_csv = self.root + 'imu0/data.csv'
        self.cam0_timestamps = []
        self.cam1_timestamps = []
        self.cam0_image_list = []
        self.cam1_image_list = []
        self.ground_truth = []
        self.imu_integrate = []
        self.imu_raw = []
        self.imu_timestamps = []
        self.Fundamentals = []
        read_cam_csv(self.cam0_csv, self.cam0_timestamps, self.cam0_image_list)
        read_cam_csv(self.cam1_csv, self.cam1_timestamps, self.cam1_image_list)
        read_gt_csv(self.ground_truth_csv, self.ground_truth)
        read_imu_csv(self.imu_csv, self.imu_timestamps, self.imu_raw)
        integrate_imu(self.imu_timestamps, self.imu_raw, self.cam0_timestamps, self.imu_integrate)

        if os.path.exists(self.fundamentals_csv):
            self.Fundamentals = torch.load(self.fundamentals_csv)
        else:
            pose_2_fundamental_matrix(self.ground_truth, self.Fundamentals, self.fx, self.fy, self.cx, self.cy)
            fundamentals = torch.cat(self.Fundamentals, dim=0)
            torch.save(fundamentals, self.fundamentals_csv)

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
            'fundamental': self.Fundamentals[item],
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
