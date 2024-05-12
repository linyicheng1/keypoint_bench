import torch.utils.data as data
import cv2
import numpy as np


class ImagePairsDataset(data.Dataset):

    def __init__(self, path_file="", gray=False, resize=None):
        self.path_file = path_file
        self.gray = gray
        # read all files in root
        self.image_list = self.read_image(self.path_file)
        self.resize = resize

    def __len__(self):
        # get image count
        return len(self.image_list)

    def __getitem__(self, item):
        # read video frame
        image = cv2.imread(self.image_list[item][0])
        assert image is not None, 'can not load: ' + self.image_list[item][0]
        image1 = cv2.imread(self.image_list[item][1])
        assert image1 is not None, 'can not load: ' + self.image_list[item][1]

        if self.gray:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype('float32') / 255.
            image = np.expand_dims(image, axis=2)
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY).astype('float32') / 255.
            image1 = np.expand_dims(image1, axis=2)
        else:
            # bgr -> rgb
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype('float32') / 255.
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB).astype('float32') / 255.
        if self.resize is not None:
            image = cv2.resize(image, self.resize)
            image1 = cv2.resize(image1, self.resize)
        # to tensor
        image = image.transpose((2, 0, 1))
        image1 = image1.transpose((2, 0, 1))

        return {'image0': image, 'image1': image1, 'dataset': 'image_pair'}

    def read_image(self, path):
        files = []
        file = open(path, 'r')
        lines = file.readlines()
        """ txt file format:
        image00 image01
        image10 image11
        ...
        """
        for line in lines:
            line = line.strip().split()
            files.append([line[0], line[1]])
        file.close()
        return files


if __name__ == '__main__':
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import matplotlib
    # matplotlib.use('TkAgg')
    dataset = ImagePairsDataset(
        path_file='/media/ddc_robot/4cda377d-28db-4424-921c-6a1e0545ceeb/Dataset/long_term_image/file.txt',
        gray=False)

    for data in tqdm(dataset):
        image0 = data['image0']
        # plt.imshow(image0.transpose(1, 2, 0))
        # plt.show()
