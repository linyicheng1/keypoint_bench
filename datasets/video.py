import torch.utils.data as data
import cv2
import numpy as np


class VideoDataset(data.Dataset):

    def __init__(self, root="", gray=False, resize=None):
        self.root = root
        self.gray = gray
        # opencv read video
        self.video = cv2.VideoCapture(self.root)
        self.resize = resize

    def __len__(self):
        # get video frame count
        return int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

    def __getitem__(self, item):
        # read video frame
        image = self.video.read()[1]
        assert image is not None, 'can not load: ' + self.root

        if self.gray:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype('float32') / 255.
            image = np.expand_dims(image, axis=2)
        else:
            # bgr -> rgb
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype('float32') / 255.
        if self.resize is not None:
            image = cv2.resize(image, self.resize)
        # to tensor
        image = image.transpose((2, 0, 1))
        return {'image0': image}


if __name__ == '__main__':
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('TkAgg')
    dataset = VideoDataset(root='/media/server/4cda377d-28db-4424-921c-6a1e0545ceeb/Dataset/light_videos/1.MP4', gray=False)
    for data in tqdm(dataset):
        image0 = data['image0']
        plt.imshow(image0.transpose(1, 2, 0))
        plt.show()

