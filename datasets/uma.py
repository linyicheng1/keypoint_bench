import torch.utils.data as data
import cv2


class UMADataset(data.Dataset):

    def __init__(self, root="", gray=False):
        self.root = root
        self.gray = gray
        self.image_list = []

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        # read image
        img = cv2.imread(self.image_list[item], cv2.IMREAD_COLOR)
        assert img is not None, 'can not load: ' + self.image_list[item]

