from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from datasets.hpatches import HPatchesDataset
from datasets.megadepth import MegaDepthDataset
from datasets.kitti import KittiDataset
from datasets.euroc import EurocDataset
from datasets.uma import UMADataset
from datasets.video import VideoDataset
from datasets.images import ImagesDataset
from datasets.tartanair import TartanAirDataset
from datasets.image_pairs import ImagePairsDataset


class DInterface(LightningDataModule):

    def __init__(self, params):
        super().__init__()
        self.num_workers = params['num_workers']
        self.batch_size = params['batch_size']
        data_type = params['data_type']
        self.test_set_param = params[data_type+'_params']

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.trainset = self.instancialize(self.trainset_param, train=True)
            self.valset = self.instancialize(self.valset_param, train=False)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test_set = self.instancialize(self.test_set_param, train=False)

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def instancialize(self, params, train=True):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.kwargs.
        """
        if params['type'] == 'hpatches':
            return HPatchesDataset(params['root'], params['alteration'], params['image_size'], params['gray'])
        elif params['type'] == 'megadepth':
            return MegaDepthDataset(params['root'], train, params['using_cache'],
                                    params['pairs_per_scene'], params['image_size'],
                                    params['colorjit'], params['gray'],
                                    params['crop_or_scale'])
        elif params['type'] == 'kitti':
            return KittiDataset(params['root'], params['gt'], params['gray'])
        elif params['type'] == 'euroc':
            return EurocDataset(params['root'], params['gray'])
        elif params['type'] == 'uma':
            return UMADataset(params['root'], params['gray'])
        elif params['type'] == 'tartanair':
            return TartanAirDataset(params['root'], params['gt'], params['gray'])
        elif params['type'] == 'video':
            return VideoDataset(params['root'], params['gray'])
        elif params['type'] == 'images':
            return ImagesDataset(params['root'], params['gray'])
        elif params['type'] == 'image_pair':
            resize = [params['image_size'], params['image_size']] if 'image_size' in params else None
            return ImagePairsDataset(params['root'], params['gray'], resize)
        else:
            raise ValueError('Invalid dataset type')


