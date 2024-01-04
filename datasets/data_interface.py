import inspect
import importlib
import pickle as pkl
import pytorch_lightning as pl
import torch.nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from pytorch_lightning import LightningDataModule
from .hpatches import HPatchesDataset
from .megadepth import MegaDepthDataset


class DInterface(LightningDataModule):

    def __init__(self, params):
        super().__init__()
        self.num_workers = params['num_workers']
        self.batch_size = params['batch_size']
        if 'train_dataset' in params:
            self.trainset_param = params['train_dataset']
        if 'val_dataset' in params:
            self.valset_param = params['val_dataset']
        if 'test_dataset' in params:
            self.testset_param = params['test_dataset']

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.trainset = self.instancialize(self.trainset_param, train=True)
            self.valset = self.instancialize(self.valset_param, train=False)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.testset = self.instancialize(self.testset_param, train=False)

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

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
        else:
            raise ValueError('Invalid dataset type')


