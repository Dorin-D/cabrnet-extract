from cabrnet.core.utils.data import DatasetManager

import pytorch_lightning as pl




class LatentDataModule(pl.LightningDataModule):
    def __init__(self, dataset_path):
        super().__init__()
        self.dataset_path = dataset_path

    def setup(self, stage):
        dataloaders = DatasetManager.get_dataloaders(config=self.dataset_path,
                                                     sampling_ratio=1)
        self.train_loader = dataloaders['train_set']
        self.test_loader = dataloaders['test_set']

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.test_loader