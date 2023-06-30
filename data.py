import cv2
from transform import get_training_augmentation, get_validation_augmentation
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np


def read_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def read_mask(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)[..., None]
    return mask


class PVDataset(Dataset):

    def __init__(self, df, transform=None, return_meta=False):
        self.csv = df.reset_index(drop=True)
        self.transform = transform
        self.return_meta = return_meta
        self.image_list = self.csv["image"]
        self.mask_list = self.csv["mask"]
        self.area_list = self.csv["area"]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        imag_path = self.image_list[idx]
        image = read_image(imag_path)
        mask = read_mask(self.mask_list[idx])
        area = self.area_list[idx]

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]

        if self.return_meta:
            return image, mask, area
        else:
            return image, mask


def get_dataloader(conf, df, test=False):
    if not test:
        dataset = PVDataset(
            df,
            get_training_augmentation(),
        )
        dataloader = DataLoader(dataset,
                                batch_size=conf["training"]["batch_size"],
                                num_workers=conf["training"]["num_workers"],
                                drop_last=True,
                                shuffle=True,
                                persistent_workers=True,
                                pin_memory=True)
    else:
        dataset = PVDataset(df,
                            get_validation_augmentation(),
                            return_meta=True)
        dataloader = DataLoader(dataset,
                                batch_size=2 * conf["training"]["batch_size"],
                                num_workers=conf["training"]["num_workers"],
                                persistent_workers=True,
                                pin_memory=True)
    return dataloader
