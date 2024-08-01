import os

import cv2
from torch.utils.data import DataLoader, Dataset

from transform import get_training_augmentation, get_validation_augmentation


class PVDataset(Dataset):

    def __init__(self, df, transform=None, return_meta=False):
        self.csv = df
        self.transform = transform
        self.return_meta = return_meta
        self.image_list = self.csv["image"]
        self.mask_list = self.csv["mask"]
        self.area_list = self.csv["area"]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = (
            self.image_list[idx]
            if os.path.exists(self.image_list[idx])
            else self.image_list[idx].replace("png", "jpg")
        )
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_list[idx], cv2.IMREAD_GRAYSCALE)[..., None]

        area = self.area_list[idx]

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]

        if self.return_meta:
            return image, mask, area
        else:
            return image, mask


def get_dataloader(conf, df, test=False):
    if test:
        dataset = PVDataset(df, get_validation_augmentation(), return_meta=True)
        dataloader = DataLoader(
            dataset,
            batch_size=2 * conf["training"]["batch_size"],
            num_workers=conf["training"]["num_workers"],
            persistent_workers=True,
            pin_memory=True,
        )
    else:
        dataset = PVDataset(
            df,
            get_training_augmentation(),
        )
        dataloader = DataLoader(
            dataset,
            batch_size=conf["training"]["batch_size"],
            num_workers=conf["training"]["num_workers"],
            drop_last=True,
            shuffle=True,
            persistent_workers=True,
            pin_memory=True,
        )
    return dataloader

