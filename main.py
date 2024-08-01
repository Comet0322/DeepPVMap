import argparse
import json
import math
import os
from glob import glob

import cv2
import lightning_fabric
import pandas as pd
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import wandb
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from dataset import get_dataloader
from model import get_model
from system import System
from utils import parse_args_as_dict, prepare_parser_from_dict


def get_optimizer(model, optimizer, lr, weight_decay):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or "bn" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    params = [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]

    return AdamW(params, lr=lr)


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
        The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
        The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
        The total number of training steps.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
        The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
        following a half-cosine).
        last_epoch (:obj:`int`, `optional`, defaults to -1):
        The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        # Warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # decadence
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_df(conf):
    area_to_process = conf["main_args"]["area_list"].split(",")
    base_dir = conf["main_args"]["data_dir"]
    df_path = os.path.join(base_dir, "df.csv")
    if os.path.exists(df_path):
        return pd.read_csv(df_path)

    image_list, mask_list, area_list = [], [], []
    for area in tqdm(area_to_process):
        mask = sorted(glob(f"{base_dir}/annotations/{area}/*/*"))
        img = [i.replace("annotations", "images") for i in mask]
        areas = [area] * len(img)

        image_list.extend(img)
        mask_list.extend(mask)
        area_list.extend(areas)

    df = pd.DataFrame(
        {
            "image": image_list,
            "mask": mask_list,
            "area": area_list,
        }
    )
    df["panel"] = df["mask"].map(lambda path: cv2.imread(path).sum() != 0)
    df["stratify"] = df["area"] + "_" + df["panel"].astype(str)
    df.to_csv(df_path, index=False)

    return df


def main(conf):
    all_df = get_df(conf)
    dev_df, test_df = train_test_split(
        all_df,
        test_size=0.1,
        random_state=conf["data"]["seed"],
        shuffle=True,
        stratify=all_df.stratify,
    )
    dev_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    skf = StratifiedKFold(n_splits=conf["training"]["n_splits"]).split(
        dev_df.index, dev_df.stratify
    )
    train_index, val_index = next(skf)
    train_df = dev_df.iloc[train_index].reset_index(drop=True)
    val_df = dev_df.iloc[val_index].reset_index(drop=True)

    lightning_fabric.utilities.seed.seed_everything(conf["data"]["seed"])
    exp_dir = conf["main_args"]["exp_dir"]
    experiment_name = conf["main_args"]["name"]
    logger = None
    wandb.init(project="PV Panel Detection5", name=f"{experiment_name}", config=conf)
    logger = WandbLogger()

    train_loader = get_dataloader(conf, train_df)
    val_loader = get_dataloader(conf, val_df, test=True)
    test_loader = get_dataloader(conf, test_df, test=True)
    model = get_model(conf)
    optimizer = get_optimizer(model, **conf["optim"])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 5000, conf["training"]["epochs"] * len(train_loader)
    )

    loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=False)

    system = System(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        config=conf,
    )

    callbacks = []
    checkpoint_dir = os.path.join(exp_dir, "checkpoints/")
    checkpoint = ModelCheckpoint(
        checkpoint_dir,
        monitor="val_panel_IOU",
        mode="max",
        save_top_k=1,
        verbose=False,
    )
    callbacks.append(checkpoint)
    callbacks.append(LearningRateMonitor(logging_interval="step"))

    trainer = pl.Trainer(
        max_epochs=conf["training"]["epochs"],
        gradient_clip_val=conf["training"]["gradient_clipping"],
        resume_from_checkpoint=conf["training"]["ckpt_path"],
        callbacks=callbacks,
        default_root_dir=exp_dir,
        accelerator="gpu",
        logger=logger,
        precision=16,
    )

    trainer.fit(system)
    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(exp_dir, f"best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)

    to_save = system.model.state_dict()
    torch.save(to_save, os.path.join(exp_dir, f"last_model.pth"))

    state_dict = torch.load(checkpoint.best_model_path)
    system.load_state_dict(state_dict=state_dict["state_dict"])
    system.cpu()
    to_save = system.model.state_dict()
    torch.save(to_save, os.path.join(exp_dir, f"best_model.pth"))

    trainer = pl.Trainer(default_root_dir=exp_dir, accelerator="gpu", logger=logger)

    area_list = test_df.area.unique()
    for area in area_list:
        a = test_df[test_df.image.map(lambda x: f"/{area}/" in x)].reset_index(
            drop=True
        )
        test_loader = get_dataloader(conf, a, test=True)
        trainer.test(system, dataloaders=test_loader, verbose=False)
    test_df["area"] = "all"
    test_loader = get_dataloader(conf, test_df, test=True)
    trainer.test(system, dataloaders=test_loader, verbose=False)
    system.cpu()

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_dir", default="exp/tmp", help="Full path to save best validation model"
    )
    parser.add_argument(
        "--data_dir", default="data", help="Full path to data directory"
    )
    parser.add_argument("--name", default="tmp", help="experiment name")
    # area list
    parser.add_argument(
        "--area_list",
        default="1+7,4,5,6,8,9,11,changhua,mingjian_s",
        help="comma separated list of areas to process",
    )
    with open("conf.yml", encoding="utf8") as f:
        def_conf = yaml.safe_load(f)
    parser = prepare_parser_from_dict(def_conf, parser=parser)
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    os.makedirs(arg_dic["main_args"]["exp_dir"], exist_ok=True)
    main(arg_dic)
