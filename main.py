import os
from glob import glob
import json
import argparse
from tqdm import tqdm
import pandas as pd
import yaml
import wandb
import torch
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, \
    LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import StratifiedKFold, train_test_split
import lightning_fabric
import cv2

from data import get_dataloader
from model import get_model
from optimizer import get_optimizer
from system import System
from utils import prepare_parser_from_dict, parse_args_as_dict, save_files
from scheduler import get_cosine_schedule_with_warmup


def run(
    conf,
    train_df,
    val_df,
    test_df,
    test=False,
):
    lightning_fabric.utilities.seed.seed_everything(conf["data"]["seed"])
    exp_dir = conf["main_args"]["exp_dir"]
    experiment_name = conf["main_args"]["name"]
    wandb.init(project="PV Panel Detection5",
               name=f"{experiment_name}",
               config=conf)
    logger = WandbLogger()

    train_loader = get_dataloader(conf, train_df)
    val_loader = get_dataloader(conf, val_df, test=True)
    test_loader = get_dataloader(conf, test_df, test=True)

    model = get_model(conf)

    optimizer = get_optimizer(model, **conf["optim"])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 5000, conf["training"]["epochs"] * len(train_loader))

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
        save_top_k=-1,
        verbose=False,
    )
    callbacks.append(checkpoint)
    callbacks.append(LearningRateMonitor(logging_interval='step'))

    trainer = pl.Trainer(
        max_epochs=conf["training"]["epochs"],
        gradient_clip_val=conf["training"]["gradient_clipping"],
        resume_from_checkpoint=conf["training"]["ckpt_path"],
        callbacks=callbacks,
        default_root_dir=exp_dir,
        accelerator="gpu",
        logger=logger,
        precision=16,
        check_val_every_n_epoch=conf["training"]["epochs"] if test else 1,
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
    trainer = pl.Trainer(default_root_dir=exp_dir,
                         accelerator="gpu",
                         logger=logger)

    for area in conf["data"]["areas"]:
        a = test_df[test_df.image.map(lambda x: f"/{area}/" in x)].reset_index(
            drop=True)
        test_loader = get_dataloader(conf, a, test=True)
        trainer.test(system, dataloaders=test_loader, verbose=False)

    test_df["area"] = "all"
    test_loader = get_dataloader(conf, test_df, test=True)
    trainer.test(system, dataloaders=test_loader, verbose=False)
    system.cpu()

    to_save = system.model.state_dict()
    torch.save(to_save, os.path.join(exp_dir, f"best_model.pth"))

    wandb.finish()

    return checkpoint.best_model_score.item()


def check_panel(path):
    if cv2.imread(path).sum() != 0:
        return 1
    else:
        return 0


def get_df(area_to_process):
    if os.path.exists("data/df.csv"):
        return pd.read_csv("data/df.csv")
    image_list = []
    mask_list = []
    area_list = []
    panel_list = []
    for area in tqdm(area_to_process):
        mask = sorted(glob(f"data/annotations/{area}/*/*"))
        img = [i.replace("annotations", "images") for i in mask]
        areas = [area] * len(img)

        image_list.extend(img)
        mask_list.extend(mask)
        area_list.extend(areas)
        panel_list.extend([check_panel(i) for i in mask])

    df = pd.DataFrame({
        "image": image_list,
        "mask": mask_list,
        "area": area_list,
        "panel": panel_list
    })
    df["stratify"] = df["area"].astype(str) + "_" + df["panel"].astype(str)
    df.to_csv("data/df.csv", index=False)

    return df


def main(conf):
    all_df = get_df(conf["data"]["areas"])

    train_df, test_df = train_test_split(all_df,
                                         test_size=0.1,
                                         random_state=conf["data"]["seed"],
                                         shuffle=True,
                                         stratify=all_df.stratify)
    skf = StratifiedKFold(n_splits=conf["training"]["n_splits"]).split(
        train_df.index, train_df.stratify)
    k, (train_index, val_index) = next(enumerate(skf))
    train = train_df.iloc[train_index]
    val = train_df.iloc[val_index]
    run(conf, train, val, test_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir",
                        default="exp/tmp",
                        help="Full path to save best validation model")
    parser.add_argument("--name", default="tmp", help="experiment name")

    with open("conf.yml") as f:
        def_conf = yaml.safe_load(f)
    parser = prepare_parser_from_dict(def_conf, parser=parser)
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    os.makedirs(arg_dic["main_args"]["exp_dir"], exist_ok=True)
    save_files(arg_dic["main_args"]["exp_dir"])
    main(arg_dic)
