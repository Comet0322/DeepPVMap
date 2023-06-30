import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from torch.optim.lr_scheduler import ReduceLROnPlateau


class System(pl.LightningModule):

    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        train_loader,
        val_loader=None,
        scheduler=None,
        config=None,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.config = {} if config is None else config
        self.default_monitor = "val_loss"
        self.segmentation_classes = ["background", "panel"]

    def forward(self, img):
        return self.model(img)

    def shared_step(self, batch, stage):
        image, mask, *_ = batch
        probs = self.forward(image)
        loss = self.loss_fn(probs, mask.float())
        preds = (probs > 0.5).long()
        tp, fp, fn, tn = smp.metrics.get_stats(preds,
                                               mask.long(),
                                               mode="binary")

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])
        loss = torch.stack([x["loss"] for x in outputs])
        if loss.isnan().any():
            print(
                f"{stage}, {len(loss[loss.isnan()])}/{len(loss)} of batch is nan"
            )
            loss = torch.mean(loss[~loss.isnan()])
        else:
            loss = torch.mean(loss)

        panelIOU = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        metrics = {
            f"{stage}_panel_IOU": panelIOU,
        }

        self.log_dict(metrics, prog_bar=True, sync_dist=True)
        self.log(f"{stage}_loss", loss, sync_dist=True)

    def test_start(self, batch, stage):
        image, mask, area = batch
        probs = self.forward(image)
        loss = self.loss_fn(probs, mask.float())
        preds = (probs > 0.5).long()
        tp, fp, fn, tn = smp.metrics.get_stats(preds,
                                               mask.long(),
                                               mode="binary")

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "area": area
        }

    def test_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])
        area = outputs[0]["area"][0]

        loss = torch.stack([x["loss"] for x in outputs]).flatten()
        loss = torch.mean(loss[~loss.isnan()])
        panelIOU = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        metrics = {
            f"{stage}_{area}_panel_IOU": panelIOU,
        }

        self.log_dict(metrics, prog_bar=True, sync_dist=True)
        self.log(f"{stage}_{area}_loss", loss, sync_dist=True)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "val")

    def test_step(self, batch, batch_idx):
        return self.test_start(batch, "test")

    def test_epoch_end(self, outputs):
        return self.test_end(outputs, "test")

    def configure_optimizers(self):
        if self.scheduler is None:
            return self.optimizer
        else:
            if isinstance(self.scheduler, ReduceLROnPlateau):
                epoch_schedulers = [{
                    "scheduler": self.scheduler,
                    "monitor": self.default_monitor
                }]
            else:
                epoch_schedulers = [{
                    "scheduler": self.scheduler,
                    "interval": "step"
                }]
            return [self.optimizer], epoch_schedulers

    def train_dataloader(self):
        """Training dataloader"""
        return self.train_loader

    def val_dataloader(self):
        """Validation dataloader"""
        return self.val_loader

    def on_save_checkpoint(self, checkpoint):
        """Overwrite if you want to save more things in the checkpoint."""
        checkpoint["training_config"] = self.config
        return checkpoint
