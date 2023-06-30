import torch
from segmentation_models_pytorch import Unet


def get_model(conf):
    model_arch = conf["model"]["arch"]
    encoder_name = conf["model"]["encoder"]
    model = eval(model_arch)(
        encoder_name,
        encoder_weights='imagenet',
        classes=1,
        activation="sigmoid",
    )

    return model
