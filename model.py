from segmentation_models_pytorch import Unet


def get_model(conf):
    encoder_name = conf["model"]["encoder"]
    model = Unet(
        encoder_name,
        encoder_weights="imagenet",
        classes=1,
        activation="sigmoid",
    )

    return model
