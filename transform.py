import albumentations as A


def get_training_augmentation():
    transform = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=45, p=0.5),
        # A.OneOf(
        #     [
        #         A.RandomBrightnessContrast(
        #             p=1, brightness_limit=0.2, contrast_limit=0.2),
        #         A.RandomGamma(p=1, gamma_limit=(80, 120)),
        #         # A.HueSaturationValue(
        #         #     p=1,
        #         #     hue_shift_limit=20,
        #         #     sat_shift_limit=30,
        #         #     val_shift_limit=20,
        #         # ),
        #     ],
        #     p=0.8,
        # ),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.Lambda(image=ToTensor, mask=ToTensor),
    ]
    return A.Compose(transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    transform = [
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.Lambda(image=ToTensor, mask=ToTensor),
    ]
    return A.Compose(transform)


def ToTensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')
