import albumentations as A


def get_training_augmentation():
    transform = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=45, p=0.5),
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
