import cv2
from albumentations import Compose, LongestMaxSize, ShiftScaleRotate

def get_augmentations(IMG_SZ):
    train_aug = Compose([ 
        LongestMaxSize(IMG_SZ, p=1.0, always_apply=True),
        ShiftScaleRotate(p=1.0, border_mode=cv2.BORDER_CONSTANT, rotate_limit=10),
    ])

    val_aug = Compose([ 
        LongestMaxSize(IMG_SZ, p=1.0, always_apply=True),
    ])

    return train_aug, val_aug