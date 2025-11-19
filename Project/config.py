import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

Config = {
    'base_dir': '/home/stud/fmarchetto/SegmentationTests/Data/',
    'datasets': {'balanced': '/home/stud/fmarchetto/SegmentationTests/Data/balanced_train_df.csv'},
    'val_csv': '/home/stud/fmarchetto/SegmentationTests/Data/val_df.csv',
    'test_csv': '/home/stud/fmarchetto/SegmentationTests/Data/test_df.csv',
    'models': [{'name': 'U-Net ResNet34', 'arch': 'Unet', 'encoder': 'resnet34'}, {'name': 'U-Net efficientnet-b2', 'arch': 'Unet', 'encoder': 'efficientnet-b2'}],
    'batch_sizes': [8, 16],
    'learning_rates': [0.001, 0.0001],
    'optimizers': [{'name': 'SGD_Momentum', 'class': 'SGD', 'extra_params': {'momentum': 0.9, 'nesterov': True}}, {'name': 'RMSprop', 'class': 'RMSprop', 'extra_params': {'momentum': 0.9}}, {'name': 'Adam', 'class': 'Adam'}],
    'epochs': 8,
    'patience_early_stop': 3,
    'device': "torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
    'wandb_project': 'Zuliani1_Marchetto2',
    'transform': 'A.Compose([\n    A.Rotate(limit=10, p=0.5), A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=0, p=0.5), A.HorizontalFlip(p=0.5),    A.RandomBrightnessContrast(p=0.5), ToTensorV2()\n])',
}
