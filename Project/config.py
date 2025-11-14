import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

Config = {
    'base_dir': '/home/stud/fmarchetto/SegmentationTests/Data/',
    'datasets': {'balanced': '/home/stud/fmarchetto/SegmentationTests/Data/balanced_train_df.csv'},
    'val_csv': '/home/stud/fmarchetto/SegmentationTests/Data/val_df.csv',
    'test_csv': '/home/stud/fmarchetto/SegmentationTests/Data/test_df.csv',
    'models': [{'name': 'U-Net ResNet34', 'arch': 'Unet', 'encoder': 'resnet34'}, {'name': 'DeepLabV3+ ResNet50', 'arch': 'DeepLabV3Plus', 'encoder': 'resnet50'}],
    'batch_sizes': [16],
    'learning_rates': [0.001, 0.0001],
    'optimizers': [{'name': 'RMSprop', 'class': 'RMSprop', 'extra_params': {'momentum': 0.9}}, {'name': 'Adam', 'class': 'Adam'}],
    'epochs': 8,
    'patience_early_stop': 3,
    'device': "torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
    'wandb_project': 'Zuliani1_Marchetto2',
    'transform': 'A.Compose([\n    A.HorizontalFlip(p=0.5),\n    A.Normalize(mean=0.5, std=0.25),\n    ToTensorV2()\n])',
}
