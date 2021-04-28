"""
This is a script of fitting DL model.
Reference: https://github.com/lukasliebel/dfc2020_baseline/blob/master/code/train.py
Author: Lei Song
Maintainer: Lei Song (lsong@clarku.edu)
"""

import argparse
from augmentation import *
from dataset import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import pickle as pkl
from models.deeplab import DeepLab
from models.unet import UNet
from train import Trainer
from loss import BalancedCrossEntropyLoss


def main():
    # Get in-line arguments
    # define and parse arguments
    parser = argparse.ArgumentParser()

    # config
    parser.add_argument('--exp_name', type=str, default="experiment",
                        help='experiment name that will be used in path names '
                             'for logs and checkpoints (default: experiment)')

    # dataset
    parser.add_argument('--data_dir', type=str, default=None,
                        help='path to dataset')
    parser.add_argument('--out_dir', type=str, default="models",
                        help='path to output dir (default: ./models)')
    parser.add_argument('--lowest_score', type=int, default=9,
                        help='lowest score to subset train dataset (default: 9)')
    parser.add_argument('--noise_ratio', type=float, default=0.2,
                        help='ratio of noise to subset train dataset (default: 0.2)')
    parser.add_argument('--rg_rotate', type=str, default='-90, 90',
                        help='ratio of noise to subset train dataset (default: -90, 90)')

    # Network
    parser.add_argument('--model', type=str, choices=['unet', 'deeplab'],
                        default="unet",
                        help='network architecture (default: unet)')
    parser.add_argument('--train_mode', type=str, choices=['single', 'double'],
                        default="single",
                        help='training mode: single model or co-teaching (default: single)')
    parser.add_argument('--out_stride', type=int, default=8,
                        help='out stride size for deeplab (default: 8)')

    # Training hyper-parameters
    parser.add_argument('--lr', type=float, default=0.001,
                        help='initial learning rate')
    parser.add_argument('--decay', type=float, default=1e-5,
                        help='decay rate')
    parser.add_argument('--val_freq', type=int, default=1000,
                        help='validation will be run every val_freq \
                        batches/optimization steps during training')
    parser.add_argument('--save_freq', type=int, default=1000,
                        help='training state will be saved every save_freq \
                        batches/optimization steps during training')
    parser.add_argument('--log_freq', type=int, default=100,
                        help='tensorboard logs will be written every log_freq \
                              number of batches/optimization steps')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size for prediction (default: 16)')
    parser.add_argument('--workers', type=int, default=1,
                        help='number of workers for data-loading (default: 4)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs (default: 100)')
    parser.add_argument('--optimizer_name', type=str, choices=['Adadelta', 'Adam'],
                        default="Adam",
                        help='optimizer (default: Adam)')
    parser.add_argument('--resume', '-r', type=str, default=None,
                        help='path to the pretrained weights file', )

    args = parser.parse_args()

    # Check inputs
    assert args.optimizer_name in ['Adadelta', 'Adam']
    assert args.train_mode in ['single', 'double']
    assert args.model in ['deeplab', 'unet']

    # Set directory for saving files
    if args.exp_name:
        args.checkpoint_dir = os.path.join(args.out_dir, args.exp_name, 'checkpoints')
        args.logs_dir = os.path.join(args.out_dir, args.exp_name, 'logs')
    else:
        args.checkpoint_dir = os.path.join(args.out_dir, args.model, 'checkpoints')
        args.logs_dir = os.path.join(args.out_dir, args.model, 'logs')

    # Create dirs if necessary
    if not os.path.isdir(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.isdir(args.logs_dir):
        os.makedirs(args.logs_dir)

    # Dir for mean and sd pickles
    args.stats_dir = os.path.join(args.out_dir, 'norm_stats')

    # Load dataset
    # synchronize transform for train dataset
    sync_transform = Compose([
        RandomScale(),
        RandomFlip(),
        RandomCenterRotate(),
        SyncToTensor()
    ])

    # synchronize transform for validate dataset
    val_transform = Compose([
        SyncToTensor()
    ])

    # Image transform
    # Load mean and sd for normalization
    # with open(os.path.join(args.stats_dir,
    #                        "means.pkl"), "rb") as input_file:
    #     mean = tuple(pkl.load(input_file))
    #
    # with open(os.path.join(args.stats_dir,
    #                        "stds.pkl"), "rb") as input_file:
    #     std = tuple(pkl.load(input_file))
    # img_transform = ImgNorm(mean, std)

    # Get train dataset
    train_dataset = NFSEN1LC(data_dir=args.data_dir,
                             usage='train',
                             lowest_score=args.lowest_score,
                             noise_ratio=args.noise_ratio,
                             rg_rotate=tuple(float(each) for each in args.rg_rotate.split(',')),
                             sync_transform=sync_transform,
                             img_transform=None,
                             label_transform=None)
    # Put into DataLoader
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              drop_last=True)

    # Get validate dataset
    validate_dataset = NFSEN1LC(data_dir=args.data_dir,
                                usage='validate',
                                sync_transform=val_transform,
                                img_transform=None,
                                label_transform=None)
    # Put into DataLoader
    validate_loader = DataLoader(dataset=validate_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 drop_last=True)

    # Set up network
    args.n_classes = train_dataset.n_classes
    args.n_channels = train_dataset.n_channels
    if args.model == "deeplab":
        model = DeepLab(num_classes=args.n_classes,
                        backbone='resnet',
                        pretrained_backbone=False,
                        output_stride=args.out_stride,
                        sync_bn=False,
                        freeze_bn=False,
                        n_in=args.n_channels)
    else:
        model = UNet(n_classes=args.n_classes,
                     n_channels=args.n_channels)
    if args.use_gpu:
        model = model.cuda()

    # Define loss function
    loss_fn = BalancedCrossEntropyLoss

    # Define optimizer
    if args.optimizer_name == 'Adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(),
                                         lr=args.lr,
                                         weight_decay=args.decay)
    elif args.optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr)
    else:
        print('Not supported optimizer, use Adam instead.')
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr)

    # Set up tensorboard logging
    writer = SummaryWriter(log_dir=os.path.join(args.logs_dir, args.exp_name))

    # Save config
    pkl.dump(args, open(os.path.join(args.checkpoint_dir, "args.pkl"), "wb"))

    # Train network
    if args.train_mode == 'single':
        step = 0
        trainer = Trainer(args)
        for epoch in range(args.epoches):
            print("=" * 20, "EPOCH", epoch + 1, "/", str(args.max_epochs), "=" * 20)
            # Run training for one epoch
            model, step = trainer.train(model, train_loader, validate_loader, loss_fn,
                                        optimizer, writer, step=step)
        # export final set of weights
        trainer.export_model(model, optimizer, name="final")

    elif args.train_model == 'double':
        pass
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
