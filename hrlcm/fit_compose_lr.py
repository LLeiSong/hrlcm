"""
This is a script of fitting DL model.
Reference: https://github.com/lukasliebel/dfc2020_baseline/blob/master/code/train.py
Author: Lei Song
Maintainer: Lei Song (lsong@clarku.edu)
"""

import argparse
from math import floor
from augmentation import *
from dataset import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pickle as pkl
from models.deeplab import DeepLab
from models.unet import UNet
from train import Trainer
import torch_optimizer as optim
from loss import weighted_loss, BalancedCrossEntropyLoss
from lr_scheduler import get_compose_lr


def main():
    # Get in-line arguments
    # define and parse arguments
    parser = argparse.ArgumentParser()

    # config
    parser.add_argument('--exp_name', type=str, default="experiment",
                        help='experiment name that will be used in path names '
                             'for logs and checkpoints (default: experiment)')

    # dataset
    parser.add_argument('--data_dir', type=str, default='results/north',
                        help='path to dataset (default: results/north)')
    parser.add_argument('--out_dir', type=str, default="results/dl",
                        help='path to output dir (default: results/dl)')
    parser.add_argument('--label_offset', type=int, default=1,
                        help='offset value to minus from label in order to start from 0 (default: 1)')
    parser.add_argument('--rg_rotate', type=str, default='-90, 90',
                        help='ratio of noise to subset train dataset (default: -90, 90)')
    parser.add_argument('--trans_prob', type=float, default=0.5,
                        help='probability to do data transformation (default:0.5)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of worker(s) to load dataset (default: 0)')

    # Network
    parser.add_argument('--model', type=str, choices=['unet', 'deeplab'],
                        default="unet",
                        help='network architecture (default: unet)')
    parser.add_argument('--out_stride', type=int, default=8,
                        help='out stride size for deeplab (default: 8)')
    parser.add_argument('--gpu_devices', type=str, default=None,
                        help='the gpu devices to use (default: None) (format: 1, 2)')

    # Training hyper-parameters
    parser.add_argument('--min_lr', type=float, default=0.0001,
                        help='minimum or last learning rate for scheduler.')
    parser.add_argument('--max_lr', type=float, default=0.001,
                        help='maximum or initial learning rate for scheduler.')
    parser.add_argument('--optimizer_name', type=str,
                        choices=['AdaBound', 'AmsBound', 'AdamP'],
                        default="AmsBound",
                        help='optimizer (default: AmsBound)')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='training state will be saved every save_freq \
                        batches during training')
    parser.add_argument('--train_batch_size', type=int, default=32,
                        help='batch size for training (default: 16)')
    parser.add_argument('--val_batch_size', type=int, default=32,
                        help='batch size for validation (default: 16)')
    parser.add_argument('--epochs', type=int, default=400,
                        help='number of training epochs (default: 400). '
                             'NOTE: The scheduler is designed best for 400.')
    parser.add_argument('--resume', '-r', type=str, default=None,
                        help='path to the pretrained weights file')

    args = parser.parse_args()

    # Check inputs
    assert args.optimizer_name.lower() in ['adabound', 'amsbound', 'adamp']
    assert args.model in ['deeplab', 'unet']
    assert args.quality_weight in [0, 1]

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
    args.stats_dir = os.path.join(args.data_dir, 'norm_stats')

    # Set flags for GPU processing if available
    args.use_gpu = torch.cuda.is_available()

    # Load dataset
    # Define rotate degrees
    args.rg_rotate = tuple(float(each) for each in args.rg_rotate.split(','))

    # synchronize transform for train dataset
    sync_transform = Compose([
        # RandomScale(prob=args.trans_prob),
        RandomFlip(prob=args.trans_prob),
        # RandomCenterRotate(degree=args.rg_rotate,
        #                    prob=args.trans_prob),
        SyncToTensor()
    ])

    # synchronize transform for validate dataset
    sync_transform_val = Compose([
        SyncToTensor()
    ])

    # Image transform
    # Load mean and sd for normalization
    with open(os.path.join(args.stats_dir,
                           "means.pkl"), "rb") as input_file:
        mean = tuple(pkl.load(input_file))

    with open(os.path.join(args.stats_dir,
                           "stds.pkl"), "rb") as input_file:
        std = tuple(pkl.load(input_file))
    img_transform = ImgNorm(mean, std)

    # Get train dataset
    train_dataset = NFSEN1LC(data_dir=args.data_dir,
                             usage='train',
                             label_offset=args.label_offset,
                             sync_transform=sync_transform,
                             img_transform=img_transform,
                             label_transform=None)
    # Put into DataLoader
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.train_batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=True)

    # Get validate dataset
    validate_dataset = NFSEN1LC(data_dir=args.data_dir,
                                usage='validate',
                                label_offset=args.label_offset,
                                sync_transform=sync_transform_val,
                                img_transform=img_transform,
                                label_transform=None)
    # Put into DataLoader
    validate_loader = DataLoader(dataset=validate_dataset,
                                 batch_size=args.val_batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers,
                                 pin_memory=True,
                                 drop_last=False)

    # Set up network
    args.n_classes = train_dataset.n_classes
    args.n_channels = train_dataset.n_channels

    # Save config
    pkl.dump(args, open(os.path.join(args.checkpoint_dir, "args.pkl"), "wb"))

    # Set up tensorboard logging
    writer = SummaryWriter(log_dir=args.logs_dir)

    # Train network
    # Define model
    if args.model == "deeplab":
        # TODO Not fully test yet
        model = DeepLab(num_classes=args.n_classes,
                        backbone='resnet',
                        pretrained_backbone=False,
                        output_stride=args.out_stride,
                        freeze_bn=False,
                        n_in=args.n_channels)
    else:
        model = UNet(n_classes=args.n_classes,
                     n_channels=args.n_channels)

    # Get devices
    if args.gpu_devices:
        args.gpu_devices = [int(each) for each in args.gpu_devices.split(',')]

    # Set model
    if args.use_gpu:
        if args.gpu_devices:
            torch.cuda.set_device(args.gpu_devices[0])
            model = torch.nn.DataParallel(model, device_ids=args.gpu_devices)
        model = model.cuda()

    # Get learning rate first
    learning_rates = get_compose_lr(model, args.epochs, args.min_lr, args.max_lr)

    # Define loss function
    loss_fn = weighted_loss
    loss_fn_valid = BalancedCrossEntropyLoss()

    # Define optimizer
    if args.optimizer_name.lower() == 'adabound':
        optimizer = optim.AdaBound(model.parameters(),
                                   lr=args.max_lr,
                                   final_lr=0.01)  # use constant final_lr 0.01
    elif args.optimizer_name.lower() == 'amsbound':
        optimizer = optim.AdaBound(model.parameters(),
                                   lr=args.max_lr,
                                   final_lr=0.01,
                                   amsbound=True)
    elif args.optimizer_name.lower() == 'adamp':
        optimizer = optim.AdamP(model.parameters(),
                                nesterov=True,
                                lr=args.max_lr)
    else:
        print('Not supported optimizer, use AdaBound instead.')
        optimizer = optim.AdaBound(model.parameters(),
                                   lr=args.max_lr,
                                   final_lr=0.01)
    # Start train
    step = 0
    epoch = 0
    # Resume model based on settings
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)

            # Get step and epoch
            if checkpoint['step'] > step:
                step = checkpoint['step']
                epoch = floor(step / floor(len(train_dataset) / args.train_batch_size))
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Load checkpoint '{}' (epoch {})".format(args.resume, epoch))
        else:
            print("No checkpoint found at '{}'".format(args.resume))

    # Do loop
    trainer = Trainer(args)
    for epoch in range(epoch + 1, args.epochs):
        # Update info
        print("[Epoch {}] lr: {}".format(
            epoch, optimizer.param_groups[0]["lr"]))

        # Run training for one epoch
        model, step = trainer.train(model, train_loader, loss_fn,
                                    optimizer, writer, step=step)
        # Run validation
        trainer.validate(model, validate_loader, step, loss_fn_valid, writer)

        # Update learning rate
        # Since learning rate is a very important hyper-parameter,
        # it is recommended to visualize learning rate first.
        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rates[epoch]

        # Save checkpoint
        if epoch % args.save_freq == 0:
            trainer.export_model(model, optimizer=optimizer, step=step, name='interim')

        # Save learning rate to scalar
        writer.add_scalar("Train/lr",
                          optimizer.param_groups[0]["lr"],
                          global_step=step)
        # Flush to disk
        writer.flush()

    # Export final set of weights
    trainer.export_model(model, optimizer, name="final")


if __name__ == "__main__":
    main()
