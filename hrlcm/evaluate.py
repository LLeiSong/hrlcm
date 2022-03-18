"""
This is a script to evaluate the final model against validate dataset.
Reference: https://github.com/schmitt-muc/SEN12MS/blob/master/classification/metrics.py
Author: Lei Song
Maintainer: Lei Song (lsong@clarku.edu)
"""

import argparse
from augmentation import *
from dataset import *
from metrics import ConfMatrix
from torch.utils.data import DataLoader
import pickle as pkl
from models.deeplab import DeepLab
from models.unet import UNet
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, \
    fbeta_score, classification_report, hamming_loss


class Precision_score(nn.Module):

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(predict_labels, true_labels):
        weighted_prec = precision_score(true_labels, predict_labels, average='weighted')

        return weighted_prec


class Recall_score(nn.Module):

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(predict_labels, true_labels):
        weighted_rec = recall_score(true_labels, predict_labels, average='weighted')

        return weighted_rec


class F1_score(nn.Module):

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(predict_labels, true_labels):
        weighted_f1 = f1_score(true_labels, predict_labels, average="weighted")

        return weighted_f1


class F2_score(nn.Module):

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(predict_labels, true_labels):
        weighted_f2 = fbeta_score(true_labels, predict_labels, beta=2, average="weighted")

        return weighted_f2


class Hamming_loss(nn.Module):

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(predict_labels, true_labels):
        return hamming_loss(true_labels, predict_labels)


class cls_report(nn.Module):
    def __init__(self, target_names):
        super().__init__()
        self.target_names = target_names

    def forward(self, predict_labels, true_labels):
        report = classification_report(true_labels, predict_labels,
                                       target_names=self.target_names,
                                       output_dict=True)

        return report


def main():
    # Get in-line arguments
    # define and parse arguments
    parser = argparse.ArgumentParser()

    # config
    parser.add_argument('--args_path', type=str, default="args.pkl",
                        help='path to config file (default: ./args.pkl)')
    parser.add_argument('--checkpoint_file', type=str, default="checkpoint.pth",
                        help='path to checkpoint file (default: ./checkpoint.pth)')

    # Dataset
    parser.add_argument('--data_dir', type=str, default=None,
                        help='path to dataset')
    parser.add_argument('--out_dir', type=str, default="models",
                        help='path to output dir (default: ./models)')
    parser.add_argument('--stats_dir', type=str, default="norm_stats",
                        help='path of normalization params (default: ./norm_stats)')
    parser.add_argument('--label_offset', type=int, default=1,
                        help='offset value to minus from label in order to start from 0 (default: 1)')
    parser.add_argument('--img_bands', type=str, choices=['all', 'nicfi'],
                        default='all',
                        help='bands of satellite images to use. \
                        all means all bands, including RGB, NIR of NICFI tiles, intercept,  \
                        cos(2t) of VV and VH. (default: all)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of worker(s) to load dataset (default: 0)')

    # Hyper-parameters of evaluation
    parser.add_argument('--batch_size', type=int, default=16,
                        help='mini-batch size (default: 16)')
    parser.add_argument('--gpu_devices', type=str, default=None,
                        help='the gpu devices to use (default: None) (format: 1, 2)')

    args = parser.parse_args()
    print("=" * 20, "EVALUATION CONFIG", "=" * 20)
    for arg in vars(args):
        print('{0:20}  {1}'.format(arg, getattr(args, arg)))
    print()

    # Load config of training
    train_args = pkl.load(open(args.args_path, "rb"))
    print("=" * 20, "TRAIN CONFIG", "=" * 20)
    for arg in vars(train_args):
        print('{0:20}  {1}'.format(arg, getattr(train_args, arg)))
    print()

    # Set flags for GPU processing if available
    args.use_gpu = torch.cuda.is_available()

    # Create output dir
    os.makedirs(args.out_dir, exist_ok=True)

    # Load dataset
    # synchronize transform for validate dataset
    val_transform = Compose([
        SyncToTensor()
    ])

    # Image transform
    id_bands = list(range(1, 13)) if args.img_bands == "all" else list(range(1, 9))
    # Load mean and sd for normalization
    with open(os.path.join(args.stats_dir,
                           "means.pkl"), "rb") as input_file:
        mean = tuple(pkl.load(input_file))
        mean = mean[0:len(id_bands)]

    with open(os.path.join(args.stats_dir,
                           "stds.pkl"), "rb") as input_file:
        std = tuple(pkl.load(input_file))
        std = std[0:len(id_bands)]
    img_transform = ImgNorm(mean, std)

    # Get validate dataset
    validate_dataset = NFSEN1LC(data_dir=args.data_dir,
                                bands=id_bands,
                                usage='validate',
                                label_offset=args.label_offset,
                                sync_transform=val_transform,
                                img_transform=img_transform,
                                label_transform=None)
    # Put into DataLoader
    validate_loader = DataLoader(dataset=validate_dataset,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers,
                                 shuffle=False,
                                 drop_last=False)

    # set up network
    if train_args.model == "deeplab":
        model = DeepLab(num_classes=train_args.n_classes,
                        backbone='resnet',
                        pretrained_backbone=False,
                        output_stride=train_args.out_stride,
                        freeze_bn=False,
                        n_in=train_args.n_channels)
    else:
        model = UNet(n_classes=train_args.n_classes,
                     n_channels=train_args.n_channels)

    # Get devices
    if args.gpu_devices:
        args.gpu_devices = [int(each) for each in args.gpu_devices.split(',')]

    if args.use_gpu:
        if args.gpu_devices:
            torch.cuda.set_device(args.gpu_devices[0])
            model = torch.nn.DataParallel(model, device_ids=args.gpu_devices)
        model = model.cuda()

    # Restore network weights
    state = torch.load(args.checkpoint_file)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    print("Loaded checkpoint from {}".format(args.checkpoint_file))

    # predict samples
    # define metrics
    prec_score_ = Precision_score()
    recal_score_ = Recall_score()
    f1_score_ = F1_score()
    f2_score_ = F2_score()
    hamming_loss_ = Hamming_loss()
    types = validate_dataset.lc_types
    classification_report_ = cls_report(types)

    # Prediction
    y_true = []
    predicted_probs = []
    conf_mat = ConfMatrix(validate_loader.dataset.n_classes)
    with torch.no_grad():
        for i, (image, labels) in enumerate(
                tqdm(validate_loader, desc="Evaluate", dynamic_ncols=True)):
            # Shrink labels
            labels = labels[:, 4:-4, 4:-4]
            
            # Move data to gpu if model is on gpu
            if args.use_gpu:
                image = image.to(torch.device("cuda"))

            # Forward pass
            logits = model(image)

            # Update confusion matrix
            conf_mat.add_batch(labels, logits.max(1)[1])

            # Convert logits to probabilities
            sm = torch.nn.Softmax(dim=1)
            probs = sm(logits).cpu().numpy()

            labels = labels.cpu().numpy()  # keep true & pred label at same loc.
            predicted_probs += list(probs)
            y_true += list(labels)

    predicted_probs = np.asarray(predicted_probs)

    # Convert predicted probabilities into one-hot labels
    y_predicted = np.argmax(predicted_probs, axis=1).flatten()
    y_true = np.asarray(y_true).flatten()

    # Evaluation with metrics
    f1 = f1_score_(y_predicted, y_true)
    f2 = f2_score_(y_predicted, y_true)
    prec = prec_score_(y_predicted, y_true)
    rec = recal_score_(y_predicted, y_true)
    hm_loss = hamming_loss_(y_predicted, y_true)
    report = classification_report_(y_predicted, y_true)
    aa = conf_mat.get_aa()

    info = {"Weighted Precision": prec,
            "Weighted Recall": rec,
            "Weighted F1": f1,
            "Weighted F2": f2,
            "Hamming Loss": hm_loss,
            "Cls Report": report,
            "Confusion matrix": conf_mat.norm_on_lines(),
            "Average Accuracy": aa}

    print("Save out metrics")
    pkl.dump(info,open(os.path.join(
                 args.out_dir,"{}_evaluation.pkl"
                     .format(train_args.exp_name)), "wb"))


if __name__ == "__main__":
    main()
