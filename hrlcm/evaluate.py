"""
This is a script to evaluate the final model against validate dataset.
Author: Lei Song
Maintainer: Lei Song (lsong@clarku.edu)
"""

import argparse
from augmentation import *
from dataset import *
from torch.utils.data import DataLoader
import pickle as pkl
from models.deeplab import DeepLab
from models.unet import UNet
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, \
    fbeta_score, classification_report, hamming_loss, confusion_matrix


def conf_mat_nor(predict_labels, true_labels, n_classes):
    """ return the normalized confusion matrix (respect to y_true)
        input labels are in one-hot encoding, n_class = number of label classes
        This function only applied to single-label
    """

    assert (np.sum(true_labels, axis=1) == 1).all()
    assert (np.sum(predict_labels, axis=1) == 1).all()

    true_idx = np.where(true_labels == 1)[1]
    pred_idx = np.where(predict_labels == 1)[1]

    con = confusion_matrix(true_idx, pred_idx, labels=np.arange(n_classes))
    b = con.sum(axis=1)[:, None]
    con_nor = np.divide(con, b, where=(b != 0))

    return con_nor


def get_AA(predict_labels, true_labels, n_classes):
    """ only applied to single-label
        zero sample classes are not excluded in the calculation
        would be 0 in the calculation
    """
    con_nor = conf_mat_nor(predict_labels, true_labels, n_classes)
    AA = np.diagonal(con_nor).sum() / n_classes

    return AA


class Precision_score(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, predict_labels, true_labels):
        weighted_prec = precision_score(true_labels, predict_labels, average='weighted')

        return weighted_prec


class Recall_score(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, predict_labels, true_labels):
        weighted_rec = recall_score(true_labels, predict_labels, average='weighted')

        return weighted_rec


class F1_score(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, predict_labels, true_labels):
        weighted_f1 = f1_score(true_labels, predict_labels, average="weighted")

        return weighted_f1


class F2_score(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, predict_labels, true_labels):
        weighted_f2 = fbeta_score(true_labels, predict_labels, beta=2, average="weighted")

        return weighted_f2


class Hamming_loss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, predict_labels, true_labels):
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

    # Hyper-parameters of evaluation
    parser.add_argument('--batch_size', type=int, default=16,
                        help='mini-batch size (default: 16)')
    parser.add_argument('--workers', type=int, default=1,
                        help='num_workers for data loading in pytorch  (default: 1)')

    args = parser.parse_args()
    print("=" * 20, "PREDICTION CONFIG", "=" * 20)
    for arg in vars(args):
        print('{0:20}  {1}'.format(arg, getattr(args, arg)))
    print()

    # Load config of training
    train_args = pkl.load(open(args.config_file, "rb"))
    print("=" * 20, "TRAIN CONFIG", "=" * 20)
    for arg in vars(train_args):
        print('{0:20}  {1}'.format(arg, getattr(train_args, arg)))
    print()

    # Cuda
    use_cuda = torch.cuda.is_available()

    # Create output dir
    os.makedirs(args.out_dir, exist_ok=True)

    # Load dataset
    # synchronize transform for validate dataset
    val_transform = Compose([
        SyncToTensor()
    ])

    # Get validate dataset
    validate_dataset = NFSEN1LC(data_dir=args.data_dir,
                                usage='validate',
                                sync_transform=val_transform,
                                img_transform=None,
                                label_transform=None)
    # Put into DataLoader
    validate_loader = DataLoader(dataset=validate_dataset,
                                 batch_size=args.batch_size,
                                 num_workers=args.workers,
                                 shuffle=False,
                                 drop_last=False)

    # set up network
    if train_args.model == "deeplab":
        model = DeepLab(num_classes=train_args.n_classes,
                        backbone='resnet',
                        pretrained_backbone=False,
                        output_stride=train_args.out_stride,
                        sync_bn=False,
                        freeze_bn=False,
                        n_in=train_args.n_inputs)
    else:
        model = UNet(n_classes=train_args.n_classes,
                     n_channels=train_args.n_inputs)
    if args.use_gpu:
        model = model.cuda()

    # Restore network weights
    state = torch.load(args.checkpoint_file)
    step = state["step"]
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    print("loaded checkpoint from step", step)

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

    with torch.no_grad():
        for image, labels in enumerate(tqdm(validate_loader, desc="evaluate")):
            # move data to gpu if model is on gpu
            if use_cuda:
                image = image.to(torch.device("cuda"))

            # forward pass
            logits = model(image)

            # convert logits to probabilities
            sm = torch.nn.Softmax(dim=1)
            probs = sm(logits).cpu().numpy()

            labels = labels.cpu().numpy()  # keep true & pred label at same loc.
            predicted_probs += list(probs)
            y_true += list(labels)

    predicted_probs = np.asarray(predicted_probs)

    # Convert predicted probabilities into one-hot labels
    loc = np.argmax(predicted_probs, axis=-1)
    y_predicted = np.zeros_like(predicted_probs).astype(np.float32)
    for i in range(len(loc)):
        y_predicted[i, loc[i]] = 1

    y_true = np.asarray(y_true)

    # Evaluation with metrics
    f1 = f1_score_(y_predicted, y_true)
    f2 = f2_score_(y_predicted, y_true)
    prec = prec_score_(y_predicted, y_true)
    rec = recal_score_(y_predicted, y_true)
    hm_loss = hamming_loss_(y_predicted, y_true)
    report = classification_report_(y_predicted, y_true)
    conf_mat = conf_mat_nor(y_predicted, y_true, n_classes=train_args.n_classes)
    # zero-sample classes are not excluded
    aa = get_AA(y_predicted, y_true, n_classes=train_args.n_classes)

    info = {"weightedPrec": prec,
            "weightedRec": rec,
            "weightedF1": f1,
            "weightedF2": f2,
            "HammingLoss": hm_loss,
            "clsReport": report,
            "conf_mat": conf_mat,
            "AverageAcc": aa}

    print("Saving metrics...")
    pkl.dump(info, open(os.path.join(args.out_dir, "test_scores.pkl"), "wb"))


if __name__ == "__main__":
    main()
