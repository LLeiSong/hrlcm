"""
This is a script to do prediction using final model.
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

    # Create output dir
    os.makedirs(args.out_dir, exist_ok=True)

    # Load dataset
    # synchronize transform for validate dataset
    pred_transform = Compose([
        SyncToTensor()
    ])

    # Get validate dataset
    predict_dataset = NFSEN1LC(data_dir=args.data_dir,
                               usage='predict',
                               sync_transform=pred_transform,
                               img_transform=None,
                               label_transform=None)
    # Put into DataLoader
    predict_loader = DataLoader(dataset=predict_dataset,
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
    n = 0
    for image, tile_id in tqdm(predict_loader, desc="[Pred]"):
        # Move data to gpu if model is on gpu
        if args.use_gpu:
            image = image.cuda()

        # forward pass
        with torch.no_grad():
            prediction = model(image)

        # convert to 256x256 numpy arrays
        prediction = prediction.cpu().numpy()
        prediction = np.argmax(prediction, axis=1)

        # UNDER CONSTRUCTION

