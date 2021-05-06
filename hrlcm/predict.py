"""
This is a script to do prediction using final model.
Author: Lei Song
Maintainer: Lei Song (lsong@clarku.edu)
"""

import argparse
from augmentation import *
from dataset import *
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import pickle as pkl
from models.deeplab import DeepLab
from models.unet import UNet
from tqdm.auto import tqdm
from sync_batchnorm import convert_model


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
    parser.add_argument('--out_dir', type=str, default="prediction",
                        help='path to output dir (default: ./prediction)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of worker(s) to load dataset (default: 0)')

    # Hyper-parameters of evaluation
    parser.add_argument('--batch_size', type=int, default=16,
                        help='mini-batch size (default: 16)')
    parser.add_argument('--gpu_devices', type=str, default=None,
                        help='the gpu devices to use (default: None) (format: 1, 2)')

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

    # Set flags for GPU processing if available
    args.use_gpu = torch.cuda.is_available()

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
    # Get devices
    if args.gpu_devices:
        args.gpu_devices = [int(each) for each in args.gpu_devices.split(',')]

    if args.use_gpu:
        if args.gpu_devices:
            torch.cuda.set_device(args.gpu_devices[0])
            model = torch.nn.DataParallel(model, device_ids=args.gpu_devices)
            if args.sync_norm:
                model = convert_model(model)
        model = model.cuda()

    # Restore network weights
    state = torch.load(args.checkpoint_file)
    step = state["step"]
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    print("Loaded checkpoint from step", step)

    # Predict
    # synchronize transform for predict dataset
    pred_transform = Compose([
        ImgToTensor()
    ])

    catalog = pd.read_csv(os.path.join(args.data_dir, 'dl_catalog_predict.csv'))
    for tile_id in catalog['tile_id']:
        # Get validate dataset
        predict_dataset = NFSEN1LC(data_dir=args.data_dir,
                                   usage='predict',
                                   sync_transform=None,
                                   img_transform=pred_transform,
                                   label_transform=None,
                                   tile_id=tile_id)
        # Put into DataLoader
        predict_loader = DataLoader(dataset=predict_dataset,
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers,
                                    shuffle=False,
                                    drop_last=False)

        # Set and crate paths
        score_path = os.path.join(args.out_dir, 'score')
        class_path = os.path.join(args.out_dir, 'class')
        if not os.path.isdir(score_path):
            os.mkdir(score_path)
        if not os.path.isdir(class_path):
            os.mkdir(class_path)

        # File names
        name_score = os.path.join(score_path, 'score_{}'.format(tile_id))
        name_class = os.path.join(class_path, 'class_{}.tif'.format(tile_id))

        model.eval()

        # Create dummy tile
        meta = predict_dataset.meta
        n_class = predict_dataset.n_classes
        canvas = np.zeros((1, meta['height'], meta['width']),
                          dtype=meta['dtype'])
        canvas_score_ls = []

        for img, index in tqdm(predict_loader, desc="[Pred]"):
            img = Variable(img, requires_grad=False)

            # GPU setting
            if args.use_gpu:
                img = img.cuda()

            out = F.softmax(model(img), 1)
            batch, n_class, width, height = out.size()
            score_width = width
            score_height = height

            # for each batch
            for i in range(batch):
                index = (index[0][i], index[1][i])
                out_predict = out.max(dim=1)[1][:, :, :].cpu().numpy()[i, :, :]
                out_predict = np.expand_dims(out_predict, axis=0)
                out_predict = out_predict.astype(np.int8)
                canvas[:, index[0]: index[0] + score_width, index[1]: index[1] + score_height] = out_predict

                # scores for each non-background class
                for n in range(n_class - 1):
                    out_score = out[:, n + 1, :, :].data[i][:, :].cpu().numpy() * 100
                    out_score = np.expand_dims(out_score, axis=0).astype(np.int8)
                    try:
                        canvas_score_ls[n][:, index[0]: index[0] + score_width, index[1]: index[1] + score_height] = \
                            out_score
                    except:
                        canvas_score_single = np.zeros((1, meta['height'], meta['width']), dtype=meta['dtype'])
                        canvas_score_single[:, index[0]: index[0] + score_width, index[1]: index[1] + score_height] = \
                            out_score
                        canvas_score_ls.append(canvas_score_single)

        # Save out
        with rasterio.open(name_class, 'w', **meta) as dst:
            dst.write(canvas)

        for n in range(n_class - 1):
            with rasterio.open('{}_class{}.tif'.format(name_score, n + 1), 'w', **meta) as dst:
                dst.write(canvas_score_ls[n])
