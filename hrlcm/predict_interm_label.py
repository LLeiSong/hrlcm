"""
This is a script to do prediction using final model.
Author: Lei Song
Maintainer: Lei Song (lsong@ucsb.edu)
"""

import argparse
from augmentation import *
from dataset import *
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pickle as pkl
from models.unet import UNet


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
    parser.add_argument('--year', type=str,
                        default='2018', help='The year for the prediction.')
    parser.add_argument('--fname_predict', type=str,
                        default='dl_catalog_predict.csv',
                        help='the csv file name to do prediction.')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='path to dataset')
    parser.add_argument('--out_dir', type=str, default="prediction",
                        help='path to output dir (default: ./prediction)')
    parser.add_argument('--stats_dir', type=str, default="norm_stats",
                        help='path of normalization params (default: ./norm_stats)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of worker(s) to load dataset (default: 0)')
    parser.add_argument('--label_offset', type=int, default=1,
                        help='offset value to minus from label in order to start from 0 (default: 1)')
    parser.add_argument('--img_bands', type=str, choices=['all', 'nicfi'],
                        default='all',
                        help='bands of satellite images to use. \
                        all means all bands, including RGB, NIR of NICFI tiles, intercept,  \
                        cos(2t) of VV and VH. (default: all)')

    # Hyper-parameters of GPU
    parser.add_argument('--batch_size', type=int, default=64,
                        help='mini-batch size (default: 64)')
    parser.add_argument('--gpu_devices', type=str, default=None,
                        help='the gpu devices to use (default: None) (format: 1, 2)')
    
    # Output format
    parser.add_argument('--label_format', type=str, choices=['full', 'intersect'],
                        default='intersect',
                        help='the format of the output labels. (default: intersect)')

    args = parser.parse_args()
    assert args.img_bands in ['all', 'nicfi']

    print("=" * 20, "PREDICTION CONFIG", "=" * 20)
    for arg in vars(args):
        print('{0:20}  {1}'.format(arg, getattr(args, arg)))
    print()

    # Load config of training
    train_args = pkl.load(open(args.args_path, "rb"))
    print("=" * 20, "TRAIN CONFIG", "=" * 20)
    for arg in vars(train_args):
        print('{0:20}  {1}'.format(arg, getattr(train_args, arg)))
    print()

    # Create output dir
    os.makedirs(args.out_dir, exist_ok=True)

    # Set flags for GPU processing if available
    args.use_gpu = torch.cuda.is_available()

    # set up network
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

    # Predict
    # Image transform
    bands_all = [1,2,3,4,5,7,9,10,11,12,13,15,17,18,19,20] + list(range(23,27))
    bands_opt = [1,2,3,4,9,10,11,12]
    id_bands = bands_all if args.img_bands == "all" else bands_opt
    
    # Load mean and sd for normalization
    with open(os.path.join(args.stats_dir, "means_2018.pkl"), "rb") as input_file:
        mean = tuple(pkl.load(input_file))
        mean = tuple([mean[i-1] for i in id_bands])

    with open(os.path.join(args.stats_dir, "stds_2018.pkl"), "rb") as input_file:
        std = tuple(pkl.load(input_file))
        std = tuple([std[i-1] for i in id_bands])
    
    # Set transform for prediction
    sync_transform = Compose([
        SyncToTensor()
    ])
    img_transform = ImgNorm(mean, std)
    
    # Get predict dataset
    predict_dataset = NFSEN1LC(data_dir=args.data_dir,
                               bands=id_bands,
                               usage='ipredict',
                               label_offset=args.label_offset,
                               sync_transform=sync_transform,
                               img_transform=img_transform,
                               label_transform=None,
                               predict_catalog=args.fname_predict)
    # Put into DataLoader
    predict_loader = DataLoader(dataset=predict_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                shuffle=False,
                                drop_last=False)
    
    # Start the prediction
    catalog = pd.read_csv(os.path.join(args.data_dir, args.fname_predict))
    for i, (image, target, tile_ids) in enumerate(predict_loader):
        # Move data to gpu if model is on gpu
        if args.use_gpu:
            image = image.cuda()

        # Forward pass
        with torch.no_grad():
            prediction = model(image)
            
        prediction = prediction.cpu().numpy()
        prediction = np.argmax(prediction, axis=1) + args.label_offset
        orig_label = target.cpu().numpy() + args.label_offset
            
        for n in range(prediction.shape[0]):
            output = prediction[n, :, :]
            output = output.astype(np.uint8)
            output = np.pad(output, 4, 'constant', constant_values = 255)
            
            if args.label_format == "intersect":
                label_to_compare = orig_label[n, :, :]
                output[output != label_to_compare] = 255
            
            # Read meta
            tile_index = tile_ids[n]
            img_path = catalog.loc[catalog['tile_id'] == tile_index]['label'].item()
            with rasterio.open(img_path, "r") as src:
                meta = src.meta
                
            # Save out
            name_class = os.path.join(args.out_dir, 'class_{}.tif'.format(tile_index))
            with rasterio.open(name_class, 'w', **meta) as dst:
                dst.write(output, indexes = 1)


if __name__ == "__main__":
    main()
