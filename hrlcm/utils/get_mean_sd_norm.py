"""
This is a script to get means and standard deviation of images
for data normalization.
Author: Lei Song
Maintainer: Lei Song (lsong@clarku.edu)
"""

import pickle as pkl
from ..augmentation import *
from ..dataset import *
from torch.utils.data import DataLoader


# Define a dummy args for testing
class args_dummy:
    def __init__(self):
        self.exp_name = 'unet_norm_calc'
        self.data_dir = 'results/north'
        self.out_dir = 'results/dl'
        self.noise_ratio = None
        self.trans_prob = 0.5
        self.label_offset = 1
        self.rg_rotate = '-90, 90'
        self.model = 'unet'
        self.train_mode = 'single'
        self.out_stride = 8
        self.lr = 0.001
        self.decay = 1e-5
        self.save_freq = 20
        self.log_feq = 20
        self.train_batch_size = 16
        self.val_batch_size = 16
        self.num_workers = 16
        self.epochs = 100
        self.optimizer_name = 'Adam'
        self.resume = None
        self.checkpoint_dir = os.path.join(self.out_dir, self.exp_name, 'checkpoints')
        self.logs_dir = os.path.join(self.out_dir, self.exp_name, 'logs')


# Initialize dummy args
args = args_dummy()
args.stats_dir = os.path.join(args.data_dir, 'norm_stats')
if not os.path.isdir(args.stats_dir):
    os.makedirs(args.stats_dir)

# Calculate mean and sd
# Load dataset
sync_transform = Compose([
    SyncToTensor()
])
train_set = NFSEN1LC(data_dir=args.data_dir,
                     usage='train',
                     label_offset=args.label_offset,
                     sync_transform=sync_transform,
                     img_transform=None,
                     label_transform=None)

valid_set = NFSEN1LC(data_dir=args.data_dir,
                     usage='validate',
                     label_offset=args.label_offset,
                     sync_transform=sync_transform,
                     img_transform=None,
                     label_transform=None)

# Fast way with enough RAM
# loader = DataLoader(train_set, batch_size=len(train_set))
# data = next(iter(loader))
# data[0].mean(), data[0].std()

# Hard way without enough RAM
loader = DataLoader(train_set, batch_size=1,
                    shuffle=False, num_workers=0)
loader_val = DataLoader(valid_set, batch_size=1,
                        shuffle=False, num_workers=0)

# Initialize
mean = torch.zeros(14)
std = torch.zeros(14)
num_pixel = 512 * 512
num_img = len(train_set) + len(valid_set)

# Mean
for data, _, _ in loader:
    mean += data.squeeze(0).sum((1, 2)) / num_pixel
for data, _ in loader_val:
    mean += data.squeeze(0).sum((1, 2)) / num_pixel
mean /= num_img
print(mean)
pkl.dump(mean.detach().cpu().tolist(),
         open(os.path.join(args.stats_dir, "means.pkl"), "wb"))

# SD
mean = mean.unsqueeze(1).unsqueeze(2)
for data, _, _ in loader:
    std += ((data.squeeze(0) - mean) ** 2).sum((1, 2)) / num_pixel
for data, _ in loader_val:
    std += ((data.squeeze(0) - mean) ** 2).sum((1, 2)) / num_pixel
std /= num_img
std = std.sqrt()
print(std)
pkl.dump(std.detach().cpu().tolist(),
         open(os.path.join(args.stats_dir, "stds.pkl"), "wb"))
