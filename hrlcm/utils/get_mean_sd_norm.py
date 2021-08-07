"""
This is a script to get means and standard deviation of images
for data normalization.
Author: Lei Song
Maintainer: Lei Song (lsong@clarku.edu)
"""

from tqdm.auto import tqdm
import pickle as pkl
from ..augmentation import *
from ..dataset import *
from torch.utils.data import DataLoader


# Define a dummy args for testing
class args_dummy:
    def __init__(self):
        self.data_dir = 'results/north'
        self.catalog = 'dl_catalog_predict.csv'


class full_tile(Dataset):
    def __init__(self,
                 catalog,
                 img_transform=None):
        self.catalog = catalog
        self.img_transform = img_transform

    def __len__(self):
        return len(self.catalog)

    def __getitem__(self, idx):
        # import
        path = self.catalog.iloc[idx]['img']
        img = load_sat(path)

        # augmentations
        if self.img_transform is not None:
            img = self.img_transform(img)

        return img


# Initialize dummy args
args = args_dummy()
args.stats_dir = os.path.join(args.data_dir, 'norm_stats')
if not os.path.isdir(args.stats_dir):
    os.makedirs(args.stats_dir)

full_catalog = pd.read_csv(os.path.join(args.data_dir, args.catalog))

# Calculate mean and sd
# Load dataset
transform = ComposeImg([
    ImgToTensor()
])
train_set = full_tile(catalog=full_catalog,
                      img_transform=transform)

# Fast way with enough RAM
# loader = DataLoader(train_set, batch_size=len(train_set))
# data = next(iter(loader))
# data[0].mean(), data[0].std()

# Hard way without enough RAM
loader = DataLoader(train_set, batch_size=1,
                    shuffle=False, num_workers=0)

# Initialize
mean = torch.zeros(14)
std = torch.zeros(14)
num_pixel = 4096 * 4096
num_img = len(train_set)

# Mean
for data in tqdm(loader):
    mean += data.squeeze(0).sum((1, 2)) / num_pixel
mean /= num_img
print(mean)
pkl.dump(mean.detach().cpu().tolist(),
         open(os.path.join(args.stats_dir, "means.pkl"), "wb"))

# SD
mean = mean.unsqueeze(1).unsqueeze(2)
for data in tqdm(loader):
    std += ((data.squeeze(0) - mean) ** 2).sum((1, 2)) / num_pixel
std /= num_img
std = std.sqrt()
print(std)
pkl.dump(std.detach().cpu().tolist(),
         open(os.path.join(args.stats_dir, "stds.pkl"), "wb"))
