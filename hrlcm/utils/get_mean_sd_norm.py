"""
This is a script to get means and standard deviation of images
for data normalization.
Author: Lei Song
Maintainer: Lei Song (lsong@clarku.edu)
"""
import sys
sys.path.append('/home/lsong36/hrlcm/hrlcm')
from tqdm.auto import tqdm
import pickle as pkl
from augmentation import *
from dataset import *
from torch.utils.data import DataLoader
import argparse


# Define a dummy args for testing
class args_dummy:
    def __init__(self, catalog):
        self.data_dir = '/scratch/lsong36/tanzania/training'
        self.catalog = catalog


class full_tile(Dataset):
    def __init__(self,
                 catalog,
                 data_dir,
                 img_transform=None):
        self.catalog = catalog
        self.data_dir = data_dir
        self.img_transform = img_transform

    def __len__(self):
        return len(self.catalog)

    def __getitem__(self, idx):
        # import
        path = self.catalog.iloc[idx]['img']
        img = load_sat(path, range(1, 29))

        # augmentations
        if self.img_transform is not None:
            img = self.img_transform(img)

        return img

def main():
    # Get in-line arguments
    # define and parse arguments
    parser = argparse.ArgumentParser()

    # config
    parser.add_argument('--year', type=str, default="2018",
                        help='The year to process.')
    year = parser.parse_args().year
                             
    # Initialize dummy args
    args = args_dummy('dl_catalog_predict_normcal_{}.csv'.format(year))
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
                          data_dir=args.data_dir,
                          img_transform=transform)
    
    # Fast way with enough RAM
    # loader = DataLoader(train_set, batch_size=len(train_set))
    # data = next(iter(loader))
    # data[0].mean(), data[0].std()
    
    # Hard way without enough RAM
    loader = DataLoader(train_set, batch_size=1,
                        shuffle=False, num_workers=32)
    
    # Initialize
    mean = torch.zeros(28)
    std = torch.zeros(28)
    num_pixel = 4096 * 4096
    num_img = len(train_set)
    
    # Mean
    for data in tqdm(loader):
        mean += data.squeeze(0).nansum((1, 2)) / num_pixel
    mean /= num_img
    print(mean)
    pkl.dump(mean.detach().cpu().tolist(),
             open(os.path.join(args.stats_dir, "means_{}.pkl".format(year)), "wb"))
    
    # SD
    mean = mean.unsqueeze(1).unsqueeze(2)
    for data in tqdm(loader):
        std += ((data.squeeze(0) - mean) ** 2).nansum((1, 2)) / num_pixel
    std /= num_img
    std = std.sqrt()
    print(std)
    pkl.dump(std.detach().cpu().tolist(),
             open(os.path.join(args.stats_dir, "stds_{}.pkl".format(year)), "wb"))

if __name__ == "__main__":
    main()
