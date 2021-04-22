import os
import torch
import random
import numpy as np
import pandas as pd
import rasterio
from augmentation import *
from torch.utils.data import Dataset
from rasterio.plot import reshape_as_image


# Land cover types
lc_types = {1: 'cropland',
            2: 'forest',
            3: 'grassland',
            4: 'shrubland',
            5: 'water',
            6: 'urban',
            7: 'bareland'}


def min_max_norm(img, nodata):
    """Data normalization with min/max method
    Params:
        img (numpy.ndarray): The targeted image for normalization
        nodata (float): the value of background of image
    Returns:
        numpy.ndarray
    """

    img_tmp = np.where(img == nodata, np.nan, img)
    img_max = np.nanmax(img_tmp)
    img_min = np.nanmin(img_tmp)
    img_norm = (img - img_min)/(img_max - img_min)

    return img_norm


def load_sat(path, norm=False):
    """Util function to load satellite image
    Args:
        path (str): the satellite image path
        norm (bool): the flag to do normalization or not.
    Returns:
        numpy.ndarray: the array of image values
    """
    # path = 'results/north/dl_train/1239-996_64_img.tif'
    with rasterio.open(path) as src:
        # reshape the image for augmentation
        sat = reshape_as_image(src.read())
        if norm:
            nodata = src.nodata
            sat = min_max_norm(sat, nodata=nodata)
    return sat


def load_label(path):
    """Util function to load label
    Args:
        path (str): the satellite image path
    Returns:
        numpy.ndarray: the array of image values
    """
    # path = 'results/north/dl_train/1239-996_64_label.tif'
    with rasterio.open(path) as src:
        label = src.read(1)
    return label


def load_tile(tile_info, unlabeled=False, norm=False):
    """Util function for reading data from single sample
    Args:
        tile_info (str): the tile index
        unlabeled (bool): the flag of this tile has label or not
        norm (bool): the flag to do normalization or not.
    Returns:
        list: the list of tile (satellite image, label and tile index)
    """
    # Load satellite image
    img = load_sat(tile_info["img"], norm)

    # load label
    if unlabeled:
        return img
    else:
        label = load_label(tile_info["label"])
        return img, label


class NFSEN1LC(Dataset):
    """PyTorch dataset class for NFSEN1LC
    Which is short for NICFI + SENTINEL-1 land cover classification
    """

    def __init__(self, data_dir,
                 usage='train',
                 lowest_score=9,
                 noise_ratio=0.2,
                 rg_rotate=(-90, 90),
                 norm=False,
                 transform=None):
        """Initialize the dataset
        Args:
            data_dir (str): the directory of all data
            usage (str): Usage of the dataset : "train", "validate" or "predict"
            lowest_score (int): the lowest value of label score, [8, 9, 10]
            noise_ratio (float): the ratio of noise in training
            rg_rotate (tuple or None): Range of degrees for rotation, e.g. (-90, 90)
            norm (bool): the flag to do normalization or not.
            transform (list or None): Data augmentation methods:
                one or multiple elements from ['vflip','hflip','dflip', 'rotate','resize']
                which represents:
                1) 'vflip', vertical flip
                2) 'hflip', horizontal flip
                3) 'dflip', diagonal flip
                4) 'rotate', rotation
                5) 'resize', rescale image fitted into the specified data size.
        """

        # Initialize
        super(NFSEN1LC, self).__init__()
        self.data_dir = data_dir
        self.usage = usage
        self.lowest_score = lowest_score
        self.noise_ratio = noise_ratio
        self.rg_rotate = rg_rotate
        self.norm = norm
        self.transform = transform
        self.n_classes = len(lc_types)
        self.lc_types = lc_types

        # Check inputs
        assert usage in ['train', 'validate', 'predict']
        assert lowest_score in [8, 9, 10]
        assert transform <= ['vflip', 'hflip', 'dflip', 'rotate', 'resize']
        assert os.path.exists(data_dir)

        # Read catalog
        if self.usage == 'train':
            catalog_nm = 'dl_catalog_train.csv'
        elif self.usage == 'validate':
            catalog_nm = 'dl_catalog_train.csv'
        elif self.usage == 'predict':
            catalog_nm = 'dl_catalog_predict.csv'
        catalog = pd.read_csv(os.path.join(self.data_dir, catalog_nm))
        self.catalog = catalog.loc[catalog['score'] >= self.lowest_score]

        # Shrink the catalog based on noise ratio

        # Set noisy_or_not
        self.noisy_or_not = []

    def __getitem__(self, index):
        """Support dataset indexing and apply transformation
        Args:
            index -- Index of each small chips in the dataset
        Returns:
            tuple
        """
        tile_info = self.catalog.iloc[index]
        if self.usage == 'train':
            img, label = load_tile(tile_info, norm=self.norm)
            if self.transform:
                # 0.5 possibility to flip
                trans_flip_ls = [m for m in self.transform if m not in ['rotate', 'resize']]
                if random.randint(0, 1) and len(trans_flip_ls) > 1:
                    trans_flip = random.sample(trans_flip_ls, 1)
                    img, label = flip(img, label, trans_flip[0])

                # 0.5 possibility to resize
                if random.randint(0, 1) and 'resize' in self.transform:
                    img, label = re_scale(img, label.astype(np.uint8),
                                          rand_resize_crop=True, diff=True, cen_locate=False)

                # 0.5 possibility to rotate
                if random.randint(0, 1) and 'rotate' in self.transform:
                    img, label = center_rotate(img, label, self.rg_rotate)

            # numpy to torch
            label = torch.from_numpy(label).long()
            img = torch.from_numpy(img.transpose((2, 0, 1))).float()

            return img, label

        elif self.usage == 'validate':
            img, label = load_tile(tile_info, norm=self.norm)

            # numpy to torch
            label = torch.from_numpy(label).long()
            img = torch.from_numpy(img.transpose((2, 0, 1))).float()

            return img, label
        else:
            img = load_tile(tile_info, unlabeled=True, norm=self.norm)
            img = torch.from_numpy(img.transpose((2, 0, 1))).float()

            return img

    def __len__(self):
        """Get number of samples in the dataset
        Returns:
            int
        """
        return self.catalog.shape[0]
