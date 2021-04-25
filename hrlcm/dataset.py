"""
This is a script to read data for deep learning classification.
Author: Lei Song
Maintainer: Lei Song (lsong@clarku.edu)
"""
import os
from math import floor
import pandas as pd
import rasterio
from torch.utils.data import Dataset
from rasterio.plot import reshape_as_image


def load_sat(path):
    """Util function to load satellite image
    Args:
        path (str): the satellite image path
    Returns:
        numpy.ndarray: the array of image values
    """
    # path = 'results/north/dl_train/1239-996_64_img.tif'
    with rasterio.open(path) as src:
        # reshape the image for augmentation
        sat = reshape_as_image(src.read())
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


def load_tile(tile_info, unlabeled=False):
    """Util function for reading data from single sample
    Args:
        tile_info (str): the tile index
        unlabeled (bool): the flag of this tile has label or not
    Returns:
        list: the list of tile (satellite image, label and tile index)
    """
    # Load satellite image
    img = load_sat(tile_info["img"])

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
                 sync_transform=None,
                 img_transform=None,
                 label_transform=None):
        """Initialize the dataset
        Args:
            data_dir (str): the directory of all data
            usage (str): Usage of the dataset : "train", "validate" or "predict"
            lowest_score (int): the lowest value of label score, [8, 9, 10], just for train.
            noise_ratio (float): the ratio of noise in training, just for train.
            rg_rotate (tuple or None): Range of degrees for rotation, e.g. (-90, 90)
            sync_transform (transform or None): Synthesize Data augmentation methods
            img_transform (transform or None): Image only augmentation methods
            label_transform (transform or None): Label only augmentation methods
        """

        # Initialize
        super(NFSEN1LC, self).__init__()
        self.data_dir = data_dir
        self.usage = usage
        self.rg_rotate = rg_rotate
        self.sync_transform = sync_transform
        self.img_transform = img_transform
        self.label_transform = label_transform
        # Land cover types
        self.lc_types = {1: 'cropland',
                         2: 'forest',
                         3: 'grassland',
                         4: 'shrubland',
                         5: 'water',
                         6: 'urban',
                         7: 'bareland'}
        self.n_classes = len(self.lc_types)
        self.n_channels = 14  # hardcoded

        # Check inputs
        assert usage in ['train', 'validate', 'predict']
        assert lowest_score in [8, 9, 10]
        assert os.path.exists(data_dir)

        # Read catalog
        if self.usage == 'train':
            catalog_nm = 'dl_catalog_train.csv'
        elif self.usage == 'validate':
            catalog_nm = 'dl_catalog_valid.csv'
        elif self.usage == 'predict':
            catalog_nm = 'dl_catalog_predict.csv'
        catalog = pd.read_csv(os.path.join(self.data_dir, catalog_nm))

        # Shrink the catalog based on noise ratio
        if self.usage == 'train':
            # Initialize values
            self.lowest_score = lowest_score
            self.noise_ratio = noise_ratio

            # Subset catalog based on score
            catalog = catalog.loc[catalog['score'] >= self.lowest_score]

            if self.lowest_score < 10:
                # Subset catalog based on noise_ratio
                catalog_perfect = catalog.loc[catalog['score'] == 10]
                catalog_rest = catalog.loc[catalog['score'] < 10]
                num_perfect = len(catalog_perfect.index)
                num_noisy = floor(num_perfect * self.noise_ratio / (1 - self.noise_ratio))
                catalog_rest = catalog_rest.sample(n=num_noisy)
                catalog_perfect = catalog_perfect.append(catalog_rest)
                self.catalog = catalog_perfect
            else:
                self.catalog = catalog

            # Set noisy_or_not
            self.noisy_or_not = self.catalog['score'] != 10
        else:
            self.catalog = catalog

    def __getitem__(self, index):
        """Support dataset indexing and apply transformation
        Args:
            index -- Index of each small chips in the dataset
        Returns:
            tuple
        """
        tile_info = self.catalog.iloc[index]
        if self.usage in ['train', 'validate']:
            img, label = load_tile(tile_info)

            # Transform
            if self.sync_transform is not None:
                img, label = self.sync_transform(img, label)
            if self.img_transform is not None:
                img = self.img_transform(img)
            if self.label_transform is not None:
                label = self.label_transform(label)

            return img, label
        else:
            img = load_tile(tile_info, unlabeled=True)
            if self.img_transform is not None:
                img = self.img_transform(img)

            return img, tile_info['tile_id']

    def __len__(self):
        """Get number of samples in the dataset
        Returns:
            int
        """
        return len(self.catalog.index)
