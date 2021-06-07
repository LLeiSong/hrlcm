"""
This is a script to read data for deep learning classification.
Author: Lei Song
Maintainer: Lei Song (lsong@clarku.edu)
"""

import os
import sys
import itertools
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


def load_tile(tile_info, unlabeled=False, offset=1):
    """Util function for reading data from single sample
    Args:
        tile_info (pandas.DataFrame): the tile info
        unlabeled (bool): the flag of this tile has label or not
        offset (int): the offset to load label in order to start with 0.
    Returns:
        list or numpy.ndarray: the list of tile (satellite image, label and tile index)
    """
    # Load satellite image
    img = load_sat(tile_info["img"])

    # load label
    if unlabeled:
        return img
    else:
        label = load_label(tile_info["label"])
        if offset > 0:
            label = label - offset
        return img, label


def get_meta(fname):
    """
    Get metadata of a prediction tile

    Params:
        fname (str):  File name of a image chip
    Returns:
        dictionary
    """

    with rasterio.open(fname, "r") as src:
        meta = src.meta
    meta.update({
        'count': 1,
        'nodata': -128,
        'dtype': 'int8'
    })

    return meta


def get_chips(img, dsize):
    """
    Generate small chips from input images and the corresponding index of each chip
    The index marks the location of corresponding upper-left pixel of a chip.
    Params:
        img (numpy.ndarray): Image or image stack in format of (H,W,C) to be crop
        dsize (int): Cropped chip size
    Returns:
        list of cropped chips and corresponding coordinates
    """

    h, w, _ = img.shape
    x_ls = range(0, h, dsize)
    y_ls = range(0, w, dsize)

    index = list(itertools.product(x_ls, y_ls))

    img_ls = []
    for i in range(len(index)):
        x, y = index[i]
        img_ls.append(img[x:x + dsize, y:y + dsize, :])

    return img_ls, index


class NFSEN1LC(Dataset):
    """PyTorch dataset class for NFSEN1LC
    Which is short for NICFI + SENTINEL-1 land cover classification
    """

    def __init__(self, data_dir,
                 usage='train',
                 highest_score=10,
                 lowest_score=9,
                 noise_ratio=0.3,
                 random_state=1,
                 label_offset=1,
                 sync_transform=None,
                 img_transform=None,
                 label_transform=None,
                 tile_id='1207-996_12'):
        """Initialize the dataset
        Args:
            data_dir (str): the directory of all data
            usage (str): Usage of the dataset : "train", "validate" or "predict"
            lowest_score (int): the lowest value of label score, [8, 9, 10], just for train.
            noise_ratio (float or None): the ratio of noise in training, just for train.
            random_state (int): the random state for pandas sampling.
            label_offset (int): the offset of label to minus in order to fit into DL model.
            sync_transform (transform or None): Synthesize Data augmentation methods
            img_transform (transform or None): Image only augmentation methods
            label_transform (transform or None): Label only augmentation methods
            tile_id (str): the tile index for prediction image. Only for predict dataset.
        """

        # Initialize
        super(NFSEN1LC, self).__init__()
        self.data_dir = data_dir
        self.usage = usage
        self.random_state = random_state
        self.label_offset = label_offset
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
        assert highest_score in [8, 9, 10]
        assert os.path.exists(data_dir)

        # Read catalog
        if self.usage == 'train':
            catalog_nm = 'dl_catalog_train.csv'
        elif self.usage == 'validate':
            catalog_nm = 'dl_catalog_valid.csv'
        elif self.usage == 'predict':
            catalog_nm = 'dl_catalog_predict.csv'
        else:
            sys.exit('Not valid usage setting.')
        catalog_full = pd.read_csv(os.path.join(self.data_dir, catalog_nm))

        # Shrink the catalog based on noise ratio
        if self.usage == 'train':
            # Initialize values
            self.highest_score = highest_score
            self.lowest_score = lowest_score
            self.noise_ratio = noise_ratio

            # Subset catalog based on score
            catalog = catalog_full.loc[(self.lowest_score <= catalog_full['score']) &
                                       (catalog_full['score'] <= self.highest_score)]

            if self.lowest_score < 10 & self.highest_score == 10:
                # Subset catalog based on noise_ratio
                catalog_perfect = catalog.loc[catalog['score'] == 10]
                catalog_rest = catalog.loc[catalog['score'] < 10]
                if self.noise_ratio is not None:
                    num_perfect = len(catalog_perfect.index)
                    num_noisy = floor(num_perfect * self.noise_ratio / (1 - self.noise_ratio))
                    catalog_rest = catalog_rest.sample(n=num_noisy, random_state=self.random_state)
                else:
                    self.noise_ratio = round(len(catalog_rest.index) / len(catalog.index), 1)
                catalog_perfect = catalog_perfect.append(catalog_rest)
                self.catalog = catalog_perfect
            else:
                self.catalog = catalog

            # Set noisy_or_not
            self.noisy_or_not = self.catalog['score'] != 10
            self.noisy_or_not = self.noisy_or_not.to_numpy()
        elif self.usage == 'validate':
            self.catalog = catalog_full
        # Prediction
        else:
            self.chip_size = 512  # image size of train
            self.tile_id = tile_id
            catalog = catalog_full.loc[catalog_full['tile_id'] == self.tile_id]
            if len(catalog.index) == 0:
                sys.exit('No such {} to prediction.'.format(self.tile_id))
            elif len(catalog.index) != 1:
                sys.exit('Duplicate catalog for tile {}.'.format(self.tile_id))
            else:
                catalog['img'][0] = os.path.join(self.data_dir, catalog['img'][0])
                self.catalog = catalog.iloc[0]
                img = load_tile(self.catalog, unlabeled=True)
                self.meta = get_meta(self.catalog['img'])
                self.img_ls, self.index_ls = get_chips(img, self.chip_size)

    def __getitem__(self, index):
        """Support dataset indexing and apply transformation
        Args:
            index -- Index of each small chips in the dataset
        Returns:
            tuple
        """
        if self.usage in ['train', 'validate']:
            tile_info = self.catalog.iloc[index]
            tile_info = tile_info.replace(tile_info['img'],
                                          os.path.join(self.data_dir, tile_info["img"]))
            tile_info = tile_info.replace(tile_info['label'],
                                          os.path.join(self.data_dir, tile_info["label"]))
            img, label = load_tile(tile_info, offset=self.label_offset)

            # Transform
            if self.sync_transform is not None:
                img, label = self.sync_transform(img, label)
            if self.img_transform is not None:
                img = self.img_transform(img)
            if self.label_transform is not None:
                label = self.label_transform(label)

            if self.usage == 'train':
                return img, label, index
            else:
                return img, label
        else:
            img = self.img_ls[index]
            if self.img_transform is not None:
                img = self.img_transform(img)
            index = self.index_ls[index]

            return img, index

    def __len__(self):
        """Get number of samples in the dataset
        Returns:
            int
        """
        if self.usage == 'predict':
            return len(self.index_ls)
        elif self.usage in ['train', 'validate']:
            return len(self.catalog.index)
        else:
            return 0
