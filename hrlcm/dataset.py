"""
This is a script to read data for deep learning classification.
Author: Lei Song
Maintainer: Lei Song (lsong@clarku.edu)
"""

import math
import os
import sys
import itertools
import numpy as np
import pandas as pd
import rasterio
from torch.utils.data import Dataset
from rasterio.plot import reshape_as_image


def logistic_scale(x, L=1, k=1.5, x0=2.5):
    """Util function to scale data using logistic
    Args:
        x (numpy.ndarray): the array to scale
        L (float): the curve's maximum value
        k (float): logistic growth rate
        x0 (float): the x value of the sigmoid midpoint
    Returns:
        numpy.ndarray: the array of image values
    """
    return L / (1 + math.e ** (-k * (x - x0)))


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


def load_sat_buf(path, paths_relate, buffer):
    data_dir = os.path.dirname(os.path.dirname(path))
    paths_split = paths_relate.split(',')
    with rasterio.open(path, "r") as src:
        meta = src.meta
    dst_ids = [[0, buffer, 0, buffer],
               [0, buffer, buffer, buffer + meta['width']],
               [0, buffer, buffer + meta['width'], meta['width'] + 2 * buffer],
               [buffer, buffer + meta['height'], 0, buffer],
               [buffer, buffer + meta['height'], buffer, buffer + meta['width']],
               [buffer, buffer + meta['height'], buffer + meta['width'], meta['width'] + 2 * buffer],
               [buffer + meta['width'], meta['width'] + 2 * buffer, 0, buffer],
               [buffer + meta['width'], meta['width'] + 2 * buffer, buffer, buffer + meta['width']],
               [buffer + meta['width'], meta['width'] + 2 * buffer,
                buffer + meta['width'], meta['width'] + 2 * buffer]]
    src_ids = [[meta['height'] - buffer, meta['height'], meta['width'] - buffer, meta['width']],
               [meta['height'] - buffer, meta['height'], 0, meta['width']],
               [meta['height'] - buffer, meta['height'], 0, buffer],
               [0, meta['height'], meta['width'] - buffer, meta['width']],
               [0, meta['height'], 0, meta['width']],
               [0, meta['height'], 0, buffer],
               [0, buffer, meta['width'] - buffer, meta['width']],
               [0, buffer, 0, meta['width']],
               [0, buffer, 0, buffer]]

    # Read central tile and expand it
    with rasterio.open(path) as src:
        # reshape the image for augmentation
        img_full = np.pad(reshape_as_image(src.read()),
                          ((buffer, buffer), (buffer, buffer), (0, 0)), mode='edge')

    for i in [1, 3, 5, 7, 0, 2, 6, 8]:  # skip central tile
        if paths_split[i] != 'None':
            with rasterio.open(os.path.join(data_dir, paths_split[i]), "r") as src:
                sat = reshape_as_image(src.read())
            img_full[dst_ids[i][0]:dst_ids[i][1], dst_ids[i][2]:dst_ids[i][3], :] = \
                sat[src_ids[i][0]:src_ids[i][1], src_ids[i][2]:src_ids[i][3], :]

    return img_full


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


def get_chips(img, dsize, buffer=0):
    """
    Generate small chips from input images and the corresponding index of each chip
    The index marks the location of corresponding upper-left pixel of a chip.
    Params:
        img (numpy.ndarray): Image or image stack in format of (H,W,C) to be crop
        dsize (int): Cropped chip size
        buffer (int): Number of overlapping pixels when extracting images chips
    Returns:
        list of cropped chips and corresponding coordinates
    """

    h, w, _ = img.shape
    x_ls = range(0, h - 2 * buffer, dsize - 2 * buffer)
    y_ls = range(0, w - 2 * buffer, dsize - 2 * buffer)

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
                 score_k=1.5,
                 hardiness_max=1.2,
                 random_state=1,
                 label_offset=1,
                 chip_buffer=64,
                 sync_transform=None,
                 img_transform=None,
                 label_transform=None,
                 tile_id='1207-996'):
        """Initialize the dataset
        Args:
            data_dir (str): the directory of all data
            usage (str): Usage of the dataset : "train", "validate" or "predict"
            score_k (float): the k param of logistic scale to calculate weight of score.
            hardiness_max (float): the rescaled max of hardiness as weight for loss calculation.
                Must be larger than 1 to be effective, otherwise no weight affect.
                E.g.hardiness = 3,
                then weight = ((hardiness_max - 1) * (hardiness - 1) / (max(hardiness) - 1) + 1)
                = 4 * 0.2 * ((2 - 1) * (4 - 1) / (5 - 1) + 1) = 1.4
            random_state (int): the random state for pandas sampling.
            label_offset (int): the offset of label to minus in order to fit into DL model.
            chip_buffer (int): buffer value to read images.
            sync_transform (transform or None): Synthesize Data augmentation methods
            img_transform (transform or None): Image only augmentation methods
            label_transform (transform or None): Label only augmentation methods
            tile_id (str): the tile index for prediction image. Only for predict dataset.
        """

        # Initialize
        super(NFSEN1LC, self).__init__()
        self.data_dir = data_dir
        self.usage = usage
        self.score_k = score_k
        self.hardiness_max = hardiness_max
        self.random_state = random_state
        self.label_offset = label_offset
        self.chip_buffer = chip_buffer
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
            self.catalog = catalog_full

            # Set score, hardiness and loss weights
            self.score = self.catalog['score'].to_numpy()
            self.score_weight = logistic_scale(self.score, k=self.score_k)
            self.hardiness = self.catalog['hardiness'].to_numpy()
            self.hardiness_weight = (self.hardiness_max - 1) * (self.hardiness - 1) / (max(self.hardiness) - 1) + 1
            self.weight = self.score_weight * self.hardiness_weight
        elif self.usage == 'validate':
            self.catalog = catalog_full
        # Prediction
        else:
            self.chip_size = 512 + self.chip_buffer * 2  # image size of train
            self.tile_id = tile_id
            catalog = catalog_full.loc[catalog_full['tile_id'] == self.tile_id]
            if len(catalog.index) == 0:
                sys.exit('No such {} to prediction.'.format(self.tile_id))
            elif len(catalog.index) != 1:
                sys.exit('Duplicate catalog for tile {}.'.format(self.tile_id))
            else:
                catalog['img'][catalog.index[0]] = os.path.join(self.data_dir, catalog['img'][catalog.index[0]])
                self.catalog = catalog.iloc[0]
                img = load_sat_buf(self.catalog['img'], self.catalog['tiles_relate'], self.chip_buffer)
                self.meta = get_meta(self.catalog['img'])
                self.img_ls, self.index_ls = get_chips(img, self.chip_size, self.chip_buffer)

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
            ind = self.index_ls[index]

            return img, ind

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
