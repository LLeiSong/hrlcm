"""
This is a script to read data for deep learning classification.
Author: Lei Song
Maintainer: Lei Song (lsong@ucsb.edu)
"""

import copy
import os
import sys
import itertools
import numpy as np
import pandas as pd
import rasterio
import multiprocessing as mp
from torch.utils.data import Dataset
from rasterio.plot import reshape_as_image


def load_sat(path, bands):
    """Util function to load satellite image
    Args:
        path (str): the satellite image path
        bands (list): the bands to read.
    Returns:
        numpy.ndarray: the array of image values
    """
    # path = 'results/north/dl_train/1239-996_64_img.tif'
    with rasterio.open(path) as src:
        # reshape the image for augmentation
        sat = reshape_as_image(src.read(bands))
    return sat


def load_sat_buf(path, bands, paths_relate, buffer):
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
        img_full = np.pad(reshape_as_image(src.read(bands)),
                          ((buffer, buffer), (buffer, buffer), (0, 0)), mode='edge')

    for i in [1, 3, 5, 7, 0, 2, 6, 8]:  # skip central tile
        if paths_split[i] != 'None':
            with rasterio.open(os.path.join(data_dir, paths_split[i]), "r") as src:
                sat = reshape_as_image(src.read(bands))
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


def load_tile(tile_info, bands, unlabeled=False, offset=1):
    """Util function for reading data from single sample
    Args:
        tile_info (pandas.DataFrame): the tile info
        unlabeled (bool): the flag of this tile has label or not
        bands (list): the bands to read.
        offset (int): the offset to load label in order to start with 0.
    Returns:
        list or numpy.ndarray: the list of tile (satellite image, label and tile index)
    """
    # Load satellite image
    img = load_sat(tile_info["img"], bands)

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
        img_each = copy.deepcopy(img[x:x + dsize, y:y + dsize, :])
        img_ls.append(img_each)

    return img_ls, index


# Use multi-threads
def _read_img_label(index, catalog, bands, label_offset, data_dir):
    tile_info = catalog.iloc[index]
    tile_info = tile_info.replace(tile_info['img'], os.path.join(data_dir, tile_info["img"]))
    tile_info = tile_info.replace(tile_info['label'], os.path.join(data_dir, tile_info["label"]))
    img, label = load_tile(tile_info, bands=bands, offset=label_offset)
    return img,label


class NFSEN1LC(Dataset):
    """PyTorch dataset class for NFSEN1LC
    Which is short for NICFI + SENTINEL-1 land cover classification
    """

    def __init__(self, 
                 data_dir,
                 bands,
                 usage='train',
                 label_offset=1,
                 chip_buffer=64,
                 sync_transform=None,
                 img_transform=None,
                 label_transform=None,
                 predict_catalog=None,
                 tile_id='1207-996'):
        """Initialize the dataset
        Args:
            data_dir (str): the directory of all data
            bands (list): The id of bands to use. 1:12 for all, 1:8 for NICFI only.
                The default is all.
            usage (str): Usage of the dataset : "train", "validate" or "predict"
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
        self.bands = bands
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
        self.n_channels = len(self.bands)

        # Check inputs
        assert usage in ['train', 'validate', 'predict', "ipredict"]
        assert os.path.exists(data_dir)

        # Read catalog
        if self.usage == 'train':
            catalog_nm = 'dl_catalog_train.csv'
        elif self.usage == 'validate':
            catalog_nm = 'dl_catalog_valid.csv'
        elif self.usage == 'predict':
            catalog_nm = predict_catalog
        elif self.usage == "ipredict":
            catalog_nm = predict_catalog
        else:
            sys.exit('Not valid usage setting.')
        catalog_full = pd.read_csv(os.path.join(self.data_dir, catalog_nm))

        # Read chips for train and validate
        if self.usage in ['train', 'validate', 'ipredict']:
            self.catalog = catalog_full
            # Load all images here before the training to save time
            args = [(index, catalog_full, bands, label_offset, data_dir) for index in range(len(catalog_full))]
            with mp.Pool(processes=35) as pool:
                results = pool.starmap(_read_img_label, args)
            self.img_ls = [result[0] for result in results]
            self.label_ls = [result[1] for result in results]
            del results
            
            # Do more for ipredict
            if self.usage == 'ipredict':
                self.tile_id_ls = [catalog_full.iloc[index]['tile_id'] for index in range(len(catalog_full))]
        # Final prediction
        else:
            self.chip_size = 512 + self.chip_buffer * 2  # image size of train, hardcoded
            self.tile_id = tile_id
            catalog = catalog_full.loc[catalog_full['tile_id'] == self.tile_id]
            if len(catalog.index) == 0:
                sys.exit('No such {} to prediction.'.format(self.tile_id))
            elif len(catalog.index) != 1:
                sys.exit('Duplicate catalog for tile {}.'.format(self.tile_id))
            else:
                catalog['img'][catalog.index[0]] = os.path.join(self.data_dir, catalog['img'][catalog.index[0]])
                self.catalog = catalog.iloc[0]
                img = load_sat_buf(self.catalog['img'], self.bands, self.catalog['tiles_relate'], self.chip_buffer)
                self.meta = get_meta(self.catalog['img'])
                self.img_ls, self.index_ls = get_chips(img, self.chip_size, self.chip_buffer)

    def __getitem__(self, index):
        """Support dataset indexing and apply transformation
        Args:
            index -- Index of each small chips in the dataset
        Returns:
            tuple
        """
        if self.usage in ['train', 'validate', 'ipredict']:
            img = self.img_ls[index]
            label = self.label_ls[index]
            # Transform
            if self.sync_transform is not None:
                img, label = self.sync_transform(img, label)
            if self.img_transform is not None:
                img = self.img_transform(img)
            if self.label_transform is not None:
                label = self.label_transform(label)

            if self.usage == 'train':
                return img, label, index
            elif self.usage == 'ipredict':
                tile_id = self.tile_id_ls[index]
                return img, label, tile_id
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
        elif self.usage in ['train', 'validate', 'ipredict']:
            return len(self.catalog.index)
        else:
            return 0
