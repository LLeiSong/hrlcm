from skimage import transform as trans
import random
import torch
import numpy as np
import cv2


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, label):
        h, w, _ = img.shape
        h2, w2 = label.shape
        assert [h, w] == [h2, w2]
        for t in self.transforms:
            img, label = t(img, label)
        return img, label


def uni_shape(img, label, dsize, tl_x=0, tl_y=0):

    """
    Unify dimension of images and labels to specified data size
    Params:
        img (numpy.ndarray): Concatenated variables or brightness value with a dimension of (H, W, C)
        label (numpy.ndarray): Ground truth with a dimension of (H,W)
        dsize (int): Target data size
        tl_x (int): Vertical offset by pixels
        tl_y (int): Horizontal offset by pixels
    Returns:
        (numpy.ndarray, numpy.ndarray) tuple of shape unified image, and label.
    """

    resize_h, resize_w, c = img.shape

    canvas_img = np.zeros((dsize, dsize, c), dtype=img.dtype)
    canvas_label = np.zeros((dsize, dsize), dtype=label.dtype)

    canvas_img[tl_x:tl_x + resize_h, tl_y:tl_y + resize_w] = img
    canvas_label[tl_x:tl_x + resize_h, tl_y:tl_y + resize_w] = label

    return canvas_img, canvas_label


class RandomCenterRotate(object):
    """Synthesize new image chips by rotating the input chip around its center randomly."""
    def __init__(self, degree=(-90, 90), prob=0.5):
        """Initialize.
        Params:
            degree (tuple or list): Range of degree for rotation
            prob (float): the probability of doing rotate, [0, 1]
        """
        self.prob = prob
        if isinstance(degree, tuple) or isinstance(degree, list):
            self.degree = random.uniform(degree[0], degree[1])

    def __call__(self, img, label):
        """Define the call.
        Params:
            img (numpy.ndarray): Concatenated variables or brightness value with a dimension of (H, W, C)
            label (numpy.ndarray): Ground truth with a dimension of (H,W)
            degree (tuple or list): Range of degree for rotation
        Returns:
            (numpy.ndarray, numpy.ndarray) tuple of rotated image, and label.
        """
        if random.random() < self.prob:
            # Get the dimensions of the image (e.g. number of rows and columns).
            h, w, _ = img.shape

            # Determine the image center.
            center = (w // 2, h // 2)

            # Grab the rotation matrix
            rot_mtrx = cv2.getRotationMatrix2D(center, self.degree, 1.0)

            # perform the actual rotation for both raw and labeled image.
            img = cv2.warpAffine(img, rot_mtrx, (w, h))
            label = cv2.warpAffine(label, rot_mtrx, (w, h))
            label = np.rint(label)

        return img, label


class RandomFlip(object):
    """Synthesize new image chips by vertical or horizontal or diagonal flipping randomly."""
    def __init__(self, prob=0.5):
        """Initialize.
        Params:
            prob (float): the probability of doing flip, [0, 1]
        """
        self.prob = prob
        self.type = random.sample(['vflip', 'hflip', 'dflip'], 1)

    def __call__(self, img, label):
        """Define the call.
        Params:
            img (numpy.ndarray): Concatenated variables or brightness value with a dimension of (H, W, C)
            label (numpy.ndarray): Ground truth with a dimension of (H,W)
        Returns:
            (numpy.ndarray, numpy.ndarray) tuple of flipped image, and label.
        """

        def _diagonal_flip(img_src):
            flipped = np.flip(img_src, 1)
            flipped = np.flip(flipped, 0)
            return flipped

        if random.random() < self.prob:
            # Horizontal flip
            if self.type == 'hflip':
                img = np.flip(img, 0)
                label = np.flip(label, 0)

            # Vertical flip
            elif self.type == 'vflip':
                img = np.flip(img, 1)
                label = np.flip(label, 1)

            # Diagonal flip
            elif self.type == 'dflip':
                img = _diagonal_flip(img)
                label = _diagonal_flip(label)

        return img.copy(), label.copy()


class RandomScale(object):
    """Synthesize new image chips by rescaling the input chip."""
    def __init__(self, scale=(0.5, 1.5), rand_resize_crop=False, diff=False, cen_locate=True, prob=0.5):
        """Initialize.
        Params:
            scale (tuple or list): Range of scale ratio
            rand_resize_crop (bool): Whether crop the rescaled image chip randomly or at the center
                if the chip is larger than input ones
            diff (bool): Whether change the aspect ratio
            cen_locate (bool): Whether locate the rescaled image chip at the center or a random position
                if the chip is smaller than input
            prob (float): the probability of doing flip, [0, 1]
        """
        # Check input
        assert isinstance(scale, tuple) or isinstance(scale, list)
        self.scale = scale
        self.rand_resize_crop = rand_resize_crop
        self.diff = diff
        self.cen_locate = cen_locate
        self.prob = prob

    def __call__(self, img, label):
        """Define the call.
        Params:
            img (numpy.ndarray): Concatenated variables or brightness value with a dimension of (H, W, C)
            label (numpy.ndarray): Ground truth with a dimension of (H,W)
        Returns:
            (numpy.ndarray, numpy.ndarray) tuple of rescaled image, and label.
        """
        if random.random() < self.prob:
            h, w, _ = img.shape
            resize_h = round(random.uniform(self.scale[0], self.scale[1]) * h)
            if self.diff:
                resize_w = round(random.uniform(self.scale[0], self.scale[1]) * w)
            else:
                resize_w = resize_h

            img_re = trans.resize(img, (resize_h, resize_w), preserve_range=True)
            label_re = trans.resize(label, (resize_h, resize_w), preserve_range=True)

            # crop image when length of side is larger than input ones
            if self.rand_resize_crop:
                x_off = random.randint(0, max(0, resize_h - h))
                y_off = random.randint(0, max(0, resize_w - w))
            else:
                x_off = max(0, (resize_h - h) // 2)
                y_off = max(0, (resize_w - w) // 2)

            img_re = img_re[x_off:x_off + min(h, resize_h), y_off:y_off + min(w, resize_w), :]
            label_re = label_re[x_off:x_off + min(h, resize_h), y_off:y_off + min(w, resize_w)]
            label_re = np.rint(label_re)

            # locate image when it is smaller than input
            if resize_h < h or resize_w < w:
                if self.cen_locate:
                    tl_x = max(0, (h - resize_h) // 2)
                    tl_y = max(0, (w - resize_w) // 2)
                else:
                    tl_x = random.randint(0, max(0, h - resize_h))
                    tl_y = random.randint(0, max(0, w - resize_w))

                # resized result
                img_re, label_re = uni_shape(img_re, label_re, h, tl_x, tl_y)

            return img_re, label_re
        else:
            return img, label


class SyncToTensor(object):
    """Convert numpy to tensor."""
    def __call__(self, img, label):
        """Define the call.
        Params:
            img (numpy.ndarray): Concatenated variables or brightness value with a dimension of (H, W, C)
            label (numpy.ndarray or None): Ground truth with a dimension of (H,W)
        Returns:
            (numpy.ndarray, numpy.ndarray or None) tuple of rescaled image, and label.
        """
        label = torch.from_numpy(label).long()
        img = torch.from_numpy(img.transpose((2, 0, 1))).float()

        return img, label


class ImgToTensor(object):
    """Convert numpy to tensor."""
    def __call__(self, img):
        """Define the call.
        Params:
            img (numpy.ndarray): Concatenated variables or brightness value with a dimension of (H, W, C)
            label (numpy.ndarray or None): Ground truth with a dimension of (H,W)
        Returns:
            (numpy.ndarray, numpy.ndarray or None) tuple of rescaled image, and label.
        """
        img = torch.from_numpy(img.transpose((2, 0, 1))).float()

        return img


class LabelToTensor(object):
    """Convert numpy to tensor."""
    def __call__(self, label):
        """Define the call.
        Params:
            img (numpy.ndarray): Concatenated variables or brightness value with a dimension of (H, W, C)
            label (numpy.ndarray or None): Ground truth with a dimension of (H,W)
        Returns:
            (numpy.ndarray, numpy.ndarray or None) tuple of rescaled image, and label.
        """
        label = torch.from_numpy(label).long()

        return label
