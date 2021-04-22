from skimage import transform as trans
import random
import numpy as np
import cv2


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


def center_rotate(img, label, degree):
    """Synthesize new image chips by rotating the input chip around its center.
    Params:
        img (numpy.ndarray): Concatenated variables or brightness value with a dimension of (H, W, C)
        label (numpy.ndarray): Ground truth with a dimension of (H,W)
        degree (tuple or list): Range of degree for rotation
    Returns:
        (numpy.ndarray, numpy.ndarray) tuple of rotated image, and label.
    """

    if isinstance(degree, tuple) or isinstance(degree, list):
        degree = random.uniform(degree[0], degree[1])

    # Get the dimensions of the image (e.g. number of rows and columns).
    h, w, _ = img.shape

    # Determine the image center.
    center = (w // 2, h // 2)

    # Grab the rotation matrix
    rot_mtrx = cv2.getRotationMatrix2D(center, degree, 1.0)

    # perform the actual rotation for both raw and labeled image.
    img = cv2.warpAffine(img, rot_mtrx, (w, h))
    label = cv2.warpAffine(label, rot_mtrx, (w, h))
    label = np.rint(label)

    return img, label


def flip(img, label, ftype):
    """Synthesize new image chips by flipping the input chip around a user defined axis.
    Params:
        img (numpy.ndarray): Concatenated variables or brightness value with a dimension of (H, W, C)
        label (numpy.ndarray): Ground truth with a dimension of (H,W)
        ftype (str): Flip type from ['vflip','hflip','dflip']
    Returns:
        (numpy.ndarray, numpy.ndarray) tuple of flipped image, and label.
    Note:
        Provided transformation are:
            1) 'vflip', vertical flip
            2) 'hflip', horizontal flip
            3) 'dflip', diagonal flip
    """

    def diagonal_flip(img_src):
        flipped = np.flip(img_src, 1)
        flipped = np.flip(flipped, 0)
        return flipped

    # Horizontal flip
    if ftype == 'hflip':
        img = np.flip(img, 0)
        label = np.flip(label, 0)

    # Vertical flip
    elif ftype == 'vflip':
        img = np.flip(img, 1)
        label = np.flip(label, 1)

    # Diagonal flip
    elif ftype == 'dflip':
        img = diagonal_flip(img)
        label = diagonal_flip(label)

    else:
        raise ValueError("Bad flip type")

    return img.copy(), label.copy()


def re_scale(img, label, scale=(0.8, 1.2), rand_resize_crop=False, diff=False, cen_locate=True):
    """Synthesize new image chips by rescaling the input chip.
    Params:
        img (numpy.ndarray): Concatenated variables or brightness value with a dimension of (H, W, C)
        label (numpy.ndarray): Ground truth with a dimension of (H,W)
        scale (tuple or list): Range of scale ratio
        rand_resize_crop (bool): Whether crop the rescaled image chip randomly or at the center
        if the chip is larger than input ones
        diff (bool): Whether change the aspect ratio
        cen_locate (bool): Whether locate the rescaled image chip at the center or a random position
        if the chip is smaller than input
    Returns:
        (numpy.ndarray, numpy.ndarray) tuple of rescaled image, and label.
    """

    h, w, _ = img.shape
    if isinstance(scale, tuple) or isinstance(scale, list):
        resize_h = round(random.uniform(scale[0], scale[1]) * h)
        if diff:
            resize_w = round(random.uniform(scale[0], scale[1]) * w)
        else:
            resize_w = resize_h
    else:
        raise Exception('Wrong scale type!')

    img_re = trans.resize(img, (resize_h, resize_w), preserve_range=True)
    label_re = trans.resize(label, (resize_h, resize_w), preserve_range=True)

    # crop image when length of side is larger than input ones
    if rand_resize_crop:
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
        if cen_locate:
            tl_x = max(0, (h - resize_h) // 2)
            tl_y = max(0, (w - resize_w) // 2)
        else:
            tl_x = random.randint(0, max(0, h - resize_h))
            tl_y = random.randint(0, max(0, w - resize_w))

        # resized result
        img_re, label_re = uni_shape(img_re, label_re, h, tl_x, tl_y)

    return img_re, label_re
