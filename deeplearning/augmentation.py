from skimage import transform as trans
import random
import numpy as np
import cv2


def uniShape(img, label, dsize, tlX=0, tlY=0):

    """
    Unify dimension of images and labels to specified data size

    Params:

    img (narray): Concatenated variables or brightness value with a dimension of (H, W, C)
    label (narray): Ground truth with a dimension of (H,W)
    dsize (int): Target data size
    tlX (int): Vertical offset by pixels
    tlY (int): Horizontal offset by pixels

    Returns:

    (narray, narray, narray) tuple of shape unified image, label and mask

    """

    resizeH, resizeW, c = img.shape

    canvas_img = np.zeros((dsize, dsize, c), dtype=img.dtype)
    canvas_label = np.zeros((dsize, dsize), dtype=label.dtype)

    canvas_img[tlX:tlX + resizeH, tlY:tlY + resizeW] = img
    canvas_label[tlX:tlX + resizeH, tlY:tlY + resizeW] = label

    return canvas_img, canvas_label


def centerRotate(img, label, degree):
    """
    Synthesize new image chips by rotating the input chip around its center.

    Args:

    img (narray): Concatenated variables or brightness value with a dimension of (H, W, C)
    label (narray): Ground truth with a dimension of (H,W)
    degree (tuple or list): Range of degree for rotation

    Returns:

    (narray, narray, narray) tuple of rotated image, label and mask

    """

    if isinstance(degree, tuple) or isinstance(degree, list):
        degree = random.uniform(degree[0], degree[1])

    # Get the dimensions of the image (e.g. number of rows and columns).
    h, w, _ = img.shape

    # Determine the image center.
    center = (w // 2, h // 2)

    # Grab the rotation matrix
    rotMtrx = cv2.getRotationMatrix2D(center, degree, 1.0)

    # perform the actual rotation for both raw and labeled image.
    img = cv2.warpAffine(img, rotMtrx, (w, h))
    label = cv2.warpAffine(label, rotMtrx, (w, h))
    label = np.rint(label)

    return img, label


def flip(img, label, ftype):
    """
    Synthesize new image chips by flipping the input chip around a user defined axis.

    Args:

        img (narray): Concatenated variables or brightness value with a dimension of (H, W, C)
        label (narray): Ground truth with a dimension of (H,W)
        ftype (str): Flip type from ['vflip','hflip','dflip']

    Returns:

        (narray, narray, narray) tuple of flipped image, label and mask

    Note:

        Provided transformation are:
            1) 'vflip', vertical flip
            2) 'hflip', horizontal flip
            3) 'dflip', diagonal flip

    """

    def diagonal_flip(img):
        flipped = np.flip(img, 1)
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


def reScale(img, label, scale=(0.8, 1.2), randResizeCrop=False, diff=False, cenLocate=True):
    """
    Synthesize new image chips by rescaling the input chip.

    Params:

        img (narray): Concatenated variables or brightness value with a dimension of (H, W, C)
        label (narray): Ground truth with a dimension of (H,W)
        scale (tuple or list): Range of scale ratio
        randResizeCrop (bool): Whether crop the rescaled image chip randomly or at the center if the chip is larger than inpput ones
        diff (bool): Whether change the aspect ratio
        cenLocate (bool): Whether locate the rescaled image chip at the center or a random position if the chip is smaller than input

    Returns:

        (narray, narray, narray) tuple of rescaled image, label and mask

    """

    h, w, _ = img.shape
    if isinstance(scale, tuple) or isinstance(scale, list):
        resizeH = round(random.uniform(scale[0], scale[1]) * h)
        if diff:
            resizeW = round(random.uniform(scale[0], scale[1]) * w)
        else:
            resizeW = resizeH
    else:
        raise Exception('Wrong scale type!')

    imgRe = trans.resize(img, (resizeH, resizeW), preserve_range=True)
    labelRe = trans.resize(label, (resizeH, resizeW), preserve_range=True)

    # crop image when length of side is larger than input ones
    if randResizeCrop:
        x_off = random.randint(0, max(0, resizeH - h))
        y_off = random.randint(0, max(0, resizeW - w))
    else:
        x_off = max(0, (resizeH - h) // 2)
        y_off = max(0, (resizeW - w) // 2)

    imgRe = imgRe[x_off:x_off + min(h, resizeH), y_off:y_off + min(w, resizeW), :]
    labelRe = labelRe[x_off:x_off + min(h, resizeH), y_off:y_off + min(w, resizeW)]
    labelRe = np.rint(labelRe)

    # locate image when it is smaller than input
    if resizeH < h or resizeW < w:
        if cenLocate:
            tlX = max(0, (h - resizeH) // 2)
            tlY = max(0, (w - resizeW) // 2)
        else:
            tlX = random.randint(0, max(0, h - resizeH))
            tlY = random.randint(0, max(0, w - resizeW))

        # resized result
        imgRe, labelRe = uniShape(imgRe, labelRe, h, tlX, tlY)

    return imgRe, labelRe
