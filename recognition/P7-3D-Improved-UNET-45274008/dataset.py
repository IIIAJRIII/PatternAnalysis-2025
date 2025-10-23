import numpy as np
import nibabel as nib
from tqdm import tqdm


def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    """
    Convert a label array to one-hot channels along a new last axis.
    """
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c][arr == c] = 1
    return res


def load_data_3D(
    imageNames,
    normImage=False,
    categorical=False,
    dtype=np.float32,
    getAffines=False,
    orient=False,
    early_stop=False,
):
    """
    Load medical image data from names (cases list provided) into a single array.

    This function pre-allocates 5D arrays for conv3d to avoid excessive memory usage.

    Args:
        imageNames: list of file paths to nifti images.
        normImage: bool, normalise the image to zero-mean unit-variance.
        categorical: bool, convert labels to one-hot channels.
        dtype: numpy dtype for output arrays. If dtype == np.uint8, assumes labels.
        getAffines: bool, return affines along with images.
        orient: bool, apply orientation/resampling (external `im.applyOrientation` expected).
        early_stop: bool, stop after ~20 cases for quick testing.

    Returns:
        images (and optionally affines if getAffines is True).
    """
    affines = []

    interp = "linear"
    if dtype == np.uint8:  # assume labels
        interp = "nearest"

    # get fixed size from first image
    num = len(imageNames)
    niftiImage = nib.load(imageNames[0])
    if orient:
        # expects an external module `im` with applyOrientation
        niftiImage = im.applyOrientation(niftiImage, interpolation=interp, scale=1)

    first_case = niftiImage.get_fdata(caching="unchanged")
    if len(first_case.shape) == 4:
        first_case = first_case[:, :, :, 0]  # sometimes extra dim, remove

    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, depth, channels = first_case.shape
        images = np.zeros((num, rows, cols, depth, channels), dtype=dtype)
    else:
        rows, cols, depth = first_case.shape
        images = np.zeros((num, rows, cols, depth), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        if orient:
            niftiImage = im.applyOrientation(niftiImage, interpolation=interp, scale=1)

        inImage = niftiImage.get_fdata(caching="unchanged")
        affine = niftiImage.affine

        if len(inImage.shape) == 4:
            inImage = inImage[:, :, :, 0]  # sometimes extra dims

        # clip slices to expected depth
        inImage = inImage[:, :, :depth]
        inImage = inImage.astype(dtype)

        if normImage:
            inImage = (inImage - inImage.mean()) / inImage.std()

        if categorical:
            inImage = to_channels(inImage, dtype=dtype)
            images[
                i,
                : inImage.shape[0],
                : inImage.shape[1],
                : inImage.shape[2],
                : inImage.shape[3],
            ] = inImage  # with pad
        else:
            images[
                i,
                : inImage.shape[0],
                : inImage.shape[1],
                : inImage.shape[2],
            ] = inImage  # with pad

        affines.append(affine)

        if i > 20 and early_stop:
            break

    if getAffines:
        return images, affines
    return images