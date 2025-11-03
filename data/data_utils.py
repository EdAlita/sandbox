import numpy as np
from numpy import typing as npt
from typing import Tuple

def create_weight_mask(
        mapped_aseg: npt.NDArray,
        max_weight: int = 5,
        max_edge_weight: int = 5,
        gradient: bool = True
) -> np.ndarray:
    """
    Create weighted mask - with median frequency balancing and edge-weighting.

    Parameters
    ----------
    mapped_aseg : np.ndarray
        Segmentation to create weight mask from.
    max_weight : int
        Maximal weight on median weights (cap at this value). (Default value = 5).
    max_edge_weight : int
        Maximal weight on gradient weight (cap at this value). (Default value = 5).
    gradient : bool
        Flag, set to create gradient mask (default = True).

    Returns
    -------
    np.ndarray
        Weights.
    """
    unique, counts = np.unique(mapped_aseg, return_counts=True)

    # Median Frequency Balancing
    class_wise_weights = np.median(counts) / counts
    class_wise_weights[class_wise_weights > max_weight] = max_weight
    (h, w, d) = mapped_aseg.shape

    weights_mask = np.reshape(class_wise_weights[mapped_aseg.ravel()], (h, w, d))

    # Gradient Weighting
    if gradient:
        (gx, gy, gz) = np.gradient(mapped_aseg)
        grad_weight = max_edge_weight * np.asarray(
            np.power(np.power(gx, 2) + np.power(gy, 2) + np.power(gz, 2), 0.5) > 0,
            dtype="float",
        )

        weights_mask += grad_weight

    return weights_mask

def get_thick_slices(
        img_data: npt.NDArray,
        slice_thickness: int = 3
) -> np.ndarray:
    """
    Extract thick slices from the image.

    Feed slice_thickness preceding and succeeding slices to network,
    label only middle one.

    Parameters
    ----------
    img_data : npt.NDArray
        3D MRI image read in with nibabel.
    slice_thickness : int
        Number of slices to stack on top and below slice of interest (default=3).

    Returns
    -------
    np.ndarray
        Image data with the thick slices of the n-th axis appended into the n+1-th axis.
    """
    img_data_pad = np.pad(img_data, ((0, 0), (0, 0), (slice_thickness, slice_thickness)), mode="edge")
    from numpy.lib.stride_tricks import sliding_window_view

    # sliding_window_view will automatically create thick slices through a sliding window, but as this in only a view,
    # less memory copies are required
    return sliding_window_view(img_data_pad, 2 * slice_thickness + 1, axis=2)

def filter_blank_slices_thick(
        img_vol: npt.NDArray,
        label_vol: npt.NDArray,
        weight_vol: npt.NDArray,
        threshold: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter blank slices from the volume using the label volume.

    Parameters
    ----------
    img_vol : npt.NDArray
        Orig image volume.
    label_vol : npt.NDArray
        Label images (ground truth).
    weight_vol : npt.NDArray
        weights.
    threshold : int
        Threshold for number of pixels needed to keep slice (below = dropped). (Default value = 50).

    Returns
    -------
    filtered img_vol : np.ndarray
        [MISSING].
    label_vol : np.ndarray
        [MISSING].
    """
    select_slices = np.sum(label_vol, axis=(1, 2)) > threshold
    non_zero = np.sum(label_vol, axis=(1, 2)) > 0.0
    not_selected = np.sum(label_vol, axis=(1, 2)) < threshold
    num = sum(select_slices)
    no_num = sum(not_selected)
    N, _, _ = label_vol.shape
    while sum(not_selected) != int(num * 0.10 + 1):
        n = np.random.randint(low=0, high=N-1)
        not_selected[n] = False

    select = select_slices + not_selected

    img_vol_out = img_vol[select, :, :, :]
    label_vol_out = label_vol[select, :, :]
    weight_vol = weight_vol[select, :, :]
    return img_vol_out, label_vol_out, weight_vol