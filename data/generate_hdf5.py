from ast import mod
from pathlib import Path
import sys
import os

import h5py
import nibabel as nib
from nibabel.processing import resample_to_output
import numpy as np
import time
from typing import Dict, Sequence, Tuple
from sklearn.model_selection import train_test_split
from collections import defaultdict

# Ensure project root is on sys.path so `utils` package can be imported when running this file directly
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from utils.gen_logger import loggen
from data_utils import create_weight_mask, get_thick_slices, filter_blank_slices_thick

logger = loggen("generate_hdf5")

class H5pyDataset:
    def __init__(self, params: Dict):
        self.dataset_path = Path(params['dataset_path'])
        self.dataset_name = params['dataset_name']
        self.slice_thickness = params["thickness"]
        self.gt_name = params["gt_name"]
        self.volume_name = params["volume_name"]
        self.max_weight = params["max_weight"]
        self.edge_weight = params["edge_weight"]
        self.gradient = params["gradient"]

        assert self.dataset_path.is_dir(), f"The provided paths are not valid: {self.dataset_path}!"

        self.subjects_dirs = sorted([d for d in self.dataset_path.iterdir() if d.is_dir()])

        self.data_set_size = len(self.subjects_dirs)

    def _load_volumes(self, subject_path: Path):

        vol_path = subject_path / f"{subject_path.name}_{self.volume_name}.nii"
        gt_path = subject_path / f"{subject_path.name}_{self.gt_name}.nii"

        volume_img = nib.load(vol_path)
        gt_img = nib.load(gt_path)

        orig_zooms = volume_img.header.get_zooms()
        orig_shape = volume_img.header.get_data_shape()

        volume_img = nib.as_closest_canonical(volume_img)
        gt_img = nib.as_closest_canonical(gt_img)

        zoom_factors = (4.0, 4.0, 4.0)
        volume_ds = resample_to_output(volume_img, voxel_sizes=zoom_factors)  # linear interpolation
        gt_ds = resample_to_output(gt_img, voxel_sizes=zoom_factors, mode='nearest')  # nearest for labels
        logger.info(f"Original size: {orig_shape}, New size: {volume_ds.shape}")

        # load as float for interpolation
        volume = np.array(volume_ds.get_fdata(), dtype=np.int16)
        gt = np.array(gt_ds.get_fdata(), dtype=np.int16)

        # downsample by factor 2 in each spatial dimension
        try:
            import scipy.ndimage as ndi
        except Exception:
            raise RuntimeError("scipy is required for downsampling (scipy.ndimage).")

        # cast back to original dtypes
        #volume[volume < 0] = 0
        gt[gt < 0] = 0
        volume[volume < 0] = 0
        gt = np.where(gt >= 1, 1, 0).astype(np.int16)

        # update zooms/shape to reflect downsampled data
        new_zooms = tuple(z * 2 for z in orig_zooms)
        new_shape = volume.shape

        return volume, gt, new_zooms, new_shape

    def _load_volumes_fused(self, subject_path: Path, modalities: Sequence[str] = None) -> Tuple[np.ndarray, np.ndarray, Tuple[float, ...], Tuple[int, ...]]:

        subject_name = subject_path.name

        if modalities is None:
            if hasattr(self, "modalities") and self.modalities:
                modalities = list(self.modalities)
            else:
                found = []
                for p in subject_path.iterdir():
                    if not p.is_file():
                        continue
                    name = p.name
                    if not name.startswith(subject_name + "_"):
                        continue
                    stem = name[len(subject_name) + 1:]
                    if stem.endswith(".nii.gz"):
                        stem = stem[:-7]
                    elif stem.endswith(".nii"):
                        stem = stem[:-4]
                    if hasattr(self, "gt_name") and stem == self.gt_name:
                        continue
                    found.append(stem)
                if not found:
                    raise FileNotFoundError(f"No modality files found for subject {subject_name} in {subject_path}")
                modalities = sorted(found)

        if isinstance(modalities, str):
            modalities = [modalities]

        logger.info(f"Loading modalities for {subject_name}: {modalities}")

        zoom_factors = (4.0, 4.0, 4.0)

        resampled_modalities = []
        orig_zooms = None
        orig_shape = None

        for mod in modalities:
            candidates = [
                subject_path / f"{subject_name}_{mod}.nii",
                subject_path / f"{subject_name}_{mod}.nii.gz"
            ]
            vol_path = None
            for c in candidates:
                if c.exists():
                    vol_path = c
                    break
            if vol_path is None:
                raise FileNotFoundError(f"Modality file for '{mod}' not found for subject {subject_name} (looked for {candidates})")

            vol_img = nib.load(str(vol_path))
            if orig_zooms is None:
                orig_zooms = vol_img.header.get_zooms()
                orig_shape = vol_img.header.get_data_shape()

            vol_img = nib.as_closest_canonical(vol_img)
            vol_ds = resample_to_output(vol_img, voxel_sizes=zoom_factors)
            arr = np.array(vol_ds.get_fdata(), dtype=np.float32)
            resampled_modalities.append(arr)
            logger.debug(f"Loaded modality {mod}, resampled shape {arr.shape}")

        if len(resampled_modalities) == 1:
            fused = resampled_modalities[0]
        else:
            stacked = np.stack(resampled_modalities, axis=0)
            fused = np.mean(stacked, axis=0)

        gt_candidates = [
            subject_path / f"{subject_name}_{self.gt_name}.nii",
            subject_path / f"{subject_name}_{self.gt_name}.nii.gz"
        ]
        gt_path = None
        for c in gt_candidates:
            if c.exists():
                gt_path = c
                break
        if gt_path is None:
            raise FileNotFoundError(f"GT file for subject {subject_name} not found (looked for {gt_candidates})")

        gt_img = nib.load(str(gt_path))
        gt_img = nib.as_closest_canonical(gt_img)
        gt_ds = resample_to_output(gt_img, voxel_sizes=zoom_factors, mode="nearest")
        gt = np.array(gt_ds.get_fdata(), dtype=np.int16)

        logger.info(f"Original size: {orig_shape}, resampled size: {fused.shape}")

        fused_ds[fused_ds < 0] = 0
        fused_ds = np.rint(fused_ds).astype(np.int16)

        gt_ds[gt_ds < 0] = 0
        gt_ds = np.where(gt_ds >= 1, 1, 0).astype(np.int16)

        if orig_zooms is None:
            raise RuntimeError("Could not determine original zooms from volume header.")
        new_zooms = tuple(float(z) * 2.0 for z in orig_zooms)
        new_shape = fused_ds.shape

        return fused_ds, gt_ds, new_zooms, new_shape
    
    def create_hdf5(self) -> None:

        data_per_idx = defaultdict(lambda: defaultdict(list))

        for idx, current_subject in enumerate(self.subjects_dirs):
            try:
                
                start_time = time.time()
                logger.info(
                    f"Volume {idx + 1}/{self.data_set_size}: {current_subject.name}"
                )
                volume, gt, zooms, shape = self._load_volumes(current_subject)
                volume = np.moveaxis(volume, [0, 1, 2], [2, 1, 0])
                gt = np.moveaxis(gt, [0, 1, 2], [2, 1, 0])
                logger.info("Unique values in GT: {}".format(np.unique(gt, return_counts=True)))

                size, _, _ = volume.shape

                weight = create_weight_mask(
                    mapped_aseg=gt,
                    max_weight=self.max_weight,
                    max_edge_weight=self.edge_weight,
                    gradient=self.gradient
                )
                weight[weight < 1] = 0.0
                weight = weight / np.max(weight)
                logger.info("Weight statistics - min: {}, max: {}, mean: {}".format(
                    np.min(weight), np.max(weight), np.mean(weight)
                ))
                logger.info(
                    "Created weights with max_w {}, gradient {},"
                    " edge_w {}".format(
                        self.max_weight,
                        self.gradient,
                        self.edge_weight))
                
                volume_thick = get_thick_slices(volume, self.slice_thickness)
                logger.info(
                    "Created thick slices with thickness {} resulting in {}.".format(
                        self.slice_thickness, volume_thick.shape
                    )
                )

                filtered_volume, filtered_gt, filtered_weight = filter_blank_slices_thick(
                    volume_thick, gt, weight
                )
                logger.info(
                    "Filtered blank slices. Original slices: {},"
                    " Remaining slices: {}".format(
                        size, filtered_volume.shape[0]
                    )
                )

                data_per_idx[f"{idx}"]['volume'].extend(filtered_volume)
                data_per_idx[f"{idx}"]['gt'].extend(filtered_gt)
                data_per_idx[f"{idx}"]['weight'].extend(filtered_weight)
                data_per_idx[f"{idx}"]['subject'].extend(
                    current_subject.name.encode("ascii", "ignore")
                )

            except Exception as e:
                logger.error(
                    f"Error processing subject {current_subject.name}: {e}"
                )
                continue
        
        n_folds = 5
        base_seed = 42
        subject_keys = list(data_per_idx.keys())
        print(len(data_per_idx.keys()))

        for fold in range(n_folds):
            seed = base_seed + fold
            train_keys, val_keys = train_test_split(
                subject_keys, test_size=1.0 / n_folds, random_state=seed
            )
            print(f"Fold {fold}: Train keys: {train_keys}, Val keys: {val_keys}")

            def concat_fields(keys_list, field):
                arrays = [data_per_idx[k][field] for k in keys_list if len(data_per_idx[k][field]) > 0]
                if not arrays:
                    return np.array([], dtype=data_per_idx[keys_list[0]][field].dtype)
                return np.concatenate(arrays, axis=0)

            train_vol = concat_fields(train_keys, 'volume')
            train_gt = concat_fields(train_keys, 'gt')
            train_weight = concat_fields(train_keys, 'weight')
            train_subject = concat_fields(train_keys, 'subject')

            val_vol = concat_fields(val_keys, 'volume')
            val_gt = concat_fields(val_keys, 'gt')
            val_weight = concat_fields(val_keys, 'weight')
            val_subject = concat_fields(val_keys, 'subject')

            train_out = f"{self.dataset_name}_fold_{fold}_train.h5"
            val_out = f"{self.dataset_name}_fold_{fold}_val.h5"

            try:
                with h5py.File(train_out, 'w') as hf_tr:
                    group = hf_tr.create_group(f"{fold}")
                    group.create_dataset('volume', data=train_vol, compression='gzip')
                    group.create_dataset('gt', data=train_gt, compression='gzip')
                    group.create_dataset('weight', data=train_weight, compression='gzip')
                    group.create_dataset('subject', data=train_subject, compression='gzip')

                with h5py.File(val_out, 'w') as hf_val:
                    group = hf_val.create_group(f"{fold}")
                    group.create_dataset('volume', data=val_vol, compression='gzip')
                    group.create_dataset('gt', data=val_gt, compression='gzip')
                    group.create_dataset('weight', data=val_weight, compression='gzip')
                    group.create_dataset('subject', data=val_subject, compression='gzip')

                logger.info(
                    f"Wrote fold {fold} -> train: {train_out} ({train_vol.shape[0]} slices), "
                    f"val: {val_out} ({val_vol.shape[0]} slices)"
                )
            except Exception as e:
                logger.error(f"Failed writing HDF5 for fold {fold}: {e}")

        logger.info(
            "Successfully written {} in {:.3f} seconds.".format(
                self.dataset_name, time.time() - start_time
            )
        )



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate HDF5 train/val folds from dataset')
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset directory (subjects)')
    parser.add_argument('--dataset_name', type=str, default='dataset', help='Name prefix for output files')
    parser.add_argument('--thickness', type=int, default=3, help='Slice thickness used when filtering')
    parser.add_argument('--gt_name', type=str, default='gt', help='Ground truth filename suffix (without extension)')
    parser.add_argument('--volume_name', type=str, default='volume', help='Volume filename suffix (without extension)')
    parser.add_argument('--max_weight', type=int, default=5, help='Maximum class weight')
    parser.add_argument('--edge_weight', type=int, default=5, help='Maximum edge weight')
    parser.add_argument('--gradient', action='store_true', default=False, help='Enable gradient weighting')

    args = parser.parse_args()

    dataset_params = {
        "dataset_name": args.dataset_name,
        "dataset_path": args.data_path,
        "thickness": args.thickness,
        "gt_name": args.gt_name,
        "volume_name": args.volume_name,
        "max_weight": args.max_weight,
        "edge_weight": args.edge_weight,
        "gradient": not args.gradient
    }

    dataset = H5pyDataset(dataset_params)
    dataset.create_hdf5()

