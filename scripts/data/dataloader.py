import nibabel as nib
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader

from monai.transforms import (
    MapTransform, Compose, LoadImaged, ToTensord, ConcatItemsd, 
    Orientationd, Spacingd, EnsureChannelFirstd,
    ConvertToMultiChannelBasedOnBratsClassesd,
    RandRotate90d, RandFlipd, RandScaleIntensityd, 
    RandShiftIntensityd, RandGaussianNoised, RandBiasFieldd, NormalizeIntensityd
)
from monai.data import Dataset as MonaiDataset

from scripts.utils.vars import DATA_PATH, PATIENT_FOLDER_NAME, STANDARDIZATION_STATS_PATH, BATCH_SIZE
from scripts.utils.utils import writeJSON, readJSON

class CustomSpatialCrop(MapTransform):
    """
    Cuts depth dimension from beginning and end.
    For example, having depth=155, start=5 and end=-6,
        the resulting depth dimension is 155-5-6=144.
    
    Parameters:
        - keys (list): Keys to standardize (e.g., ["modalities"]).
        - start (int): Dimension to start from.
        - end (int): Dimension to end at.
    """
    def __init__(self, keys, start=5, end=-6, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.start = start
        self.end = end

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key in d:
                # crop the spatial dimensions
                d[key] = d[key][:, :, :, self.start:self.end]
        return d

class DepthwiseStandardization(MapTransform):
    def __init__(self, keys, mean, std):
        """
        Custom depth-wise standardization.
        
        Parameters:
            - keys (list): Keys to standardize (e.g., ["modalities"]).
            - mean (np.array): Per-depth mean (channels, depth).
            - std (np.array): Per-depth std (channels, depth).
        """
        super().__init__(keys)
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, data):
        for key in self.keys:
            modality = data[key]  # (channels, width, height, depth)
            standardized = np.zeros_like(modality)
            # standardize each channel and depth
            for c in range(modality.shape[0]):  # channels
                for d in range(modality.shape[3]):  # depth
                    slice_mean = self.mean[c, d]
                    slice_std = self.std[c, d]
                    if slice_std > 0:  # to avoid division by zero
                        standardized[c, :, :, d] = (modality[c, :, :, d] - slice_mean) / slice_std
                    else:
                        standardized[c, :, :, d] = modality[c, :, :, d] - slice_mean
            data[key] = standardized
        return data

def computeStandardizationStats(patient_ids: list, modalities: list, save_path: str):
    """
    Compute the per-channel and per-depth mean and standard deviation from the training dataset.

    Parameters:
        - patient_ids (list): List of patient IDs in the training set.
        - modalities (list): List of modalities to include as channels (e.g., ["t1", "t2", "flair"]).
        - save_path (str): Path where to save the stats.

    Returns:
        - mean (np.array): Per-channel and per-depth mean (channels, depth).
        - std (np.array): Per-channel and per-depth std (channels, depth).
    """
    num_depths = None  # will be determined after loading the first image
    # initialize accumulators for sums and squared sums
    sum_data = None
    sumsq_data = None
    total_voxels = None
    # iterate through each patient modalities image
    for pid in tqdm(patient_ids, desc="Processing patients data"):
        patient_modalities = []
        for modality in modalities:
            # define the path to the modality file
            modality_file = f"{DATA_PATH}/{PATIENT_FOLDER_NAME}{pid}/{PATIENT_FOLDER_NAME}{pid}_{modality}.nii"
            # load the modality image (width, height, depth)
            modality_img = nib.load(modality_file).get_fdata()
            patient_modalities.append(modality_img)
        # stack modalities along the channel axis (channels, width, height, depth)
        patient_channels = np.stack(patient_modalities, axis=0)
        channels, width, height, depth = patient_channels.shape
        if num_depths is None:
            num_depths = depth
            # initialize accumulators based on the dimensions
            sum_data = np.zeros((channels, num_depths), dtype=np.float64)
            sumsq_data = np.zeros((channels, num_depths), dtype=np.float64)
            total_voxels = np.zeros((channels, num_depths), dtype=np.int64)
        # process each depth slice individually
        for z in range(num_depths):
            # extract the slice for all channels (channels, width, height)
            slice_data = patient_channels[:, :, :, z]
            # compute sum and sum of squares over width and height dimensions
            sum_data[:, z] += slice_data.sum(axis=(1, 2))
            sumsq_data[:, z] += np.square(slice_data).sum(axis=(1, 2))
            # update the total number of voxels
            num_voxels = slice_data.shape[1] * slice_data.shape[2]
            total_voxels[:, z] += num_voxels
    # compute mean and standard deviation
    mean = sum_data / total_voxels
    variance = (sumsq_data / total_voxels) - np.square(mean)
    std = np.sqrt(variance)
    # prepare the statistics dictionary
    stats = {"mean": mean.tolist(), "std": std.tolist()}
    # save the statistics to a JSON file and return them
    writeJSON(stats, save_path)
    return mean, std

def loadStandardizationStats(patient_ids: list, modalities: list):
    """
    Load global mean and std from the file.
    If file does not exist, compute global mean and std values using computeStandardizationStats() function.

    Parameters:
        - patient_ids (list): List of patient IDs in the training set.
        - modalities (list): List of modalities to include as channels (e.g., ["t1", "t2", "flair"]).

    Returns:
        - mean (float): Per-channel and per-depth mean.
        - std (float): Per-channel and per-depth std.
    """
    standardization_stats_modalities_path = STANDARDIZATION_STATS_PATH.split(".")[0]
    for modality in modalities:
        standardization_stats_modalities_path += "_" + modality
    standardization_stats_modalities_path += ".txt"
    try:
        stats = readJSON(standardization_stats_modalities_path)
        mean, std = stats["mean"], stats["std"]
    except FileNotFoundError:
        mean, std = computeStandardizationStats(patient_ids, modalities, standardization_stats_modalities_path)
    return mean, std

def defineTransforms(modalities: list, train: bool = True):
    """
    Defines data transformations for MONAI dataset.

    Parameters:
        - modalities (list): List of modalities to include as channels (e.g., ["t1", "t2", "flair"]).
        - train (bool): Whether to define transformations for training or validation/testing dataloaders.
    
    Returns:
        - transforms (monai.transforms.Compose): Data transformations.
    """
    if train:
        transforms = Compose([
            LoadImaged(keys=[f"modality_{i}" for i in range(len(modalities))] + ["mask"]), # load images
            EnsureChannelFirstd(keys=[f"modality_{i}" for i in range(len(modalities))]), # ensure channels first
            ConvertToMultiChannelBasedOnBratsClassesd(keys="mask"), # to 3 channels for mask (3 classes): necrotic: 1, edema: 2, enhancing: 4
            ConcatItemsd(keys=[f"modality_{i}" for i in range(len(modalities))], name="modalities", dim=0),
            Orientationd(keys=["modalities", "mask"], axcodes="RAS"), # standardize orientation to RAS
            Spacingd(
                keys=["modalities", "mask"],
                pixdim=(1.0, 1.0, 1.0), # resample to 1mm isotropic spacing
                mode=("bilinear", "nearest"), # use bilinear for images, nearest for labels
            ),
            RandRotate90d(keys=["modalities", "mask"], prob=0.5), # random rotation
            RandFlipd(keys=["modalities", "mask"], prob=0.5, spatial_axis=1), # random flip 
            RandFlipd(keys=["modalities", "mask"], prob=0.5, spatial_axis=2), # random flip 
            RandScaleIntensityd(keys="modalities", factors=0.1, prob=0.25), # scale pixel intensity
            RandShiftIntensityd(keys="modalities", offsets=0.1, prob=0.25), # shift pixel intensity 
            RandGaussianNoised(keys="modalities", prob=0.25), # gaussian noise
            RandBiasFieldd(keys="modalities", prob=0.5, degree=2),  # bias field augmentation
            NormalizeIntensityd(keys="modalities", nonzero=True, channel_wise=True), # normalization
            CustomSpatialCrop(keys=["modalities", "mask"], start=5, end=-6), # remove first 5 and last 6 dimensions from depth to ensure compatibility with MONAI UNet model
            ToTensord(keys=["modalities", "mask"]) # convert to tensors
        ])
    else:
        transforms = Compose([
            LoadImaged(keys=[f"modality_{i}" for i in range(len(modalities))] + ["mask"]), # load images
            EnsureChannelFirstd(keys=[f"modality_{i}" for i in range(len(modalities))]), # ensure channels first
            ConvertToMultiChannelBasedOnBratsClassesd(keys="mask"), # to 3 channels for mask (3 classes): necrotic: 1, edema: 2, enhancing: 4
            ConcatItemsd(keys=[f"modality_{i}" for i in range(len(modalities))], name="modalities", dim=0),
            Orientationd(keys=["modalities", "mask"], axcodes="RAS"), # standardize orientation to RAS
            Spacingd(
                keys=["modalities", "mask"],
                pixdim=(1.0, 1.0, 1.0), # resample to 1mm isotropic spacing
                mode=("bilinear", "nearest"), # use bilinear for images, nearest for labels
            ),
            NormalizeIntensityd(keys="modalities", nonzero=True, channel_wise=True), # normalization
            CustomSpatialCrop(keys=["modalities", "mask"], start=5, end=-6), # remove first 5 and last 6 dimensions from depth to ensure compatibility with MONAI UNet model
            ToTensord(keys=["modalities", "mask"]) # convert to tensors
        ])
    return transforms

def createDataloader(patient_ids: list, modalities: list, batch_size: int = BATCH_SIZE, train: bool = True):
    """
    Creates a PyTorch DataLoader for the BraTS dataset with standardization.
    Defines separate configurations for training (with augmentations) and validation/test (no augmentations).
    For possible transformations see https://docs.monai.io/en/latest/transforms.html.

    Parameters:
        - patient_ids (list): List of patient IDs to include in the dataloader.
        - modalities (list): List of modalities to include as channels (e.g., ["t1", "t2", "flair"]).
        - batch_size (int): Batch size for the dataloader.
        - train (bool): Train or validation/test set.

    Returns:
        - dataloader (torch.utils.data.DataLoader): Configured dataloader for the given dataset.
    """
    # prepare data in MONAI format
    data_dicts = [
        {
            f"modality_{i}": f"{DATA_PATH}/{PATIENT_FOLDER_NAME}{pid}/{PATIENT_FOLDER_NAME}{pid}_{modality}.nii.gz"
            for i, modality in enumerate(modalities)
        } | {
            "mask": f"{DATA_PATH}/{PATIENT_FOLDER_NAME}{pid}/{PATIENT_FOLDER_NAME}{pid}_seg.nii.gz"
        }
        for pid in patient_ids
    ]
    # load mean and std for each modality and depth
    # mean, std = loadStandardizationStats(patient_ids=patient_ids, modalities=modalities)
    # define data augmentation and transforms
    transforms = defineTransforms(modalities=modalities, train=train)
    # create MONAI dataset and DataLoader, return DataLoader
    monai_dataset = MonaiDataset(data=data_dicts, transform=transforms)
    dataloader = DataLoader(monai_dataset, batch_size=batch_size, shuffle=train)
    return dataloader
